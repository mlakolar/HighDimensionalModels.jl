# Which tests and CI jobs a change needs (test/impact.toml, test/testsuite.toml), and the rule of
# the CI gate. Plain Julia: TOML and the git command line, no package environment.
#
#   julia scripts/ci_plan.jl                        local changes against origin/master
#   julia scripts/ci_plan.jl --base=REV --head=REV  a range of commits
#   julia scripts/ci_plan.jl --event=pull_request   in CI (see .github/workflows/CI.yml)
#   julia scripts/ci_plan.jl --gate                 the gate job: reads the results from ENV
#
# It prints the plan and its reasons. With --github-output and --summary it also writes the CI
# outputs and a Markdown job summary. When the changes or the tracked files cannot be determined
# it plans the full suite; it never plans "no tests" because something failed. An unclassified test
# file, an invalid map, or a source file whose rules select no tests is an error.
module CIPlan

using TOML

include(joinpath(@__DIR__, "..", "test", "testsuite.jl"))
using .TestSuite

const LEVELS = ("none", "focused", "full")
const JOBS = ("test", "test-cross", "benchmark-integrity")
# A full plan runs these; test-cross (macOS, Windows) only for the nightly and manual runs.
const FULL_JOBS = ["test", "benchmark-integrity"]
const ROOT = normpath(joinpath(@__DIR__, ".."))

struct Rule
  paths::Vector{String}
  level::String
  families::Vector{String}
  tests::Vector{String}
  jobs::Vector{String}
  reason::String
end

struct Impact
  smoke::Vector{String}
  source::Vector{String}
  families::Dict{String,Vector{String}}
  rules::Vector{Rule}
end

"A changed path: `status` is A, M, D or R (renamed: `path` is the new path, `old` the old one)."
struct Change
  status::Char
  path::String
  old::String
end
Change(status::Char, path::AbstractString) = Change(status, String(path), "")

struct Plan
  level::String
  "Test names, or `nothing` for every test file."
  tests::Union{Nothing,Vector{String}}
  jobs::Vector{String}
  "(path, reason) for every changed path."
  reasons::Vector{Tuple{String,String}}
  notes::Vector{String}
  coverage::Bool
end

# `*` matches within a path segment, `**` across segments.
function glob_regex(glob::AbstractString)
  parts = split(glob, "**")
  inner = map(p -> join(map(s -> replace(s, r"[\\^$.|?+()\[\]{}]" => x -> "\\" * x), split(p, '*')), "[^/]*"), parts)
  return Regex("^" * join(inner, ".*") * "\$")
end

matches(glob, path) = occursin(glob_regex(glob), path)

levelrank(level) = something(findfirst(==(level), LEVELS), 0)
maxlevel(a, b) = levelrank(a) >= levelrank(b) ? a : b

strings(d, key, where) = begin
  v = get(d, key, String[])
  v isa Vector && all(x -> x isa String, v) || error("$where: `$key` must be a list of strings")
  String.(v)
end

"Read test/impact.toml and check it against the inventory; throws on any inconsistency."
function load_impact(path::AbstractString, inventory::TestSuite.Inventory)
  data = TOML.parsefile(path)
  unknown = setdiff(keys(data), ("smoke", "source", "families", "rules", "miss"))
  isempty(unknown) || error("$path: unknown keys $(join(sort!(collect(unknown)), ", "))")
  families = get(data, "families", Dict{String,Any}())
  impact = Impact(strings(data, "smoke", path), strings(data, "source", path),
                  Dict(String(k) => strings(families, k, "$path [families]") for k in keys(families)),
                  [rule(r, "$path rule $i") for (i, r) in enumerate(get(data, "rules", Any[]))])
  return validate(impact, inventory, path)
end

function rule(r, where)
  r isa Dict || error("$where: not a table")
  unknown = setdiff(keys(r), ("paths", "level", "families", "tests", "jobs", "reason"))
  isempty(unknown) || error("$where: unknown keys $(join(sort!(collect(unknown)), ", "))")
  level = get(r, "level", "")
  level in LEVELS || error("$where: level must be one of $(join(LEVELS, ", ")), not `$level`")
  reason = get(r, "reason", "")
  reason isa String && !isempty(reason) || error("$where: needs a reason")
  paths = strings(r, "paths", where)
  isempty(paths) && error("$where: needs paths")
  Rule(paths, level, strings(r, "families", where), strings(r, "tests", where), strings(r, "jobs", where), reason)
end

function validate(impact::Impact, inventory::TestSuite.Inventory, where)
  tests = Set(TestSuite.alltests(inventory))
  problems = String[]
  known(name, ctx) = name in tests || push!(problems, "$ctx: no test file `$name` in test/testsuite.toml")
  foreach(t -> known(t, "smoke"), impact.smoke)
  foreach(t -> known(t, "source"), impact.source)
  isempty(impact.smoke) && push!(problems, "smoke must name at least one test file")
  for (family, names) in impact.families
    isempty(names) && push!(problems, "family $family is empty")
    foreach(t -> known(t, "family $family"), names)
  end
  for r in impact.rules
    ctx = "rule for $(join(r.paths, ", "))"
    foreach(f -> haskey(impact.families, f) || push!(problems, "$ctx: no family `$f`"), r.families)
    foreach(t -> known(t, ctx), r.tests)
    foreach(j -> j in JOBS || push!(problems, "$ctx: no CI job `$j`"), r.jobs)
    if r.level in ("none", "full") && !(isempty(r.families) && isempty(r.tests))
      push!(problems, "$ctx: a $(r.level) rule cannot also select families or tests")
    end
  end
  # Every test can be reached from a source change (or runs with every plan).
  reachable = union(Set(impact.smoke), Set(impact.source), Set(Iterators.flatten(values(impact.families))))
  for name in sort!(collect(setdiff(tests, reachable)))
    push!(problems, "test `$name` is in no family, smoke or source list")
  end
  isempty(problems) || error("$where is inconsistent:\n  " * join(problems, "\n  "))
  return impact
end

ispackagecode(path) = startswith(path, "src/")

"""
    source_problems(impact, files) -> Vector{String}

One message for each source file in `files` whose rules, taken together, do not
select its tests. The rules matching a file must include a full rule, or focused rules naming at
least one family or test between them; a rule that only adds jobs may accompany them.
A none rule never covers package code. Otherwise a change to the file would run only the smoke
test and the package-wide checks.
"""
function source_problems(impact::Impact, files)
  problems = String[]
  for f in files
    ispackagecode(f) || continue
    rules = [r for r in impact.rules if r.level != "none" && any(g -> matches(g, f), r.paths)]
    if isempty(rules)
      push!(problems, "$f has no rule: add one naming the families or tests that exercise it")
    elseif !any(r -> r.level == "full" || !isempty(r.families) || !isempty(r.tests), rules)
      push!(problems, "$f: its rules select no families or tests, so a change would run only the smoke and package-wide checks")
    end
  end
  return problems
end
testname(path) = (m = match(r"^test/(.+)\.jl$", path)) === nothing ? nothing : String(m[1])

"""
    plan(changes, impact, inventory)

The plan for a list of changed paths: the union of the rules matching each path. A path that
no rule matches, a deleted or renamed source or test file, and an empty change set select the
full suite.
"""
function plan(changes::Vector{Change}, impact::Impact, inventory::TestSuite.Inventory)
  alltests = Set(TestSuite.alltests(inventory))
  level = "none"
  tests = Set{String}()
  jobs = Set{String}()
  reasons = Tuple{String,String}[]
  notes = String[]
  if isempty(changes)
    level = "full"
    push!(reasons, ("(no changed files)", "The change set is empty; the full suite runs rather than none."))
  end
  for c in changes
    paths = c.status == 'R' ? [c.old, c.path] : [c.path]
    for (k, path) in enumerate(paths)
      deleted = c.status == 'D' || (c.status == 'R' && k == 1)
      matched = [r for r in impact.rules if any(g -> matches(g, path), r.paths)]
      why = String[]
      if deleted && (ispackagecode(path) || testname(path) !== nothing)
        level = "full"
        push!(why, "$(c.status == 'R' ? "Renamed" : "Deleted") source or test file: the full suite runs.")
      elseif isempty(matched)
        level = "full"
        push!(why, "No rule in test/impact.toml matches this path: the full suite runs. Add a rule.")
      end
      for r in matched
        level = maxlevel(level, r.level)
        for f in r.families
          union!(tests, impact.families[f])
        end
        union!(tests, r.tests)
        union!(jobs, r.jobs)
        push!(why, r.reason)
      end
      push!(reasons, (path, join(why, " ")))
    end
  end
  if any(c -> ispackagecode(c.path), changes) && !any(c -> startswith(c.path, "test/"), changes)
    push!(notes, "Package code changed without a test change: say in the PR how the change is tested.")
  end
  return finish(level, tests, jobs, reasons, notes, impact, changes, alltests)
end

function finish(level, tests, jobs, reasons, notes, impact, changes, alltests)
  if level == "none"
    return Plan("none", String[], sort!(collect(jobs)), reasons, notes, false)
  elseif level == "full"
    return Plan("full", nothing, sort!(collect(union(Set(jobs), FULL_JOBS))), reasons, notes, true)
  end
  union!(tests, impact.smoke)
  any(c -> ispackagecode(c.path) || (c.status == 'R' && ispackagecode(c.old)), changes) && union!(tests, impact.source)
  push!(jobs, "test")
  selected = sort!(collect(tests))
  # A selection that covers every file is the full suite.
  Set(selected) == alltests && return Plan(level, nothing, sort!(collect(jobs)), reasons, notes, true)
  return Plan(level, selected, sort!(collect(jobs)), reasons, notes, false)
end

"The plan of the nightly and manual runs: the complete suite and every job, on every platform."
function full_plan(why::AbstractString; cross::Bool = true)
  jobs = sort!(union(FULL_JOBS, cross ? ["test-cross"] : String[]))
  Plan("full", nothing, jobs, [("(event)", String(why))], String[], true)
end

# ---- git -----------------------------------------------------------------------------------

"The output of a git command, or `nothing` if it fails or git cannot be started."
function git(args::AbstractVector{<:AbstractString}; dir = ROOT)
  out = IOBuffer()
  ok = try
    success(pipeline(Cmd(`git -C $dir $args`; ignorestatus = true); stdout = out, stderr = devnull))
  catch err
    err isa Base.IOError || rethrow()
    false
  end
  return ok ? String(take!(out)) : nothing
end

"The tracked files of the repository at `dir`, or `nothing` if git cannot list them."
function tracked_files(dir = ROOT)
  text = git(["ls-files"]; dir)
  return text === nothing ? nothing : String.(split(text, '\n'; keepempty = false))
end

"Parse `git diff --name-status -M` output; `nothing` if a line is malformed."
function parse_name_status(text::AbstractString)
  changes = Change[]
  for line in split(text, '\n'; keepempty = false)
    fields = split(line, '\t')
    status = isempty(fields[1]) ? ' ' : fields[1][1]
    if status in ('R', 'C') && length(fields) == 3
      push!(changes, status == 'R' ? Change('R', fields[3], fields[2]) : Change('A', fields[3]))
    elseif status in ('A', 'M', 'D', 'T') && length(fields) == 2
      push!(changes, Change(status == 'T' ? 'M' : status, fields[2]))
    else
      return nothing
    end
  end
  return changes
end

"The changes from `base` to `head` (a commit, or `nothing` for the working tree with untracked files)."
function changed_files(base::AbstractString, head::Union{Nothing,AbstractString}; dir = ROOT)
  occursin(r"^[0]+$", base) && return nothing
  git(["rev-parse", "--verify", "--quiet", base * "^{commit}"]; dir) === nothing && return nothing
  range = head === nothing ? [base] : [base, head]
  text = git(["diff", "--no-renames", "--name-status", range...]; dir)
  # Renames detected separately so a rename is never mistaken for an edit.
  renames = git(["diff", "-M", "--diff-filter=R", "--name-status", range...]; dir)
  (text === nothing || renames === nothing) && return nothing
  changes = parse_name_status(renames)
  plain = parse_name_status(text)
  (changes === nothing || plain === nothing) && return nothing
  renamed = Set(Iterators.flatten((c.old, c.path) for c in changes))
  append!(changes, filter(c -> !(c.path in renamed), plain))
  if head === nothing
    untracked = git(["ls-files", "--others", "--exclude-standard"]; dir)
    untracked === nothing && return nothing
    append!(changes, [Change('A', p) for p in split(untracked, '\n'; keepempty = false)])
  end
  return changes
end

# ---- output --------------------------------------------------------------------------------

function describe(io::IO, p::Plan; markdown::Bool = false)
  b(s) = markdown ? "**$s**" : s
  ntests = p.tests === nothing ? "every test file" : isempty(p.tests) ? "no package tests" : "$(length(p.tests)) test files"
  println(io, markdown ? "### Test plan: $(p.level)" : "Test plan: $(p.level) ($ntests)")
  markdown && println(io, "\n$(uppercasefirst(ntests)); CI jobs: $(isempty(p.jobs) ? "none" : join(p.jobs, ", ")).\n")
  markdown || println(io, "CI jobs: ", isempty(p.jobs) ? "none" : join(p.jobs, ", "))
  if p.tests !== nothing && !isempty(p.tests)
    println(io, markdown ? "$(b("Selected:")) `$(join(p.tests, "`, `"))`\n" : "Selected: $(join(p.tests, " "))")
  end
  for note in p.notes
    println(io, markdown ? "> [!WARNING]\n> $note\n" : "Note: $note")
  end
  println(io, markdown ? "| Changed | Reason |\n|---|---|" : "Changed:")
  for (path, why) in p.reasons
    println(io, markdown ? "| `$path` | $why |" : "  $path\n      $why")
  end
end

"The sentinel that stands for an empty job or test list in the outputs, so that a missing output differs from an empty one."
const NONE = "none"

function write_outputs(path, p::Plan)
  open(path, "a") do io
    println(io, "level=", p.level)
    println(io, "tests=", p.tests === nothing ? "all" : isempty(p.tests) ? NONE : join(p.tests, ","))
    println(io, "jobs=", isempty(p.jobs) ? NONE : join(p.jobs, " "))
    println(io, "coverage=", p.coverage)
    println(io, "ok=true")
  end
end

# ---- command line --------------------------------------------------------------------------

function option(args, name, default = nothing)
  for a in args
    startswith(a, "--$name=") && return String(split(a, '='; limit = 2)[2])
  end
  return default
end

const EVENTS = ("local", "pull_request", "push", "schedule", "workflow_dispatch")

"""
    compute(args; dir, list_files = tracked_files) -> Plan

`--event=` local (default), pull_request, push, schedule or workflow_dispatch; `--base=` and
`--head=` the revisions to compare (the default for a local run is the merge base with
origin/master and the working tree). A schedule or workflow_dispatch event plans the complete
suite on every platform without looking at changes.

Throws if a test file is unclassified, the map is invalid, or a source file's rules select no
tests. If `list_files(dir)` returns `nothing` (git cannot list the tracked files), that last
check cannot be made and the plan is the full suite.
"""
function compute(args; dir = ROOT, list_files = tracked_files)
  event = option(args, "event", "local")
  event in EVENTS || error("--event must be one of $(join(EVENTS, ", ")), not `$event`")
  inventory = TestSuite.load(joinpath(dir, "test", "testsuite.toml"))
  TestSuite.check(inventory, TestSuite.discover(joinpath(dir, "test")))
  impact = load_impact(joinpath(dir, "test", "impact.toml"), inventory)
  files = list_files(dir)
  if files === nothing
    return audit_failed(plan_event(event, args, impact, inventory, dir))
  end
  problems = source_problems(impact, files)
  isempty(problems) || error("test/impact.toml does not select tests for every source file:\n  " * join(problems, "\n  "))
  return plan_event(event, args, impact, inventory, dir)
end

const AUDIT_FAILED = "git could not list the tracked files, so the source files were not checked against test/impact.toml: the full suite runs."

"`p` widened to the full suite because the source-map check could not be made."
function audit_failed(p::Plan)
  jobs = sort!(union(p.jobs, FULL_JOBS))
  Plan("full", nothing, jobs, vcat([("(source map check)", AUDIT_FAILED)], p.reasons), vcat([AUDIT_FAILED], p.notes), true)
end

function plan_event(event, args, impact, inventory, dir)
  event == "schedule" && return full_plan("The nightly run tests every file on every platform.")
  event == "workflow_dispatch" && return full_plan("A manual run tests every file on every platform.")
  base = option(args, "base", event == "local" ? something(git(["merge-base", "origin/master", "HEAD"]; dir), "origin/master") : nothing)
  head = option(args, "head", event == "local" ? nothing : "HEAD")
  head = head == "" ? nothing : head
  base === nothing && return full_plan("No base revision to compare with: the full suite runs."; cross = false)
  changes = changed_files(strip(base), head === nothing ? nothing : strip(head); dir)
  if changes === nothing
    return full_plan("The changed files could not be determined (base `$base`): the full suite runs."; cross = false)
  end
  return plan(changes, impact, inventory)
end

# ---- gate ----------------------------------------------------------------------------------

const RESULT_WORDS = ("success", "failure", "cancelled", "skipped")

"""
    parse_results(text) -> (results, problems)

The `job=result` words of the gate's RESULTS. A word that is not `job=result`, a result that is
not one of `RESULT_WORDS`, and a job that appears twice are problems.
"""
function parse_results(text::AbstractString)
  results = Dict{String,String}()
  problems = String[]
  for word in split(text)
    m = match(r"^([^=]+)=([^=]+)$", word)
    if m === nothing
      push!(problems, "the result `$word` is not of the form job=result.")
      continue
    end
    job, result = String(m[1]), String(m[2])
    result in RESULT_WORDS || push!(problems, "the result of $job is `$result`, not one of $(join(RESULT_WORDS, ", ")).")
    haskey(results, job) && push!(problems, "the result of $job is given twice.")
    results[job] = result
  end
  return results, problems
end

"The names of the test files of `test/testsuite.toml`: the selections a plan can name."
inventory_names(dir = ROOT) = TestSuite.alltests(TestSuite.load(joinpath(dir, "test", "testsuite.toml")))

"""
    plan_problems(level, tests, jobs, names = inventory_names()) -> (required, problems)

The jobs a plan requires, from its outputs `level`, `tests` and `jobs` as `write_outputs` writes
them, and what is wrong with them. The valid outputs are the combinations that `plan`,
`finish` and `full_plan` emit, in these terms (`names` are the inventory's test names):

| output | syntax |
|---|---|
| `level` | one of `LEVELS` |
| `tests` | `none`, `all`, or distinct names from `names` joined by `,` without empty items or spaces |
| `jobs` | `none`, or distinct names from `JOBS` joined by single spaces (no leading or trailing space) |

| level | `tests` | `jobs` |
|---|---|---|
| `none` | `none` | `none`, or jobs without `test` and `test-cross` |
| `focused` | `all` or names | contain `test` |
| `full` | `all` | contain every job of `FULL_JOBS` |

A plan that requires `test-cross` requires `test` too. Anything else is a problem, among them an
empty output, a name that is not in the inventory, a repeated name, and a selection combined
with the wrong level or job list.
"""
function plan_problems(level::AbstractString, tests::AbstractString, jobs::AbstractString,
                       names::AbstractVector{<:AbstractString} = inventory_names())
  problems = String[]
  bad(msg) = push!(problems, msg)
  isempty(level) ? bad("the plan's level is missing or empty.") :
    level in LEVELS || bad("the plan level is `$level`, not one of $(join(LEVELS, ", ")).")
  any(in((NONE, "all")), names) && bad("the inventory has a test named `$NONE` or `all`, which the plan outputs use as keywords.")
  # The selection: none, all, or a list of distinct inventory names.
  if isempty(tests)
    bad("the plan's test selection is missing or empty.")
  elseif tests != NONE && tests != "all"
    items = split(tests, ','; keepempty = true)
    for item in items
      item in names || bad("the plan's test selection has `$item`, which is neither `$NONE`, `all` nor a test file of the inventory (items are separated by `,` without spaces).")
    end
    allunique(items) || bad("the plan's test selection names a test twice.")
  end
  # The job list: none, or distinct jobs separated by single spaces.
  required = String[]
  words = split(jobs)
  if isempty(words)
    bad("the plan's job list is missing or empty; a plan without jobs says `$NONE`.")
  elseif jobs != NONE
    join(words, " ") == jobs || bad("the plan's job list `$jobs` is not `$NONE` or single-space separated names without leading or trailing space.")
    for j in words
      j in JOBS || bad("the plan names an unknown job `$j`.")
    end
    allunique(words) || bad("the plan names a job twice.")
    required = String.(words)
  end
  if isempty(problems)
    hastest = "test" in required
    if level == "none"
      tests == NONE || bad("a `none` plan selects tests (`$tests`).")
      (hastest || "test-cross" in required) && bad("a `none` plan requires the `test` or `test-cross` job.")
    elseif level == "focused"
      tests != NONE || bad("a `focused` plan selects no tests.")
      hastest || bad("a `focused` plan lacks the `test` job.")
    else
      tests == "all" || bad("a `full` plan selects `$tests`, not `all`.")
      for j in FULL_JOBS
        j in required || bad("a `full` plan lacks the `$j` job.")
      end
    end
    "test-cross" in required && !hastest && bad("the plan requires test-cross without the `test` job.")
  end
  return required, problems
end

"""
    verdict(plan_result, plan_ok, plan_level, plan_tests, plan_jobs, results; problems = String[], names = inventory_names()) -> (passed, lines)

The required check of a CI run. It passes only if the plan job succeeded and wrote its outputs,
the plan's level, selection and jobs are one of the combinations the planner emits (the table of
`plan_problems`), every job the plan required
succeeded, and every other job was skipped: a job that ran without being required is a mismatch
between the plan and the workflow. `results` maps each CI job to its `needs.<job>.result`; a job
missing from it fails the check, and so does any entry of `problems` (from `parse_results`).
"""
function verdict(plan_result::AbstractString, plan_ok::AbstractString, plan_level::AbstractString,
                 plan_tests::AbstractString, plan_jobs::AbstractString, results::AbstractDict;
                 problems::AbstractVector{<:AbstractString} = String[],
                 names::AbstractVector{<:AbstractString} = inventory_names())
  lines = String[]
  fail(msg) = push!(lines, "FAIL: " * msg)
  plan_result == "success" || fail("the plan job ended with `$(isempty(plan_result) ? "no result" : plan_result)`.")
  plan_ok == "true" || fail("the plan did not write a complete plan.")
  required, plan_issues = plan_problems(plan_level, plan_tests, plan_jobs, names)
  foreach(fail, plan_issues)
  foreach(fail, problems)
  for j in JOBS
    r = get(results, j, "")
    if j in required
      r == "success" ? push!(lines, "ok: $j (required) succeeded.") : fail("$j is required, but its result is `$(isempty(r) ? "missing" : r)`.")
    elseif r == "skipped"
      push!(lines, "ok: $j (not required) was skipped.")
    else
      fail("$j was not required and should have been skipped, but its result is `$(isempty(r) ? "missing" : r)`.")
    end
  end
  for j in sort!(collect(keys(results)))
    j in JOBS || fail("unexpected job `$j` in the results: update CIPlan.JOBS.")
  end
  return !any(l -> startswith(l, "FAIL"), lines), lines
end

"The gate job: reads PLAN_RESULT, PLAN_OK, PLAN_LEVEL, PLAN_TESTS, PLAN_JOBS and RESULTS (`job=result` words) from ENV."
function gate()
  results, problems = parse_results(get(ENV, "RESULTS", ""))
  passed, lines = verdict(get(ENV, "PLAN_RESULT", ""), get(ENV, "PLAN_OK", ""), get(ENV, "PLAN_LEVEL", ""),
                          get(ENV, "PLAN_TESTS", ""), get(ENV, "PLAN_JOBS", ""), results; problems)
  foreach(println, lines)
  println(passed ? "Every job the plan required passed." : "The required jobs did not all pass.")
  summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
  isempty(summary) || open(summary, "a") do io
    println(io, "### Gate: ", passed ? "passed" : "failed", "\n")
    foreach(l -> println(io, "- ", l), lines)
  end
  return passed
end

function main(args)
  if "--gate" in args
    gate() || exit(1)
    return nothing
  end
  p = compute(args)
  describe(stdout, p)
  out = option(args, "github-output")
  out === nothing || write_outputs(out, p)
  summary = option(args, "summary")
  summary === nothing || open(io -> describe(io, p; markdown = true), summary, "a")
  return p
end

end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && CIPlan.main(ARGS)
