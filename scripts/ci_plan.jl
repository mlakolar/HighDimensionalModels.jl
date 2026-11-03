# Which tests and CI jobs a change needs (test/impact.toml, test/testsuite.toml). Plain Julia:
# TOML and the git command line, no package environment.
#
#   julia scripts/ci_plan.jl                       local changes against origin/master
#   julia scripts/ci_plan.jl --base=REV --head=REV  a range of commits
#   julia scripts/ci_plan.jl --event=pull_request   in CI (see .github/workflows/CI.yml)
#
# It prints the plan and its reasons. With --github-output and --summary it also writes the CI
# outputs and a Markdown job summary. When the changes cannot be determined it plans the full
# suite; it never plans "no tests" because something failed.
module CIPlan

using TOML

include(joinpath(@__DIR__, "..", "test", "testsuite.jl"))
using .TestSuite

const LEVELS = ("none", "focused", "broad", "full")
const JOBS = ("test", "test-cross", "master-integration", "knockoff-extension", "examples")
# A full plan runs these; test-cross (macOS, Windows) only on request or by rule.
const FULL_JOBS = ["test", "master-integration", "knockoff-extension", "examples"]
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
    r.level == "none" && !isempty(r.jobs) && push!(problems, "$ctx: a none rule cannot add jobs")
  end
  # Every test can be reached from a source change (or runs with every plan).
  reachable = union(Set(impact.smoke), Set(impact.source), Set(Iterators.flatten(values(impact.families))))
  for name in sort!(collect(setdiff(tests, reachable)))
    push!(problems, "test `$name` is in no family, smoke or source list")
  end
  isempty(problems) || error("$where is inconsistent:\n  " * join(problems, "\n  "))
  return impact
end

"Source and extension files without a rule of level focused or above (should be none)."
function unmapped_sources(impact::Impact, files)
  filter(files) do f
    (startswith(f, "src/") || startswith(f, "ext/")) || return false
    !any(r -> r.level != "none" && any(g -> TestSuite.matches(g, f), r.paths), impact.rules)
  end
end

ispackagecode(path) = startswith(path, "src/") || startswith(path, "ext/")
testname(path) = (m = match(r"^test/(.+)\.jl$", path)) === nothing ? nothing : String(m[1])

"""
    plan(changes, impact, inventory; base_inventory=nothing, explicit_full=false)

The plan for a list of changed paths. `base_inventory` is the test inventory before the change
(`nothing` if unknown), used when test/testsuite.toml changed.
"""
function plan(changes::Vector{Change}, impact::Impact, inventory::TestSuite.Inventory;
              base_inventory::Union{Nothing,TestSuite.Inventory} = nothing, explicit_full::Bool = false)
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
      matched = [r for r in impact.rules if any(g -> TestSuite.matches(g, path), r.paths)]
      name = testname(path)
      why = String[]
      if deleted && (ispackagecode(path) || name !== nothing)
        level = "full"
        push!(why, "$(c.status == 'R' ? "Renamed" : "Deleted") source or test file: the full suite runs.")
      elseif path == "test/testsuite.toml"
        added = inventory_additions(base_inventory, inventory)
        if added === nothing
          level = "full"
          push!(why, "The test inventory changed beyond adding parallel test files: the full suite runs.")
        else
          level = maxlevel(level, "focused")
          union!(tests, added)
          push!(why, isempty(added) ? "The test inventory changed without changing what runs." :
                     "The test inventory adds $(join(added, ", ")).")
        end
      elseif name !== nothing && name in alltests && isempty(matched)
        level = maxlevel(level, "focused")
        push!(tests, name)
        push!(why, "Test file changed: it runs itself.")
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
  if explicit_full
    level = "full"
    push!(jobs, "test-cross")
    push!(reasons, ("(request)", "Full validation was requested, on every platform."))
  end
  if any(c -> ispackagecode(c.path), changes) && !any(c -> startswith(c.path, "test/"), changes)
    push!(notes, "Package code changed without a test change: say in the PR how the change is tested.")
  end
  return finish(level, tests, jobs, reasons, notes, impact, changes, alltests; coverage = false)
end

function finish(level, tests, jobs, reasons, notes, impact, changes, alltests; coverage)
  if level == "none"
    return Plan("none", String[], String[], reasons, notes, false)
  elseif level == "full"
    return Plan("full", nothing, sort!(collect(union(Set(jobs), FULL_JOBS))), reasons, notes, coverage)
  end
  union!(tests, impact.smoke)
  any(c -> ispackagecode(c.path) || (c.status == 'R' && ispackagecode(c.old)), changes) && union!(tests, impact.source)
  push!(jobs, "test")
  selected = sort!(collect(tests))
  # A selection that covers every file is the full suite.
  Set(selected) == alltests && return Plan(level, nothing, sort!(collect(jobs)), reasons, notes, coverage)
  return Plan(level, selected, sort!(collect(jobs)), reasons, notes, coverage)
end

"The parallel test files added by the change, or `nothing` if it changed anything else."
function inventory_additions(base, head)
  base === nothing && return nothing
  base.serial == head.serial && base.subprocess == head.subprocess && base.not_tests == head.not_tests || return nothing
  issubset(base.parallel, head.parallel) || return nothing
  return sort!(collect(setdiff(head.parallel, base.parallel)))
end

"The comprehensive plan of schedules, tags and requested full runs."
function full_plan(why::AbstractString; cross::Bool, coverage::Bool = true)
  jobs = sort!(union(FULL_JOBS, cross ? ["test-cross"] : String[]))
  Plan("full", nothing, jobs, [("(event)", String(why))], String[], coverage)
end

# ---- git -----------------------------------------------------------------------------------

function git(args::AbstractVector{<:AbstractString}; dir = ROOT)
  out = IOBuffer()
  ok = success(pipeline(Cmd(`git -C $dir $args`; ignorestatus = true); stdout = out, stderr = devnull))
  return ok ? String(take!(out)) : nothing
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

"The inventory at a revision, or `nothing` if it cannot be read."
function inventory_at(rev::AbstractString; dir = ROOT)
  text = git(["show", "$rev:test/testsuite.toml"]; dir)
  text === nothing && return nothing
  try
    mktempdir() do tmp
      path = joinpath(tmp, "testsuite.toml")
      write(path, text)
      TestSuite.load(path)
    end
  catch
    nothing
  end
end

"Whether the commit message of `rev` asks for full validation (a line `CI: full`)."
function requests_full(rev::AbstractString; dir = ROOT)
  message = git(["log", "-1", "--format=%B", rev]; dir)
  message !== nothing && any(l -> strip(l) == "CI: full", split(message, '\n'))
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

function write_outputs(path, p::Plan)
  open(path, "a") do io
    println(io, "level=", p.level)
    println(io, "tests=", p.tests === nothing ? "all" : join(p.tests, ","))
    println(io, "jobs=", join(p.jobs, " "))
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

"""
    compute(args) -> Plan

`--event=` local (default), pull_request, push, schedule, workflow_dispatch or tag;
`--base=`/`--head=` the revisions to compare; `--schedule=` nightly or weekly; `--full`.
"""
function compute(args; dir = ROOT)
  inventory = TestSuite.load(joinpath(dir, "test", "testsuite.toml"))
  TestSuite.check(inventory, TestSuite.discover(joinpath(dir, "test")))
  impact = load_impact(joinpath(dir, "test", "impact.toml"), inventory)
  files = split(something(git(["ls-files"]; dir), ""), '\n'; keepempty = false)
  missing = unmapped_sources(impact, files)
  isempty(missing) || error("test/impact.toml has no rule for $(join(missing, ", "))")
  event = option(args, "event", "local")
  if event == "schedule"
    schedule = option(args, "schedule", "")
    schedule in ("nightly", "weekly") || error("--schedule must be nightly or weekly")
    return full_plan("The $schedule run tests every file."; cross = schedule == "weekly")
  elseif event == "tag"
    return full_plan("A tag is a release: every file on every platform."; cross = true)
  elseif "--full" in args
    return full_plan("Full validation was requested."; cross = true)
  end
  base = option(args, "base", event == "local" ? something(git(["merge-base", "origin/master", "HEAD"]; dir), "origin/master") : nothing)
  head = option(args, "head", event == "local" ? nothing : "HEAD")
  head = head == "" ? nothing : head
  base === nothing && return full_plan("No base revision to compare with: the full suite runs."; cross = false, coverage = false)
  changes = changed_files(strip(base), head === nothing ? nothing : strip(head); dir)
  if changes === nothing
    return full_plan("The changed files could not be determined (base `$base`): the full suite runs."; cross = false, coverage = false)
  end
  explicit = head !== nothing && requests_full(option(args, "request-sha", head); dir)
  return plan(changes, impact, inventory; base_inventory = inventory_at(strip(base); dir), explicit_full = explicit)
end

function main(args)
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
