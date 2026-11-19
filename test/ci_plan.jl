module CIPlanTest
# The test planner and the rule of the CI gate (scripts/ci_plan.jl), and the maps the planner
# reads (test/impact.toml, test/testsuite.toml). Pure Julia and git; no package code.
using Test

include(joinpath(@__DIR__, "..", "scripts", "ci_plan.jl"))
using .CIPlan: Change, plan, load_impact, parse_name_status, changed_files, source_problems,
               full_plan, write_outputs, describe, FULL_JOBS, compute, tracked_files, Rule, Impact
const TS = CIPlan.TestSuite

const ROOT = normpath(joinpath(@__DIR__, ".."))
const INVENTORY = TS.load(joinpath(@__DIR__, "testsuite.toml"))
const IMPACT = load_impact(joinpath(@__DIR__, "impact.toml"), INVENTORY)
const ALL = Set(TS.alltests(INVENTORY))
# An installed copy of the package has no git checkout; the git cases need one.
const HASGIT = Sys.which("git") !== nothing
const CHECKOUT = HASGIT && success(pipeline(`git -C $ROOT rev-parse --is-inside-work-tree`; stdout = devnull, stderr = devnull))
function tracked()
  CHECKOUT && return split(read(`git -C $ROOT ls-files`, String), '\n'; keepempty = false)
  [replace(relpath(joinpath(r, f), ROOT), '\\' => '/') for d in ("src", "test", "scripts")
   for (r, _, fs) in walkdir(joinpath(ROOT, d)) for f in fs]
end

p(changes...; kw...) = plan(Change[changes...], IMPACT, INVENTORY; kw...)
M(path) = Change('M', path)
selected(pl) = pl.tests === nothing ? ALL : Set(pl.tests)
isfull(pl) = pl.level == "full" && pl.tests === nothing && issubset(FULL_JOBS, pl.jobs) && pl.coverage
isnone(pl) = pl.level == "none" && isempty(pl.tests) && !pl.coverage

function gitrepo(f)
  mktempdir() do dir
    g(args...) = run(pipeline(`git -C $dir -c user.name=t -c user.email=t@t.invalid -c commit.gpgsign=false $args`; stdout = devnull, stderr = devnull))
    g("init", "-q", "-b", "master")
    f(dir, g)
  end
end

@testset "test planner" begin
  @testset "the real maps are complete" begin
    files = tracked()
    @test isempty(source_problems(IMPACT, files))
    @test !isempty(filter(f -> startswith(f, "src/"), files))
    # Every tracked path alone gives a plan that runs the tests, unless a none rule covers it; then
    # no tests, and the path is neither package code, a test, nor a script.
    for f in files
      pl = p(M(f))
      if pl.level == "none"
        @test isnone(pl) && issubset(pl.jobs, ["benchmark-integrity"])
        @test !any(prefix -> startswith(f, prefix), ("src/", "test/", "scripts/"))
      else
        @test pl.tests === nothing || !isempty(pl.tests)
        @test "test" in pl.jobs
        @test issubset(IMPACT.smoke, selected(pl))
      end
    end
  end

  @testset "the real map: package, tests, scripts and CI run the complete suite" begin
    for f in ("src/HighDimensionalModels.jl", "src/lasso.jl", "Project.toml", "test/lasso.jl", "test/runtests.jl",
              "test/testsuite.jl", "test/testsuite.toml", "test/impact.toml", "test/dependencies.jl",
              "scripts/test.jl", "scripts/dependencies.jl", "scripts/ci_plan.jl", ".github/workflows/CI.yml")
      pl = p(M(f))
      @test isfull(pl)
      @test pl.jobs == ["benchmark-integrity", "test"]
      @test !("test-cross" in pl.jobs)
    end
    # Documentation and the other workflows: no test, no job.
    for f in ("docs/src/index.md", "docs/make.jl", "docs/Project.toml", "README.md", "NEWS.md", "LICENSE", ".gitignore",
              ".github/workflows/Documentation.yml", ".github/workflows/CompatHelper.yml", ".github/workflows/TagBot.yml")
      pl = p(M(f))
      @test isnone(pl) && isempty(pl.jobs)
    end
    # The benchmark harness: no package test, but its integrity job.
    pl = p(M("benchmark/cases.jl"))
    @test isnone(pl) && pl.jobs == ["benchmark-integrity"]
    @test isnone(p(M("benchmark/cases.jl"), M("README.md")))
    # Mixed changes take the union.
    @test isfull(p(M("README.md"), M("src/lasso.jl")))
    mixed = p(M("README.md"), M("benchmark/run.jl"))
    @test isnone(mixed) && mixed.jobs == ["benchmark-integrity"]
    @test isfull(p(M("benchmark/run.jl"), M("test/lasso.jl")))
    # A package change without a test change is flagged.
    @test p(M("src/lasso.jl")).notes == ["Package code changed without a test change: say in the PR how the change is tested."]
    @test isempty(p(M("src/lasso.jl"), M("test/lasso.jl")).notes)
  end

  @testset "new, unknown, renamed and deleted paths fail safe" begin
    new = p(Change('A', "Makefile"))
    @test isfull(new)
    @test occursin("No rule", only(new.reasons)[2])
    @test isfull(p(M("somewhere/unknown.txt")))
    @test isfull(p(Change('A', "newtop/unknown.jl")))
    @test isfull(p(Change('A', "src/newfile.jl")))
    @test isfull(p(Change('R', "src/lasso2.jl", "src/lasso.jl")))
    @test isfull(p(Change('D', "test/lasso.jl")))
    @test isfull(p(Change('D', "src/lasso.jl")))
    # A file renamed out of the documentation into the package, and a deleted document.
    @test isfull(p(Change('R', "src/extra.jl", "docs/extra.md")))
    @test isnone(p(Change('D', "docs/old.md")))
    # A new test file not in the inventory has no entry in it (compute stops, see below), and here it is a test change.
    @test isfull(p(Change('A', "test/newtest.jl")))
  end

  @testset "focused rules narrow the plan (for a larger package)" begin
    map(rules...) = Impact(["smoke"], ["aqua"], Dict("lasso" => ["lasso", "statsapi"], "inv" => ["testsuite_validation"]), Rule[rules...])
    R(paths, level; families = String[], tests = String[], jobs = String[]) = Rule(paths, level, families, tests, jobs, "r")
    narrow = map(R(["src/lasso.jl"], "focused"; families = ["lasso"]),
                 R(["src/other.jl"], "focused"; tests = ["aqua"], jobs = ["benchmark-integrity"]),
                 R(["docs/**"], "none"), R(["test/**"], "full"))
    pl = plan([M("src/lasso.jl")], narrow, INVENTORY)
    @test pl.level == "focused" && pl.tests == ["aqua", "lasso", "smoke", "statsapi"] && pl.jobs == ["test"] && !pl.coverage
    # Source changes add the package-wide checks.
    pl = plan([M("src/other.jl")], narrow, INVENTORY)
    @test pl.tests == ["aqua", "smoke"] && pl.jobs == ["benchmark-integrity", "test"]
    # Several rules take the union; the strongest level wins; a path no rule matches is full.
    both = plan([M("src/lasso.jl"), M("src/other.jl")], narrow, INVENTORY)
    @test both.level == "focused" && both.tests == ["aqua", "lasso", "smoke", "statsapi"] && allunique(both.tests) && issorted(both.tests)
    @test plan([M("src/lasso.jl"), M("test/lasso.jl")], narrow, INVENTORY).level == "full"
    @test plan([M("src/lasso.jl"), M("docs/a.md")], narrow, INVENTORY).level == "focused"
    @test isfull(plan([M("src/lasso.jl"), M("elsewhere.txt")], narrow, INVENTORY))
    @test isfull(plan([Change('D', "src/lasso.jl")], narrow, INVENTORY))
    @test isnone(plan([M("docs/a.md")], narrow, INVENTORY))
    # A selection that covers every test file is the complete suite.
    everything = map(R(["src/lasso.jl"], "focused"; tests = collect(ALL)))
    pl = plan([M("src/lasso.jl")], everything, INVENTORY)
    @test pl.level == "focused" && pl.tests === nothing && pl.coverage
  end

  @testset "the inventory" begin
    base = INVENTORY
    @test TS.alltests(base) == ["aqua", "ci_plan", "dependencies", "elastic_net", "group_lasso", "lasso", "lasso_intercept", "lasso_reference", "smoke", "statsapi", "testsuite_validation"]
    @test isempty(base.serial) && isempty(base.subprocess)
    # The checks of the inventory itself.
    @test_throws ErrorException TS.check(INVENTORY, vcat(TS.discover(@__DIR__), ["brand_new"]))
    err = try TS.check(INVENTORY, vcat(TS.discover(@__DIR__), ["data/new_regression"])); "" catch e; sprint(showerror, e) end
    @test occursin("test/data/new_regression.jl is not classified", err)
    @test_throws ErrorException TS.check(INVENTORY, filter(!=("lasso"), TS.discover(@__DIR__)))
    mktempdir() do dir
      bad = joinpath(dir, "testsuite.toml")
      write(bad, "parallel = [\"a\", \"a\"]\n")
      @test_throws ErrorException TS.load(bad)
      write(bad, "parallel = [\"a\"]\nunknown = 1\n")
      @test_throws ErrorException TS.load(bad)
    end
  end

  @testset "empty and malformed input" begin
    @test isfull(p())
    @test parse_name_status("M\tsrc/lasso.jl\nR087\tsrc/a.jl\tsrc/b.jl\n") ==
          [Change('M', "src/lasso.jl"), Change('R', "src/b.jl", "src/a.jl")]
    @test parse_name_status("") == Change[]
    @test parse_name_status("garbage") === nothing
    @test parse_name_status("M\ta\tb") === nothing
    @test parse_name_status("X\tsrc/lasso.jl") === nothing
    sched = full_plan("nightly")
    @test isfull(sched) && "test-cross" in sched.jobs
    @test !("test-cross" in full_plan("push"; cross = false).jobs)
  end

  @testset "outputs" begin
    mktempdir() do dir
      out = joinpath(dir, "out")
      write_outputs(out, p(M("src/lasso.jl")))
      @test readlines(out) == ["level=full", "tests=all", "jobs=benchmark-integrity test", "coverage=true", "ok=true"]
      write(out, ""); write_outputs(out, p(M("README.md")))
      # An empty job or test list is written as `none`, so that a missing output differs from it.
      @test readlines(out) == ["level=none", "tests=none", "jobs=none", "coverage=false", "ok=true"]
      write(out, ""); write_outputs(out, p(M("benchmark/run.jl")))
      @test "jobs=benchmark-integrity" in readlines(out) && "tests=none" in readlines(out)
      write(out, ""); write_outputs(out, full_plan("nightly"))
      @test "jobs=benchmark-integrity test test-cross" in readlines(out)
    end
    text = sprint(describe, p(M("src/lasso.jl")))
    @test occursin("Test plan: full", text) && occursin("src/lasso.jl", text) && occursin("every test", text)
    @test occursin("| `src/lasso.jl` |", sprint(io -> describe(io, p(M("src/lasso.jl")); markdown = true)))
  end

  @testset "the gate" begin
    jobs = CIPlan.JOBS
    allskipped = Dict(j => "skipped" for j in jobs)
    with(d...) = merge(allskipped, Dict(d...))
    # The verdict of a plan (level, tests, jobs; a full pull-request plan by default) and results.
    v(results; result = "success", ok = "true", level = "full", tests = "all", jobs = "benchmark-integrity test", kw...) =
      CIPlan.verdict(result, ok, level, tests, jobs, results; kw...)
    docs = (level = "none", tests = "none", jobs = "none")
    full = with("test" => "success", "benchmark-integrity" => "success")
    nightly = with("test" => "success", "test-cross" => "success", "benchmark-integrity" => "success")
    # Documentation only: nothing required, everything skipped.
    @test v(allskipped; docs...)[1]
    # Benchmark files only: the benchmark job is required.
    @test v(with("benchmark-integrity" => "success"); level = "none", tests = "none", jobs = "benchmark-integrity")[1]
    # A full pull-request plan: the required jobs passed, the cross-platform job was skipped.
    @test v(full)[1]
    # A focused plan, with a test selection.
    @test v(with("test" => "success"); level = "focused", tests = "lasso,smoke", jobs = "test")[1]
    # The nightly plan requires every job.
    @test v(nightly; jobs = "benchmark-integrity test test-cross")[1]
    @test !v(full; jobs = "benchmark-integrity test test-cross")[1]    # test-cross skipped although required
    # A required job that failed, was cancelled, skipped or is missing.
    for r in ("failure", "cancelled", "skipped", "")
      @test !v(merge(full, Dict("test" => r)))[1]
      @test !v(merge(full, Dict("benchmark-integrity" => r)))[1]
    end
    @test !v(delete!(copy(full), "benchmark-integrity"))[1]
    # A job that was not required must have been skipped: one that ran, failed, was cancelled
    # or is missing fails the gate.
    for r in ("success", "failure", "cancelled", "")
      @test !v(with("test" => "success", "test-cross" => r); level = "focused", tests = "lasso", jobs = "test")[1]
    end
    @test !v(full; level = "focused", tests = "lasso", jobs = "test")[1]   # benchmark-integrity ran without being required
    @test v(with("test" => "success"); level = "focused", tests = "lasso", jobs = "test")[1]
    # A failed, cancelled or incomplete plan fails even if nothing else ran.
    for (result, ok) in (("failure", ""), ("cancelled", ""), ("skipped", ""), ("", ""), ("success", ""), ("success", "false"))
      @test !v(allskipped; result, ok, docs...)[1]
    end
    # A plan naming an unknown job, repeating a job, or mixing the `none` sentinel with jobs.
    @test !v(with("test" => "success"); jobs = "test deploy")[1]
    @test !v(with("test" => "success"); jobs = "test test")[1]
    @test !v(allskipped; level = "none", tests = "none", jobs = "none test")[1]
    @test !v(with("test" => "success", "deploy" => "skipped"); jobs = "test", level = "focused", tests = "lasso")[1]   # results of an unknown job
    passed, lines = v(with("test" => "failure"); level = "focused", tests = "lasso", jobs = "test")
    @test !passed && any(l -> occursin("test is required, but its result is `failure`", l), lines)
    # Missing or empty plan outputs are not an empty plan (the documentation-only plan says `none`).
    for (level, tests, jobs) in (("", "none", "none"), ("none", "", "none"), ("none", "none", ""), ("", "", ""))
      passed, lines = v(allskipped; level, tests, jobs)
      @test !passed && any(l -> occursin("missing or empty", l), lines)
    end
    @test !v(allskipped; level = "sometimes", tests = "none", jobs = "none")[1]
    # Contradictions between level, test selection and jobs, each rejected.
    @test !v(allskipped; level = "full", tests = "all", jobs = "none")[1]              # full without the test job
    @test !v(allskipped; level = "focused", tests = "lasso", jobs = "benchmark-integrity")[1]
    @test !v(full; level = "none", tests = "none", jobs = "benchmark-integrity test")[1]   # none with the test job
    @test !v(full; level = "none", tests = "all", jobs = "benchmark-integrity test")[1]    # none with tests
    @test !v(allskipped; level = "none", tests = "all", jobs = "none")[1]              # tests without the test job
    @test !v(full; level = "full", tests = "none", jobs = "benchmark-integrity test")[1]   # the test job without tests
    @test !v(with("test-cross" => "success"); level = "none", tests = "none", jobs = "test-cross")[1]
    @test !v(with("test-cross" => "success"); level = "full", tests = "all", jobs = "test-cross")[1]
    passed, lines = v(full; level = "none", tests = "none", jobs = "benchmark-integrity test")
    @test any(l -> occursin("a `none` plan requires the `test` or `test-cross` job", l), lines)
    # The reviewed counterexamples and their neighbours: only the combinations the planner emits pass.
    pp(level, tests, jobs) = CIPlan.plan_problems(level, tests, jobs, TS.alltests(INVENTORY))[2]
    @test isempty(pp("full", "all", "benchmark-integrity test"))
    @test isempty(pp("full", "all", "benchmark-integrity test test-cross"))
    @test isempty(pp("focused", "lasso,smoke", "test"))
    @test isempty(pp("focused", "all", "test"))
    @test isempty(pp("focused", "smoke", "benchmark-integrity test"))
    @test isempty(pp("none", "none", "none"))
    @test isempty(pp("none", "none", "benchmark-integrity"))
    for (what, level, tests, jobs) in (
        ("full, a name list instead of all", "full", "smoke", "benchmark-integrity test"),
        ("full, a name list of every file", "full", join(TS.alltests(INVENTORY), ","), "benchmark-integrity test"),
        ("full without benchmark-integrity", "full", "all", "test"),
        ("full without the test job", "full", "all", "benchmark-integrity"),
        ("full with none", "full", "none", "benchmark-integrity test"),
        ("none with tests", "none", "smoke", "none"),
        ("none with all", "none", "all", "none"),
        ("none with the test job", "none", "none", "test"),
        ("none with test-cross", "none", "none", "test-cross"),
        ("focused with none", "focused", "none", "test"),
        ("focused without the test job", "focused", "smoke", "benchmark-integrity"),
        ("an unknown test name", "focused", "nosuchtest", "test"),
        ("a duplicate name", "focused", "smoke,smoke", "test"),
        ("a trailing comma", "focused", "smoke,", "test"),
        ("a leading comma", "focused", ",smoke", "test"),
        ("an empty item", "focused", "smoke,,lasso", "test"),
        ("a space in the list", "focused", "smoke, lasso", "test"),
        ("a whitespace-only selection", "full", "   ", "benchmark-integrity test"),
        ("a whitespace-only selection, focused", "focused", " ", "test"),
        ("all with a name", "focused", "all,smoke", "test"),
        ("none with a name", "focused", "none,smoke", "test"),
        ("test-cross without test", "focused", "smoke", "test-cross"),
        ("a job twice", "full", "all", "benchmark-integrity test test"),
        ("a double space between jobs", "full", "all", "benchmark-integrity  test"),
        ("a leading space in the jobs", "full", "all", " benchmark-integrity test"),
        ("none among jobs", "full", "all", "benchmark-integrity none test"),
        ("an unknown job", "full", "all", "benchmark-integrity deploy test"),
        ("a padded none job list", "none", "none", " none "),
        ("a job list with trailing space after none", "none", "none", "none "),
        ("a job list with leading space before none", "none", "none", " none"),
        ("a padded none selection", "none", " none ", "none"),
        ("a selection with leading space before none", "none", " none", "none"),
        ("a selection with trailing space after none", "none", "none ", "none"),
        ("a padded all selection", "full", " all ", "benchmark-integrity test"),
        ("a selection with trailing space after all", "full", "all ", "benchmark-integrity test"),
        ("an unknown level", "partial", "all", "test"),
        ("an empty level", "", "all", "test"))
      @test !isempty(pp(level, tests, jobs))
      @test !v(merge(allskipped, Dict("test" => "success", "benchmark-integrity" => "success")); level, tests, jobs)[1]
    end
    # An inventory with a test named like a keyword cannot be told apart from the keyword.
    @test !isempty(CIPlan.plan_problems("focused", "smoke", "test", ["smoke", "all"])[2])
    # Property: every plan the planner produces for these changes and events, through `write_outputs`, is valid. Changes and events
    # span documentation, the benchmark, source, tests, scripts, the workflow, unknown paths,
    # the nightly and manual runs, and a map with focused rules.
    narrow_rules = [Rule(["src/lasso.jl"], "focused", ["lasso"], String[], String[], "r"),
                    Rule(["src/other.jl"], "focused", String[], ["aqua"], ["benchmark-integrity"], "r"),
                    Rule(["src/cross.jl"], "focused", String[], ["smoke"], ["test-cross"], "r"),
                    Rule(["src/everything.jl"], "focused", String[], collect(ALL), String[], "r"),
                    Rule(["docs/**"], "none", String[], String[], String[], "r"),
                    Rule(["bench/**"], "none", String[], String[], ["benchmark-integrity"], "r"),
                    Rule(["test/**"], "full", String[], String[], String[], "r")]
    narrow = Impact(["smoke"], ["aqua"], Dict("lasso" => ["lasso", "statsapi"]), narrow_rules)
    changes = [Change[], [M("README.md")], [M("benchmark/cases.jl")], [M("benchmark/cases.jl"), M("README.md")],
               [M("src/lasso.jl")], [M("test/lasso.jl")], [M("test/testsuite.toml")], [M("scripts/test.jl")],
               [M(".github/workflows/CI.yml")], [M("Project.toml")], [M("somewhere/unknown.txt")],
               [Change('D', "src/lasso.jl")], [Change('R', "src/a.jl", "docs/a.md")]]
    narrow_changes = [[M("src/lasso.jl")], [M("src/other.jl")], [M("src/cross.jl")], [M("src/everything.jl")],
                      [M("docs/a.md")], [M("bench/x.jl")], [M("src/lasso.jl"), M("docs/a.md")],
                      [M("src/lasso.jl"), M("src/other.jl")], [M("elsewhere.txt")], [M("test/x.jl")]]
    plans = vcat([plan(c, IMPACT, INVENTORY) for c in changes], [plan(c, narrow, INVENTORY) for c in narrow_changes],
                 [full_plan("nightly"), full_plan("manual"), full_plan("push"; cross = false)])
    @test length(unique(pl.level for pl in plans)) == 3          # none, focused and full all occur
    @test any(pl -> pl.tests === nothing && pl.level == "focused", plans)
    mktempdir() do dir
      out = joinpath(dir, "out")
      for pl in plans
        write(out, ""); write_outputs(out, pl)
        o = Dict(String(first(split(l, '='))) => String(last(split(l, '='; limit = 2))) for l in readlines(out))
        required, problems = CIPlan.plan_problems(o["level"], o["tests"], o["jobs"], TS.alltests(INVENTORY))
        @test isempty(problems)
        @test required == (o["jobs"] == "none" ? String[] : String.(split(o["jobs"])))
        # The gate accepts the plan when its required jobs succeeded and the others were skipped.
        res = Dict(j => (j in required ? "success" : "skipped") for j in jobs)
        @test v(res; level = o["level"], tests = o["tests"], jobs = o["jobs"])[1]
      end
    end
    # Results: duplicates, malformed words, unknown results.
    res, problems = CIPlan.parse_results("test=success test-cross=skipped benchmark-integrity=skipped")
    @test isempty(problems) && res == Dict("test" => "success", "test-cross" => "skipped", "benchmark-integrity" => "skipped")
    for text in ("test=failure test=success", "test=success test=success", "test", "test=", "=success", "test=success=failure",
                 "test=weird", "test==success")
      res, problems = CIPlan.parse_results(text)
      @test !isempty(problems)
      @test !v(merge(full, res); problems)[1]
    end
    @test occursin("given twice", only(CIPlan.parse_results("test=failure test=success")[2]))
    # The gate job reads the plan and the results from the environment, as the workflow writes them.
    function gate(results; level = "focused", tests = "lasso", jobs = "test", plan = "success", ok = "true")
      vars = ["PLAN_RESULT" => plan, "PLAN_OK" => ok, "PLAN_LEVEL" => level, "PLAN_TESTS" => tests, "PLAN_JOBS" => jobs,
              "RESULTS" => results, "GITHUB_STEP_SUMMARY" => nothing]
      withenv(vars...) do
        redirect_stdout(() -> CIPlan.gate(), devnull)
      end
    end
    ok = "test=success test-cross=skipped benchmark-integrity=skipped"
    @test gate(ok)
    @test !gate("test=failure test-cross=skipped benchmark-integrity=skipped")
    @test !gate("test=skipped test-cross=skipped benchmark-integrity=skipped")
    @test !gate("test=failure test=success test-cross=skipped benchmark-integrity=skipped")   # a duplicate does not overwrite
    @test !gate("test=success test-cross=skipped")                                           # a job is missing
    @test !gate("")                                              # no results at all
    @test gate("test=skipped test-cross=skipped benchmark-integrity=skipped"; level = "none", tests = "none", jobs = "none")
    @test !gate(ok; jobs = "")                                   # an empty job list is not the documentation plan
    @test !gate("test=skipped test-cross=skipped benchmark-integrity=skipped"; level = "none", tests = "none", jobs = "")
    # An output that is not set at all is read as empty.
    for var in ("PLAN_LEVEL", "PLAN_TESTS", "PLAN_JOBS")
      vars = ["PLAN_RESULT" => "success", "PLAN_OK" => "true", "PLAN_LEVEL" => "none", "PLAN_TESTS" => "none",
              "PLAN_JOBS" => "none", "RESULTS" => "test=skipped test-cross=skipped benchmark-integrity=skipped",
              "GITHUB_STEP_SUMMARY" => nothing]
      withenv(vars..., var => nothing) do
        @test !redirect_stdout(() -> CIPlan.gate(), devnull)
      end
    end
    # Every job the workflow defines is a known job, the gate needs all of them, and the
    # workflow keeps the properties the gate relies on.
    workflow = joinpath(ROOT, ".github", "workflows", "CI.yml")
    if isfile(workflow)
      text = replace(read(workflow, String), "\r\n" => "\n")   # Windows checkouts have CRLF line endings
      needs = match(r"\n  gate:\n(?:.*\n)*?    needs: \[([^\]]*)\]", text)
      @test needs !== nothing && Set(strip.(split(needs[1], ','))) == Set(["plan"; collect(jobs)])
      for j in jobs
        @test occursin("\n  $j:\n", text)
        @test occursin("$j=\${{ needs.$j.result }}", text) || occursin("$j=\${{ needs['$j'].result }}", text)
      end
      for (var, output) in ("PLAN_RESULT" => "needs.plan.result", "PLAN_OK" => "needs.plan.outputs.ok",
                            "PLAN_LEVEL" => "needs.plan.outputs.level", "PLAN_TESTS" => "needs.plan.outputs.tests",
                            "PLAN_JOBS" => "needs.plan.outputs.jobs")
        @test occursin("$var: \${{ $output }}", text)             # the gate reads the whole plan
      end
      @test !occursin("continue-on-error", text)                   # no test step may fail silently
      @test !occursin("tags:", text)
      cross = match(r"\n  test-cross:\n((?:    .*\n|\n)*)", text)
      @test cross !== nothing && occursin("github.event_name != 'pull_request'", cross[1])
    end
  end

  @testset "invalid maps are rejected" begin
    good = read(joinpath(@__DIR__, "impact.toml"), String)
    cases = [
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\nfamilies = [\"nosuchfamily\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\ntests = [\"nosuchtest\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"sometimes\"\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"broad\"\ntests = [\"lasso\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\njobs = [\"deploy\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\n",
      "[[rules]]\nlevel = \"focused\"\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"full\"\ntests = [\"lasso\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\nreason = \"r\"\ncolour = \"red\"\n",
    ]
    mktempdir() do dir
      path = joinpath(dir, "impact.toml")
      for case in cases
        write(path, good * "\n" * case)
        @test_throws ErrorException load_impact(path, INVENTORY)
      end
      # A test in no family, smoke or source list cannot be reached from a source change.
      write(path, replace(good, "planner = [\"ci_plan\"]" => "planner = [\"smoke\"]"))
      @test_throws ErrorException load_impact(path, INVENTORY)
      write(path, "surprise = 1\n" * good)
      @test_throws ErrorException load_impact(path, INVENTORY)
      # An unknown test file in smoke or source, an empty smoke list, an empty family.
      write(path, replace(good, "smoke = [\"smoke\", \"dependencies\"]" => "smoke = [\"smoke\", \"nosuchtest\"]"))
      @test_throws ErrorException load_impact(path, INVENTORY)
      write(path, replace(good, "source = [\"aqua\"]" => "source = [\"nosuchtest\"]"))
      @test_throws ErrorException load_impact(path, INVENTORY)
      write(path, replace(good, "smoke = [\"smoke\", \"dependencies\"]" => "smoke = []"))
      @test_throws ErrorException load_impact(path, INVENTORY)
      write(path, replace(good, "planner = [\"ci_plan\"]" => "planner = []"))
      @test_throws ErrorException load_impact(path, INVENTORY)
    end
  end

  @testset "source rules must select tests" begin
    R(paths, level; families = String[], tests = String[], jobs = String[]) = Rule(paths, level, families, tests, jobs, "r")
    map(rules...) = Impact(["smoke"], ["aqua"], Dict("lasso" => ["lasso", "statsapi"]), Rule[rules...])
    new = ["src/new_estimator.jl"]
    vacuous = "select no families or tests"
    @test occursin(vacuous, only(source_problems(map(R(new, "focused")), new)))
    @test occursin(vacuous, only(source_problems(map(R(new, "focused"; jobs = ["benchmark-integrity"])), new)))
    @test isempty(source_problems(map(R(new, "focused"; families = ["lasso"])), new))
    @test isempty(source_problems(map(R(new, "focused"; tests = ["lasso"])), new))
    @test isempty(source_problems(map(R(new, "full")), new))
    @test occursin("has no rule", only(source_problems(map(R(new, "none")), new)))
    @test occursin("has no rule", only(source_problems(map(), new)))
    # Rules are judged together: a job-only rule may accompany one that selects tests.
    both = map(R(new, "focused"; families = ["lasso"]), R(["src/*.jl"], "focused"; jobs = ["benchmark-integrity"]))
    @test isempty(source_problems(both, new))
    pl = plan([M(new[1])], both, INVENTORY)
    @test Set(pl.tests) == Set(["lasso", "statsapi", "smoke", "aqua"]) && "benchmark-integrity" in pl.jobs
    # Overlapping rules select the union of their tests.
    two = map(R(new, "focused"; families = ["lasso"]), R(["src/new_*.jl"], "focused"; tests = ["ci_plan"]))
    @test Set(plan([M(new[1])], two, INVENTORY).tests) == Set(["lasso", "statsapi", "ci_plan", "smoke", "aqua"])
    # Job-only and test-less rules stay valid for paths outside src/.
    @test isempty(source_problems(map(R(["scripts/test.jl"], "focused"; jobs = ["benchmark-integrity"])), ["scripts/test.jl"]))
    # The real map covers any file under src/ with its full rule, existing or not.
    @test isempty(source_problems(IMPACT, vcat(tracked(), new)))
    @test isfull(plan([Change('A', new[1])], IMPACT, INVENTORY))
  end

  @testset "tracked files" begin
    CHECKOUT && @test "src/HighDimensionalModels.jl" in tracked_files(ROOT)
    mktempdir() do dir
      @test tracked_files(dir) === nothing                         # not a repository
      @test tracked_files(joinpath(dir, "no", "such", "dir")) === nothing
      if HASGIT
        run(pipeline(`git -C $dir init -q`; stdout = devnull, stderr = devnull))
        @test tracked_files(dir) == String[]                         # an empty repository
        @test isempty(source_problems(IMPACT, tracked_files(dir)))
      end
    end
    # git that cannot be started is a failure, not an empty result.
    Sys.isunix() && withenv("PATH" => mktempdir()) do
      @test CIPlan.git(["--version"]) === nothing
    end
  end

  # compute() on a small repository of its own: inventory, map, source files and fixtures.
  HASGIT && @testset "planning a repository" begin
    gitrepo() do dir, g
      put(path, text) = (mkpath(dirname(joinpath(dir, path))); write(joinpath(dir, path), text))
      put("src/a.jl", "a\n"); put("src/b.jl", "b\n"); put("README.md", "docs\n"); put("benchmark/run.jl", "bench\n")
      for t in ("a", "b", "smoke"); put("test/$t.jl", "# $t\n"); end
      put("test/data/baseline.jl", "# writes fixtures\n"); put("test/data/x.csv", "1\n")
      inventory = "parallel = [\"a\", \"b\", \"smoke\"]\n[not_tests]\n\"data/baseline\" = \"fixture generator\"\n"
      put("test/testsuite.toml", inventory)
      map = """
        smoke = ["smoke"]
        [families]
        a = ["a"]
        b = ["b"]
        [[rules]]
        paths = ["src/a.jl"]
        level = "focused"
        families = ["a"]
        reason = "a"
        [[rules]]
        paths = ["src/b.jl"]
        level = "focused"
        families = ["b"]
        reason = "b"
        [[rules]]
        paths = ["test/data/**"]
        level = "focused"
        tests = ["a"]
        reason = "fixtures of a"
        [[rules]]
        paths = ["test/impact.toml", "test/testsuite.toml"]
        level = "focused"
        tests = ["smoke"]
        reason = "the maps"
        [[rules]]
        paths = ["README.md"]
        level = "none"
        reason = "docs"
        [[rules]]
        paths = ["benchmark/**"]
        level = "none"
        jobs = ["benchmark-integrity"]
        reason = "the harness"
        """
      put("test/impact.toml", map)
      g("add", "."); g("commit", "-qm", "base")
      head() = strip(read(`git -C $dir rev-parse HEAD`, String))
      base = head()
      pr(; kw...) = compute(["--event=pull_request", "--base=$base", "--head=HEAD"]; dir, kw...)
      fails(f) = try f(); "" catch e; sprint(showerror, e) end

      put("src/a.jl", "a2\n"); g("commit", "-qam", "change a")
      ok = pr()
      @test ok.level == "focused" && ok.tests == ["a", "smoke"] && ok.jobs == ["test"]
      # The tracked files cannot be listed: full suite, said in both outputs.
      blind = pr(list_files = _ -> nothing)
      @test isfull(blind)
      @test blind.reasons[1][1] == "(source map check)" && occursin("git could not list the tracked files", blind.reasons[1][2])
      @test occursin("git could not list the tracked files", sprint(describe, blind))
      @test occursin("git could not list the tracked files", sprint(io -> describe(io, blind; markdown = true)))

      # Documentation and benchmark changes: no test, and only the benchmark job for the latter.
      put("README.md", "docs2\n"); g("commit", "-qam", "docs")
      docs = compute(["--event=pull_request", "--base=HEAD^", "--head=HEAD"]; dir)
      @test isnone(docs) && isempty(docs.jobs)
      put("benchmark/run.jl", "bench2\n"); g("commit", "-qam", "bench")
      bench = compute(["--event=pull_request", "--base=HEAD^", "--head=HEAD"]; dir)
      @test isnone(bench) && bench.jobs == ["benchmark-integrity"]

      # Events: schedule and manual runs plan everything on every platform; a push or pull request
      # without a usable base is the Linux jobs only; an unknown event is an error.
      for event in ("schedule", "workflow_dispatch")
        every = compute(["--event=$event"]; dir)
        @test isfull(every) && "test-cross" in every.jobs
      end
      zero = compute(["--event=push", "--base=0000000000000000000000000000000000000000", "--head=HEAD"]; dir)
      @test isfull(zero) && !("test-cross" in zero.jobs)
      @test isfull(compute(["--event=push"]; dir))                    # no base given
      @test occursin("--event must be one of", fails(() -> compute(["--event=release"]; dir)))
      @test occursin("--event must be one of", fails(() -> compute(["--event="]; dir)))

      # A source file without a rule, or with a rule selecting no tests, stops the plan.
      put("src/c.jl", "c\n"); g("add", "."); g("commit", "-qm", "add c")
      @test occursin("src/c.jl has no rule", fails(pr))
      put("test/impact.toml", map * "[[rules]]\npaths = [\"src/c.jl\"]\nlevel = \"focused\"\nreason = \"New estimator\"\n")
      g("commit", "-qam", "vacuous rule for c")
      @test occursin("src/c.jl: its rules select no families or tests", fails(pr))
      @test isfull(pr(list_files = _ -> nothing))
      g("rm", "-q", "src/c.jl"); put("test/impact.toml", map); g("commit", "-qam", "drop c")

      # An unlisted test file stops the plan for a pull request, a nightly and a manual run; so does a listed file that is missing.
      put("test/data/new_regression.jl", "# a new test\n"); g("add", "."); g("commit", "-qm", "new fixture test")
      @test occursin("test/data/new_regression.jl is not classified", fails(pr))
      @test occursin("is not classified", fails(() -> pr(list_files = _ -> nothing)))
      @test occursin("is not classified", fails(() -> compute(["--event=schedule"]; dir)))
      g("rm", "-q", "test/data/new_regression.jl")
      rm(joinpath(dir, "test", "b.jl")); g("commit", "-qam", "remove a listed test file")
      @test occursin("lists `b`, but test/b.jl does not exist", fails(pr))
      @test occursin("lists `b`", fails(() -> compute(["--event=workflow_dispatch"]; dir)))
      # Once the file is listed it is planned like the others.
      put("test/b.jl", "# b\n"); put("test/data/new_regression.jl", "# a new test\n")
      put("test/testsuite.toml", replace(inventory, "\"smoke\"]" => "\"smoke\", \"data/new_regression\"]"))
      put("test/impact.toml", replace(map, "a = [\"a\"]" => "a = [\"a\", \"data/new_regression\"]"))
      g("add", "."); g("commit", "-qm", "classify it")
      classified = pr()      # the change to src/a.jl selects family a, which now includes the new file
      @test classified.level == "focused" && classified.tests == ["a", "data/new_regression", "smoke"]
    end
  end

  HASGIT && @testset "changed files from git" begin
    gitrepo() do dir, g
      write(joinpath(dir, "a.jl"), "a\n"); write(joinpath(dir, "b.jl"), "b\n" ^ 20)
      g("add", "."); g("commit", "-qm", "one")
      first = strip(read(`git -C $dir rev-parse HEAD`, String))
      write(joinpath(dir, "a.jl"), "a2\n"); g("commit", "-qam", "two")
      g("mv", "b.jl", "c.jl"); g("commit", "-qm", "three")
      # Every commit of the range counts, and a rename is one change.
      @test Set(changed_files(first, "HEAD"; dir)) == Set([Change('M', "a.jl"), Change('R', "c.jl", "b.jl")])
      # Missing and all-zero bases are unknown, never empty.
      @test changed_files("0000000000000000000000000000000000000000", "HEAD"; dir) === nothing
      @test changed_files("1234567890abcdef1234567890abcdef12345678", "HEAD"; dir) === nothing
      @test changed_files("HEAD~5", "HEAD"; dir) === nothing
      @test changed_files("HEAD", "HEAD"; dir) == Change[]
      # The working tree, with untracked files.
      write(joinpath(dir, "d.jl"), "new\n"); rm(joinpath(dir, "a.jl"))
      @test Set(changed_files("HEAD", nothing; dir)) == Set([Change('D', "a.jl"), Change('A', "d.jl")])
    end
  end
end

end
