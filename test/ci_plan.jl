module CIPlanTest
# The test planner (scripts/ci_plan.jl) and the maps it reads (test/impact.toml,
# test/testsuite.toml). Pure Julia and git; no package code.
using Test

include(joinpath(@__DIR__, "..", "scripts", "ci_plan.jl"))
using .CIPlan: Change, plan, load_impact, parse_name_status, changed_files, source_problems,
               requests_full, inventory_at, full_plan, write_outputs, describe, FULL_JOBS,
               compute, tracked_files, Rule, Impact
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
  [replace(relpath(joinpath(r, f), ROOT), '\\' => '/') for d in ("src", "ext", "test", "scripts", "examples")
   for (r, _, fs) in walkdir(joinpath(ROOT, d)) for f in fs]
end

p(changes...; kw...) = plan(Change[changes...], IMPACT, INVENTORY; kw...)
M(path) = Change('M', path)
selected(pl) = pl.tests === nothing ? ALL : Set(pl.tests)
isfull(pl) = pl.level == "full" && pl.tests === nothing && issubset(FULL_JOBS, pl.jobs)

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
    # Every tracked path alone gives a plan with tests, unless a none rule covers it; then no
    # tests and no jobs, and it is not package code or a test.
    for f in files
      pl = p(M(f))
      if pl.level == "none"
        @test isempty(pl.tests) && isempty(pl.jobs)
        @test !startswith(f, "src/") && !startswith(f, "ext/") && !startswith(f, "test/")
      else
        @test pl.tests === nothing || !isempty(pl.tests)
        @test "test" in pl.jobs
        @test issubset(IMPACT.smoke, selected(pl))
      end
    end
  end

  @testset "single source file: GLM" begin
    pl = p(M("src/glm.jl"))
    @test pl.level == "broad"
    @test issubset(["glm", "glm_penalties", "cv_glm", "cv", "quantile", "statsapi", "smoke", "aqua", "deprecated"], selected(pl))
    # A GLM change does not rerun unrelated low-rank or structured tests.
    @test !any(t -> startswith(t, "lowrank") || startswith(t, "structured"), selected(pl))
    @test issubset(["test", "examples", "master-integration"], pl.jobs)
    @test !("test-cross" in pl.jobs)
  end

  @testset "focused source file: structured" begin
    pl = p(M("src/structured.jl"))
    @test pl.level == "focused"
    @test selected(pl) == union(Set(["structured", "structured_kernels", "structured_golden", "structured_cv_masked", "smoke"]),
                                IMPACT.source)
    @test pl.jobs == ["examples", "test"]
  end

  @testset "several, overlapping and multiple families" begin
    a, b = p(M("src/structured.jl")), p(M("src/slope.jl"))
    both = p(M("src/structured.jl"), M("src/slope.jl"))
    @test selected(both) == union(selected(a), selected(b))
    @test allunique(both.tests) && issorted(both.tests)
    g = p(M("src/group.jl"), M("src/sparsegroup.jl"))
    @test issubset(["group", "sparsegroup", "varying", "statsapi", "formula"], selected(g))
    q = p(M("src/quantile.jl"), M("src/lowrank.jl"))
    @test q.level == "focused"
    @test issubset(["quantile_refit", "lowrank_solvers", "extensions_absent"], selected(q))
    @test p(M("src/structured.jl"), M("src/cv.jl")).level == "broad"
  end

  @testset "focused source file: low-rank" begin
    pl = p(M("src/lowrank.jl"))
    @test pl.level == "focused"
    @test issubset(IMPACT.families["lowrank"], selected(pl))
    @test !any(t -> startswith(t, "structured") || startswith(t, "glm") || startswith(t, "quantile"), selected(pl))
  end

  @testset "broad source file: cross-validation" begin
    pl = p(M("src/cv.jl"))
    @test pl.level == "broad"
    for family in ("cv", "lasso_api", "glm", "multitask", "lowrank", "quantile", "structured")
      @test issubset(IMPACT.families[family], selected(pl))
    end
    @test !issubset(ALL, selected(pl))
  end

  @testset "shared code is full" begin
    for f in ("src/utils.jl", "src/lasso.jl", "src/statsapi.jl", "src/HighDimensionalModels.jl")
      @test isfull(p(M(f)))
    end
  end

  @testset "new, unknown, renamed and deleted paths fail safe" begin
    new = p(Change('A', "src/newestimator.jl"))
    @test isfull(new)
    @test occursin("No rule", only(new.reasons)[2])
    @test isfull(p(M("somewhere/unknown.txt")))
    @test isfull(p(Change('R', "src/structured2.jl", "src/structured.jl")))
    @test isfull(p(Change('D', "test/glm.jl")))
    @test isfull(p(Change('D', "src/varying.jl")))
    # A new test file not in the inventory has no rule.
    @test isfull(p(Change('A', "test/newtest.jl")))
  end

  @testset "test changes" begin
    pl = p(M("test/lasso.jl"))
    @test pl.level == "focused"
    @test pl.tests == ["lasso", "smoke"]
    @test pl.jobs == ["test"]
    @test isempty(pl.notes)
    @test selected(p(M("test/kernels.jl"))) == Set(["naming", "varying", "smoke"])
    @test selected(p(M("test/golden/structured/fused/beta.csv"))) == Set(["structured_golden", "smoke"])
    @test selected(p(M("test/data/hdm/forced/x.csv"))) == Set(["hdm_golden", "smoke"])
    @test isfull(p(M("test/runtests.jl")))
    @test isfull(p(M("test/testsuite.jl")))
    # The inventory's validation runs when the inventory or the validation changes.
    @test selected(p(M("test/testsuite.toml"); base_inventory = INVENTORY)) == Set(["testsuite_validation", "smoke"])
    @test selected(p(M("test/testsuite_validation.jl"))) == Set(["testsuite_validation", "smoke"])
    @test "ci_plan" in selected(p(M("test/impact.toml")))
    kn = p(M("test/knockoff_extension.jl"))
    @test kn.tests == ["smoke"] && "knockoff-extension" in kn.jobs
  end

  @testset "the inventory" begin
    base = INVENTORY
    added = TS.Inventory(vcat(base.parallel, ["zz_new"]), base.serial, base.subprocess, base.not_tests)
    pl = plan([M("test/testsuite.toml")], IMPACT, INVENTORY; base_inventory = TS.Inventory(base.parallel[2:end], base.serial, base.subprocess, base.not_tests))
    @test pl.level == "focused" && base.parallel[1] in pl.tests
    @test CIPlan.inventory_additions(base, added) == ["zz_new"]
    @test CIPlan.inventory_additions(added, base) === nothing           # a test removed
    moved = TS.Inventory(base.parallel, base.serial, Dict("deprecated" => ["--depwarn=no"]), base.not_tests)
    @test CIPlan.inventory_additions(base, moved) === nothing
    @test isfull(p(M("test/testsuite.toml")))                          # base unknown
    # The checks of the inventory itself.
    @test_throws ErrorException TS.check(INVENTORY, vcat(TS.discover(@__DIR__), ["brand_new"]))
    # A new Julia file among the fixtures is not exempt (#203 review).
    err = try TS.check(INVENTORY, vcat(TS.discover(@__DIR__), ["data/hdm/new_regression"])); "" catch e; sprint(showerror, e) end
    @test occursin("test/data/hdm/new_regression.jl is not classified", err)
    @test_throws ErrorException TS.check(INVENTORY, filter(!=("glm"), TS.discover(@__DIR__)))
    mktempdir() do dir
      bad = joinpath(dir, "testsuite.toml")
      write(bad, "parallel = [\"a\", \"a\"]\n")
      @test_throws ErrorException TS.load(bad)
      write(bad, "parallel = [\"a\"]\nunknown = 1\n")
      @test_throws ErrorException TS.load(bad)
    end
  end

  @testset "dependencies, workflows and documentation" begin
    dep = p(M("Project.toml"))
    @test isfull(dep) && "test-cross" in dep.jobs
    ci = p(M(".github/workflows/CI.yml"))
    @test isfull(ci) && "test-cross" in ci.jobs
    @test isfull(p(M("scripts/ci_plan.jl")))
    for f in ("docs/src/index.md", "README.md", "NEWS.md", "benchmark/run.jl", ".github/workflows/Documentation.yml")
      pl = p(M(f))
      @test pl.level == "none" && isempty(pl.tests) && isempty(pl.jobs)
    end
    mixed = p(M("README.md"), M("src/transfer.jl"))
    @test mixed.level == "focused" && "transfer" in mixed.tests
    @test p(M("examples/01_lasso.jl")).jobs == ["examples", "test"]
    @test "master-integration" in p(M("scripts/test_master.jl")).jobs
  end

  @testset "empty and malformed input, explicit escalation" begin
    @test isfull(p())
    @test parse_name_status("M\tsrc/glm.jl\nR087\tsrc/a.jl\tsrc/b.jl\n") ==
          [Change('M', "src/glm.jl"), Change('R', "src/b.jl", "src/a.jl")]
    @test parse_name_status("") == Change[]
    @test parse_name_status("garbage") === nothing
    @test parse_name_status("M\ta\tb") === nothing
    @test parse_name_status("X\tsrc/glm.jl") === nothing
    esc = p(M("test/lasso.jl"); explicit_full = true)
    @test isfull(esc) && "test-cross" in esc.jobs
    @test p(M("src/transfer.jl")).notes == ["Package code changed without a test change: say in the PR how the change is tested."]
    @test isempty(p(M("src/transfer.jl"), M("test/transfer.jl")).notes)
    sched = full_plan("nightly"; cross = false)
    @test isfull(sched) && sched.coverage && !("test-cross" in sched.jobs)
    @test "test-cross" in full_plan("weekly"; cross = true).jobs
  end

  @testset "outputs" begin
    mktempdir() do dir
      out = joinpath(dir, "out")
      write_outputs(out, p(M("src/structured.jl")))
      lines = readlines(out)
      @test "level=focused" in lines && "ok=true" in lines && "jobs=examples test" in lines
      @test "tests=" * join(sort!(union(["smoke", "structured", "structured_cv_masked", "structured_golden", "structured_kernels"], IMPACT.source)), ",") in lines
      write(out, ""); write_outputs(out, p(M("src/utils.jl")))
      @test "tests=all" in readlines(out)
      write(out, ""); write_outputs(out, p(M("README.md")))
      @test "tests=" in readlines(out) && "jobs=" in readlines(out)
    end
    text = sprint(describe, p(M("src/glm.jl")))
    @test occursin("Test plan: broad", text) && occursin("src/glm.jl", text) && occursin("GLM estimators", text)
    @test occursin("| `src/glm.jl` |", sprint(io -> describe(io, p(M("src/glm.jl")); markdown = true)))
  end

  @testset "invalid maps are rejected" begin
    good = read(joinpath(@__DIR__, "impact.toml"), String)
    cases = [
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\nfamilies = [\"nosuchfamily\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\ntests = [\"nosuchtest\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"sometimes\"\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\njobs = [\"deploy\"]\nreason = \"r\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"focused\"\n",
      "[[rules]]\npaths = [\"src/x.jl\"]\nlevel = \"full\"\ntests = [\"glm\"]\nreason = \"r\"\n",
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
    end
  end

  @testset "the gate" begin
    v = CIPlan.verdict
    jobs = CIPlan.JOBS
    allskipped = Dict(j => "skipped" for j in jobs)
    with(d...) = merge(allskipped, Dict(d...))
    # Documentation only: nothing required, everything skipped.
    @test v("success", "true", "", allskipped)[1]
    # A focused plan: the required jobs passed, the others were skipped.
    @test v("success", "true", "examples test", with("test" => "success", "examples" => "success"))[1]
    # A required job that failed, was cancelled, skipped or is missing.
    for r in ("failure", "cancelled", "skipped", "")
      @test !v("success", "true", "examples test", with("test" => r, "examples" => "success"))[1]
    end
    @test !v("success", "true", "test", delete!(with("test" => "success"), "examples"))[1]
    # A job that was not required may be skipped or pass, but not fail or be cancelled.
    @test v("success", "true", "test", with("test" => "success", "examples" => "success"))[1]
    @test !v("success", "true", "test", with("test" => "success", "examples" => "failure"))[1]
    @test !v("success", "true", "test", with("test" => "success", "examples" => "cancelled"))[1]
    # A failed, cancelled or incomplete plan fails even if nothing else ran.
    for (result, ok) in (("failure", ""), ("cancelled", ""), ("skipped", ""), ("", ""), ("success", ""), ("success", "false"))
      @test !v(result, ok, "", allskipped)[1]
    end
    # A plan naming an unknown job, and results from an unknown job.
    @test !v("success", "true", "test deploy", with("test" => "success"))[1]
    @test !v("success", "true", "test", with("test" => "success", "deploy" => "skipped"))[1]
    passed, lines = v("success", "true", "test", with("test" => "failure"))
    @test !passed && any(l -> occursin("test is required, but its result is `failure`", l), lines)
    # Every job the workflow defines is a known job, and the gate needs all of them.
    workflow = joinpath(ROOT, ".github", "workflows", "CI.yml")
    if isfile(workflow)
      text = replace(read(workflow, String), "\r\n" => "\n")   # Windows checkouts have CRLF line endings
      needs = match(r"\n  gate:\n(?:.*\n)*?    needs: \[([^\]]*)\]", text)
      @test needs !== nothing && Set(strip.(split(needs[1], ','))) == Set(["plan"; collect(jobs)])
      for j in jobs
        @test occursin("\n  $j:\n", text)
        @test occursin("$j=\${{ needs.$j.result }}", text) || occursin("$j=\${{ needs['$j'].result }}", text)
      end
    end
  end

  @testset "source rules must select tests" begin
    R(paths, level; families = String[], tests = String[], jobs = String[]) = Rule(paths, level, families, tests, jobs, "r")
    map(rules...) = Impact(["smoke"], ["aqua"], Dict("glm" => ["glm", "cv_glm"]), Rule[rules...])
    new = ["src/new_estimator.jl"]
    vacuous = "select no families or tests"
    @test occursin(vacuous, only(source_problems(map(R(new, "focused")), new)))
    @test occursin(vacuous, only(source_problems(map(R(new, "broad")), new)))
    @test occursin(vacuous, only(source_problems(map(R(new, "broad"; jobs = ["examples"])), new)))
    @test isempty(source_problems(map(R(new, "focused"; families = ["glm"])), new))
    @test isempty(source_problems(map(R(new, "focused"; tests = ["lasso"])), new))
    @test isempty(source_problems(map(R(new, "full")), new))
    @test occursin("has no rule", only(source_problems(map(R(new, "none")), new)))
    @test occursin("has no rule", only(source_problems(map(), ["ext/NewExt.jl"])))
    # Rules are judged together: a job-only rule may accompany one that selects tests.
    both = map(R(new, "focused"; families = ["glm"]), R(["src/*.jl"], "focused"; jobs = ["master-integration"]))
    @test isempty(source_problems(both, new))
    pl = plan([M(new[1])], both, INVENTORY)
    @test Set(pl.tests) == Set(["glm", "cv_glm", "smoke", "aqua"]) && "master-integration" in pl.jobs
    # Overlapping rules select the union of their tests.
    two = map(R(new, "focused"; families = ["glm"]), R(["src/new_*.jl"], "broad"; tests = ["lasso"]))
    pl = plan([M(new[1])], two, INVENTORY)
    @test pl.level == "broad" && Set(pl.tests) == Set(["glm", "cv_glm", "lasso", "smoke", "aqua"])
    # Job-only and test-less rules stay valid for paths outside src/ and ext/.
    @test isempty(source_problems(map(R(["scripts/test_master.jl"], "focused"; jobs = ["master-integration"])), ["scripts/test_master.jl"]))
    # The real map accepts a vacuous rule for a file that does not exist yet, and rejects the file once it does.
    good = read(joinpath(@__DIR__, "impact.toml"), String)
    mktempdir() do dir
      path = joinpath(dir, "impact.toml")
      write(path, good * "\n[[rules]]\npaths = [\"src/new_estimator.jl\"]\nlevel = \"focused\"\nreason = \"New estimator\"\n")
      impact = load_impact(path, INVENTORY)
      @test isempty(source_problems(impact, tracked()))
      @test occursin(vacuous, only(source_problems(impact, vcat(tracked(), new))))
    end
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
      put("src/a.jl", "a\n"); put("src/b.jl", "b\n")
      for t in ("a", "b", "smoke"); put("test/$t.jl", "# $t\n"); end
      put("test/data/hdm/baseline.jl", "# writes fixtures\n"); put("test/data/hdm/x.csv", "1\n")
      inventory = "parallel = [\"a\", \"b\", \"smoke\"]\n[not_tests]\n\"data/hdm/baseline\" = \"fixture generator\"\n"
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
        paths = ["test/data/hdm/**"]
        level = "focused"
        tests = ["a"]
        reason = "fixtures of a"
        [[rules]]
        paths = ["test/impact.toml", "test/testsuite.toml"]
        level = "focused"
        tests = ["smoke"]
        reason = "the maps"
        """
      put("test/impact.toml", map)
      g("add", "."); g("commit", "-qm", "base")
      head() = strip(read(`git -C $dir rev-parse HEAD`, String))
      base = head()
      pr(; kw...) = compute(["--event=pull_request", "--base=$base", "--head=HEAD"]; dir, kw...)
      fails(f) = try f(); "" catch e; sprint(showerror, e) end

      put("src/a.jl", "a2\n"); g("commit", "-qam", "change a")
      ok = pr()
      @test ok.level == "focused" && ok.tests == ["a", "smoke"]
      # The tracked files cannot be listed: full suite, said in both outputs.
      blind = pr(list_files = _ -> nothing)
      @test isfull(blind) && blind.coverage == ok.coverage
      @test blind.reasons[1][1] == "(source map check)" && occursin("git could not list the tracked files", blind.reasons[1][2])
      @test occursin("git could not list the tracked files", sprint(describe, blind))
      @test occursin("git could not list the tracked files", sprint(io -> describe(io, blind; markdown = true)))

      # A source file without a rule, or with a rule selecting no tests, stops the plan.
      put("src/c.jl", "c\n"); g("add", "."); g("commit", "-qm", "add c")
      @test occursin("src/c.jl has no rule", fails(pr))
      put("test/impact.toml", map * "[[rules]]\npaths = [\"src/c.jl\"]\nlevel = \"focused\"\nreason = \"New estimator\"\n")
      g("commit", "-qam", "vacuous rule for c")
      @test occursin("src/c.jl: its rules select no families or tests", fails(pr))
      @test isfull(pr(list_files = _ -> nothing))
      g("rm", "-q", "src/c.jl"); put("test/impact.toml", map); g("commit", "-qam", "drop c")

      # A new Julia file among the fixtures must be classified before anything is planned ...
      put("test/data/hdm/new_regression.jl", "# a new test\n"); g("add", "."); g("commit", "-qm", "new fixture test")
      @test occursin("test/data/hdm/new_regression.jl is not classified", fails(pr))
      @test occursin("is not classified", fails(() -> pr(list_files = _ -> nothing)))
      # ... and once it is a test, it runs.
      put("test/testsuite.toml", replace(inventory, "\"smoke\"]" => "\"smoke\", \"data/hdm/new_regression\"]"))
      put("test/impact.toml", replace(map, "a = [\"a\"]" => "a = [\"a\", \"data/hdm/new_regression\"]"))
      g("commit", "-qam", "classify it")
      classified = pr()
      @test classified.level == "focused"
      @test classified.tests == ["a", "data/hdm/new_regression", "smoke"]
    end
  end

  HASGIT && @testset "changed files from git" begin
    gitrepo() do dir, g
      write(joinpath(dir, "a.jl"), "a\n"); write(joinpath(dir, "b.jl"), "b\n" ^ 20)
      g("add", "."); g("commit", "-qm", "one")
      first = strip(read(`git -C $dir rev-parse HEAD`, String))
      write(joinpath(dir, "a.jl"), "a2\n"); g("commit", "-qam", "two")
      g("mv", "b.jl", "c.jl"); g("commit", "-qm", "three\n\nCI: full")
      # Every commit of the range counts, and a rename is one change.
      @test Set(changed_files(first, "HEAD"; dir)) == Set([Change('M', "a.jl"), Change('R', "c.jl", "b.jl")])
      @test requests_full("HEAD"; dir)
      @test !requests_full("HEAD^"; dir)
      # Missing and all-zero bases are unknown, never empty.
      @test changed_files("0000000000000000000000000000000000000000", "HEAD"; dir) === nothing
      @test changed_files("1234567890abcdef1234567890abcdef12345678", "HEAD"; dir) === nothing
      @test changed_files("HEAD~5", "HEAD"; dir) === nothing
      @test changed_files("HEAD", "HEAD"; dir) == Change[]
      # The working tree, with untracked files.
      write(joinpath(dir, "d.jl"), "new\n"); rm(joinpath(dir, "a.jl"))
      @test Set(changed_files("HEAD", nothing; dir)) == Set([Change('D', "a.jl"), Change('A', "d.jl")])
      @test inventory_at("HEAD"; dir) === nothing
    end
    CHECKOUT && @test inventory_at("HEAD"; dir = ROOT) isa TS.Inventory
  end
end

end
