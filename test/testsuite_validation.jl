# Tests of the test inventory and the runner's selection (test/testsuite.jl, test/runtests.jl):
# every check that keeps a test file from being skipped silently. Inventories are built in
# temporary directories; the few child processes list files or run two tiny files, never the suite.
using Test, TOML
using ParallelTestRunner: parse_args, filter_tests!

include(joinpath(@__DIR__, "testsuite.jl"))
const TS = TestSuite

# A directory with these Julia files (relative names without `.jl`) and a runtests.jl.
function tree(names; contents = Dict{String,String}())
  dir = mktempdir()
  for name in vcat(names, "runtests")
    path = joinpath(dir, name * ".jl")
    mkpath(dirname(path))
    write(path, get(contents, name, "# $name\n"))
  end
  return dir
end

# Load an inventory from TOML text.
function inventory(text)
  path = joinpath(mktempdir(), "testsuite.toml")
  write(path, text)
  return TS.load(path)
end

# The error message of `f()`, or `nothing` if it does not throw.
function message(f)
  try
    f()
  catch err
    return sprint(showerror, err)
  end
  return nothing
end

const VALID = """
parallel = ["a", "sub/c"]
serial = ["b"]
[subprocess]
x = ["--depwarn=yes"]
[not_tests]
"helper" = "a helper module"
"data/hdm/fixture" = "a fixture generator, run by hand"
"""
const FILES = ["a", "b", "sub/c", "x", "helper", "data/hdm/fixture"]

@testset "inventory loading" begin
  inv = inventory(VALID)
  @test inv.parallel == ["a", "sub/c"]
  @test inv.serial == ["b"]
  @test inv.subprocess == Dict("x" => ["--depwarn=yes"])
  @test inv.not_tests == Dict("helper" => "a helper module", "data/hdm/fixture" => "a fixture generator, run by hand")
  @test TS.alltests(inv) == ["a", "b", "sub/c", "x"]
  @test isempty(TS.alltests(inventory("")))

  @test_throws TOML.ParserError inventory("parallel = [\"a\"")
  @test_throws TOML.ParserError inventory("[subprocess]\nx = []\nx = []")
  @test occursin("unknown keys tests", message(() -> inventory("tests = [\"a\"]")))
  @test occursin("`parallel` must be a list", message(() -> inventory("parallel = \"a\"")))
  @test occursin("`serial` must be a list", message(() -> inventory("serial = [1]")))
  @test occursin("subprocess.x must be a list", message(() -> inventory("[subprocess]\nx = \"--depwarn=yes\"")))
  @test occursin("needs a reason", message(() -> inventory("[not_tests]\nhelper = \"\"")))
  @test occursin("needs a reason", message(() -> inventory("[not_tests]\nhelper = 1")))

  # Exact names only: a pattern would exempt files added later.
  for pattern in ("data/**", "golden/*", "*")
    @test occursin("patterns are not accepted", message(() -> inventory("[not_tests]\n\"$pattern\" = \"fixtures\"")))
  end

  @test occursin("listed more than once: a", message(() -> inventory("parallel = [\"a\", \"a\"]")))
  @test occursin("listed more than once: a", message(() -> inventory("parallel = [\"a\"]\nserial = [\"a\"]")))
  @test occursin("listed more than once: a", message(() -> inventory("parallel = [\"a\"]\n[subprocess]\na = []")))
  @test occursin("listed more than once: a, b",
                 message(() -> inventory("parallel = [\"a\", \"b\"]\nserial = [\"b\", \"a\"]")))
end

@testset "discovery" begin
  dir = tree(FILES)
  write(joinpath(dir, "sub", "notes.md"), "not Julia")
  @test TS.discover(dir) == sort(FILES)
  @test !("runtests" in TS.discover(dir))
end

@testset "inventory checks" begin
  inv = inventory(VALID)
  @test TS.check(inv, TS.discover(tree(FILES))) === nothing

  # A new file anywhere, including a fixture directory, needs a decision.
  for new in ("new", "sub/new", "data/hdm/new_regression", "golden/structured/new")
    err = message(() -> TS.check(inv, TS.discover(tree(vcat(FILES, new)))))
    @test occursin("test/$new.jl is not classified", err)
  end

  err = message(() -> TS.check(inv, TS.discover(tree(setdiff(FILES, ["sub/c"])))))
  @test occursin("lists `sub/c`, but test/sub/c.jl does not exist", err)
  err = message(() -> TS.check(inv, TS.discover(tree(setdiff(FILES, ["x"])))))
  @test occursin("lists `x`, but test/x.jl does not exist", err)

  err = message(() -> TS.check(inv, TS.discover(tree(setdiff(FILES, ["data/hdm/fixture"])))))
  @test occursin("not_tests entry \"data/hdm/fixture\", but test/data/hdm/fixture.jl does not exist", err)

  overlap = inventory(VALID * "\"a\" = \"also a helper\"\n")
  err = message(() -> TS.check(overlap, TS.discover(tree(FILES))))
  @test occursin("`a` is listed both as a test and in not_tests", err)

  # Every problem is reported at once.
  err = message(() -> TS.check(inv, TS.discover(tree(vcat(setdiff(FILES, ["b", "helper"]), "new")))))
  @test occursin("test/new.jl is not classified", err)
  @test occursin("lists `b`", err)
  @test occursin("not_tests entry \"helper\"", err)
end

@testset "this repository's inventory" begin
  inv = TS.load(joinpath(@__DIR__, "testsuite.toml"))
  @test TS.check(inv, TS.discover(@__DIR__)) === nothing
  @test TS.alltests(inv) == ["aqua", "ci_plan", "dependencies", "lasso", "lasso_reference", "smoke", "statsapi", "testsuite_validation"]
  @test isempty(inv.serial) && isempty(inv.subprocess)
  # Helpers and fixtures are named one by one, with a reason.
  @test Set(keys(inv.not_tests)) == Set(["testsuite"])
end

@testset "test expressions" begin
  dir = tree(FILES)
  suite = TS.testsuite(inventory(VALID), dir)
  @test sort!(collect(keys(suite))) == ["a", "b", "sub/c", "x"]
  @test suite["sub/c"] == :(include($(joinpath(dir, "sub/c.jl"))))
  @test !haskey(suite, "helper")
  cmd = string(suite["x"])
  @test occursin("\"--startup-file=no\", \"--depwarn=yes\", \"--project=\"", cmd)
  @test occursin(repr(joinpath(dir, "x.jl")), cmd)   # as a string literal: Windows paths print escaped
end

@testset "--only" begin
  suite() = Dict(name => :(nothing) for name in ("lowrank", "lowrank_sparse", "lasso", "cv"))
  @test sort!(collect(keys(TS.select_only!(suite(), "lowrank")))) == ["lowrank"]
  @test sort!(collect(keys(TS.select_only!(suite(), " lasso, cv ,")))) == ["cv", "lasso"]
  @test occursin("no test file lowrnk, cvv", message(() -> TS.select_only!(suite(), "lowrnk,lasso,cvv")))
  @test occursin("--only needs test names", message(() -> TS.select_only!(suite(), "")))
  @test occursin("--only needs test names", message(() -> TS.select_only!(suite(), " , ")))
  @test occursin("--only needs test names", message(() -> TS.select_only!(suite(), nothing)))

  # Positional arguments are prefixes (ParallelTestRunner's filter_tests!, as in runtests.jl).
  prefixed = suite()
  filter_tests!(prefixed, parse_args(["lowrank"]))
  @test sort!(collect(keys(prefixed))) == ["lowrank", "lowrank_sparse"]
  excluded = suite()
  filter_tests!(excluded, parse_args(["lowrank", "!lowrank_sparse"]))
  @test sort!(collect(keys(excluded))) == ["lowrank"]
  listed = suite()
  filter_tests!(listed, parse_args(["--list", "lowrank"]))
  @test length(listed) == 4
end

@testset "worker count" begin
  @test TS.worker_threads("2") == 2
  @test TS.worker_threads("1") == 1
  for bad in ("0", "-1", "two", "", "1.5")
    @test occursin("HDM_TEST_THREADS must be a positive integer", message(() -> TS.worker_threads(bad)))
  end

  # Workers × threads stay within the CPU budget, with at least one worker.
  @test TS.max_workers(2, 4) == 2      # a 4-CPU container on a larger host
  @test TS.max_workers(2, 32) == 16
  @test TS.max_workers(2, 3) == 1
  @test TS.max_workers(2, 1) == 1
  @test TS.max_workers(1, 12) == 12
  @test TS.max_workers(4, 12) == 3
  @test 1 <= TS.cpu_threads() <= Sys.CPU_THREADS
  if isdefined(Sys, :EFFECTIVE_CPU_THREADS)
    @test TS.cpu_threads() == Sys.EFFECTIVE_CPU_THREADS
  else
    @test TS.cpu_threads() == Sys.CPU_THREADS
  end
  @test TS.max_workers(2) == max(1, TS.cpu_threads() ÷ 2)
end

@testset "the runner's report is checked against the selection" begin
  table(names) = ["HighDimensionalModels tests: a selection, $(length(names)) of 8 files",
                  "  workers: 2 Julia threads and 1 BLAS thread each",
                  TS.SELECTION_PREFIX * join(names, " "),
                  "Running $(length(names)) tests using 2 parallel jobs.",
                  "                   │   Test   │ ──────────────── CPU ──────────────── │",
                  "Test      (Worker) │ time (s) │ GC (s) │ GC % │ Alloc (MB) │ RSS (MB) │",
                  ["$(rpad(name, 14)) ($i) │     0.46 │   0.01 │  2.5 │      53.32 │   394.35 │" for (i, name) in enumerate(names)]...,
                  "", "Test Summary: | Pass  Total  Time", "  Overall     |    4      4  4.2s", "    SUCCESS"]
  @test TS.check_run(table(["lasso", "smoke"])) == ["lasso", "smoke"]
  @test TS.check_run(table(["lasso", "smoke"]), ["smoke", "lasso"]) == ["lasso", "smoke"]
  # Colored output (CI runs with --color=yes) and the detail lines of --verbose.
  colored = replace.(table(["lasso", "smoke"]), "lasso   " => "\e[32mlasso\e[39m   ")
  @test TS.check_run(colored) == ["lasso", "smoke"]
  verbose = vcat(table(["lasso"])[1:6], ["lasso          (1) │         started at 2026-11-17T17:55:48.954"], table(["lasso"])[7:end])
  @test TS.check_run(verbose) == ["lasso"]
  # The selection differs from the expectation, is missing, is empty, or is printed twice.
  @test occursin("selected lasso, expected lasso smoke", message(() -> TS.check_run(table(["lasso"]), ["lasso", "smoke"])))
  @test occursin("selected lasso smoke, expected lasso", message(() -> TS.check_run(table(["lasso", "smoke"]), ["lasso"])))
  @test occursin("found 0", message(() -> TS.check_run(filter(l -> !startswith(l, TS.SELECTION_PREFIX), table(["lasso"])))))
  @test occursin("selected no test file", message(() -> TS.check_run(table(String[]))))
  @test occursin("found 2", message(() -> TS.check_run(vcat(table(["lasso"]), table(["lasso"])))))
  # A selected file the runner did not report, whether the table is short or absent.
  short = table(["lasso", "smoke"])
  short[3] = TS.SELECTION_PREFIX * "lasso smoke statsapi"
  @test occursin("not reported by the runner: statsapi", message(() -> TS.check_run(short)))
  @test occursin("not reported by the runner: lasso", message(() -> TS.check_run(table(["lasso"])[1:3])))
  # A name that only prefixes a reported file does not count.
  prefix = table(["lasso_reference"])
  prefix[3] = TS.SELECTION_PREFIX * "lasso"
  @test occursin("not reported by the runner: lasso", message(() -> TS.check_run(prefix)))
end

# A child Julia in this test environment, without coverage (later flags win).
julia(args...) = Cmd([Base.julia_cmd().exec..., "--startup-file=no", "--code-coverage=none",
                      "--project=" * dirname(Base.active_project()), args...])

# Run `cmd`; return its exit code and its combined output.
function output(cmd)
  io = IOBuffer()
  process = run(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
  return process.exitcode, String(take!(io))
end

@testset "subprocess failures fail the run" begin
  dir = tree(["passes", "fails", "ok"];
             contents = Dict("passes" => "exit(0)\n", "fails" => "exit(3)\n", "ok" => "using Test\n@test true\n"))
  inv = TS.Inventory(String[], String[], Dict("passes" => String[], "fails" => String[]), Dict{String,String}())
  suite = TS.testsuite(inv, dir)
  @test Core.eval(Module(), suite["passes"]) isa Base.Process
  @test_throws ProcessFailedException Core.eval(Module(), suite["fails"])

  # The same expressions under ParallelTestRunner: the run fails, and the passing file ran.
  script = joinpath(dir, "probe.jl")
  write(script, """
    using ParallelTestRunner, Test
    include($(repr(joinpath(@__DIR__, "testsuite.jl"))))
    inv = TestSuite.Inventory(["ok"], String[], Dict("fails" => String[]), Dict{String,String}())
    suite = TestSuite.testsuite(inv, $(repr(dir)))
    runtests(Main, parse_args(["--jobs=1", "--verbose"]); testsuite = suite, history_key = "hdm-testsuite-validation")
    """)
  code, out = output(julia(script))
  @test code != 0
  @test occursin("ProcessExited(3)", out)
  @test occursin("Test run finished with errors", out)
  @test occursin(r"ok\s+\(\d+\)", out)
end

@testset "runtests.jl arguments" begin
  runtests = joinpath(@__DIR__, "runtests.jl")
  code, out = output(julia(runtests, "--list"))
  @test code == 0
  @test occursin("testsuite_validation", out) && occursin("lasso_reference", out)
  code, out = output(julia(runtests, "--only=lasso,no_such_test"))
  @test code != 0
  @test occursin("--only: no test file no_such_test", out)
  code, out = output(julia(runtests, "no_such_prefix"))
  @test code != 0
  @test occursin("no test file matches the selection: no_such_prefix", out)
end
