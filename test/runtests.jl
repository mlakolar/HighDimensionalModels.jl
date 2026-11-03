# Runs the test files of test/testsuite.toml in parallel worker processes (ParallelTestRunner).
# With no arguments, as from a plain `Pkg.test()`, it runs every test file.
#
#   julia scripts/test_registered.jl                     every test file
#   julia scripts/test_registered.jl --only=lasso,cv     exactly these files
#   julia scripts/test_registered.jl lowrank             files whose names start with `lowrank`
#   julia scripts/test_registered.jl --list              the available test files
#   julia scripts/test_registered.jl --jobs=4 --verbose  worker count, per-file detail
#
# Each worker runs with HDM_TEST_THREADS Julia threads (default 2) and one BLAS thread; the
# default worker count keeps workers × threads within the CPU threads. `--jobs=N` or
# PTR_NUM_JOBS sets the worker count. CONTRIBUTING.md describes the testing workflow.
using HighDimensionalModels
using ParallelTestRunner

include(joinpath(@__DIR__, "testsuite.jl"))
using .TestSuite

inventory = TestSuite.load(joinpath(@__DIR__, "testsuite.toml"))
# An unclassified or missing file stops every run, whatever the arguments.
TestSuite.check(inventory, TestSuite.discover(@__DIR__))
testsuite = TestSuite.testsuite(inventory, @__DIR__)

args = parse_args(ARGS; custom = ["only"])
only = args.custom["only"]
if only !== nothing && args.list === nothing
  value = something(only)
  names = value === nothing ? String[] : filter!(!isempty, strip.(split(value, ',')))
  isempty(names) && error("--only needs test names, for example --only=lasso,cv")
  unknown = filter(name -> !haskey(testsuite, name), names)
  isempty(unknown) || error("--only: no test file $(join(unknown, ", ")); `--list` shows the names")
  filter!(((name, _),) -> name in names, testsuite)
end
filter_tests!(testsuite, args)
isempty(testsuite) && error("no test file matches the selection: $(join(ARGS, " "))")

threads = parse(Int, get(ENV, "HDM_TEST_THREADS", "2"))
threads >= 1 || error("HDM_TEST_THREADS must be a positive integer")
if args.list === nothing
  selected = sort!(collect(keys(testsuite)))
  all_selected = length(selected) == length(alltests(inventory))
  commit = haskey(ENV, "GITHUB_SHA") ? ", commit $(ENV["GITHUB_SHA"])" : ""
  cpu = "$(strip(first(Sys.cpu_info()).model)), $(Sys.CPU_THREADS) threads"
  println("HighDimensionalModels tests: $(all_selected ? "every file" : "a selection"), ",
          "$(length(selected)) of $(length(alltests(inventory))) files; Julia $VERSION on $(Sys.MACHINE) ($cpu)$commit")
  println("  workers: $threads Julia threads and 1 BLAS thread each")
  println("  ", join(selected, " "))
end

runtests(HighDimensionalModels, args; testsuite,
         serial = filter(name -> haskey(testsuite, name), inventory.serial), serial_position = :after,
         # Threads exercise the threaded code paths (cross-validation folds, stability selection).
         exeflags = ["--threads=$threads"],
         init_worker_code = :(using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)),
         max_default_jobs = max(1, Sys.CPU_THREADS ÷ threads))
