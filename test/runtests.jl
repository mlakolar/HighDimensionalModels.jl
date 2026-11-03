# Runs the test files of test/testsuite.toml in parallel worker processes (ParallelTestRunner).
# With no arguments, as from a plain `Pkg.test()`, it runs every test file.
#
#   julia scripts/test_registered.jl                     every test file
#   julia scripts/test_registered.jl --only=lasso,cv     exactly these files
#   julia scripts/test_registered.jl lowrank             files whose names start with `lowrank`
#   julia scripts/test_registered.jl --list              the available test files
#   julia scripts/test_registered.jl --jobs=4 --verbose  worker count, per-file detail
#
# Each worker runs with HDM_TEST_THREADS Julia threads (default 2) and one BLAS thread. The
# default worker count keeps workers × threads within the CPU threads available to this process
# (Sys.EFFECTIVE_CPU_THREADS where Julia has it, so a container's CPU quota counts), and
# ParallelTestRunner lowers it further when memory is short. `--jobs=N` or PTR_NUM_JOBS sets the
# worker count. CONTRIBUTING.md describes the testing workflow.
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
# `--list` shows every file, as ParallelTestRunner's own filter does.
only !== nothing && args.list === nothing && TestSuite.select_only!(testsuite, something(only))
filter_tests!(testsuite, args)
isempty(testsuite) && error("no test file matches the selection: $(join(ARGS, " "))")

threads = TestSuite.worker_threads()
if args.list === nothing
  selected = sort!(collect(keys(testsuite)))
  all_selected = length(selected) == length(alltests(inventory))
  commit = haskey(ENV, "GITHUB_SHA") ? ", commit $(ENV["GITHUB_SHA"])" : ""
  cpu = "$(strip(first(Sys.cpu_info()).model)), $(TestSuite.cpu_threads()) of $(Sys.CPU_THREADS) threads available"
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
         max_default_jobs = TestSuite.max_workers(threads))
