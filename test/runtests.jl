# Runs the test files of test/testsuite.toml in parallel worker processes (ParallelTestRunner).
# Run it through scripts/test.jl, which builds and verifies the environment (the checkouts of
# ProximalBase and ProximalSolvers that CONTRIBUTING.md describes). With no arguments it runs
# every test file. A plain `Pkg.test()` has no verified environment, and test/dependencies.jl
# fails there.
#
#   julia scripts/test.jl                          every test file
#   julia scripts/test.jl --only=lasso,smoke       exactly these files
#   julia scripts/test.jl lasso                    files whose names start with `lasso`
#   julia scripts/test.jl --list                   the available test files
#   julia scripts/test.jl --jobs=2 --verbose       worker count, per-file detail
#
# Each worker runs with one Julia thread and one BLAS thread: neither the package nor its
# tests start threads. The default worker count is at most the CPU threads available to this
# process (Sys.EFFECTIVE_CPU_THREADS where Julia has it, so a container's CPU quota counts), and
# ParallelTestRunner lowers it further when memory is short. `--jobs=N` or PTR_NUM_JOBS sets the
# worker count.
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

if args.list === nothing
  selected = sort!(collect(keys(testsuite)))
  all_selected = length(selected) == length(alltests(inventory))
  commit = haskey(ENV, "GITHUB_SHA") ? ", commit $(ENV["GITHUB_SHA"])" : ""
  cpu = "$(strip(first(Sys.cpu_info()).model)), $(TestSuite.cpu_threads()) of $(Sys.CPU_THREADS) threads available"
  println("HighDimensionalModels tests: $(all_selected ? "every file" : "a selection"), ",
          "$(length(selected)) of $(length(alltests(inventory))) files; Julia $VERSION on $(Sys.MACHINE) ($cpu)$commit")
  println("  workers: 1 Julia thread and 1 BLAS thread each")
  # scripts/test.jl compares this line with the files the runner reports (TestSuite.check_run).
  println(TestSuite.SELECTION_PREFIX, join(selected, " "))
end

runtests(HighDimensionalModels, args; testsuite,
         serial = filter(name -> haskey(testsuite, name), inventory.serial), serial_position = :after,
         exeflags = ["--threads=1"],
         init_worker_code = :(using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1)),
         max_default_jobs = TestSuite.cpu_threads())
