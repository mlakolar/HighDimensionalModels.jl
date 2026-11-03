# The test inventory (test/testsuite.toml): which Julia files under test/ are test files and
# how each one runs. It depends only on TOML, so CI scripts can load it without installing the
# package. test/runtests.jl refuses to run when a file is unclassified or a listed file is
# missing, so a new test file cannot be silently skipped. The selection and worker-count helpers
# at the end are used by test/runtests.jl; test/testsuite_validation.jl tests all of it.
module TestSuite

using TOML

export Inventory, discover, check, testsuite, alltests

struct Inventory
  "Test files run by the parallel runner, each in a fresh module on a reused worker."
  parallel::Vector{String}
  "Test files run one at a time after the parallel batch."
  serial::Vector{String}
  "Test files run in a fresh child process with these extra Julia flags."
  subprocess::Dict{String,Vector{String}}
  "Julia files that are not test files (name => reason), each named exactly; there are no patterns."
  not_tests::Dict{String,String}
end

const KEYS = ("parallel", "serial", "subprocess", "not_tests")

function load(path::AbstractString)
  data = TOML.parsefile(path)
  unknown = setdiff(keys(data), KEYS)
  isempty(unknown) || error("$path: unknown keys $(join(sort!(collect(unknown)), ", ")); expected $(join(KEYS, ", "))")
  names(key) = begin
    v = get(data, key, String[])
    v isa Vector && all(x -> x isa String, v) || error("$path: `$key` must be a list of test names")
    String.(v)
  end
  subprocess = Dict{String,Vector{String}}()
  for (name, flags) in get(data, "subprocess", Dict{String,Any}())
    flags isa Vector && all(x -> x isa String, flags) || error("$path: subprocess.$name must be a list of Julia flags")
    subprocess[name] = String.(flags)
  end
  not_tests = Dict{String,String}()
  for (name, reason) in get(data, "not_tests", Dict{String,Any}())
    # Exact names only: a pattern would also exempt files added later, without a decision.
    occursin('*', name) && error("$path: not_tests.\"$name\": name each file; patterns are not accepted")
    reason isa String && !isempty(reason) || error("$path: not_tests.\"$name\" needs a reason")
    not_tests[name] = reason
  end
  inv = Inventory(names("parallel"), names("serial"), subprocess, not_tests)
  seen = Dict{String,Int}()
  for name in alltests(inv; sorted = false)
    seen[name] = get(seen, name, 0) + 1
  end
  dup = sort!([n for (n, c) in seen if c > 1])
  isempty(dup) || error("$path: listed more than once: $(join(dup, ", "))")
  return inv
end

"Every test name of the inventory, sorted."
function alltests(inv::Inventory; sorted::Bool = true)
  names = vcat(inv.parallel, inv.serial, collect(keys(inv.subprocess)))
  return sorted ? sort!(names) : names
end

"The Julia files under `dir`, as paths relative to it without `.jl`, with `/` separators."
function discover(dir::AbstractString)
  found = String[]
  for (root, _, files) in walkdir(dir), file in files
    endswith(file, ".jl") || continue
    rel = replace(relpath(joinpath(root, file), dir), '\\' => '/')
    rel == "runtests.jl" && continue
    push!(found, rel[1:end-3])
  end
  return sort!(found)
end

"""
    check(inv, discovered)

Throw an error listing every problem: a discovered file that is neither a test nor a declared
non-test (or is both), a listed test without a file, and a non-test entry matching no file.
"""
function check(inv::Inventory, discovered)
  problems = String[]
  tests = Set(alltests(inv))
  for name in sort!(collect(intersect(tests, keys(inv.not_tests))))
    push!(problems, "`$name` is listed both as a test and in not_tests")
  end
  for name in discovered
    name in tests || haskey(inv.not_tests, name) ||
      push!(problems, "test/$name.jl is not classified: add it to a list in test/testsuite.toml")
  end
  for name in sort!(collect(setdiff(tests, discovered)))
    push!(problems, "test/testsuite.toml lists `$name`, but test/$name.jl does not exist")
  end
  for name in sort!(collect(setdiff(keys(inv.not_tests), discovered)))
    push!(problems, "test/testsuite.toml: not_tests entry \"$name\", but test/$name.jl does not exist")
  end
  isempty(problems) || error("The test inventory is inconsistent:\n  " * join(problems, "\n  "))
  return nothing
end

"The test expressions for ParallelTestRunner, one per test name."
function testsuite(inv::Inventory, dir::AbstractString)
  suite = Dict{String,Expr}()
  for name in vcat(inv.parallel, inv.serial)
    suite[name] = :(include($(joinpath(dir, name * ".jl"))))
  end
  for (name, flags) in inv.subprocess
    path = joinpath(dir, name * ".jl")
    # The worker's flags (check-bounds, depwarn, coverage) come first; the later flags win.
    suite[name] = :(run(Cmd([Base.julia_cmd().exec..., "--startup-file=no", $(flags...),
                             "--project=" * dirname(Base.active_project()), $path])))
  end
  return suite
end

"""
    select_only!(suite, value)

Keep exactly the tests named in `value` (`--only=a,b`), a comma-separated list. An empty list
or a name that is not a test is an error, so a typo cannot select nothing and pass.
"""
function select_only!(suite::AbstractDict, value::Union{Nothing,AbstractString})
  names = value === nothing ? String[] : filter!(!isempty, strip.(split(value, ',')))
  isempty(names) && error("--only needs test names, for example --only=lasso,cv")
  unknown = filter(name -> !haskey(suite, name), names)
  isempty(unknown) || error("--only: no test file $(join(unknown, ", ")); `--list` shows the names")
  filter!(((name, _),) -> name in names, suite)
  return suite
end

"The Julia threads of each worker: `HDM_TEST_THREADS`, default 2."
function worker_threads(value::AbstractString = get(ENV, "HDM_TEST_THREADS", "2"))
  threads = tryparse(Int, value)
  threads !== nothing && threads >= 1 || error("HDM_TEST_THREADS must be a positive integer, got \"$value\"")
  return threads
end

"The CPU threads this process may use: `Sys.EFFECTIVE_CPU_THREADS` (cgroup and affinity limits)
where Julia has it, else `Sys.CPU_THREADS` (Julia 1.10)."
cpu_threads() = @static isdefined(Sys, :EFFECTIVE_CPU_THREADS) ? Sys.EFFECTIVE_CPU_THREADS : Sys.CPU_THREADS

"""
    max_workers(threads, cpus = cpu_threads())

The default worker count's upper bound: the CPU budget divided among workers of `threads`
Julia threads each, at least one. ParallelTestRunner applies its memory bound on top, and
`--jobs`/`PTR_NUM_JOBS` override both.
"""
max_workers(threads::Integer, cpus::Integer = cpu_threads()) = max(1, cpus ÷ threads)

end
