# The test inventory (test/testsuite.toml): which Julia files under test/ are test files and
# how each one runs. It depends only on TOML, so CI scripts can load it without installing the
# package. test/runtests.jl refuses to run when a file is unclassified or a listed file is
# missing, so a new test file cannot be silently skipped.
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
  "Julia files that are not test files (glob => reason)."
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
  for (glob, reason) in get(data, "not_tests", Dict{String,Any}())
    reason isa String && !isempty(reason) || error("$path: not_tests.\"$glob\" needs a reason")
    not_tests[glob] = reason
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

# `*` matches within a path segment, `**` across segments.
function glob_regex(glob::AbstractString)
  parts = split(glob, "**")
  inner = map(p -> join(map(s -> replace(s, r"[\\^$.|?+()\[\]{}]" => x -> "\\" * x), split(p, '*')), "[^/]*"), parts)
  return Regex("^" * join(inner, ".*") * "\$")
end

matches(glob, name) = occursin(glob_regex(glob), name)

"""
    check(inv, discovered)

Throw an error listing every problem: a discovered file that is neither a test nor a declared
non-test (or is both), a listed test without a file, and a non-test entry matching no file.
"""
function check(inv::Inventory, discovered)
  problems = String[]
  tests = Set(alltests(inv))
  for name in discovered
    globs = [g for g in keys(inv.not_tests) if matches(g, name)]
    if name in tests
      isempty(globs) || push!(problems, "test/$name.jl is a test and matches not_tests $(join(globs, ", "))")
    elseif isempty(globs)
      push!(problems, "test/$name.jl is not classified: add it to a list in test/testsuite.toml")
    end
  end
  for name in sort!(collect(setdiff(tests, discovered)))
    push!(problems, "test/testsuite.toml lists `$name`, but test/$name.jl does not exist")
  end
  for glob in sort!(collect(keys(inv.not_tests)))
    any(name -> matches(glob, name), discovered) ||
      push!(problems, "test/testsuite.toml: not_tests entry \"$glob\" matches no file")
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

end
