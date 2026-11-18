# Run the test suite against fixed ProximalBase and ProximalSolvers commits:
#
#   julia scripts/test.jl [runner arguments]
#
# It resolves the two commits (HDM_PROXIMALBASE_REV, HDM_PROXIMALSOLVERS_REV, else the tip of
# master, once), prepares their checkouts in HDM_DEPS_DIR, builds a test environment that
# develops them and this package by path, checks the manifest and the loaded packages against
# the commits, writes a receipt next to the environment (receipt.toml, which also lists the test
# files that ran), and runs test/runtests.jl in a child process with `--check-bounds=yes
# --depwarn=error`. Runner arguments pass through: `--only=lasso,smoke`, `--jobs=2`, `--list`,
# `--verbose`. JULIA_TEST_COVERAGE=true collects coverage. Any setup failure, failing test, or
# selected test file missing from the runner's report ends with a nonzero exit code.
# See CONTRIBUTING.md.
include(joinpath(@__DIR__, "dependencies.jl"))
include(joinpath(@__DIR__, "..", "test", "testsuite.jl"))
using .Dependencies, .TestSuite, TOML

"The test files that the arguments should select, or `nothing` if only the runner can tell."
function expected_selection(args)
  only = [a[length("--only=")+1:end] for a in args if startswith(a, "--only=")]
  length(only) > 1 && error("more than one --only argument")
  isempty(only) || return String.(filter!(!isempty, strip.(split(only[1], ','))))
  all(startswith("-"), args) || return nothing
  return TestSuite.alltests(TestSuite.load(joinpath(Dependencies.ROOT, "test", "testsuite.toml")))
end

function main(args)
  revs, paths = Dependencies.prepare()
  env = joinpath(Dependencies.deps_dir(), "test-env", "v$(VERSION.major).$(VERSION.minor)")
  project = joinpath(Dependencies.ROOT, "Project.toml")
  original = read(project)
  Dependencies.write_test_project(env)
  Dependencies.develop(env, paths)
  read(project) == original || error("the test setup changed Project.toml of the package")
  # The manifest, and the loaded packages in a process of the environment (also test/dependencies.jl).
  manifest = Dependencies.verify_environment(env, revs, paths)
  receipt = Dependencies.write_receipt(env, revs, paths, manifest)
  println("\nReceipt ($receipt):\n", read(receipt, String))
  Dependencies.summary(["### Dependencies", "",
                        ["- $name `$(revs[name])`" for name in Dependencies.PACKAGES]..., ""])

  coverage = get(ENV, "JULIA_TEST_COVERAGE", "false") == "true" ? "user" : "none"
  color = get(stdout, :color, false) ? "yes" : "no"
  runtests = joinpath(Dependencies.ROOT, "test", "runtests.jl")
  cmd = `$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --depwarn=error --code-coverage=$coverage
         --color=$color --project=$env $runtests $args`
  lines = String[]
  process = open(cmd, "r")
  for line in eachline(process)
    println(line)
    push!(lines, line)
  end
  wait(process)
  success(process) || (println(stderr, "ERROR: the test run failed (exit code $(process.exitcode))"); return false)
  any(a -> a in ("--list", "--help", "-h"), args) && return true

  ran = TestSuite.check_run(lines, expected_selection(args))
  data = TOML.parsefile(receipt)
  data["tests"] = Dict("ran" => ran)
  open(io -> TOML.print(io, data; sorted = true), receipt, "w")
  println("\nTest files run (", length(ran), "): ", join(ran, " "))
  return true
end

try
  main(ARGS) || exit(1)
catch err
  println(stderr, "ERROR: ", sprint(showerror, err))
  exit(1)
end
