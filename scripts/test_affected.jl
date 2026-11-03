# Run the tests your local changes can affect (test/impact.toml), with registered dependencies.
#
#   julia scripts/test_affected.jl                    changes since the merge base with origin/master,
#                                                     including uncommitted and untracked files
#   julia scripts/test_affected.jl --base=HEAD~3      changes since another revision
#   julia scripts/test_affected.jl --dry-run          print the plan only
#
# Other arguments go to the test runner (e.g. `--jobs=4 --verbose`). The plan and its reasons are
# printed first. It is the selection CI makes for a pull request; CI also runs the other jobs the
# plan names (master-dependency integration, the Knockoffs extension, the examples).
include(joinpath(@__DIR__, "ci_plan.jl"))

own = filter(a -> startswith(a, "--base=") || a == "--dry-run", ARGS)
plan = CIPlan.compute(filter(a -> startswith(a, "--base="), own))
CIPlan.describe(stdout, plan)
if "--dry-run" in own
  exit(0)
elseif plan.level == "none"
  println("\nNo package test is affected.")
  exit(0)
end
runner_args = filter(a -> !(a in own), ARGS)
plan.tests === nothing || push!(runner_args, "--only=" * join(plan.tests, ","))
println()
run(`$(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, "test_registered.jl")) $runner_args`)
