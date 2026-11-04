# Integration with the dependency master checkouts, sibling directories of this checkout (cloned
# from master when missing). The non-blocking "Upstream master migration" workflow runs this.
include(joinpath(@__DIR__, "integration.jl"))
test_integration(ARGS; label = "Master", root = joinpath(PACKAGE_ROOT, ".master-test-env"),
                 workspace = dirname(PACKAGE_ROOT))
