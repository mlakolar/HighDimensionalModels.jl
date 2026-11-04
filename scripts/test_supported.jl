# Integration with the supported dependency commits of scripts/supported_dependencies.toml, the
# required CI check. The commits are cloned into .supported-test-env/deps (never your sibling
# checkouts) and verified, exactly, before and after the tests. Arguments as for
# scripts/test_master.jl: `--setup-only`, or test file selections such as `--only=lasso,cv`.
include(joinpath(@__DIR__, "integration.jl"))
const SUPPORTED = joinpath(@__DIR__, "supported_dependencies.toml")
revisions = TOML.parsefile(SUPPORTED)["revisions"]
for (name, revision) in revisions
    haskey(WORKSPACE_PACKAGES, name) || error("$SUPPORTED: $name is not a workspace package")
    revision isa String && occursin(r"^[0-9a-f]{40}$", revision) ||
        error("$SUPPORTED: $name needs a full 40-character commit SHA, not $(repr(revision))")
end
root = joinpath(PACKAGE_ROOT, ".supported-test-env")
workspace = mkpath(joinpath(root, "deps"))
test_integration(ARGS; label = "Supported", root, workspace, revisions)
