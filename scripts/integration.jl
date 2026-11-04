# Integration tests in a dedicated environment: scripts/test_master.jl (dependency master
# checkouts) and scripts/test_supported.jl (dependencies at pinned commits). The environment keeps
# extra-only workspace dependencies reachable on Julia 1.10 and leaves the package Project.toml
# unchanged for Aqua checks.
using Pkg, TOML
include(joinpath(@__DIR__, "master_dependencies.jl"))

const PACKAGE_ROOT = dirname(@__DIR__)

"""
    test_integration(args; label, root, workspace, revisions = nothing)

Build the test environment `root/vX.Y` with the workspace dependencies from the checkouts in
`workspace` (master checkouts, or with `revisions` the given commits, see `instantiate_master`),
check it, and unless `--setup-only` is in `args`, run test/runtests.jl with the other `args`.
A failing test file fails the call.
"""
function test_integration(args; label, root, workspace, revisions = nothing)
    project_path = joinpath(PACKAGE_ROOT, "Project.toml")
    original_project = read(project_path, String)
    project = TOML.parse(original_project)
    deps = copy(get(project, "deps", Dict{String,Any}()))
    extras = merge(get(project, "weakdeps", Dict()), get(project, "extras", Dict()))
    for name in get(get(project, "targets", Dict()), "test", String[])
        deps[name] = extras[name]
    end
    deps[project["name"]] = project["uuid"]
    compat = Dict(name => value for (name, value) in get(project, "compat", Dict())
                  if name == "julia" || haskey(deps, name))
    checkout(name) = name == project["name"] ? PACKAGE_ROOT : joinpath(workspace, WORKSPACE_PACKAGES[name][1])
    sources = Dict{String,Any}()
    for name in keys(deps)
        if name == project["name"] || haskey(WORKSPACE_PACKAGES, name)
            sources[name] = Dict("path" => checkout(name))
        elseif haskey(get(project, "sources", Dict()), name)
            source = copy(project["sources"][name])
            haskey(source, "path") && (source["path"] = abspath(joinpath(PACKAGE_ROOT, source["path"])))
            sources[name] = source
        end
    end
    environment = joinpath(root, "v$(VERSION.major).$(VERSION.minor)")
    mkpath(environment)
    open(joinpath(environment, "Project.toml"), "w") do io
        TOML.print(io, Dict("deps" => deps, "compat" => compat, "sources" => sources))
    end
    instantiate_master(environment; workspace, owner = PACKAGE_ROOT, revisions)
    read(project_path, String) == original_project || error("test setup changed package Project.toml")

    # Every workspace package in the environment is the expected checkout, at the expected commit.
    installed = Dict(info.name => info for info in values(Pkg.dependencies()))
    for name in keys(deps)
        haskey(WORKSPACE_PACKAGES, name) && !haskey(installed, name) && error("$name is missing from the test environment")
    end
    function verify_checkouts()
        rows = String[]
        for name in sort!(filter(in(keys(WORKSPACE_PACKAGES)), collect(keys(installed))))
            info = installed[name]
            realpath(info.source) == realpath(checkout(name)) ||
                error("test environment did not retain the checkout $(checkout(name)) of $name")
            name == project["name"] && continue
            revision = revisions === nothing ? readchomp(`git -C $(checkout(name)) rev-parse HEAD`) :
                       verify_revision(name, checkout(name), revisions[name])
            push!(rows, "| $name | $(info.version) | `$revision` |")
        end
        return rows
    end
    report = join(["$label dependencies:", "", "| package | version | commit |", "|---|---|---|",
                   verify_checkouts()...], "\n")
    println(report, "\n\n$label test environment: ", environment)
    if "--setup-only" in args
        # CI runs the setup as its own step, which records the commits in the job summary.
        summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
        isempty(summary) || open(io -> println(io, report, "\n"), summary, "a")
        return nothing
    end

    # CI must exercise the certified smoothed backend, rather than silently
    # taking the supported older-release fallback with an unmerged workspace.
    certified = Core.eval(Main, Expr(:toplevel, :(using ProximalBase, ProximalSolvers),
        :(all(kernel -> ProximalSolvers._hasconjugate(ProximalBase.ConvolutionCheck(0.5, 0.2; kernel)),
              (:uniform, :epanechnikov)))))
    certified || error("workspace ProximalSolvers must include convolution certificates (PR #96)")
    depwarn = get(ENV, "JULIA_TEST_DEPWARN", "yes")
    depwarn in ("yes", "no", "error") || error("invalid JULIA_TEST_DEPWARN")
    coverage = get(ENV, "JULIA_TEST_COVERAGE", "false") == "true" ? "user" : "none"
    test_file = joinpath(PACKAGE_ROOT, "test", "runtests.jl")
    # Other arguments select test files, as for test/runtests.jl (e.g. `--only=lasso,cv`).
    test_args = filter(!=("--setup-only"), args)
    run(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --depwarn=$depwarn --code-coverage=$coverage --project=$environment $test_file $test_args`)
    # The tests ran against the verified commits throughout.
    verify_checkouts()
    return nothing
end
