# Run from a package's scripts/instantiate.jl, or pass an environment directory.
# Workspace packages come from master checkouts, or, with `revisions`, from checkouts at exact
# commits (scripts/test_supported.jl).
using Pkg, TOML

const WORKSPACE_PACKAGES = Dict(
    "ProximalBase" => ("ProximalBase.jl", "ProximalBase.jl"),
    "ProximalSolvers" => ("ProximalSolvers.jl", "ProximalSolvers.jl"),
    "HighDimensionalModels" => ("HighDimensionalModels.jl", "HighDimensionalModels.jl"),
    "CovarianceSelection" => ("CovarianceSelection.jl", "CovarianceSelection.jl"),
    "DiffPrecTest" => ("DiffPrecTest.jl", "DiffPrecTest.jl"),
    "KernelSmoothing" => ("KernelSmoothing.jl", "KernelSmoothing.jl"),
    "FunctionalDataAnalysis" => ("FunctionalDataAnalysis.jl", "FunctionalDataAnalysis.jl"),
    "FunctionalGraphicalModels" => ("FunctionalGraphicalModels.jl", "FunctionalGraphicalModels.jl"),
)

repository_url(name) = "https://github.com/mlakolar/$(WORKSPACE_PACKAGES[name][2])"

# Clone `name` at the full commit SHA `revision` into `checkout`, detached.
function clone_revision(name, checkout, revision)
    mkpath(dirname(checkout))
    run(`git init --quiet $checkout`)
    run(`git -C $checkout fetch --quiet --depth=1 $(repository_url(name)) $revision`)
    run(`git -C $checkout checkout --quiet --detach FETCH_HEAD`)
end

# Error unless `checkout` is at exactly `revision` with no local changes.
function verify_revision(name, checkout, revision)
    head = readchomp(`git -C $checkout rev-parse HEAD`)
    head == revision || error("$name: $checkout is at $head, not the revision $revision; " *
                              "remove the checkout to clone the revision again")
    changes = readchomp(`git -C $checkout status --porcelain --untracked-files=all`)
    isempty(changes) || error("$name: $checkout has local changes:\n$changes")
    return head
end

# Without `revisions`, workspace packages come from master checkouts in `workspace` (cloned when
# missing). With `revisions` (package name => full commit SHA), every workspace dependency must
# have a revision, and its checkout in `workspace` is cloned at that commit when missing and
# verified to be at it, clean, otherwise; existing checkouts are never reset.
function instantiate_master(environment::AbstractString;
                            workspace::Union{Nothing,AbstractString}=nothing,
                            owner::AbstractString=environment,
                            revisions::Union{Nothing,AbstractDict}=nothing)
    environment = realpath(environment)
    owner = realpath(owner)
    workspace = workspace === nothing ? dirname(owner) : realpath(workspace)
    owner_project = TOML.parsefile(joinpath(owner, "Project.toml"))
    owner_name = get(owner_project, "name", "")
    specs = Dict{String,Pkg.PackageSpec}()
    visiting = Set{String}()
    function visit(project_path)
        project = TOML.parsefile(project_path)
        deps = merge(get(project, "deps", Dict()), get(project, "extras", Dict()))
        for name in sort!(collect(keys(deps)))
            haskey(WORKSPACE_PACKAGES, name) || continue
            name in visiting && continue
            push!(visiting, name)
            folder, _ = WORKSPACE_PACKAGES[name]
            is_owner = name == owner_name
            checkout = is_owner ? owner : joinpath(workspace, folder)
            if is_owner
                # The package itself: this checkout, as it is.
            elseif revisions !== nothing
                haskey(revisions, name) || error("$name has no pinned revision")
                isdir(checkout) || clone_revision(name, checkout, revisions[name])
                verify_revision(name, checkout, revisions[name])
            else
                if !isdir(checkout)
                    url = repository_url(name)
                    run(`git clone --branch master --single-branch $url $checkout`)
                end
                branch = strip(read(`git -C $checkout branch --show-current`, String))
                branch == "master" || error("$name must use its master checkout: $checkout is on $branch")
            end
            dependency = TOML.parsefile(joinpath(checkout, "Project.toml"))
            dependency["name"] == name || error("wrong package name at $checkout")
            dependency["uuid"] == deps[name] || error("wrong UUID for $name at $checkout")
            specs[name] = Pkg.PackageSpec(; path=checkout)
            visit(joinpath(checkout, "Project.toml"))
        end
    end
    project_path = joinpath(environment, "Project.toml")
    original = read(project_path, String)
    visit(project_path)
    Pkg.activate(environment)
    try
        if isempty(specs)
            Pkg.resolve()
        else
            Pkg.develop([specs[name] for name in sort!(collect(keys(specs)))])
        end
        Pkg.instantiate()
    finally
        # Transitive master dependencies must not become stale direct dependencies.
        write(project_path, original)
    end
    # Keep extra-only master paths in the manifest for Julia 1.10 test sandboxes.
    # Resolving again after restoring Project.toml would prune those entries.
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: julia instantiate_master.jl PACKAGE_DIRECTORY [WORKSPACE_DIRECTORY]")
    instantiate_master(ARGS[1]; workspace=length(ARGS) > 1 ? ARGS[2] : dirname(realpath(ARGS[1])))
end
