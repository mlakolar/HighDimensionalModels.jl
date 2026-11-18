using Pkg, TOML

const PACKAGE = "HighDimensionalModels"
# The dependencies developed from checkouts, in the order of the command line
const SOURCES = ("ProximalBase", "ProximalSolvers")

# The manifest of an environment directory, by the names Pkg looks for
function manifest_path(environment)
  v = "v$(VERSION.major).$(VERSION.minor)"
  for name in ("JuliaManifest-$v.toml", "Manifest-$v.toml", "JuliaManifest.toml", "Manifest.toml")
    path = joinpath(environment, name)
    isfile(path) && return path
  end
  error("no manifest in $environment")
end

# Every manifest entry except the measured package, by name. A path that a
# package is tracked by is resolved against the manifest's folder, so that two
# environments in different folders compare by the checkout, not by its spelling.
function other_entries(path)
  manifest = TOML.parsefile(path)
  haskey(manifest, "deps") || error("$path is not a manifest of format 2")
  entries = Dict{String,Any}()
  for (name, list) in manifest["deps"]
    name == PACKAGE && continue
    entries[name] = [haskey(e, "path") ? merge(e, Dict("path" => realpath(joinpath(dirname(path), e["path"])))) : e
                     for e in list]
  end
  entries
end

# The names of the manifest's packages that are tracked by path
function tracked_by_path(path)
  manifest = TOML.parsefile(path)
  sort!([name for (name, list) in manifest["deps"] for e in list if haskey(e, "path")])
end

function checkout(name, root, uuids)
  root = realpath(root)
  project = TOML.parsefile(joinpath(root, "Project.toml"))
  get(project, "name", "") == name && get(project, "uuid", "") == uuids[name] ||
    error("the checkout $root must be a $name checkout")
  root
end

function setup_environment(args)
  length(args) in (4, 5) ||
    error("usage: julia benchmark/setup.jl PACKAGE_ROOT ENV_DIR PROXIMALBASE_CHECKOUT " *
          "PROXIMALSOLVERS_CHECKOUT [REFERENCE_ENV]")
  environment = abspath(args[2])
  project = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
  uuids = project["deps"]
  roots = Dict(PACKAGE => checkout(PACKAGE, args[1], uuids),
               SOURCES[1] => checkout(SOURCES[1], args[3], uuids),
               SOURCES[2] => checkout(SOURCES[2], args[4], uuids))
  length(unique(values(roots))) == 3 || error("the three checkouts must be different directories")
  reference = length(args) == 5 ? manifest_path(realpath(args[5])) : nothing
  target = joinpath(environment, "Project.toml")
  ispath(target) && error("Refusing an existing ENV_DIR/Project.toml")
  if reference !== nothing
    julia = get(TOML.parsefile(reference), "julia_version", "")
    julia == string(VERSION) ||
      error("REFERENCE_ENV was resolved by Julia $julia; use one made with Julia $VERSION")
  end
  mkpath(environment)
  open(target, "w") do io
    TOML.print(io, project)
  end
  Pkg.activate(environment)
  # Explicit develop calls by path, which also work on Julia 1.10.
  specs = [PackageSpec(path = roots[name]) for name in (SOURCES..., PACKAGE)]
  if reference === nothing
    Pkg.develop(specs)
  else
    # Start from the reference resolution and change only the measured package.
    cp(reference, joinpath(environment, "Manifest.toml"))
    Pkg.develop(specs; preserve = Pkg.PRESERVE_ALL)
  end
  Pkg.instantiate()
  manifest = manifest_path(environment)
  paths = sort!([PACKAGE, SOURCES...])
  tracked_by_path(manifest) == paths ||
    error("the manifest must track exactly $(join(paths, ", ")) by path, " *
          "but tracks $(join(tracked_by_path(manifest), ", "))")
  for info in values(Pkg.dependencies())
    haskey(roots, info.name) || continue
    info.is_tracking_path && realpath(info.source) == roots[info.name] ||
      error("$(info.name) resolves to $(info.source), not $(roots[info.name])")
  end
  if reference !== nothing
    expected, actual = other_entries(reference), other_entries(manifest)
    changed = sort!([name for name in union(keys(expected), keys(actual))
                     if get(expected, name, nothing) != get(actual, name, nothing)])
    isempty(changed) ||
      error("manifest entries differ from REFERENCE_ENV: $(join(changed, ", "))")
  end
  Pkg.status(mode = Pkg.PKGMODE_MANIFEST)
end

if abspath(PROGRAM_FILE) == @__FILE__
  setup_environment(ARGS)
end
