# The dependency checkouts, and environments built from them (scripts/dependencies.jl).
#
#   julia scripts/instantiate.jl ENV_DIR      develop the package and the two checkouts of
#                                             ProximalBase and ProximalSolvers by path into the
#                                             existing environment ENV_DIR (for example `docs`),
#                                             verify it as scripts/test.jl does, and write
#                                             ENV_DIR/receipt.toml
#   julia scripts/instantiate.jl --checkouts  only prepare the checkouts and print their paths
#
# Revisions and the checkouts directory are chosen as for scripts/test.jl: HDM_PROXIMALBASE_REV,
# HDM_PROXIMALSOLVERS_REV and HDM_DEPS_DIR. The Project.toml of ENV_DIR is restored after the
# setup. The manifest is checked to track the checkouts by path, and a Julia process that uses
# ENV_DIR must load them at the requested commits with clean trees.
include(joinpath(@__DIR__, "dependencies.jl"))
using .Dependencies

function main(args)
  length(args) == 1 || error("usage: julia scripts/instantiate.jl ENV_DIR | --checkouts")
  revs, paths = Dependencies.prepare()
  Dependencies.summary(["### Dependencies", "",
                        ["- $name `$(revs[name])`" for name in Dependencies.PACKAGES]..., ""])
  if args[1] == "--checkouts"
    for name in Dependencies.PACKAGES
      println("$name checkout: $(paths[name]) at $(revs[name])")
    end
    return nothing
  end
  env = abspath(args[1])
  project = joinpath(env, "Project.toml")
  isfile(project) || error("$env has no Project.toml")
  original = read(project)
  try
    Dependencies.develop(env, paths)
  finally
    write(project, original)
  end
  manifest = Dependencies.verify_environment(env, revs, paths)
  receipt = Dependencies.write_receipt(env, revs, paths, manifest)
  println("\nEnvironment $env tracks the checkouts above. Receipt ($receipt):\n", read(receipt, String))
end

try
  main(ARGS)
catch err
  println(stderr, "ERROR: ", sprint(showerror, err))
  exit(1)
end
