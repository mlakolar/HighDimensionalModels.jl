# The ProximalBase and ProximalSolvers checkouts that every environment of this repository is
# built from, and the checks that a run uses those commits. Plain Julia (standard libraries and
# the git command line); scripts/test.jl, scripts/instantiate.jl and test/dependencies.jl use it.
#
# A run uses one fixed pair of commits. HDM_PROXIMALBASE_REV and HDM_PROXIMALSOLVERS_REV give
# them as full 40-character commit SHAs; an unset variable is resolved once, at the start of the
# run, to the tip of the repository's master branch. A failure to resolve is an error. The
# checkouts live in HDM_DEPS_DIR (default `.deps` in the package root), one directory per
# package. The package is developed by path together with them, so no registered release of
# ProximalBase or ProximalSolvers can take their place.
module Dependencies

using Dates, Pkg, TOML

const ROOT = realpath(joinpath(@__DIR__, ".."))
const GITHUB = "https://github.com/mlakolar"
const PACKAGES = ["ProximalBase", "ProximalSolvers"]

rev_var(name) = "HDM_" * uppercase(name) * "_REV"
checkout_var(name) = "HDM_" * uppercase(name) * "_CHECKOUT"
repository(name) = "$GITHUB/$name.jl"
issha(s) = occursin(r"^[0-9a-f]{40}$", s)

"The deps directory: `HDM_DEPS_DIR`, default `.deps` in the package root."
deps_dir() = abspath(get(ENV, "HDM_DEPS_DIR", joinpath(ROOT, ".deps")))
checkout_path(name, dir = deps_dir()) = joinpath(dir, name * ".jl")

"The `uuid` that the package's Project.toml gives to the dependency `name`."
function uuid(name)
  deps = get(TOML.parsefile(joinpath(ROOT, "Project.toml")), "deps", Dict{String,Any}())
  haskey(deps, name) || error("$name is not a dependency in Project.toml")
  return deps[name]
end

# ---- git ----------------------------------------------------------------------------------

"The output of `git -C dir args...`; throws if git fails."
git(dir, args...) = readchomp(Cmd(["git", "-C", dir, args...]))

"Whether `git -C dir args...` succeeds."
git_ok(dir, args...) = success(pipeline(Cmd(Cmd(["git", "-C", dir, args...]); ignorestatus = true); stdout = devnull, stderr = devnull))

"The changes in the working tree of `dir`, tracked or untracked and not ignored (empty if clean)."
changes(dir) = git(dir, "status", "--porcelain", "--untracked-files=all")

# ---- revisions ----------------------------------------------------------------------------

"The tip of `refs/heads/master` of `url`; throws if it cannot be read."
function master_revision(url)
  out = try
    read(Cmd(["git", "ls-remote", url, "refs/heads/master"]), String)
  catch err
    error("cannot resolve master of $url (git ls-remote failed); set the revision explicitly ",
          "or retry: ", sprint(showerror, err))
  end
  lines = split(out, '\n'; keepempty = false)
  fields = length(lines) == 1 ? split(lines[1]) : String[]
  length(fields) == 2 && fields[2] == "refs/heads/master" && issha(fields[1]) ||
    error("git ls-remote $url refs/heads/master did not return one commit: ", repr(out))
  return String(fields[1])
end

"""
    revisions() -> Dict(name => sha)

The commit of each package: its environment variable (a full lowercase 40-hex SHA, else an
error), or, if the variable is unset, the tip of master. Prints the pair.
"""
function revisions()
  revs = Dict{String,String}()
  for name in PACKAGES
    var = rev_var(name)
    if haskey(ENV, var)
      issha(ENV[var]) || error("$var must be a full 40-character lowercase hexadecimal commit SHA, got \"$(ENV[var])\"")
      revs[name] = ENV[var]
      println("$name: $(revs[name]) ($var)")
    else
      revs[name] = master_revision(repository(name))
      println("$name: $(revs[name]) (master, resolved now)")
    end
  end
  return revs
end

# ---- checkouts ----------------------------------------------------------------------------

"Fetch the commit `rev` of `name` into the repository at `dir` (a no-op if it is there)."
function fetch_revision(name, dir, rev)
  git_ok(dir, "cat-file", "-e", rev * "^{commit}") && return
  git_ok(dir, "fetch", "--quiet", "--depth=1", repository(name), rev) ||
    error("$name: cannot fetch commit $rev from $(repository(name)); is it a commit of that repository?")
end

"Clone `name` at `rev` into `path`, detached, through a temporary directory that is removed on failure."
function clone_revision(name, path, rev)
  mkpath(dirname(path))
  partial = path * ".partial"
  rm(partial; recursive = true, force = true)
  try
    mkpath(partial)
    git_ok(partial, "init", "--quiet") || error("$name: git init failed in $partial")
    fetch_revision(name, partial, rev)
    git_ok(partial, "checkout", "--quiet", "--detach", rev) || error("$name: cannot check out $rev")
    mv(partial, path)
  finally
    rm(partial; recursive = true, force = true)
  end
end

"""
    checkout(name, dir, rev) -> path

The checkout of `name` under `dir`, at exactly commit `rev` with a clean tree. A missing
checkout is cloned. An existing one must be a git repository of its own, with the right name and
UUID in its Project.toml; it is moved to `rev` (detached) only if it has no tracked or untracked
changes, and otherwise is an error.
"""
function checkout(name, dir, rev)
  path = checkout_path(name, dir)
  ispath(path) || (println("$name: cloning $rev into $path"); clone_revision(name, path, rev))
  isdir(path) || error("$name: $path is not a directory")
  # A directory inside another repository (such as the package's own) would answer for that one.
  ispath(joinpath(path, ".git")) && realpath(git(path, "rev-parse", "--show-toplevel")) == realpath(path) ||
    error("$name: $path is not the top of a git checkout")
  project = joinpath(path, "Project.toml")
  isfile(project) || error("$name: $path has no Project.toml")
  info = TOML.parsefile(project)
  get(info, "name", "") == name && get(info, "uuid", "") == uuid(name) ||
    error("$name: $path holds another package ($(get(info, "name", "?")), $(get(info, "uuid", "?")))")
  head = git(path, "rev-parse", "HEAD")
  dirty = changes(path)
  if head != rev
    isempty(dirty) || error("$name: $path is at $head, not $rev, and has local changes:\n$dirty\n",
                            "commit or remove them, or use another HDM_DEPS_DIR")
    println("$name: moving $path from $head to $rev")
    fetch_revision(name, path, rev)
    git_ok(path, "checkout", "--quiet", "--detach", rev) || error("$name: cannot check out $rev in $path")
    head = git(path, "rev-parse", "HEAD")
    dirty = changes(path)
  end
  head == rev || error("$name: $path is at $head, not $rev")
  isempty(dirty) || error("$name: $path has local changes:\n$dirty\ncommit or remove them, or use another HDM_DEPS_DIR")
  return path
end

"Resolve the revisions, prepare both checkouts under `dir`, and return `(revs, paths)`."
function prepare(dir = deps_dir())
  revs = revisions()
  paths = Dict{String,String}()
  # A loop, not a Dict comprehension: that would run a failing checkout twice.
  for name in PACKAGES
    paths[name] = checkout(name, dir, revs[name])
  end
  return revs, paths
end

# ---- environments -------------------------------------------------------------------------

"The manifest next to `Project.toml` in `env`, as Julia chooses it."
function manifest_file(env)
  for name in ("Manifest-v$(VERSION.major).$(VERSION.minor).toml", "Manifest.toml")
    isfile(joinpath(env, name)) && return joinpath(env, name)
  end
  error("no manifest in $env")
end

"""
    write_test_project(env)

Create `env` with a fresh Project.toml that lists the package, its dependencies and the packages
of its test target, with the package's `[compat]` bounds, and no manifest. The package's own
Project.toml is not changed.
"""
function write_test_project(env)
  project = TOML.parsefile(joinpath(ROOT, "Project.toml"))
  deps = copy(get(project, "deps", Dict{String,Any}()))
  extras = get(project, "extras", Dict{String,Any}())
  for name in get(get(project, "targets", Dict()), "test", String[])
    deps[name] = extras[name]
  end
  deps[project["name"]] = project["uuid"]
  compat = Dict(name => bound for (name, bound) in get(project, "compat", Dict()) if name == "julia" || haskey(deps, name))
  mkpath(env)
  for file in readdir(env; join = true)
    startswith(basename(file), "Manifest") && rm(file)
  end
  open(io -> TOML.print(io, Dict("deps" => deps, "compat" => compat)), joinpath(env, "Project.toml"), "w")
  return env
end

"""
    develop(env, paths)

Add to the environment `env` (an existing project) the packages at `paths` (name => directory)
and the package itself, each by path, and install the rest. Does not read `[sources]` entries
of any Project.toml except that of `env`.
"""
function develop(env, paths)
  specs = [Pkg.PackageSpec(; path = paths[name]) for name in PACKAGES]
  push!(specs, Pkg.PackageSpec(; path = ROOT))
  Pkg.activate(env)
  Pkg.develop(specs)
  Pkg.instantiate()
  return nothing
end

"""
    verify_manifest(env, paths)

Throw unless the manifest of `env` lists the two dependencies and the package once each, tracked
by path (no registry version, no repository revision) at exactly the directories of `paths` and
the package root.
"""
function verify_manifest(env, paths)
  file = manifest_file(env)
  deps = get(TOML.parsefile(file), "deps", Dict{String,Any}())
  isempty(deps) && error("$file has no `deps` table (manifest format 2 expected)")
  expected = [(name, paths[name], uuid(name)) for name in PACKAGES]
  push!(expected, ("HighDimensionalModels", ROOT, TOML.parsefile(joinpath(ROOT, "Project.toml"))["uuid"]))
  for (name, dir, id) in expected
    entries = get(deps, name, Any[])
    length(entries) == 1 || error("$file lists $name $(length(entries)) times; expected once")
    entry = only(entries)
    for key in ("repo-url", "repo-rev", "repo-subdir", "git-tree-sha1")
      haskey(entry, key) && error("$file: $name is not tracked by path alone ($key = $(repr(entry[key])))")
    end
    haskey(entry, "path") || error("$file: $name is not tracked by path (a registered version?)")
    entry["uuid"] == id || error("$file: $name has UUID $(entry["uuid"]), expected $id")
    found = realpath(joinpath(dirname(file), entry["path"]))
    found == realpath(dir) || error("$file: $name is tracked at $found, expected $dir")
  end
  return file
end

# ---- the loaded packages ------------------------------------------------------------------

"""
    expectations() -> Dict(name => (rev, path))

The revisions and checkout directories that scripts/test.jl exports to the test processes, from
the environment variables HDM_<PACKAGE>_REV and HDM_<PACKAGE>_CHECKOUT. Throws if one is missing.
"""
function expectations()
  expected = Dict{String,Tuple{String,String}}()
  for name in PACKAGES
    rev, path = get(ENV, rev_var(name), ""), get(ENV, checkout_var(name), "")
    issha(rev) && !isempty(path) ||
      error("no verified dependency setup: $(rev_var(name)) and $(checkout_var(name)) are not set. ",
            "Run the tests with `julia scripts/test.jl`; an environment made by hand is not verified.")
    expected[name] = (rev, path)
  end
  return expected
end

"""
    verify_loaded(expected = expectations(); env = dirname(Base.active_project()))

In the running process: the manifest of `env`, and for each package the loaded module's
directory is the expected checkout, at the expected commit with a clean tree, and
ProximalBase is a development version (0.6.0-DEV or later). Returns the lines to print.
"""
function verify_loaded(expected = expectations(); env = dirname(Base.active_project()))
  paths = Dict(name => path for (name, (_, path)) in expected)
  verify_manifest(env, paths)
  lines = String[]
  for name in PACKAGES
    rev, path = expected[name]
    mod = Base.require(Base.PkgId(Base.UUID(uuid(name)), name))
    dir = pkgdir(mod)
    dir !== nothing && realpath(dir) == realpath(path) ||
      error("$name is loaded from $dir, expected the checkout $path")
    head = git(path, "rev-parse", "HEAD")
    head == rev || error("$name: the checkout $path is at $head, expected $rev")
    isempty(changes(path)) || error("$name: the checkout $path has local changes:\n$(changes(path))")
    version = pkgversion(mod)
    name == "ProximalBase" && !(version >= v"0.6.0-") && error("ProximalBase $version is not a 0.6 development version")
    push!(lines, "$name $version at $head ($dir)")
  end
  return lines
end

# ---- the receipt --------------------------------------------------------------------------

"The package's own commit and whether its working tree has changes (`\"unknown\"` and `nothing` without git)."
function package_state()
  try
    return git(ROOT, "rev-parse", "HEAD"), !isempty(changes(ROOT))
  catch
    return "unknown", nothing
  end
end

"""
    write_receipt(env, revs, paths, manifest) -> path

Write `env/receipt.toml`: UTC time, Julia, platform, the package's commit and whether its
working tree has changes (`dirty`), and for each dependency the requested commit, the commit and tree found, the version and the path.
"""
function write_receipt(env, revs, paths, manifest)
  commit, dirty = package_state()
  receipt = Dict{String,Any}(
    "generated" => Dates.format(now(UTC), dateformat"yyyy-mm-dd\THH:MM:SS\Z"),
    "julia" => string(VERSION),
    "platform" => "$(Sys.KERNEL) $(Sys.ARCH)",
    "manifest" => manifest,
    "HighDimensionalModels" => Dict{String,Any}("commit" => commit, "path" => ROOT))
  dirty === nothing || (receipt["HighDimensionalModels"]["dirty"] = dirty)
  for name in PACKAGES
    path = paths[name]
    receipt[name] = Dict("requested" => revs[name], "commit" => git(path, "rev-parse", "HEAD"),
                         "tree" => git(path, "rev-parse", "HEAD^{tree}"),
                         "version" => TOML.parsefile(joinpath(path, "Project.toml"))["version"], "path" => path)
  end
  file = joinpath(env, "receipt.toml")
  open(io -> TOML.print(io, receipt; sorted = true), file, "w")
  return file
end

"Append `lines` to the GitHub job summary when running in Actions."
function summary(lines)
  file = get(ENV, "GITHUB_STEP_SUMMARY", "")
  isempty(file) || open(io -> foreach(l -> println(io, l), lines), file, "a")
end

end # module

# `julia --project=ENV scripts/dependencies.jl` checks, in this process, the loaded packages
# against the expectations in the environment variables (scripts/test.jl runs this).
# `julia scripts/dependencies.jl --resolve` only resolves the revisions: it prints them and, in
# GitHub Actions, writes them to the step outputs `proximalbase` and `proximalsolvers` and to
# the job summary (the plan job of .github/workflows/CI.yml).
if abspath(PROGRAM_FILE) == @__FILE__
  try
    if ARGS == ["--resolve"]
      revs = Dependencies.revisions()
      output = get(ENV, "GITHUB_OUTPUT", "")
      isempty(output) || open(output, "a") do io
        foreach(name -> println(io, lowercase(name), "=", revs[name]), Dependencies.PACKAGES)
      end
      Dependencies.summary(["### Dependencies", "", ["- $name `$(revs[name])`" for name in Dependencies.PACKAGES]..., ""])
    elseif isempty(ARGS)
      foreach(println, Dependencies.verify_loaded())
    else
      error("usage: julia scripts/dependencies.jl [--resolve]")
    end
  catch err
    println(stderr, "ERROR: ", sprint(showerror, err))
    exit(1)
  end
end
