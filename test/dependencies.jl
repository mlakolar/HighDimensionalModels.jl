module DependenciesTest
# scripts/dependencies.jl: its checks fail closed (first testset, on small repositories made
# here), and the packages this process loaded are the checkouts that scripts/test.jl set up (last
# testset): the manifest tracks them by path, and their directories are at the requested commits
# with clean trees. Those expectations come from environment variables that scripts/test.jl
# sets. An environment made by hand has none, and the last testset fails there with a message
# that names scripts/test.jl.
using HighDimensionalModels, Test, TOML

include(joinpath(@__DIR__, "..", "scripts", "dependencies.jl"))
const D = Dependencies

git(dir, args...) = run(pipeline(Cmd(["git", "-C", dir, "-c", "user.name=t", "-c", "user.email=t@t.invalid",
                                      "-c", "commit.gpgsign=false", args...]); stdout = devnull, stderr = devnull))

# A repository with a Project.toml of package `name` (by default with the UUID the package expects).
function repo(dir, name; uuid = D.uuid(name))
  mkpath(dir)
  git(dir, "init", "-q", "-b", "master")
  write(joinpath(dir, "Project.toml"), "name = \"$name\"\nuuid = \"$uuid\"\nversion = \"0.6.0-DEV\"\n")
  git(dir, "add", "Project.toml")
  git(dir, "commit", "-qm", "one")
  return D.git(dir, "rev-parse", "HEAD")
end

message(f) = try f(); nothing catch err sprint(showerror, err) end

@testset "dependency setup fails closed" begin
  @testset "revisions" begin
    sha = "0123456789abcdef0123456789abcdef01234567"
    @test D.issha(sha)
    for bad in ("", "abc", sha[1:39], sha * "0", uppercase(sha), "g" * sha[2:end], " " * sha)
      @test !D.issha(bad)
      withenv("HDM_PROXIMALBASE_REV" => bad, "HDM_PROXIMALSOLVERS_REV" => sha) do
        @test occursin("HDM_PROXIMALBASE_REV must be a full 40-character", message(() -> redirect_stdout(D.revisions, devnull)))
      end
    end
    withenv("HDM_PROXIMALBASE_REV" => sha, "HDM_PROXIMALSOLVERS_REV" => reverse(sha)) do
      @test redirect_stdout(D.revisions, devnull) == Dict("ProximalBase" => sha, "ProximalSolvers" => reverse(sha))
    end
  end

  @testset "master" begin
    mktempdir() do dir
      sha = repo(joinpath(dir, "origin"), "ProximalBase")
      @test D.master_revision(joinpath(dir, "origin")) == sha
      @test occursin("cannot resolve master", message(() -> D.master_revision(joinpath(dir, "missing"))))
      git(joinpath(dir, "origin"), "branch", "-m", "main")
      @test occursin("did not return one commit", message(() -> D.master_revision(joinpath(dir, "origin"))))
    end
  end

  @testset "checkouts" begin
    mktempdir() do deps
      path = joinpath(deps, "ProximalBase.jl")
      sha = repo(path, "ProximalBase")
      @test D.checkout("ProximalBase", deps, sha) == path
      # Untracked and tracked changes stop the run, at the requested commit and at another one.
      write(joinpath(path, "scratch.txt"), "x")
      @test occursin("has local changes", message(() -> D.checkout("ProximalBase", deps, sha)))
      other = "0123456789abcdef0123456789abcdef01234567"
      @test occursin("is at $sha, not $other, and has local changes", message(() -> D.checkout("ProximalBase", deps, other)))
      rm(joinpath(path, "scratch.txt"))
      write(joinpath(path, "Project.toml"), read(joinpath(path, "Project.toml"), String) * "# edit\n")
      @test occursin("has local changes", message(() -> D.checkout("ProximalBase", deps, sha)))
      git(path, "checkout", "--", "Project.toml")
      @test D.checkout("ProximalBase", deps, sha) == path
    end
    mktempdir() do deps
      # Another package, another UUID, and a directory that is no checkout of its own.
      sha = repo(joinpath(deps, "ProximalBase.jl"), "ProximalSolvers")
      @test occursin("holds another package", message(() -> D.checkout("ProximalBase", deps, sha)))
      sha = repo(joinpath(deps, "ProximalSolvers.jl"), "ProximalSolvers"; uuid = "00000000-0000-0000-0000-000000000000")
      @test occursin("holds another package", message(() -> D.checkout("ProximalSolvers", deps, sha)))
    end
    mktempdir() do deps
      git(deps, "init", "-q", "-b", "master")
      mkpath(joinpath(deps, "ProximalBase.jl"))
      write(joinpath(deps, "ProximalBase.jl", "Project.toml"), "name = \"ProximalBase\"\nuuid = \"$(D.uuid("ProximalBase"))\"\n")
      @test occursin("is not the top of a git checkout", message(() -> D.checkout("ProximalBase", deps, "0"^40)))
      write(joinpath(deps, "ProximalSolvers.jl"), "a file")
      @test occursin("is not a directory", message(() -> D.checkout("ProximalSolvers", deps, "0"^40)))
    end
  end

  @testset "manifests" begin
    mktempdir() do dir
      paths = Dict(name => mkpath(joinpath(dir, name * ".jl")) for name in D.PACKAGES)
      hdm = TOML.parsefile(joinpath(D.ROOT, "Project.toml"))["uuid"]
      # The path is written as a TOML basic string: its backslashes (Windows) are escaped.
      entry(name, path, id; extra = "") =
        "[[deps.$name]]\nuuid = \"$id\"\npath = \"$(escape_string(path))\"\nversion = \"0.6.0-DEV\"\n$extra"
      function manifest(; pb = entry("ProximalBase", paths["ProximalBase"], D.uuid("ProximalBase")),
                         ps = entry("ProximalSolvers", paths["ProximalSolvers"], D.uuid("ProximalSolvers")),
                         me = entry("HighDimensionalModels", D.ROOT, hdm))
        write(joinpath(dir, "Manifest.toml"), "julia_version = \"1.10.0\"\nmanifest_format = \"2.0\"\n\n" * join([pb, ps, me], "\n"))
        return D.verify_manifest(dir, paths)
      end
      @test manifest() == joinpath(dir, "Manifest.toml")
      # A dependency taken from a repository revision (what a `[sources]` pin would produce), or
      # from the registry, or from another directory, or listed twice, or with another UUID.
      pinned = entry("ProximalBase", paths["ProximalBase"], D.uuid("ProximalBase");
                     extra = "repo-rev = \"a19c59aa723d085d5d7cf5242895bda16eb77214\"\nrepo-url = \"https://github.com/mlakolar/ProximalBase.jl\"\n")
      @test occursin("not tracked by path alone (repo-", message(() -> manifest(pb = pinned)))
      registered = "[[deps.ProximalBase]]\nuuid = \"$(D.uuid("ProximalBase"))\"\nversion = \"0.5.1\"\ngit-tree-sha1 = \"0123456789abcdef0123456789abcdef01234567\"\n"
      @test occursin("is not tracked by path alone (git-tree-sha1", message(() -> manifest(pb = registered)))
      @test occursin("is not tracked by path", message(() -> manifest(pb = "[[deps.ProximalBase]]\nuuid = \"$(D.uuid("ProximalBase"))\"\nversion = \"0.5.1\"\n")))
      elsewhere = entry("ProximalBase", mkpath(joinpath(dir, "other")), D.uuid("ProximalBase"))
      @test occursin("is tracked at", message(() -> manifest(pb = elsewhere)))
      @test occursin("lists ProximalBase 2 times", message(() -> manifest(pb = entry("ProximalBase", paths["ProximalBase"], D.uuid("ProximalBase")) * "\n" * elsewhere)))
      @test occursin("lists ProximalSolvers 0 times", message(() -> manifest(ps = "")))
      @test occursin("has UUID", message(() -> manifest(ps = entry("ProximalSolvers", paths["ProximalSolvers"], "00000000-0000-0000-0000-000000000000"))))
      @test occursin("is tracked at", message(() -> manifest(me = entry("HighDimensionalModels", dir, hdm))))
      rm(joinpath(dir, "Manifest.toml"))
      @test occursin("no manifest", message(() -> D.verify_manifest(dir, paths)))
    end
  end

  @testset "expectations and the receipt" begin
    sha = "0123456789abcdef0123456789abcdef01234567"
    names = [D.rev_var.(D.PACKAGES); D.checkout_var.(D.PACKAGES)]
    # Each variable once (withenv restores duplicated names wrongly), and the others unset.
    expect(set...) = withenv(collect(merge(Dict{String,Any}(name => nothing for name in names), Dict{String,Any}(set)))...) do
      message(() -> D.expectations())
    end
    @test occursin("scripts/test.jl", expect())
    @test occursin("scripts/test.jl", expect(D.rev_var("ProximalBase") => sha, D.rev_var("ProximalSolvers") => sha,
                                             D.checkout_var("ProximalBase") => "/x"))       # a checkout is missing
    @test occursin("scripts/test.jl", expect(D.rev_var("ProximalBase") => "short", D.rev_var("ProximalSolvers") => sha,
                                             D.checkout_var("ProximalBase") => "/x", D.checkout_var("ProximalSolvers") => "/y"))
    all = Dict{String,Any}(D.rev_var("ProximalBase") => sha, D.rev_var("ProximalSolvers") => sha,
               D.checkout_var("ProximalBase") => "/x", D.checkout_var("ProximalSolvers") => "/y")
    withenv(collect(all)...) do
      @test D.expectations() == Dict("ProximalBase" => (sha, "/x"), "ProximalSolvers" => (sha, "/y"))
    end
    mktempdir() do dir
      paths = Dict(name => joinpath(dir, name * ".jl") for name in D.PACKAGES)
      revs = Dict(name => repo(paths[name], name) for name in D.PACKAGES)
      file = D.write_receipt(dir, revs, paths, joinpath(dir, "Manifest.toml"))
      receipt = TOML.parsefile(file)
      @test receipt["julia"] == string(VERSION) && occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", receipt["generated"])
      for name in D.PACKAGES
        @test receipt[name]["commit"] == receipt[name]["requested"] == revs[name]
        @test receipt[name]["tree"] == D.git(paths[name], "rev-parse", "HEAD^{tree}")
        @test receipt[name]["version"] == "0.6.0-DEV" && receipt[name]["path"] == paths[name]
      end
      @test haskey(receipt["HighDimensionalModels"], "commit") && receipt["HighDimensionalModels"]["path"] == D.ROOT
      @test receipt["manifest"] == joinpath(dir, "Manifest.toml")
    end
  end
end

@testset "the loaded packages are the verified checkouts" begin
  expected = D.expectations()
  lines = D.verify_loaded(expected)
  @test length(lines) == length(D.PACKAGES)
  @test realpath(pkgdir(HighDimensionalModels)) == realpath(D.ROOT)
end

end
