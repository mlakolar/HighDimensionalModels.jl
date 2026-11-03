# Run the public Pkg.test entry point using registered dependencies only. Arguments are
# passed to test/runtests.jl, e.g. `--only=lasso,cv --jobs=2`; none runs every test file.
using Pkg
package_root = dirname(@__DIR__)
Pkg.activate(package_root)
Pkg.instantiate()
for info in values(Pkg.dependencies())
    info.name == "HighDimensionalModels" && continue
    (info.is_tracking_path || info.is_tracking_repo) &&
        error("release tests require registered dependencies: $(info.name)")
end
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
depwarn = get(ENV, "JULIA_TEST_DEPWARN", "error")
depwarn in ("yes", "no", "error") || error("invalid JULIA_TEST_DEPWARN")
Pkg.test(; coverage=get(ENV, "JULIA_TEST_COVERAGE", "false") == "true",
         julia_args=["--startup-file=no", "--check-bounds=yes", "--depwarn=$depwarn"],
         test_args=ARGS)
