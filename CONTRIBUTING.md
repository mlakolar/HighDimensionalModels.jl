# Development conventions

Types and modules use UpperCamelCase. Descriptive functions, fields and keyword
arguments use snake_case; mutating functions end in `!`. Retain standard numerical
abbreviations, mathematical identifiers and names required by upstream interfaces.

The package is the Gaussian lasso on top of
[ProximalBase](https://github.com/mlakolar/ProximalBase.jl) and
[ProximalSolvers](https://github.com/mlakolar/ProximalSolvers.jl) and owns no
optimization code. The full earlier package, with its tests and fixtures, is on
the branch `legacy/full` (tag `archive/pre-minimal-core/master`).

## Dependencies

The package requires the development versions of ProximalBase (0.6.0-DEV, not
registered) and ProximalSolvers, so the scripts below build their environments from
checkouts of the two repositories (`master` unless a commit is pinned), developed by path. While ProximalBase
0.6 is unregistered, a plain `Pkg.test()` or `] instantiate` cannot resolve the
package, and `test/dependencies.jl` fails in an environment made by hand; use the
scripts below.

- A run uses one pair of commits. `HDM_PROXIMALBASE_REV` and
  `HDM_PROXIMALSOLVERS_REV` give them as full 40-character commit SHAs; a
  variable that is unset is resolved once, at the start of the run, to the tip of
  `master` with `git ls-remote` (a failure to resolve is an error), and both
  commits are printed. To refresh, run again; to pin a pair, set both variables.
- The checkouts live in `HDM_DEPS_DIR` (default `.deps/` in the repository, which
  git ignores), as `ProximalBase.jl` and `ProximalSolvers.jl`. A missing checkout
  is cloned at its commit (detached). An existing one is moved to the requested
  commit only if it has no tracked or untracked changes; otherwise the script
  stops. The scripts look for checkouts only there, not next to this repository.
- `scripts/instantiate.jl ENV_DIR` refuses an existing `ENV_DIR` that has a
  `JuliaProject.toml`, a top-level `manifest` key in its `Project.toml`, or a
  `[workspace]` table in a `Project.toml` above it: these can change which project or
  manifest Julia uses, so the setup refuses them (a conservative rule).
  `scripts/test.jl` regenerates the project of its own environment (removing a
  `JuliaProject.toml` there) and then applies the same check; both reach it before
  `Pkg.develop`. Both compare the manifest they check with the one Julia selects for
  the environment, and check that it tracks the two checkouts and this package by path (so the check fails if
  ProximalSolvers' own `[sources]` pin of ProximalBase substitutes another commit),
  and that a Julia process using the environment loads those checkouts at the
  requested commits with clean trees. An environment that cannot be built from the
  checkouts is an error; the scripts do not fall back to a registered release.
- A receipt (`receipt.toml` next to the environment, uploaded by CI together with
  the manifest) records the time, Julia version, platform, this package's commit
  and whether its tree has changes, and each dependency's commit, tree, version and
  path. The test run's receipt also lists the test files that ran.

`julia scripts/instantiate.jl docs` develops the same checkouts into the
documentation environment (`docs/receipt.toml`); `julia scripts/instantiate.jl
--checkouts` only prepares the checkouts and prints their paths.

## Running tests

```sh
julia scripts/test.jl                          # every test file
julia scripts/test.jl --only=lasso,smoke       # exactly these files
julia scripts/test.jl lasso                    # every file starting with `lasso`
julia scripts/test.jl --list                   # the test files
julia scripts/test.jl --jobs=2 --verbose
```

Use Julia 1.10 (`julia +lts`) and the current release (`julia +release`); CI runs
both. `scripts/test.jl` sets up and verifies the environment, writes the receipt,
and runs `test/runtests.jl` with `--check-bounds=yes --depwarn=error` in a child
process (`JULIA_TEST_COVERAGE=true` collects coverage). It fails if a setup check
fails, a test fails, or a selected test file has no completed row in the runner's
report. The test files run in parallel worker processes
([ParallelTestRunner](https://github.com/JuliaTesting/ParallelTestRunner.jl)), each
with one Julia thread and one BLAS thread; `--jobs=N` or `PTR_NUM_JOBS` sets the
worker count.

Every Julia file under `test/` except `runtests.jl` is listed in
`test/testsuite.toml` by its exact path, and the runner and the CI plan stop if one
is not, or if a listed file is missing. A test file runs on its own: it declares every `using` it needs, uses
explicit RNGs (`Xoshiro(seed)`, not the global one), restores any global state it
changes (`ENV`, precision), and relies on no other file.

To add a test file: put it in `test/`, list it in `test/testsuite.toml`, and name
it in a family, `smoke` or `source` list of `test/impact.toml`
(`test/ci_plan.jl` checks that every test can be reached). A helper or fixture
script that is not a test goes under `not_tests` with a reason. To add a source
file, `src/**` is already covered by a rule that runs the complete suite; a rule
that narrows it must name the tests that exercise the file.

## CI

`.github/workflows/CI.yml` runs a plan job (`scripts/ci_plan.jl`,
`test/impact.toml`), then the jobs the plan requires, then `gate`. The plan job
also resolves the two dependency commits once; every job of the run uses that pair.

- A pull request runs Linux (x86-64) only: the complete suite with Julia 1.10 and
  the current release, and the benchmark integrity checks. A change to
  documentation alone runs no package test and no benchmark job (the plan, the gate
  and the separate documentation workflow still run); a change to `benchmark/` alone
  runs only the benchmark integrity checks. `gate` is the one required check: it
  passes only if the plan's level, test selection and jobs are one of the
  combinations the planner emits (`CIPlan.plan_problems` lists them), every job the
  plan required succeeded and every other job was skipped.
- Every night, and for a manual run, the same jobs run with macOS (arm64) and
  Windows added. A manual run can pin the commits with its inputs:
  `gh workflow run CI.yml --ref <branch>` (add
  `-f proximalbase_rev=<sha> -f proximalsolvers_rev=<sha>`).
- A failing run on master (push, nightly or manual) opens an issue titled "CI
  failing on master", or comments on the lowest-numbered open issue with exactly
  that title (the lookup reads every open issue, page by page, and stops without
  creating anything if the listing fails). Within this workflow the reports go
  through one concurrency group, one at a time, and GitHub keeps at most one pending
  job per group, so with several simultaneous failures a report can be dropped, which
  loses a comment. An issue with that title that someone closes, or creates by hand,
  is outside what the job controls.

`julia scripts/test_affected.jl` prints the plan for your local changes and runs
the test files it selects; `--dry-run` prints the plan only.

Benchmarks: see [`benchmark/README.md`](benchmark/README.md). Active manifests are
generated locally and not committed.
