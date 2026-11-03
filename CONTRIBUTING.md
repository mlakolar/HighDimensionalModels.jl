# Development conventions

Types and modules use UpperCamelCase. Descriptive functions, fields and keyword
arguments use snake_case; mutating functions end in `!`. Retain standard numerical
abbreviations, mathematical identifiers and names required by upstream interfaces.
Run `python3 scripts/check_naming.py` to check new declarations.

Release tests use registered dependencies and the ordinary Pkg test entry point:

```sh
julia scripts/test_registered.jl
```

This rejects path or repository overrides of runtime dependencies and runs the
suite with deprecations as errors. Unit tests use explicit kernel functions;
KernelSmoothing integration is covered by the numbered examples.

## Running tests

The test files run in parallel worker processes
([ParallelTestRunner](https://github.com/JuliaTesting/ParallelTestRunner.jl)).
`Pkg.test()` and the scripts without arguments run every file; arguments select
files and are passed on by `scripts/test_registered.jl` and `scripts/test_master.jl`:

```sh
julia scripts/test_registered.jl --only=lasso,cv   # exactly these files
julia scripts/test_registered.jl lowrank           # every file starting with `lowrank`
julia scripts/test_registered.jl --list            # the test files
julia scripts/test_registered.jl --jobs=4 --verbose
```

Within Julia, `Pkg.test(test_args=["--only=glm"])` does the same. Each worker
uses `HDM_TEST_THREADS` Julia threads (default 2) and one BLAS thread; the
default worker count keeps workers × threads within the CPU threads, and
`--jobs=N` or `PTR_NUM_JOBS` overrides it.

Every Julia file under `test/` is listed in `test/testsuite.toml`, and the
runner stops if one is not. A test file must run on its own: it declares every
`using` it needs, loads the optional backend it tests (HiGHS, ProximalAlgorithms,
StatsModels), uses explicit RNGs or seeds its own, and does not rely on another
file. Workers are reused, so an extension another file loaded may already be
loaded; tests of behaviour without an extension belong in
`test/extensions_absent.jl`, which runs in a fresh process. The design and its
reasons are in `docs/development/testing-design.md`.

### Tests affected by a change

`test/impact.toml` maps source, test, fixture and infrastructure paths to the
test files they can affect; `scripts/ci_plan.jl` turns a diff into a plan and
prints the reasons. To run what your local changes (committed, uncommitted and
untracked, since the merge base with `origin/master`) can affect:

```sh
julia scripts/test_affected.jl              # plan, then run the selected files
julia scripts/test_affected.jl --dry-run    # the plan only
julia scripts/test_affected.jl --base=HEAD~2 --jobs=4
```

A plan runs the selected families plus `test/smoke.jl`, and Aqua and the
deprecation test when package code changed. Shared code (`src/utils.jl`,
`src/losses.jl`, `src/lasso.jl`, `src/penalties.jl`, `src/statsapi.jl`, the
module file), `Project.toml`, the runner, the planner and CI select every file;
so does a path no rule matches, and a deleted or renamed source or test file.
A changed test file selects itself; adding a parallel test file to
`test/testsuite.toml` selects it.

When you add a source file, add a rule for it to `test/impact.toml` (the
planner's tests fail otherwise), naming the families whose tests call into it,
directly or through other source files. When you add a test file, put it in a
family. Selection is a convenience, not proof: the nightly full run is the
check, and a failure it finds that a selective run missed is a bug in the map,
recorded under `[[miss]]` in `test/impact.toml`.

## Continuous integration

`.github/workflows/CI.yml` starts with a plan job that runs the same planner and
writes the plan, with its reasons, to the run's summary page.

| Event | What runs |
|---|---|
| Pull request, push to master | the planned test files on Linux (Julia lts and 1, two workers each), and the jobs the plan names: master-dependency integration (on the same files), the Knockoffs extension, the examples |
| Documentation only | no package test; the gate still reports |
| Nightly (Mon-Sat) | every file on Linux with coverage, the Knockoffs extension, master-dependency integration |
| Weekly (Sun), tag, manual run | the nightly set plus macOS, Windows and the examples |

Overrides:

- `CI: full` on a line of the PR's head commit message runs everything,
  macOS and Windows included. An empty commit
  (`git commit --allow-empty -m "Run the full suite" -m "CI: full"`) works.
  Labels are not used, because a label starts a second run on the same commit
  whose pending check would not hide the first run's green one.
- `gh workflow run CI.yml --ref <branch>` runs the full set on any branch,
  e.g. before tagging a release. Its check is named `gate (workflow_dispatch)`
  and does not count as a pull request's check.
- Locally, `scripts/test_affected.jl` shows and runs the plan, and
  `scripts/test_registered.jl` with no arguments runs every file.

The `gate` job passes only if the plan succeeded and every job it required
succeeded; an unrequired job must have been skipped. A failed, cancelled or
missing required job, or a failed plan, fails it. Re-running failed jobs
re-runs the gate. A failing run on master, a tag or a schedule opens or
updates the issue "CI failing on master".

### Required checks

The branch ruleset requires no status check today. After the workflow has run
on a few pull requests, require exactly one check: `gate`, from the GitHub
Actions app, on pull requests into master. The job names of the test matrix
may change freely; only `gate` is stable. Do not require `gate (push)` or the
other event-suffixed names.

For examples or documentation, run `julia scripts/instantiate_release.jl examples`
or `julia scripts/instantiate_release.jl docs`, respectively. These environments
use this checkout and registered runtime dependencies, with KernelSmoothing
fetched at a pinned commit until it is registered.

For integration with dependency master checkouts, use a separate checkout and
run `julia scripts/test_master.jl`. Missing sibling checkouts are cloned from
master; existing checkouts are never reset or silently updated. This workflow
intentionally uses path overrides and is kept as a separate CI job.

Active manifests are generated locally. Historical benchmark lockfiles and
recorded source identities, the evidence of their original runs, are frozen in
the [benchmark snapshot](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark).
[`benchmark/README.md`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md)
describes the current benchmark harness.
HighDimensionalModels starts at 0.1.0 under a new UUID; CoordinateDescent retains
its original UUID and registered versions.

Studies and validation campaigns keep their records out of Git. The repository
keeps what detects regressions or informs users: tests, harness cases, and
summary tables of comparisons with other packages in the documentation. Each
study's records go into one archive attached to the `benchmark-evidence`
release, and a pull request commits at most about 1 MB of study material; see
[Studies and campaigns](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md#studies-and-campaigns).
