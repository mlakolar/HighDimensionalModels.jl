# Testing and CI design

This note records how the tests run, how a change selects its tests, and how CI decides what is
required. CONTRIBUTING.md has the commands; this file has the reasons. It is not part of the
documentation site.

## 1. Dependencies

HighDimensionalModels (HDM) depends on ProximalBase (PB) and ProximalSolvers (PS) and its `[compat]`
admits only PB 0.6, which is not registered. While that holds, a plain resolution fails instead of
taking a registered PB 0.5, and the test and documentation environments are built from checkouts of
the two repositories (`scripts/dependencies.jl`; the benchmark environment takes the checkouts as arguments):

- **Commits.** `HDM_PROXIMALBASE_REV` and `HDM_PROXIMALSOLVERS_REV` (full SHAs, anything else is an
  error), else the tip of `master`, resolved once at the start of a run. In CI the plan job resolves
  them and passes the pair to every job, so a push to PB between two jobs does not split a run.
- **Checkouts.** `HDM_DEPS_DIR/<package>.jl`, cloned at the commit when missing, moved to it
  only if the tree is clean. A checkout with changes stops the run: a silent reset would destroy work,
  and testing a modified tree would make the recorded commit untrue. The scripts do not look for
  checkouts next to the package; on a developer machine those are working copies.
- **Environment.** `Pkg.develop` of the two checkouts and the package into a generated project with
  the test target's packages. Julia 1.10 ignores `[sources]`, so path development, which works on
  both versions, is used for both. The package's `Project.toml` is not changed.
- **Verification.** The manifest lists PB, PS and HDM once each, tracked by path at the checkouts
  (no registry version, no repository revision). PS's `Project.toml` has a `[sources]` entry that pins
  PB to an older commit; in the manifests generated here with Julia 1.10 (which ignores `[sources]`)
  and 1.13, PB is tracked by path at the requested commit, not at that pin. The check fails if a
  manifest ever tracks PB another way. `scripts/instantiate.jl ENV_DIR` refuses an existing
  `ENV_DIR` that has a feature which can change which project or manifest Julia uses (a conservative
  rule): a `JuliaProject.toml`, a top-level `manifest` key in its `Project.toml`, or a `[workspace]` table
  in a project file above it. `scripts/test.jl` regenerates the project of its own environment (removing a
  `JuliaProject.toml` there) and then applies the same check; both reach it before `Pkg.develop`. On Julia 1.13 a `[sources]` entry for PB in the environment's own
  `Project.toml` did not replace the path development (the manifest tracked the path), and a
  `[workspace]` table in the environment's own project leaves its manifest next to it, so neither is refused;
  `verify_manifest` also fails unless its manifest is the one `Base.project_file_manifest_path` selects.
  Then, in a process of the environment (`scripts/dependencies.jl`,
  run by both entry points, and again inside the test run by `test/dependencies.jl`),
  `pkgdir(ProximalBase)` and `pkgdir(ProximalSolvers)` are the checkouts, whose `HEAD` equals the
  requested commits and whose trees are clean, and PB is a 0.6 development version. In an accepted
  environment the manifest is the first existing file of Julia's own list of manifest names
  (`JuliaManifest-vX.Y.toml`, `Manifest-vX.Y.toml`, `JuliaManifest.toml`, `Manifest.toml`).
- **Receipt.** The commits, tree SHAs, versions and paths, the package's commit and whether its tree
  has changes, Julia and the platform, and the manifest path; the test run's receipt also lists the
  test files that ran. CI uploads the receipt and the manifest of the test, cross-platform and
  documentation jobs and writes the two commits to the job summary.

`test/dependencies.jl` takes its expectations from environment variables that `scripts/test.jl` sets
and fails without them, with a message naming the script: a hand-made environment is not a verified one.

## 2. Execution model

`test/runtests.jl` runs the test files with
[ParallelTestRunner](https://github.com/JuliaTesting/ParallelTestRunner.jl) (PTR). Each test file is
`include`d into a fresh module on a worker process; workers are separate Julia processes, reused
across files. Arguments select files: `--only=a,b` (exact names), positional prefixes, `--list`,
`--jobs=N`, `--verbose`. Workers inherit the parent's `--check-bounds`, `--depwarn` and
`--code-coverage` flags. Each worker has one Julia thread and one BLAS thread, because nothing in the
package or its tests starts threads; the default worker count is at most the CPU budget
(`Sys.EFFECTIVE_CPU_THREADS` where Julia has it, `Sys.CPU_THREADS` on Julia 1.10), and
ParallelTestRunner lowers it when memory is short.

`scripts/test.jl` is the entry point. It runs `test/runtests.jl` in a child process with
`--check-bounds=yes --depwarn=error`, reads the runner's output, and fails unless the exit code is
zero and the runner's table has a completed row (the numeric columns after the worker) for every
file of the selection the runner printed; for `--only` the selection must equal the names given, and
without selecting arguments it must be every file of the inventory. The runner prints a different row
when a file starts (`started at`) or fails (`failed at`), and a crashed file's message goes to stderr,
which is not read: none of these counts as completed. A failing test fails the run through the exit
status; the row check is the separate evidence that each selected file finished.

A test file must run alone (it declares its `using`s) and leaves no global state, since workers are
reused. Process-level settings are child processes: an inventory entry under `subprocess` runs in a
fresh process with extra flags (none today).

## 3. Inventory

`test/testsuite.toml` lists every Julia file under `test/` except `runtests.jl` exactly once, by path
without `.jl`:

| class | runs as |
|---|---|
| `parallel` | a PTR test file |
| `serial` | a PTR test file run alone after the parallel batch (empty) |
| `subprocess` | a fresh child process with the listed flags (empty) |
| `not_tests` | helpers and fixture generators, with a reason (`testsuite` only) |

The runner and the planner stop when a file is unclassified, listed twice, listed as both a test and
a non-test, or listed but missing, so a new test file is not left out of a complete run silently. A
selection that matches no file is an error. `test/testsuite_validation.jl` tests these checks, `--only` and prefix
selection, the check of the runner's report, and that a failing child-process test fails the run.

The nine files are `aqua`, `ci_plan`, `dependencies`, `lasso`, `lasso_intercept`, `lasso_reference`,
`smoke`, `statsapi` and `testsuite_validation`. The historical suite and its fixtures are on `legacy/full`.

## 4. Source-to-test mapping

`test/impact.toml` is the dependency map: *families* (sets of test files) and *rules* that match
path globs and select families, test files, extra CI jobs, or the whole suite, each with a reason.
The planner (`scripts/ci_plan.jl`) takes the union of every rule matching every changed path; there
is no first-match ordering. Levels are none, focused and full.

With nine test files, a change outside documentation and `benchmark/` selects the complete suite: `src/**`,
`Project.toml`, `test/**`, `scripts/**` and `CI.yml` are full rules. Documentation and the other
workflows select no test. `benchmark/**` selects no package test but the benchmark-integrity job.
Rules that narrow a plan are kept for growth, with these protections:

- a path that no rule matches, a deleted or renamed source or test file, and an empty change set
  select the full suite;
- every file under `src/` needs a full rule, or focused rules naming at least one family or test
  (otherwise a change would run only the smoke and package-wide checks), and the planner stops on one
  that has not;
- a focused plan also runs the `smoke` tests (`smoke`, `dependencies`) and, for source changes, the
  `source` tests (`aqua`);
- every test must be reachable from a rule, family, `smoke` or `source` list.

## 5. Change detection

- Pull request: the files changed between the merge commit's parents (`HEAD^1..HEAD` of the
  checked-out merge commit). Renames count as both paths.
- Push to master: `before..after`. A missing or unfetchable `before` means full.
- Nightly and manual runs: full, on every platform, without a diff.
- A failed `git` command, an empty diff, an unknown event or an unparsable map: full or an error;
  not "no tests" when something failed. If `git ls-files` fails, the source files cannot be
  checked against the map, so the plan is full and says why.

## 6. GitHub Actions

One workflow, `CI.yml`:

| job | runs when |
|---|---|
| `plan` | always; computes the plan, resolves the PB and PS commits, writes the summary |
| `test` (Linux x86-64, Julia lts and 1) | the plan requires it (any change outside documentation and `benchmark/`) |
| `benchmark-integrity` (Linux, Julia 1) | the plan requires it (full plans and `benchmark/**`) |
| `test-cross` (macOS arm64, Windows x86-64, Julia 1) | nightly and manual runs only, never `pull_request` |
| `gate` | always (`if: always()`), named `gate` on pull requests |
| `report` | a failing run on master that is not a pull request |

**Gate.** `gate` needs every job and passes only if the plan succeeded and wrote a complete plan,
every planned job succeeded, and every unplanned job was skipped. A failed, cancelled or unexpectedly
skipped required job fails it (`always()` runs the gate after its dependencies have finished, with any result, and it sees only their
results). It reads `needs.*.result` and the plan outputs, not the
REST API. The plan's `level`, `tests` and `jobs` outputs must be one of the combinations the planner
emits (the tables in the docstring of `CIPlan.plan_problems`, derived from `LEVELS`, `JOBS`,
`FULL_JOBS` and the inventory): `none` has tests `none` and no `test` or `test-cross` job; `focused`
has `all` or a comma-separated list of distinct inventory names, and the `test` job; `full` has `all`
and every job of `FULL_JOBS`; `test-cross` needs `test`. A plan without jobs or tests writes `none`,
so a missing or empty output is not read as an empty plan, and a whitespace-only, repeated, empty or
unknown item fails the gate. The results must be one `job=result` word per job with a known result, and a repeated
job is an error rather than an overwrite. The rule is `CIPlan.verdict` in `scripts/ci_plan.jl`,
tested in `test/ci_plan.jl`, which also checks that the workflow defines every job the rule knows,
that the gate needs them all and receives the plan's outputs, that no step has `continue-on-error`
and that `test-cross` excludes pull requests. The gate is named `gate` only for
`pull_request` runs, so a manual run on a PR branch cannot satisfy the PR's required check.

**Test jobs.** The tests of one Julia version run in one job with two workers, which shares
precompilation across files. The job passes the plan's selection to `scripts/test.jl`, uploads the
receipt and manifest, and uploads coverage from complete suites. Coverage comes from both Julia
versions; Codecov merges the uploads of a commit.

**Benchmark integrity.** `benchmark/check_analysis.py`, the certificate controls, and a small run
with record validation, in an environment that develops the two checkouts. No timing is compared and
nothing here is a performance gate.

**Concurrency and permissions.** A new commit cancels the PR's older run; every other run has its own
group. The workflow uses `pull_request` only (no `pull_request_target`), `contents: read`, and
`actions: write` only where `julia-actions/cache` prunes its caches; `report` has `issues: write` and
nothing else. In the `run:` steps, event data and plan outputs reach scripts through `env:`, not interpolated into the shell line.

**Failures on master.** A failing push, nightly or manual run on master opens an issue titled "CI
failing on master" or comments on the open one, found by its exact title in the list of open issues.
The job reads every open issue (the paginated REST API, excluding pull requests), takes the
lowest-numbered one with exactly that title, and creates an issue only when there is none; if the
listing fails the step fails and nothing is created. The `report` jobs of different runs share one
concurrency group and run one at a time. GitHub keeps at most one pending job per group, so with
several simultaneous failures a report can be dropped, which loses a comment. An issue with that
title that someone closes, or creates by hand, is outside what the job controls. The nightly full runs are the audit of the map: when
a full run fails on a commit whose PR passed a selective run, record the miss in `test/impact.toml`
under `[[miss]]` and fix the rule.
