# Testing and CI design

This note records how the tests run, how a change selects its tests, and how CI decides what is
required. CONTRIBUTING.md has the commands; this file has the reasons. It is not part of the
documentation site.

## 1. Dependencies

HighDimensionalModels (HDM) depends on ProximalBase (PB) and ProximalSolvers (PS) and its `[compat]`
admits only PB 0.6, which is not registered. A plain resolution fails instead of taking a registered
PB 0.5, and every environment is built from checkouts of the two repositories (`scripts/dependencies.jl`):

- **Commits.** `HDM_PROXIMALBASE_REV` and `HDM_PROXIMALSOLVERS_REV` (full SHAs, anything else is an
  error), else the tip of `master`, resolved once at the start of a run. In CI the plan job resolves
  them and passes the pair to every job, so a push to PB between two jobs cannot split a run.
- **Checkouts.** `HDM_DEPS_DIR/<package>.jl`, cloned at the commit when missing, moved to it
  only if the tree is clean. A checkout with changes stops the run: a silent reset would destroy work,
  and testing a modified tree would make the recorded commit untrue. The scripts do not look for
  checkouts next to the package; on a developer machine those are working copies.
- **Environment.** `Pkg.develop` of the two checkouts and the package into a generated project with
  the test target's packages. Julia 1.10 ignores `[sources]`, so path development is the one
  mechanism that works on both versions. The package's `Project.toml` is not changed.
- **Verification.** The manifest lists PB, PS and HDM once each, tracked by path at the checkouts
  (no registry version, no repository revision). PS's `Project.toml` has a `[sources]` entry that pins
  PB to an older commit; in the manifests generated here with Julia 1.10 (which ignores `[sources]`)
  and 1.13, PB is tracked by path at the requested commit, not at that pin. The check fails if a
  manifest ever tracks PB another way. Then, in a process of the environment (`scripts/dependencies.jl`, and again
  inside the test run by `test/dependencies.jl`), `pkgdir(ProximalBase)` and `pkgdir(ProximalSolvers)`
  are the checkouts, whose `HEAD` equals the requested commits and whose trees are clean, and PB is
  a 0.6 development version.
- **Receipt.** The commits, tree SHAs, versions and paths, the package's commit and whether its tree
  has changes, Julia and the platform, the manifest path, and the test files that ran. CI uploads the
  receipt and the manifest and writes the two commits to the job summary.

`test/dependencies.jl` takes its expectations from environment variables that `scripts/test.jl` sets
and fails without them, with a message naming the script: a hand-made environment is not a verified one.

## 2. Execution model

`test/runtests.jl` runs the test files with
[ParallelTestRunner](https://github.com/JuliaTesting/ParallelTestRunner.jl) (PTR). Each test file is
`include`d into a fresh module on a worker process; workers are separate Julia processes, reused
across files. Arguments select files: `--only=a,b` (exact names), positional prefixes, `--list`,
`--jobs=N`, `--verbose`. Workers inherit the parent's `--check-bounds`, `--depwarn` and
`--code-coverage` flags. Each worker has `HDM_TEST_THREADS` Julia threads (default 2) and one BLAS
thread; the default worker count is the CPU budget (`Sys.EFFECTIVE_CPU_THREADS` where Julia has it,
`Sys.CPU_THREADS` on Julia 1.10) divided by the threads, at least one.

`scripts/test.jl` is the entry point. It runs `test/runtests.jl` in a child process with
`--check-bounds=yes --depwarn=error`, reads the runner's output, and fails unless the exit code is
zero and the runner's table has a row for every file of the selection the runner printed; for
`--only` the selection must equal the names given, and without arguments it must be every file of the
inventory. A run that dropped a file fails even when the runner's own summary reports success.

A test file must run alone (it declares its `using`s) and leaves no global state, since workers are
reused. Process-level settings are child processes: an inventory entry under `subprocess` runs in a
fresh process with extra flags (none today).

## 3. Inventory

`test/testsuite.toml` lists every Julia file under `test/` exactly once, by path without `.jl`:

| class | runs as |
|---|---|
| `parallel` | a PTR test file |
| `serial` | a PTR test file run alone after the parallel batch (empty) |
| `subprocess` | a fresh child process with the listed flags (empty) |
| `not_tests` | helpers and fixture generators, with a reason (`testsuite` only) |

The runner and the planner stop when a file is unclassified, listed twice, listed as both a test and
a non-test, or listed but missing, so a new test file is not left out of a complete run silently. A selection that
matches no file is an error. `test/testsuite_validation.jl` tests these checks, `--only` and prefix
selection, the check of the runner's report, and that a failing child-process test fails the run.

The eight files are `aqua`, `ci_plan`, `dependencies`, `lasso`, `lasso_reference`, `smoke`,
`statsapi` and `testsuite_validation`. The historical suite and its fixtures are on `legacy/full`.

## 4. Source-to-test mapping

`test/impact.toml` is the dependency map: *families* (sets of test files) and *rules* that match
path globs and select families, test files, extra CI jobs, or the whole suite, each with a reason.
The planner (`scripts/ci_plan.jl`) takes the union of every rule matching every changed path; there
is no first-match ordering. Levels are none, focused and full.

With eight test files, a change outside documentation and `benchmark/` selects the complete suite: `src/**`,
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
  never "no tests" because something failed. If `git ls-files` fails, the source files cannot be
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
every planned job succeeded, and every unplanned job was skipped. A failed, cancelled
or unexpectedly skipped required job fails it; so does a cancelled run, because `always()` runs the
gate anyway. It reads `needs.*.result` and the plan outputs, not the REST API. The rule is
`CIPlan.verdict` in `scripts/ci_plan.jl`, tested in `test/ci_plan.jl`, which also checks that the
workflow defines every job the rule knows, that the gate needs them all, that no step has
`continue-on-error` and that `test-cross` excludes pull requests. The gate is named `gate` only for
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

**Failures on master.** A failing push, nightly or manual run on master opens one issue titled "CI
failing on master" or comments on the open one. The nightly full runs are the audit of the map: when
a full run fails on a commit whose PR passed a selective run, record the miss in `test/impact.toml`
under `[[miss]]` and fix the rule.
