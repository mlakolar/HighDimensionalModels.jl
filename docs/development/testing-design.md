# Testing and CI design

This note records how the tests run, how a change selects its tests, and how CI decides what is
required. CONTRIBUTING.md has the commands; this file has the reasons. It is not part of the
documentation site.

Status: sections 1-3 are implemented by PR #203 (#196, the parallel runner), sections 4-6 by
PR #204 (#197, the impact map and planner) and sections 7-11 by PR #205 (#198, selective CI).
#195 tracks the whole.

## 1. Execution model

`test/runtests.jl` runs the test files with
[ParallelTestRunner](https://github.com/JuliaTesting/ParallelTestRunner.jl) (PTR), as ProximalBase
does. Each test file is `include`d into a fresh module on a worker process; workers are separate
Julia processes, reused across files and replaced when their memory exceeds a threshold. PTR prints
each file's time, compile share, GC and memory, and a summary of every testset.

- `Pkg.test()` with no arguments runs every test file. Arguments select files: `--only=a,b` (exact
  names), positional prefixes (`lowrank` runs every `lowrank*` file), `--list`, `--jobs=N`,
  `--verbose`. `scripts/test_registered.jl` and `scripts/test_master.jl` pass their arguments on.
- Workers inherit the parent's `--check-bounds`, `--depwarn` and `--code-coverage` flags
  (`Base.julia_cmd()`), so CI's `--depwarn=error` and coverage apply to every file.
- Threads: each worker has `HDM_TEST_THREADS` Julia threads (default 2, what CI used before) and
  one BLAS thread. Cross-validation folds and stability selection run threaded code; one thread
  would leave it untested. The default worker count is the CPU budget divided by
  `HDM_TEST_THREADS`, at least one, and PTR lowers it when memory is short. The budget is
  `Sys.EFFECTIVE_CPU_THREADS` where Julia has it, so a container's or job's CPU quota counts rather
  than the host's threads, and `Sys.CPU_THREADS` on Julia 1.10. `--jobs=N` or `PTR_NUM_JOBS`
  overrides it; CI uses two workers on four-core runners.

## 2. Isolation

A test file must run alone: it declares its `using`s, loads the optional backends it tests, and
uses nothing defined by another file. PTR seeds the global RNG with `Random.seed!(1)` before each
file; files that draw from the global RNG seed it themselves, and the rest use explicit RNGs, so the
results do not depend on the order or the worker.

Extension loading is process state, and a reused worker keeps whatever an earlier file loaded.
Hence two rules:

- A file that needs HiGHS, ProximalAlgorithms or StatsModels loads it.
- A contract about an extension being *absent* is tested only in a fresh child process
  (`test/extensions_absent.jl`); never in a worker.

Process-level settings are child processes too: `test/deprecated.jl` runs with `--depwarn=yes`
whatever the parent's setting. `test/knockoff_extension.jl` needs Knockoffs, which is not in the
test environment, so it keeps its own environment (`scripts/test_knockoff_extension.jl`).

## 3. Inventory

`test/testsuite.toml` lists every Julia file under `test/` exactly once:

| class | runs as |
|---|---|
| `parallel` | a PTR test file |
| `serial` | a PTR test file run alone after the parallel batch (empty: no file needs the machine) |
| `subprocess` | a fresh child process with the listed flags |
| `not_tests` | helpers, files with their own environment, fixture generators (with a reason) |

Every entry is an exact path; `not_tests` accepts no patterns, so a new file in a fixture directory
needs its own entry like any other. `runtests.jl` stops before running anything when a file is
unclassified, listed twice, listed as both a test and a non-test, or listed but missing
(`test/testsuite.jl`, TOML only), so a new test file cannot be skipped silently.
`test/testsuite_validation.jl` tests each of these checks, `--only` and prefix selection, and that
a failing child-process test fails the run.

## 4. Source-to-test mapping

`test/impact.toml` is the single dependency map. It names *families* (sets of test files) and
*rules*: a rule matches path globs and selects families, test files, extra CI jobs, or the whole
suite, with a one-line reason. The planner (`scripts/ci_plan.jl`) takes the union of every rule
matching every changed path; there is no first-match ordering. The map is curated from a reading of
the call graph (who calls which internal helper), not inferred from coverage, which misses dynamic
dispatch and unexercised branches.

Escalation is by the map, not by tiers:

- **focused**: a family's own files (e.g. `src/structured.jl` selects the structured family);
- **broad**: shared code with bounded consumers (e.g. `src/cv.jl`, `src/glm.jl`,
  `src/multitask*.jl`) selects every consuming family;
- **full**: code nearly every estimator calls (`utils`, `losses`, `lasso`, `penalties`,
  `statsapi`, the module file), the dependencies, the test runner, the planner and CI.

`src/lasso.jl` is full: its helpers (`_loadings`, `_usegram`, `lasso`, `LassoPath`) are called by
eleven families. Moving those helpers to a shared file would let ordinary lasso changes be
selective; that is a production refactor and is left for later.

Every selection adds the smoke file (`test/smoke.jl`: one cheap call per estimator family), Aqua
and the deprecation child for source changes.

## 5. Change detection

- Pull request: the files changed between the merge commit's parents (`HEAD^1..HEAD` of the
  checked-out merge commit), so the whole PR is compared with the base it merges into, however many
  commits, rebases or merges it has. Renames count as both paths.
- Push to master: `before..after`. A missing or unfetchable `before` (new branch, force push) means
  full.
- Tags, the schedules, and manual runs with `full`: full, without a diff.
- A failed `git` command, an empty diff on a code event, or an unparsable map: full. The planner
  never answers "no tests" because something failed. If `git ls-files` fails, the source files
  cannot be checked against the map (section 6), so the plan is full and says why.

## 6. Risk policy for test changes

- A changed test file selects itself (plus smoke), and whatever a rule for its directory adds.
- A helper or fixture selects the files that use it (rules in the map).
- The runner, the inventory, the map, the planner, `Project.toml` and workflows: full.
- A deleted test file: full; deletion is rare and a deleted file cannot say what it covered.
- A changed source file with no rule (a new file): full, and the planner says so.
- Every file under `src/` and `ext/` must be matched by a full rule, or by focused and broad rules
  that name at least one family or test between them; a rule that only adds CI jobs may accompany
  them, and a none rule never counts. Otherwise the planner stops with an error (and its tests
  fail), because changing the file would run only the smoke and package-wide checks. Paths outside
  `src/` and `ext/` may have job-only or test-less rules.
- An unclassified Julia file anywhere under `test/` stops the planner as it stops the runner
  (section 3).
- Source changed without a test change: reported in the plan summary.
- Documentation only (`docs/**`, Markdown, `LICENSE`): no package tests; the gate still reports.

## 7. GitHub Actions

One workflow, `CI.yml`:

| job | runs when |
|---|---|
| `plan` | always; computes the plan, writes the summary, exports outputs |
| `test` (Linux, Julia lts and 1) | the plan selects any test file |
| `test-cross` (macOS, Windows) | weekly, tags, manual runs, `Project.toml` and `CI.yml` changes, `CI: full` |
| `master-integration` | the change touches the ProximalBase/ProximalSolvers interface; nightly |
| `knockoff-extension` (lts, 1) | knockoff sources, the Knockoffs extension, full changes |
| `examples` | the public APIs the examples use, `examples/**`; full plans; weekly |
| `report` | a failing run on master (push, schedule, manual) or a tag: opens or comments on "CI failing on master"; not a manual run on another branch |
| `gate` | always (`if: always()`), named `gate` on pull requests |

The test jobs run the selected files with two workers in one job per Julia version, which shares
the precompilation and startup across files; a job per file would repeat them.

**Gate.** `gate` needs every job and passes only if the plan succeeded and wrote a complete plan,
every planned job succeeded, and every unplanned job was skipped (or succeeded). A failed,
cancelled or missing required job fails it; so does a cancelled run, because `always()` runs the
gate anyway (a skipped required check would count as passing). It reads `needs.*.result` and the
plan outputs, not the REST API, so an API failure cannot turn into a pass.

It is named `gate` only for `pull_request` runs, and `gate (<event>)` otherwise, so a manual run
on a PR branch cannot satisfy the PR's required check. A new commit gets a new SHA with no check
yet, so an older green cannot stand in for it. The workflow does not run on label events, because a
second run on the same commit would leave the first run's green check visible until the second
finished. Re-running failed jobs re-runs the gate (it depends on them).

Concurrency: a new commit cancels the PR's older run; master pushes, schedules and manual runs each
get their own group and are never cancelled.

Security: `pull_request` only (no `pull_request_target`); `contents: read`, and `actions: write`
only where `julia-actions/cache` prunes its caches. Event data reaches scripts through `env:`,
never interpolated into a shell line. Fork PRs have no secrets and need none: coverage is not
collected on PRs.

## 8. Scheduled and release validation

- Nightly (Monday to Saturday): the full registered suite on Linux, Julia lts and 1, with
  coverage; the Knockoffs job; master-dependency integration (full suite), which
  records the ProximalBase and ProximalSolvers commits it tested.
- Weekly (Sunday): the nightly jobs plus macOS and Windows and the examples.
- Release: a tag push runs the weekly set; before tagging, run it by hand
  (`gh workflow run CI.yml --ref <branch>`).

Schedules do not skip a commit that already passed: registered dependencies and dependency masters
change between nights, so the commit alone does not identify what was tested.

Benchmark correctness checks stay with #138. The planner's families map onto the harness groups
(lasso, glm, group, multitask, lowrank, quantile, structured, selection, inference), so a
`benchmark` job could take the families of a plan as `HDM_GROUPS`; that job and its acceptance
belong to #138.

## 9. Coverage

Collected only in full runs (the nightly and weekly Linux jobs on Julia lts and 1, and the Knockoffs
job for the extension), so every upload comes from the same, complete suite. Both Julia versions
upload, as before: lts counts some method-definition lines that Julia 1 does not, and without the
lts upload about 50 such lines would show as missed although their bodies are covered. Selective
PR runs collect none; a PR whose head commit says `CI: full` collects it.

## 10. Performance measurement

PTR prints per-file wall, compile, GC and RSS figures; the CI test step's job summary lists them.
The comparison with the sequential runner is in the PR descriptions. No CI step asserts on time.

## 11. Failure handling and audit

- The plan job fails (and so the gate) on an invalid inventory or map, rather than guessing.
- The nightly and weekly full runs are the audit of the map. When a full run fails on a commit
  whose PR passed a selective run, the miss is recorded in `test/impact.toml` (`[[miss]]`
  entries: commit, test, why it was not selected, the rule fixed) and, where useful, a planner test
  reproduces it. A miss is a map bug, not a flaky test.
