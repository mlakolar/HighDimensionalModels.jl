# Benchmarks

This suite measures the time and Julia heap allocation of the public call
`lasso(X, y, λ, solver)`. The output of each call is checked by an accuracy
certificate computed from the inputs and the returned coefficients (and, for the
case with an intercept, the returned intercept),
independently of the solver's stopping rule and of its convergence flag; an
output that misses its target is a recorded failure, not a timing. It does not
measure native (non-Julia) allocations or the statistical quality of the
estimates. Peak memory is recorded only as the process-wide `maxrss_bytes`.

The package provides one estimator, the Gaussian lasso, on ProximalBase and
ProximalSolvers. The cases, certificates and evidence of the estimators that
the package no longer contains are on the branch `legacy/full` and in the GitHub
release `benchmark-evidence`.

## Setup and selections

From the package root, with Julia 1.10 or later and Python 3.11 or later, and
checkouts of ProximalBase and ProximalSolvers:

```sh
julia benchmark/setup.jl "$PWD" benchmark/results/env "$PROXIMALBASE_CHECKOUT" "$PROXIMALSOLVERS_CHECKOUT"
JULIA_NUM_THREADS=1 julia --project=benchmark/results/env benchmark/run.jl
python3 benchmark/summarize.py benchmark/results benchmark/results/summary
```

`setup.jl PACKAGE_ROOT ENV_DIR PROXIMALBASE_CHECKOUT PROXIMALSOLVERS_CHECKOUT
[REFERENCE_ENV]` creates a fresh environment with the dependencies of
`benchmark/Project.toml` and develops the three checkouts into it by path with
explicit `Pkg.develop` calls, which also work on Julia 1.10. It fails unless the
manifest tracks exactly those three packages by path and each resolves to its
given checkout, and it refuses an existing `ENV_DIR/Project.toml`. It also refuses a
ProximalBase or ProximalSolvers checkout that is not the root of a git
repository or has tracked or untracked changes (`git status --porcelain
--untracked-files=all` not empty): commit the changes or use a clean checkout.
The runner
needs one Julia thread, and sets one BLAS thread itself. The default is one
quick Float64 repetition: an exploratory check, with no between-run interval.
Use a fresh output directory for each measurement series; the runner stops
instead of writing over records with the same file prefix.

| Setting | Default | Use |
|---|---|---|
| `HDM_SUITE` | `quick` | `extended` runs the quick cases and adds the tall, wide, correlated, large and Float32 cases. |
| `HDM_GROUPS` | `lasso` | Comma-separated groups (`lasso`, `controls`), selected before any data are drawn. |
| `HDM_CASES` | All cases | Comma-separated globs on case ids (`*`, `?`), within the suite and groups. |
| `HDM_REPETITION` | `1` | Process and input draw, numbered 1–10. |
| `HDM_OUTPUT` | `benchmark/results` | Directory for the process records. |
| `HDM_SAMPLES` | `50` quick; `100` extended | Sample limit, with at least three samples. |
| `HDM_SECONDS` | `1` quick; `3` extended | Limit on the accumulated timed seconds of a case. |
| `HDM_CASE_SECONDS` | `60` | Limit on a single compiled call; see `timeout` below. |
| `HDM_PROFILE_SECONDS` | `0` | Positive values write diagnostic profiles of each `ok` case. |
| `HDM_EXPECTED_PACKAGE_ROOT` | Unset | The HighDimensionalModels checkout that must be loaded, verified before any case is built. The runner also refuses to measure, with exit 3 and before any case is built, when the loaded ProximalBase or ProximalSolvers is not the root of a git checkout or has tracked or untracked changes. |
| `HDM_LIST` | `false` | `true` prints the selected case ids, suites and targets, and exits. |

A case id is `group/estimator/variant`. For example, list the extended cases
without measuring them:

```sh
HDM_SUITE=extended HDM_GROUPS=lasso,controls HDM_LIST=true \
  julia --project=benchmark/results/env benchmark/run.jl
```

## Cases

| Case | Suite | Scenario | Target (relative gap) |
|---|---|---|---|
| `lasso/lasso/dense` | quick | `ls-200x50` | 1e-6 |
| `lasso/lasso/dense-standardized` | quick | `ls-200x50`, raw inputs, `intercept = true, standardize_x = true` | 1e-6 |
| `lasso/lasso/tall` | extended | `ls-tall-5000x40` | 1e-6 |
| `lasso/lasso/wide` | extended | `ls-wide-100x400` (p > n) | 1e-6 |
| `lasso/lasso/correlated` | extended | `ls-corr-300x100` (equicorrelated columns, ρ = 0.5) | 1e-6 |
| `lasso/lasso/large` | extended | `ls-2000x1000` | 1e-6 |
| `lasso/lasso/float32` | extended | `ls-200x50`, Float32 | 1e-4 |
| `controls/lasso/one-pass` | quick, with `HDM_GROUPS=controls` | `ls-200x50` | 1e-8, must end `target_missed` |

The scenarios of `problems.jl` draw columns with shifted means and scales and a
response with an intercept. `dense-standardized` passes them raw to
`lasso(X, y, λ, solver; intercept = true, standardize_x = true)` at the level of
`dense` (below). The other cases of the table and the control call `lasso` with its
defaults (no intercept, no standardization), so each prepares its inputs, untimed: the
columns are centered and scaled to mean square one, the response is centered, and the penalty level is
`λ = 0.1 λmax` with `λmax = maxⱼ |xⱼ'y|/n` of those inputs. For the Float32 case
the standardized Float64 inputs are rounded to Float32 first, `λmax` is computed
from the rounded inputs (in Float64), and the level is rounded to Float32. In
these coordinates the earlier lasso with an intercept and standardization
solved the same scenarios at the same ratio `λ/λmax`, which is the level of
`dense-standardized` as well. The solver is
`CDSolver(rng = Xoshiro(...))` with the package's defaults otherwise (random
coordinate order, `tol = 1e-7`); the Float32 case sets `tol = 1e-6`. The
generator is a Xoshiro seeded from a SHA-256 digest (`harness_rng("solver")`)
and restarted for each sample.

Observation, not a guarantee: with ProximalBase
`13891ccfd5c33a042c5df82d670abf1f2d786a0f` and ProximalSolvers
`dac9f4b6819d9321117ee3cf3549e9e46a043f01`, on Julia 1.10.13 and 1.13.4, the six
lasso cases `dense`, `tall`, `wide`, `correlated`, `large` and `float32` met
their targets in each of the repetitions 1 to 10. The smoke run of the integrity checks
(repetition 1, Julia 1.13.4, ProximalBase `51165b9eb0e1a100d03b1cee49e2853961779776`
and ProximalSolvers `e8569b34ac622b164bddf2604ea563bb903f965a`) ended `ok` for
`dense` and `dense-standardized`, with relative gaps of `9.9e-9` for both. The largest relative gap among the
five Float64 cases was `4.2e-7` (`wide`), and `4.0e-7` for `float32`. The control ends
`target_missed` (relative gap about `0.25` in repetition 1). Other dependency
commits can give other values.

The cases correspond to those of `legacy/full` as follows. The targets are the
earlier ones.

| Earlier case | Case now | Change |
|---|---|---|
| `lasso/lasso/dense-standardized` | the same id | Raw inputs, `intercept = true, standardize_x = true`, certified at the stored intercept and coefficients (below). |
| (none) | `lasso/lasso/dense` | The default call on inputs prepared as above, which the case with the keywords does not replace. |
| `lasso/lasso/wide`, `correlated`, `large` | the same ids | Inputs prepared as above. |
| `lasso/lasso/gram-tall` | `lasso/lasso/tall` | The Gram option does not exist; the plain call. |
| `lasso/lasso/float32` | the same id | Inputs prepared as above, then rounded to Float32. |
| `controls/lasso/max-iter-1` | `controls/lasso/one-pass` | `CDSolver(maxpasses = 1)`; the solver's warning is silenced. |
| `sparse-csc`, `weights-offset`, `sqrt_lasso`, `scaled_lasso`, `LassoPath`, `refit_lasso_path`, and every other group | none | The package has no such estimator, storage or option. |

## Certificates and targets

`problems.jl` draws the inputs of each scenario from Xoshiro streams seeded by
its key and the repetition, using scalar draws only.
`certificates.jl` states the objective of `lasso`, `‖y - Xβ‖²/(2n) + λ‖β‖₁`, and
computes the certificate from the arrays it is given: the prepared inputs, the
penalty level and the coefficient vector that the case takes from the returned
fit. The cases pass no other field of the fit, and nothing in the harness reads
a field of ProximalBase or ProximalSolvers. The certificate is the relative
duality gap `(P - D)/|P|`, where `P` is the objective at the returned
coefficients and `D` is the dual value at the point `tθ`, `θ = r/n` for the
residuals `r = y - Xβ`, with `t = min(1, λ/‖X'θ‖∞)` so that the point is dual
feasible: `D = t y'θ - t² n‖θ‖²/2`. `D` is a lower bound on the minimum, so in
exact arithmetic a nonoptimal point has a positive gap; the computed gap is
clamped to zero when it is negative by less than a rounding bound, and a more
negative one (a violation of weak duality) makes the case `uncertifiable`. For
`λ = 0` the certificate is the KKT residual `‖X'r/n‖∞` of the least squares,
relative to `max(1, ‖X'y‖∞/n)`, not a duality gap. `check_certificates.jl` tests it with
analytic problems, problems whose solution is known by construction,
nonoptimal points, weak duality, dual feasibility and nonfinite inputs, and
runs the `dense`, `dense-standardized` and `float32` cases and the control once against their targets.

The certificate of `dense-standardized` (`lasso_gap_standardized`) is stated for the
objective `‖y - β₀1 - Xβ‖²/(2n) + λ Σⱼ sⱼ|βⱼ|`, with `sⱼ` the root mean square of the
centered column `xⱼ - x̄ⱼ`. A column whose entries are all equal has that value as its mean,
centers to exact zeros and has scale 0, which the penalty ignores (a constant response is
centered the same way). The primal and dual core, from the stored inputs (converted exactly to
BigFloat) to the verdict, is arithmetic in 256-bit BigFloat, independent of the package, in which
each computed quantity carries a hand-derived absolute error bound (the error model is written in a
comment above `lasso_standardized_parts` in `certificates.jl`). With unit roundoff `u = 2⁻²⁵⁶` and
`G(k) = 8ku`, a sum of `k` terms or a dot product of `k` pairs has error at most `G(k)Σ|tᵢ|`, with
the magnitudes `Σ|aᵢ||bᵢ|` and not the computed sum carrying the cancellation; operand errors
propagate by `|âᵀb̂ - aᵀb| ≤ Σ(|âᵢ| ebᵢ + |b̂ᵢ| eaᵢ + eaᵢ ebᵢ)`. The bounds cover the means, centered
columns and scales (`esⱼ = εxⱼ + G(n+2)ŝⱼ`), the fitted values and residuals, the primal value
`P = ‖r‖²/(2n) + λ Σⱼ sⱼ|βⱼ|` on the raw inputs at the stored intercept `β₀` and coefficients `β` (not
`coef(out)`, which has `p + 1` entries), the correlations, the dual point and the dual value. The dual
point is `tθ` with `θ = (r - r̄)/n`, and `t` the largest value up to 1, times `1 - 2⁻¹⁰⁰` when below 1,
with `t(|ĝⱼ| + egⱼ) ≤ λ(ŝⱼ - esⱼ)` for the columns with `sⱼ > 0`; that inequality is checked, and it gives
`|x̃ⱼ'(tθ)| ≤ λ sⱼ` in exact arithmetic (a constant column has `x̃ⱼ = 0`). The exact dual value is
`D = ỹ'(tθ) - n‖tθ‖²/2`, which is at most the objective at every `(β₀, β)`, and in particular at the
profiled minimum. Let `P_hi = P̂ + eP`, `P_lo = P̂ - eP` and `D_lo = D̂ - eD`. The relative gap of the
harness is `max(P - D, 0)/max(P, floatmin)`; with `den_lo = max(P_lo, floatmin)` and
`num_hi = max(P_hi - D_lo, 0)`, the value `num_hi/den_lo`, rounded up to a Float64, is at least the true
relative gap. The certificate returns it with `ok` and does not pass through the shared
`relative_gap`, whose rounding allowance and Float64 clamp would break this argument. It is
`uncertifiable` when `(eP + eD)/den_lo > 1e-8` (1% of the smallest target, `1e-6`), when
`P_hi < D_lo`, when the dual point cannot be shown feasible, or when a scale is not a normal Float64,
is `0` for a column that is not constant, or is not above its error bound. A reported value therefore
never underestimates the true relative gap, so an `ok` is not false, and a miss can be spurious only
by a small amount in the value: it exceeds the true gap by at most `2(1 + gap)·1e-8` (from
`num_hi ≤ num + 2(eP + eD)` and `den_lo ≥ den - 2eP`). The steps outside the BigFloat core are the exact input conversion, the
validation of the scales in Float64, and the report of `P`, `D`, `θ` and `t` and of the rounded-up
verdict. In exact arithmetic the minimum of the problem over `β₀` is the minimum of the centered
problem, so `D` bounds it from below. A shift `δ` of `β₀` from the profiled optimum for the fixed `β`
raises `P` (by `δ²/2` when the residuals have mean zero) and with it the absolute duality gap; a
shift toward that optimum lowers them (the relative gap need not change in the same direction). The
target is the relative gap
`1e-6`, as for the other cases. `check_certificates.jl` builds a problem whose solution
`(β₀, β)` is known by construction and shows that a perturbed `β₀` misses the target; it also
checks a column whose squares underflow, a dual scaling near `1e-250`, a cancellation in `Xβ`
(`repeat([1e120 1 -1e120], 2, 1)`, which is `uncertifiable`), the exact constant rule, the uncertifiable
scales, and that the value is an upper bound of the exact relative gap of small problems computed in
rational arithmetic.

A coordinate-change test of the solver, such as `fit.result.converged`, is not
an optimality certificate for this problem and decides no status.

| Status | Meaning |
|---|---|
| `ok` | The certificate is valid and its value meets the threshold target. |
| `target_missed` | The certificate is valid and its value exceeds the threshold. |
| `uncertifiable` | The certificate is outside its domain or nonfinite. |
| `error` | Building the inputs, the call or the certification throws. |
| `timeout` | A compiled call took longer than `HDM_CASE_SECONDS`; the case stops after that call, which is not interrupted. |

A case takes the most severe status of its observations, in the order `error`,
`timeout`, `uncertifiable`, `target_missed`. Only `ok` samples are
timing-eligible; failed samples are recorded and never filtered.
`HDM_GROUPS=controls` checks this end to end: its case ends `target_missed`,
with no timing-eligible sample, and the runner exits with 2.

## Compare revisions

Keep one copy of this harness fixed and measure two HighDimensionalModels
checkouts with it, in two fresh environments. The second environment copies the
manifest of the first, so every dependency other than HighDimensionalModels is
identical: the same ProximalBase and ProximalSolvers checkouts must be given
to both, and `setup.jl` fails if any other manifest entry changes, including the
path of either. Interleave the repetitions in fresh processes, alternating which
revision runs first:

```sh
before=/absolute/path/to/before
after=/absolute/path/to/after
julia benchmark/setup.jl "$before" benchmark/results/env-before "$PROXIMALBASE_CHECKOUT" "$PROXIMALSOLVERS_CHECKOUT"
julia benchmark/setup.jl "$after" benchmark/results/env-after "$PROXIMALBASE_CHECKOUT" "$PROXIMALSOLVERS_CHECKOUT" \
  benchmark/results/env-before
runs=3
for rep in $(seq 1 "$runs"); do
  if [ "$((rep % 2))" -eq 1 ]; then set -- before after; else set -- after before; fi
  for side in "$@"; do
    code=0
    case "$side" in before) checkout="$before" ;; after) checkout="$after" ;; esac
    HDM_EXPECTED_PACKAGE_ROOT="$checkout" HDM_REPETITION="$rep" \
      HDM_OUTPUT="benchmark/results/$side" JULIA_NUM_THREADS=1 \
      julia --project="benchmark/results/env-$side" benchmark/run.jl || code=$?
    [ "$code" -eq 0 ] || [ "$code" -eq 2 ] || exit "$code"
  done
done
python3 benchmark/compare.py benchmark/results/before benchmark/results/after \
  benchmark/results/comparison --expected-runs "$runs"
```

`compare.py` refuses series that differ in the commit or version of ProximalBase
or ProximalSolvers; `summarize.py` summarizes each series separately.

Choose `runs` from 1 to 10. Use the same suite, groups, sample and time limits,
machine, and Julia and BLAS versions on both sides; different Julia versions need
separate environments (`setup.jl` refuses a reference made by another version).
The runner checks `HDM_EXPECTED_PACKAGE_ROOT` before building any case. For an
A/A comparison, set `before` and `after` to the same revision; it shows the
noise level of the machine.

`compare.py` writes `REPORT.md`, `comparison.csv` and `coverage.json` to a fresh
directory. It reports after/before ratios of the repetition medians with paired
95% intervals on log ratios, the worst certificate values, and the mean and
maximum bytes with their changes. Single-run ratios are exploratory, and an
interval containing one is inconclusive. A case with a non-`ok` repetition is
excluded from the ratios, with its reasons in the report. Invalid records,
missing repetitions, unmatched cases, or differences in settings, catalog,
data hashes, the commit or version of ProximalBase or ProximalSolvers, or the
versions and tree hashes of the other dependencies prevent a comparison. A
dependency tracked by path has an empty tree hash and is compared by version
only; the commits of ProximalBase and ProximalSolvers identify them. The
tree of a dependency is identified by its commit only when it is clean, so a
record whose ProximalBase or ProximalSolvers tree has tracked or untracked
changes is invalid and cannot be compared. The measured HighDimensionalModels
checkout may have changes; `package_hash` identifies its `Project.toml` and
`src/`.

## Measurement and evidence

The phase is always `call`: the warmed public call `lasso(X, y, λ, solver)`,
including its own setup and result construction. Drawing the inputs, preparing
them, constructing the solver, compilation, resets and certification are
outside timing. The first call of a case compiles; it is certified but
contributes no sample. Each sample then starts from fresh copies of the prepared
inputs, including the solver and its random number generator (the default
generator is reseeded too), times one call with `@timed`, and certifies that
output. Sampling stops at `HDM_SAMPLES` or `HDM_SECONDS`, with at least three
samples unless an error or timeout stops the case. The order of the cases is
shuffled per repetition with a seeded generator.

Intervals describe variation across processes and input draws, including
machine noise; they do not correct for correlated noise or multiple
comparisons. Profiles are diagnostic, not latency estimates. The default suite is
not a substitute for the package tests or for broad coverage of parameter
regimes. Do not select the fastest run or discard adverse runs; put the concise
finding and its limitations in the pull request, and archive the records,
manifests and any uncommitted sources behind a claim as described in
[Studies and campaigns](#studies-and-campaigns).

## Records

Each process writes files with the prefix `julia-<VERSION>-rep<NN>` under
`HDM_OUTPUT`. The TSV files have headers, empty missing fields, integer
nanoseconds and bytes, and round-trip floats:

| Suffix | Columns or contents |
|---|---|
| `-catalog.tsv` | `id group scenario estimator variant phase suite inputs measure target` |
| `-metrics.tsv` | `id group phase rep status samples ok_samples median_ns mean_ns sd_ns min_ns max_ns p95_ns gc_share median_bytes mean_bytes min_bytes max_bytes measure target worst_value median_value data_hash` |
| `-samples.tsv` | `id rep sample time_ns gc_ns bytes value status` |
| `-errors.tsv` | `id rep status exception stack`; one row per non-`ok` case |
| `-profiles.txt` | Optional profiles, with `HDM_PROFILE_SECONDS > 0` |
| `-metadata-start.toml`, `-metadata.toml` | Provenance at the start; the same with the final keys at the end |

The metadata records the settings; the Julia, BLAS and CPU identity; and, for
HighDimensionalModels (`package_*`), ProximalBase (`proximalbase_*`) and
ProximalSolvers (`proximalsolvers_*`), the loaded path, the git commit of its
checkout (`unavailable` where the path is not the root of a repository), whether
the tree differs from it (`*_dirty`; untracked files count for ProximalBase and
ProximalSolvers, tracked changes only for HighDimensionalModels) and the version.
A record with `unavailable` or a dirty tree for ProximalBase or ProximalSolvers
fails validation. It also records
SHA-1 hashes of the package's `Project.toml` and `src/` (`package_hash`) and of
the Julia files of this folder (`harness_hash`), and a `[dependencies]` table
with the version and tree hash of every package of the manifest that is not a
standard library, other than HighDimensionalModels (a package tracked by path
has an empty tree hash). The SHA-256 `data_hash` of a case covers its scenario
inputs and penalty level; for repetition 1 it is the same on Julia 1.10.13 and
1.13.4 for every case, and `compare.py` requires equal hashes. The final metadata
adds `cases`, `ok_cases`,
`nonok_cases`, `elapsed_s`, `maxrss_bytes` and `finished = true`; start
metadata without it marks an interrupted process, which is kept and restarted
in a fresh series.

The runner exits with 0 when every case is `ok`, 2 for complete records with
non-`ok` cases, and 3 when it cannot start or write its records. Validation
checks the integrity of records separately from numerical success, so records
with failed cases can be valid:

```sh
python3 benchmark/validate.py benchmark/results benchmark/results/validation.json
python3 benchmark/check_analysis.py
JULIA_NUM_THREADS=1 julia --project=benchmark/results/env benchmark/check_certificates.jl
```

Pass `--expected-runs N` to `validate.py`, `summarize.py` and `compare.py` for a
series of `N` repetitions; `validate.py` and `summarize.py` also take
`--versions VERSION ...`. `summarize.py` writes `SUMMARY.md` and `summary.csv`.
Generated results, environments and profiles are ignored by Git.

## CI integrity checks

Linux CI runs these commands, in this order, from the package root, with
`PROXIMALBASE_CHECKOUT` and `PROXIMALSOLVERS_CHECKOUT` the checkouts of the
run's dependency commits (clean checkouts: `setup.jl` and the runner refuse a
ProximalBase or ProximalSolvers tree with changes) and `julia` the Julia version
of the job:

```sh
ENV_DIR="$RUNNER_TEMP/hdm-bench-env"
OUT="$RUNNER_TEMP/hdm-bench-out"
python3 benchmark/check_analysis.py
julia benchmark/setup.jl "$PWD" "$ENV_DIR" "$PROXIMALBASE_CHECKOUT" "$PROXIMALSOLVERS_CHECKOUT"
JULIA_NUM_THREADS=1 julia --project="$ENV_DIR" benchmark/check_certificates.jl
export HDM_EXPECTED_PACKAGE_ROOT="$PWD" HDM_SAMPLES=5 HDM_SECONDS=0.2 JULIA_NUM_THREADS=1
HDM_GROUPS=lasso HDM_OUTPUT="$OUT/lasso" julia --project="$ENV_DIR" benchmark/run.jl
python3 benchmark/validate.py "$OUT/lasso" "$OUT/lasso.json"
code=0
HDM_GROUPS=controls HDM_OUTPUT="$OUT/controls" julia --project="$ENV_DIR" benchmark/run.jl > "$OUT/controls.log" || code=$?
cat "$OUT/controls.log"
[ "$code" -eq 2 ]
grep -qx 'CASE controls/lasso/one-pass status=target_missed' "$OUT/controls.log"
python3 benchmark/validate.py "$OUT/controls" "$OUT/controls.json"
```

In order: the Python controls of the record tools; the environment; the
certificate controls; the quick lasso cases, which must end `ok` (the runner exits
with 0); the validation of its records; the control case, which must end
`target_missed` (the runner exits with 2); and the validation of its records.
Under `bash -e`, a failing command other than the guarded control run
(`|| code=$?`) ends the sequence and fails the job.

These checks test the correctness of the harness: the record tools, the
certificate, the targets of the quick case and the control that must fail. No
timing from a shared runner is a merge gate: the smoke run's samples are few and
are not compared with anything. Formal performance studies are run only when
authorized for a specific question, as described below.

## Studies and campaigns

Studies beyond the harness, such as validation campaigns, comparisons with
other packages and profiling studies, keep their records out of Git.

- **The repository** keeps what detects regressions or informs users: tests
  with small reference values; cases of this harness; and, for comparisons
  with other packages, summary tables in the documentation, each built from a
  small committed file and stating the package versions, the date and the
  hardware class. Timings are not kept as regression baselines: compare the
  base and the head on one machine with `compare.py`.
- **The archive** of a study is one compressed file attached to the GitHub
  release `benchmark-evidence` of this repository, with its SHA-256 in the pull
  request. It holds the per-replication or per-sample results that the
  conclusions are computed from, the harness records behind a cited comparison,
  the scripts and environment files used, and the profiles or logs that a claim
  rests on. Release assets are public, so remove host names, paths and
  scheduler settings first.
- **Nothing else is kept:** exploratory, canary and superseded runs,
  intermediate files, and profiles that no claim rests on.

A pull request commits at most about 1 MB of study material: scripts,
aggregated tables and its report, removed before merge unless they belong in
the repository. A pull request that has already committed larger records
removes them and is squash-merged, so that they stay out of the history of
master; its commits remain reachable from the pull request.
