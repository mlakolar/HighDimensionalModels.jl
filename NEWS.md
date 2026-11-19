# HighDimensionalModels 0.2.0-DEV

Breaking development reset. The package now has one estimator, the Gaussian lasso at a single
penalty level, `lasso(X, y, λ, solver = CDSolver())`, minimizing `‖y - Xβ‖² / (2n) + λ‖β‖₁` for a
dense `Matrix{Float32}` or `Matrix{Float64}`. It returns a `LassoSolution` (coefficients,
residuals, `λ` and the unchanged ProximalSolvers result) with the StatsAPI methods `coef`,
`predict`, `residuals` and `nobs`.

- Intercept and column scales: `lasso(X, y, λ, solver; intercept = false, standardize_x = false)`
  minimizes `‖y - β₀1 - Xβ‖² / (2n) + λ Σⱼ sⱼ|βⱼ|`, with `β₀ = 0` unless `intercept` and `sⱼ = 1`
  unless `standardize_x`. The scale `sⱼ` is the root mean square of the centred column (of the
  raw column without an intercept). With an intercept, a column whose entries are all equal has
  that value as its mean and centres to exact zeros; with `standardize_x = true` its scale is 0.
  Its coefficient is 0. The solve is the unweighted lasso
  on the centred and scaled data with the given `CDSolver`; with both keywords `false` the call
  is the one above. `LassoSolution` has two more fields, `intercept` and `hasintercept`. `coef` is
  `[intercept; coefficients]` with an intercept, `predict` checks the number of columns and adds
  the intercept, and `dof` and `deviance` (StatsAPI) are new. With `standardize_x = true`,
  `fit.result.minimizer` is the vector of the scaled coordinates, not `coefficients`.
- Elastic net: `lasso(X, y, λ, solver; alpha = 1, intercept = false, standardize_x = false)` minimizes
  `‖y - β₀1 - Xβ‖² / (2n) + λ(α‖θ‖₁ + (1 - α)‖θ‖²/2)` with `θⱼ = sⱼβⱼ`, for a finite `alpha` with
  `0 < alpha ≤ 1`; any other value, including one that is not a `Real`, throws an `ArgumentError`.
  The penalty is ProximalBase's `ElasticNet(λ₁, λ₂)` with `λ₁ = λα` and `λ₂ = λ(1 - α)` computed in
  the element type of the data, solved by the same `CDSolver` on the same transformed data. An `alpha`
  that is `1` in that type gives the lasso; one that is `0`, or an ℓ₁ level `λα` that underflows to
  `0` for `λ > 0`, would be a ridge penalty alone and throws an `ArgumentError` (ridge alone is not
  supported). `LassoSolution` has a new field `alpha`, placed after `λ`, with the effective
  `T(alpha)` (`1` for the lasso); `show` prints it when it is below 1. `dof` is the number of nonzero
  slopes, plus one with an intercept; for `alpha < 1` the number of nonzero slopes is the count that
  glmnet reports, and the count is not an unbiased estimate. No agreement with glmnet
  is claimed: glmnet scales `y` internally, which changes the ridge part unless `sd(y) = 1`.
- The loss and the penalty are ProximalBase's and the solver is ProximalSolvers' `CDSolver`; the
  package contains no optimization code. Runtime dependencies are ProximalBase, ProximalSolvers
  and StatsAPI. It needs the development `master` branches of ProximalBase (0.6 series) and
  ProximalSolvers 0.3.2 and does not resolve against the registry.
- Removed: paths, screening and cross-validation; weights;
  the weighted, adaptive, feasible and scaled lasso; nonconvex
  penalties; generalized linear models; group, sparse-group and multitask regression; quantile
  regression; SLOPE and knockoffs; structured, fused, low-rank and varying-coefficient models;
  post-selection inference; the general coordinate-descent interfaces and the deprecated names;
  the HiGHS, Knockoffs, ProximalAlgorithms and StatsModels extensions.
- The previous package, version `0.1.0` (unreleased), with its tests, documentation, examples and
  the changelog that this file held, is preserved on the branch `legacy/full` and the tag
  `archive/pre-minimal-core/master`.

## Square-root lasso (#223)

- **Breaking.** The square-root lasso is restored as `sqrt_lasso(X, y, λ, solver = CDSolver();
  intercept = false, standardize_x = true)`, with a normalized penalty level. The earlier function
  minimized `‖y - β₀ - Xβ‖ + λ Σⱼ ωⱼ|βⱼ|`. This one minimizes `‖y - β₀1 - Xβ‖/√n + λ‖θ‖₁`, with `θ` the
  coefficients in the coordinates of the penalty (`θⱼ = sⱼβⱼ` with `standardize_x = true`, as for `lasso`),
  so the same estimate needs `λ = λ_earlier/√n`: an earlier `sqrt_lasso(X, y, 20.0)` on `n = 400` rows is now
  `sqrt_lasso(X, y, 1.0)`. The noise estimate `σ` of the result is now `‖r‖/√n` for the residuals `r`, not their
  sample standard deviation. The defaults, `standardize_x = true` and `intercept = false`, are those of the
  earlier function.
- It returns a `SqrtLassoSolution` (`coefficients`, `intercept`, `hasintercept`, `residuals`, `λ`, `σ` and the
  unchanged ProximalSolvers result) with the StatsAPI methods `coef`, `predict`, `residuals`, `nobs`, `dof` and
  `deviance`. The solve is ProximalSolvers' square-root coordinate descent on `SqrtLeastSquaresLoss` at the
  level `√n λ`, in `Float64`, on the centred and scaled data of `lasso`. `λ` must be positive and finite
  (`lasso` fits `λ = 0`). The level is converted to `Float64` before the product with `√n`, which `sqrt_lasso`
  refuses with its own error where `Float64(λ)` or the product is not finite and positive, and the product
  must be representable in the element type (ProximalSolvers' error is passed on).
- `fit.result.converged` and `fit.result.gap` are ProximalSolvers' duality certificate of the transformed
  problem, unchanged: `converged = true` means that it proved a relative gap of at most `1e-4`, and a stop
  without a certificate (for example at an exact fit, where the residual is zero) or at `maxpasses` reports
  `converged = false`. The accuracy in the original coordinates is not certified; the documentation page
  gives the sizes of the rounding terms and a duality gap that a user can compute from the raw data.
