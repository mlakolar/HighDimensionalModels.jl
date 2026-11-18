# HighDimensionalModels 0.2.0-DEV

Breaking development reset. The package's main estimator is the Gaussian lasso at a single
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
- Group lasso: `group_lasso(X, y, λ, groups, solver = CDSolver(); intercept = true,
  standardize = :orthonormal, groupweights = nothing)` for a dense `Matrix{Float32}` or
  `Matrix{Float64}`, over ProximalBase's `GroupNormL2` and `BlockIterate` and ProximalSolvers' block
  coordinate descent. For disjoint nonempty `groups` that contain all the columns (contiguous or
  not) and weights `wₖ` (default `√|Gₖ|`), it minimizes `‖y - β₀1 - Xβ‖² / (2n) +
  λ Σₖ wₖ‖X̃_{Gₖ}β_{Gₖ}‖ / √n` with `:orthonormal` (the penalty is on the contribution of each group to the
  fitted values, so the fit depends on a group through its column span and is unchanged, up to rounding and the solver's tolerance, by an
  invertible recombination of its columns when the rank rule retains the same span), where `X̃` has centred columns
  with an intercept, or `‖y - β₀1 - Xβ‖² / (2n) + λ Σₖ wₖ‖β_{Gₖ}‖` with `:none`. With `:orthonormal` each
  group is factored by a pivoted QR in `T`; the rank is the number of diagonal entries of `R` above the
  tolerance `c√max(n, |G|) eps(T)|R₁₁|` (`c` is a provisional constant), the columns the rank decision leaves
  out get the coefficient 0, and a group of rank 0 stays in the fit as one column of zeros. It returns a
  `GroupLassoSolution` (coefficients, intercept, residuals, `λ`, the groups, the weights, the active groups,
  `standardize` and the unchanged ProximalSolvers result) with the StatsAPI methods `coef`, `predict`,
  `residuals`, `nobs`, `dof` and `deviance`.
- The loss and the penalty are ProximalBase's and the solver is ProximalSolvers' `CDSolver`; the
  package contains no optimization code. Runtime dependencies are ProximalBase, ProximalSolvers,
  StatsAPI and the standard library LinearAlgebra. It needs the development `master` branches of ProximalBase (0.6 series) and
  ProximalSolvers 0.3.2 and does not resolve against the registry.
- Removed: paths, screening and cross-validation; weights;
  the weighted, adaptive, feasible, scaled and square-root lasso; the elastic net and nonconvex
  penalties; generalized linear models; sparse-group and multitask regression; quantile
  regression; SLOPE and knockoffs; structured, fused, low-rank and varying-coefficient models;
  post-selection inference; the general coordinate-descent interfaces and the deprecated names;
  the HiGHS, Knockoffs, ProximalAlgorithms and StatsModels extensions.
- The previous package, version `0.1.0` (unreleased), with its tests, documentation, examples and
  the changelog that this file held, is preserved on the branch `legacy/full` and the tag
  `archive/pre-minimal-core/master`.
