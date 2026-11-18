# The lasso

## Objective

`lasso(X, y, λ)` minimizes, over `β ∈ ℝᵖ`,

```math
\frac{1}{2n}\lVert y - X\beta\rVert_2^2 + \lambda \lVert \beta \rVert_1, \qquad n = \texttt{size}(X, 1),
```

by coordinate descent from `β = 0`. The factor `1/(2n)` is ProximalBase's `LeastSquaresLoss`
(which is `½‖y - Xβ‖²`) scaled by `1/n`. With this normalization `β = 0` is a minimizer in exact arithmetic if and only if
`λ ≥ λmax = maximum(abs, X' * y) / n`. At `λ = λmax` itself the solver can return a coefficient
of the order of `eps(T)` (seen with `Float32`), where `λmax` computed outside the solver differs
from the solver's own correlations by rounding.

The keywords `intercept` and `standardize_x`, both `false` by default, give

```math
\frac{1}{2n}\lVert y - \beta_0\mathbf{1} - X\beta\rVert_2^2 + \lambda \sum_{j=1}^{p} s_j |\beta_j|,
```

where `β₀ = 0` unless `intercept = true` (the intercept is not penalized) and `sⱼ = 1` unless
`standardize_x = true`. With standardization `sⱼ` is the root mean square of the column
`xⱼ - x̄ⱼ`, where `x̄ⱼ` is the mean of the column with an intercept and `0` without one: the
centred column with an intercept, the column itself without. With the
weights, `β = 0` (with `β₀ = ȳ`, or `0` without an intercept) is a minimizer in exact arithmetic
if and only if `λ ≥ λmax = maxⱼ |x̃ⱼ'ỹ| / (n sⱼ)` over the columns with `sⱼ > 0`, where `x̃ⱼ` and
`ỹ` are the centred (or raw, without an intercept) column and response; `λmax = 0` when no column has
a positive scale (the slopes are then zero for every `λ ≥ 0`; the intercept can still be nonzero).

The call solves an ordinary unweighted lasso on transformed data. Let `ȳ` be the mean of `y`
(`0` without an intercept), `Z` the matrix with the columns `(xⱼ - x̄ⱼ)/sⱼ` (`xⱼ - x̄ⱼ` without
standardization, `xⱼ` with neither keyword) and `ỹ = y - ȳ` (`y` itself without an intercept).
The solver minimizes `‖ỹ - Zθ‖²/(2n) + λ‖θ‖₁` and the call returns `βⱼ = θⱼ/sⱼ` and
`β₀ = ȳ - x̄ᵀβ`. With both keywords `false`, `X` and `y` are not copied: the solver works on them
directly.

- **Constant columns.** With an intercept, a column whose entries are all equal has that value
  as its mean, so that it centres to exact zeros (a constant response centres to zeros in the
  same way). The solver keeps a column of zeros at zero, so the coefficient of such a column is
  `0`; with `standardize_x = true` its scale is `0` and no division takes place. A zero
  column also gets `0` without an intercept. Without an intercept, a constant column that is not
  zero is an ordinary column: with `standardize_x = true`, `fill(0.1, n)` has the scale `0.1` up to rounding
  (the computed root mean square of the stored entries).
- **Zero scale and the weight.** No tolerance decides a zero scale, only exact zeros after
  centring. With `standardize_x = true`, the scale `sⱼ` of any other column is the root mean
  square of its centred entries (of the column itself without an intercept), and the penalty
  weight on `βⱼ` is `λsⱼ`. A column whose scale underflows to zero while it is not all zeros,
  and a design that overflows when centred or when the fit is scaled back, are refused with an
  `ArgumentError` that names `X` or `y` and the step.
- **Overflow.** The means and the scales are computed from the entries divided by a power of
  two not above the largest entry, so that the terms of the sums are at most 2 in absolute
  value (at most 4 for the squares in a scale). A design whose squares overflow, for example `Float32` columns near `1e20`, is
  refused with an `ArgumentError` by the default call, because the solver's curvature
  overflows; with `standardize_x = true` the matrix `Z` is finite, and such data can be fit,
  provided that the centring, the scales, the solve and the back-map are representable. Centring
  is computed in `T`, and an `ArgumentError` is thrown when a centred entry is not finite: for
  example, `Float32[floatmax(Float32), floatmax(Float32), -floatmax(Float32)]` as a column with
  `intercept = true` throws it, with either value of `standardize_x`.
- **Limit of the back-map.** `βⱼ = θⱼ/sⱼ` is formed in the type `T`. A coefficient whose exact
  value is below the range of `T` underflows to a subnormal number or `0`, and the residuals on
  the original data, which use the stored `β`, then differ from those of the solver's problem.
  For `X = reshape([-1e300, 1e300], 2, 1)`, `y = [-1e-100, 1e-100]`, `λ = 0` and
  `standardize_x = true`, the solver returns `θ = 1e-100` with minimum `0` and `converged = true`,
  while the stored coefficient is `0` (the exact value `1e-400` underflows) and
  `sum(abs2, residuals(fit))` is `2e-200`. The result of the solver is left as it returned it.

## The elastic net

The keyword `alpha` (`1` by default) mixes the ℓ₁ penalty with a ridge penalty. With
`θⱼ = sⱼβⱼ` (so `θ = β` unless `standardize_x = true`), `lasso` minimizes

```math
\frac{1}{2n}\lVert y - \beta_0\mathbf{1} - X\beta\rVert_2^2
  + \lambda \Bigl(\alpha \lVert \theta \rVert_1 + \frac{1-\alpha}{2} \lVert \theta \rVert_2^2\Bigr).
```

`alpha` is a finite `Real` with `0 < alpha ≤ 1`. `alpha = 1` is the lasso above, and `alpha < 1`
adds the ridge term. Any other value, including a value that is not a `Real`, throws an
`ArgumentError` that names the range; `alpha = 0`, a ridge penalty alone, is not supported.

- **The effective parameters.** `alpha` is checked as given. Then `α = T(alpha)` and `λ_T = T(λ)`
  are formed in the element type `T` of the data, and these two numbers are what the fit stores
  as `fit.alpha` and `fit.λ`. If `α` is `1` in `T` (for example `alpha = 1 - 1e-9` with `Float32`
  data), the fit is the lasso and `fit.alpha` is `1`. If `α` is `0` in `T` (for example
  `alpha = 1e-50` with `Float32` data), or `λ > 0` and `λ_T * α` is `0`, the penalty would be a ridge
  alone and an `ArgumentError` says so. Otherwise the penalty is ProximalBase's
  `ElasticNet(λ₁, λ₂)` with `λ₁ = λ_T * α` and `λ₂ = λ_T * (1 - α)`, computed in `T`, so that
  `fit.λ * fit.alpha` is `λ₁` and `fit.λ * (1 - fit.alpha)` is `λ₂`. `λ = 0` gives least squares.
  With `α < 1` in `T`, `lasso` itself refuses a negative `λ` (`λ < 0`; `-0.0` is not negative) with an
  `ArgumentError` that names `λ`, also when it would round to `-0.0` or `0` in `T`; a `NaN`, an
  infinite `λ`, and a `λ` whose levels `λ₁`, `λ₂` are not representable in `T` are refused by
  ProximalBase and ProximalSolvers on those levels.
- **The solve.** The same `CDSolver` solves the problem on the same transformed data as for the
  lasso (`Z`, `ỹ`, or `X`, `y` themselves with `intercept` and `standardize_x` both `false`), with `ElasticNet(λ₁, λ₂)` in
  place of the ℓ₁ penalty, and the back-transformation and residuals are those of the lasso.
  With `alpha` omitted, or equal to `1` (or to a value that is `1` in `T`), the penalty is
  `NormL1(λ)` and the call is the lasso call.
- **The zero solution.** With the weights `sⱼ`, `β = 0` (with `β₀ = ȳ`, or `0` without an
  intercept) is a minimizer in exact arithmetic if and only if
  `λ ≥ λmax = maxⱼ |x̃ⱼ'ỹ| / (n sⱼ α)`, over the columns with `sⱼ > 0`.
- **Optimality conditions.** With `zⱼ = x̃ⱼ/sⱼ` (`x̃ⱼ` and `ỹ` as above) and `r = ỹ - Zθ`, a minimizer
  has `|zⱼ'r/n| ≤ λα` where `θⱼ = 0` and `zⱼ'r/n - λ(1 - α)θⱼ = λα sign(θⱼ)` where `θⱼ ≠ 0`.
- **Not glmnet.** No agreement with glmnet is claimed: glmnet scales `y` internally, which changes
  the ridge part unless `sd(y) = 1`. Also, `dof(fit)` is the number of nonzero slopes, plus one with an
  intercept; for `alpha < 1` the number of nonzero slopes is the count that glmnet reports, and the
  count is not an unbiased estimate of the degrees of freedom of the fit.

Three nearly collinear columns, with a response that depends on all three, fit by the lasso and
by the elastic net at the same `λ`:

```@example lasso
using HighDimensionalModels, Random

solver = CDSolver(order = :cyclic, momentum = false)
rng3 = Xoshiro(2)
n3, p3 = 100, 6
z3 = randn(rng3, n3)
Xc = hcat(z3 .+ 0.1 .* randn(rng3, n3, 3), randn(rng3, n3, p3 - 3))   # columns 1-3 share the factor z3
yc = Xc[:, 1:3] * [1.0, 1.0, 1.0] + 0.5 * randn(rng3, n3)
lasso_fit = lasso(Xc, yc, 0.2, solver)
en_fit = lasso(Xc, yc, 0.2, solver; alpha = 0.5)
(lasso = round.(lasso_fit.coefficients, digits = 3), elastic_net = round.(en_fit.coefficients, digits = 3))
```

For these data the lasso sets the third coefficient to `0` and gives the second the largest
share, while the elastic net keeps the three correlated columns with coefficients of the same
order. The printed form of a fit shows `α` when it is below `1`:

```@example lasso
en_fit
```

## Inputs

- `X::Matrix{T}` and `y::Vector{T}` with `T` equal to `Float32` or `Float64`; the fit has the
  same `T`. Other array types and mixed element types throw an `ArgumentError` that names the
  supported types.
- `λ::Real`, finite, nonnegative and representable in `T` (not overflowing to `Inf` and not
  underflowing to `0` from a nonzero value). With `α = 1` in `T` it reaches the solver in its own type, so an `Int`,
  a `Rational` or a `BigFloat` level that `T` represents is accepted, and the stored `fit.λ` is
  `T(λ)`.
- `size(X, 1) ≥ 1` is checked by this package. The remaining checks (a response of length `n`,
  `p ≥ 1`, finite entries of `X` and `y`, the level `λ`) are those of ProximalBase and
  ProximalSolvers; their errors are passed on unchanged.
- `alpha` is a finite `Real` with `0 < alpha ≤ 1` (see [The elastic net](@ref)); it is converted to
  `T`, and the stored `fit.alpha` is `T(alpha)`.
- `intercept` and `standardize_x` are `true` or `false`; any other value throws an
  `ArgumentError` that names the keyword. With `intercept` or `standardize_x` set, `size(y, 1) == size(X, 1)` and
  `size(X, 2) ≥ 1` are checked by this package (a `DimensionMismatch`), and the entries of `X`
  (and of `y` with an intercept) must be finite.
- `X` and `y` are not modified and the fit keeps no reference to them.

## Result

```@example lasso
using HighDimensionalModels, Random

rng = Xoshiro(1)
n, p = 100, 20
X = randn(rng, n, p)
y = X[:, 1:3] * [2.0, -1.5, 1.0] + 0.5 * randn(rng, n)

solver = CDSolver(order = :cyclic, momentum = false)
fit = lasso(X, y, 0.1, solver)
fit
```

[`LassoSolution`](@ref) has the fields `coefficients`, `residuals` (`y - intercept - X * coefficients`,
on the data given to `lasso`, computed once), `λ`, `alpha` (`1` for a lasso fit), `intercept`,
`hasintercept` and `result`, the ProximalSolvers result unchanged. The StatsAPI methods are:

- `coef(fit)`: `[intercept; coefficients]` with an intercept, and `coefficients` (the same
  vector) without one;
- `predict(fit, Xnew)`: `Xnew * coefficients`, plus the intercept with one; a `Xnew` with another
  number of columns throws a `DimensionMismatch`;
- `residuals(fit)`, `nobs(fit)`;
- `dof(fit)`: the number of nonzero slopes, plus one with an intercept (for `alpha < 1` the number of
  nonzero slopes is the count that glmnet reports, and the count is not an unbiased estimate);
- `deviance(fit)`: the sum of the squared residuals.

With an intercept and standardization:

```@example lasso
scales = range(0.5, 4.0; length = p)
Xr = X .* transpose(scales) .+ 5          # columns of different scales and means
yr = y .+ 10
fit2 = lasso(Xr, yr, 0.1, solver; intercept = true, standardize_x = true)
(intercept = fit2.intercept, coefficients = fit2.coefficients[1:5],
 length_of_coef = length(coef(fit2)), coef_head = coef(fit2)[1:3])
```

`fit2.coefficients` and `fit2.intercept` are those of the original columns. The columns of this
design have different scales, and the penalty `λ sⱼ|βⱼ|` is weighted by the scale of each one.

## Convergence and optimality

The result of the solver is `fit.result`, a `ProximalSolvers.CDResults`, with `converged`,
`passes`, `maxchange`, `tolerance` and `minimum`. For this problem `converged = true` means that
the largest change of a coefficient in a full pass, in the solver's scaling of the changes, was
below `tolerance` (the `CDSolver` documentation of ProximalSolvers defines both); it is not an
optimality certificate, and this package adds no convergence flag of its own. A run that reaches
`maxpasses` without that test being met returns `converged = false` and ProximalSolvers warns;
a run that meets the test within the cap, even after one pass, returns `converged = true`.

With `intercept = true` or `standardize_x = true`, `fit.result` is the result of ProximalSolvers
on the transformed problem `‖ỹ - Zθ‖²/(2n) + λ‖θ‖₁` (with the penalty
`λ(α‖θ‖₁ + (1 - α)‖θ‖²/2)` for `alpha < 1`). With `standardize_x = true`,
`fit.result.minimizer` is `θ`, and not `coefficients` (in exact arithmetic `θⱼ = sⱼ βⱼ` for the
mathematical back-map `βⱼ = θⱼ/sⱼ`; the stored slopes are the rounded divisions, with `0` for a column
of scale `0`, and `θⱼ = sⱼ βⱼ` need not hold for them, see the limit of the back-map above) (with `intercept = true`
alone, `θ = β` and the two are the same vector); `fit.result.minimum` is
the value of that problem's objective at `θ`, which equals the weighted objective above at `(β₀, β)` in
exact arithmetic, not the plain lasso objective of `β`.

`maxchange` and `tolerance` are the solver's. With `relative = true` (the default of `CDSolver`)
and `ỹ ≠ 0`, `maxchange` is the largest `|Δθⱼ|‖Zⱼ‖/√n` of a pass and `tolerance` is
`tol‖ỹ‖/√n`, where `Zⱼ` is column `j` of the matrix the solver sees: `Z` with
`standardize_x = true` (`‖Zⱼ‖ = √n` up to rounding for a column that is not zero, so the changes
are in units of `θ`), and `X` or the centred `X` otherwise (so the changes are multiplied by
`‖xⱼ‖/√n` or `‖x̃ⱼ‖/√n`). With `relative = false`, or `ỹ = 0`, the threshold is the absolute `tol`
and the changes are not multiplied. For `X = reshape([1., 3., 5.], 3, 1)`, `y = [1., 2., 6.]`,
`λ = 0.1`, `intercept = true`, one cyclic pass (`maxpasses = 1`, which makes the solver warn) and
`tol = 1e-4`, both settings return the same
`θ`, with `(maxchange, tolerance)` of about `(1.98, 2.2e-4)` with relative scaling and
`(1.2125, 1e-4)` without.

To check a fit without the solver's stopping rule, compare the correlations of the residuals
with the penalty level: at a minimizer, `|Xⱼ'r|/n ≤ λ` for every column and `Xⱼ'r/n = λ sign(βⱼ)`
where `βⱼ ≠ 0`, with `r = y - Xβ`. (With the keywords, the same holds for the columns `x̃ⱼ` of the
objective above, centred or raw, `r = y - β₀ - Xβ` and `λ sⱼ` in place of `λ`; for `alpha < 1` the conditions
are those of [The elastic net](@ref).) Rounding limits the accuracy that can be attained, but what a
run achieved depends also on `tol`, `maxpasses` and the conditioning of `X`: a loose `tol` can
report `converged = true` with large deviations. The deviations have to be measured, as below.

```@example lasso
g = X' * residuals(fit) / n
S = findall(!iszero, coef(fit))
(converged = fit.result.converged, tolerance = fit.result.tolerance,
 largest_excess = maximum(abs, g) - fit.λ,
 support_deviation = maximum(abs.(g[S] .- fit.λ .* sign.(coef(fit)[S]))))
```

## The solver argument

The optional fourth argument is a `CDSolver` from ProximalSolvers, with its options (`maxpasses`,
`tol`, `order`, `relative`, `momentum`, `rng`). `CDSolver()` visits the coordinates in random
order, drawn from a borrowed random number generator, by default the one of the task, which
advances when the ordering draws from it (a design with one column needs no draw, and a solver
with `maxpasses = 0` draws nothing). Two default calls on a design with several columns need not
return identical coefficients. `CDSolver(order = :cyclic, momentum = false)` visits the
coordinates in a fixed order. `tol` is the threshold of the change test of a coefficient, not a
bound on the distance to the minimizer (see the `CDSolver` documentation of ProximalSolvers for
its meaning and for the limits of `Float32`).

## Reference

```@docs
lasso
LassoSolution
dof(::LassoSolution)
deviance(::LassoSolution)
```
