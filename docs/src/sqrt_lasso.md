# The square-root lasso

## Breaking change from the earlier `sqrt_lasso`

`sqrt_lasso` now minimizes a normalized objective, and the penalty level of the earlier function
does not carry over. The earlier function minimized

```math
\lVert y - \beta_0 - X\beta\rVert_2 + \lambda \sum_{j=1}^{p} \omega_j |\beta_j| ,
```

and this one minimizes

```math
\frac{1}{\sqrt{n}}\lVert y - \beta_0\mathbf{1} - X\beta\rVert_2 + \lambda \lVert \theta \rVert_1 ,
\qquad n = \texttt{size}(X, 1),
```

with `θ` the coefficients in the coordinates of the penalty (`θⱼ = sⱼβⱼ`, defined below). The same
estimate, with the same penalty weights, needs `λ = λ_earlier/√n`. A call of the earlier function with `n = 400` rows and
`λ = 20.0` becomes `sqrt_lasso(X, y, 1.0)`; in general

```julia
sqrt_lasso(X, y, λ_earlier / sqrt(size(X, 1)))
```

The noise estimate `σ` of the result is now `‖r‖/√n` for the residuals `r`, the root mean square
of the residuals, and not their sample standard deviation. The defaults are those of the earlier
function, `standardize_x = true` and `intercept = false`. The function returns a
[`SqrtLassoSolution`](@ref).

## Objective

`sqrt_lasso(X, y, λ)` minimizes, over `β₀` and `β ∈ ℝᵖ`,

```math
\frac{1}{\sqrt{n}}\lVert y - \beta_0\mathbf{1} - X\beta\rVert_2 + \lambda \sum_{j=1}^{p} s_j |\beta_j| ,
```

where `β₀ = 0` unless `intercept = true` (the intercept is not penalized) and `sⱼ = 1` unless
`standardize_x = true`, as for [`lasso`](@ref): with standardization `sⱼ` is the root mean square of
the column `xⱼ - x̄ⱼ`, where `x̄ⱼ` is the mean of the column with an intercept and `0` without one. Then
`Σⱼ sⱼ|βⱼ| = ‖θ‖₁` for the standardized coefficients `θⱼ = sⱼβⱼ`. Without an intercept the scale is
the uncentred root mean square of the column. A column whose centred entries are all zero (a zero
column, or a constant column with an intercept) has scale `0` with `standardize_x = true`; with either value of
`standardize_x` the solver keeps such a column at zero, and its coefficient is `0`.

The loss has no `1/√n` in ProximalBase: `SqrtLeastSquaresLoss(y, X)` is `‖y - Xβ‖₂`, and
ProximalSolvers refuses a scaled square-root loss and asks for the unscaled loss at a rescaled level.
So `sqrt_lasso` solves, on the centred and scaled data `Z` (columns `(xⱼ - x̄ⱼ)/sⱼ`) and `ỹ`
(`y - ȳ` with an intercept),

```julia
solve!(θ, SqrtLeastSquaresLoss(ỹ, Z), NormL1(λps), solver),     λps = sqrt(Float64(n)) * Float64(λ),
```

and returns `βⱼ = θⱼ/sⱼ` and `β₀ = ȳ - x̄ᵀβ`. The level `λps` is a `Float64` for either element type
of the data. In exact arithmetic the objective that ProximalSolvers minimizes is `√n` times the one above:
`fit.result.minimum` is on that scale, and `fit.result.minimum/√n` is the objective of `sqrt_lasso`
at `θ`. With both keywords `false`, `X` and `y` are not copied and the solver works on them directly.

With `ỹ ≠ 0`, `β = 0` is a minimizer in exact arithmetic if and only if
`λ ≥ λmax = maxⱼ |zⱼᵀỹ|/(√n‖ỹ‖)` over the columns with `sⱼ > 0`, with `zⱼ` the column `j` of `Z`: the
level that sets the coefficients to zero does not depend on the scale of `y`. In exact arithmetic the
estimate is scale-equivariant in `y` at a fixed `λ`: for `c > 0` the fit to `c y` has the coefficients,
the intercept and `σ` of the fit to `y` multiplied by `c`.

## Inputs

- `X::Matrix{T}` and `y::Vector{T}` with `T` equal to `Float32` or `Float64`; the fit has the same `T`.
  Other array types and mixed element types throw an `ArgumentError` that names the supported types.
- `λ::Real`, positive and finite. A `λ` that is zero, negative, `NaN` or infinite throws an `ArgumentError`
  that states this and points to [`lasso`](@ref) for `λ = 0`: ProximalSolvers' square-root lasso needs
  a positive level. The stored `fit.λ` is `T(λ)`.
- The level is formed in stages. `λ` is converted to `Float64`, and `λps = sqrt(Float64(n)) * Float64(λ)` is
  computed in `Float64`. If `Float64(λ)` or the product is not finite and positive (a `BigFloat` or `Rational` `λ`
  below the `Float64` range, such as `big"1e-324"` with `n = 100`, or a `Float64` `λ` whose product overflows,
  such as `1e308` with `n = 40`), `sqrt_lasso` throws its own `ArgumentError`, "the level √n λ for n = … and
  λ = … is not a finite positive Float64", and ProximalSolvers is not called. Otherwise ProximalSolvers checks that
  `λps` is representable in `T` (a finite positive normal or subnormal number), and its error ("not
  representable without overflow or underflow") is passed on unchanged. That limit is on `√n λ` computed in
  this way, not on `λ`: for `Float32` data with `n = 40`, `λ = 1e-47` gives `λps ≈ 6.3e-47`, which rounds to
  zero in `Float32`.
- `n ≥ 1`, `length(y) == n` and `p ≥ 1` are checked by this package (a `DimensionMismatch`).
  With a keyword set, a design (or, with an intercept, a response) that does not preprocess (a non-finite
  entry, a mean or scale that overflows) is refused with the `ArgumentError` of `lasso`, which names the column
  or the step and starts with `lasso:`. The other checks of the data (for example a non-finite response
  without an intercept, or any non-finite entry with both keywords `false`) are ProximalSolvers', and their
  errors are passed on.
- `intercept` and `standardize_x` are `true` or `false`; any other value throws an `ArgumentError`
  that names the keyword.
- `X` and `y` are not modified and the fit keeps no reference to them.

## Result

```@example sqrt_lasso
using HighDimensionalModels, Random

rng = Xoshiro(1)
n, p = 100, 20
X = randn(rng, n, p)
y = X[:, 1:3] * [2.0, -1.5, 1.0] + 0.5 * randn(rng, n)

solver = CDSolver(order = :cyclic, momentum = false)
fit = sqrt_lasso(X, y, 0.3, solver)
fit
```

[`SqrtLassoSolution`](@ref) has the fields `coefficients`, `intercept`, `hasintercept`, `residuals`
(`y - intercept - X * coefficients`, on the data given to `sqrt_lasso`, computed once), `λ`, `σ` and
`result`, the ProximalSolvers result unchanged. The StatsAPI methods are defined as for
[`LassoSolution`](@ref): `coef`, `predict`, `residuals`, `nobs`, `dof` (the number of nonzero
coefficients, plus one with an intercept) and `deviance` (the sum of the squared residuals).

```@example sqrt_lasso
(nonzero = dof(fit), σ = fit.σ, converged = fit.result.converged, relative_gap_bound = fit.result.gap,
 minimum_over_sqrt_n = fit.result.minimum / sqrt(n))
```

## Certification and accuracy

Two statements are kept apart, here, in the docstring of `SqrtLassoSolution` and in `show`.

**Certification of the transformed problem.** `fit.result.converged` and `fit.result.gap` are
ProximalSolvers' and are neither recomputed nor changed. `converged = true` means that the
duality certificate of ProximalSolvers proved that the relative gap `(P(θ) - P*)/P*` of the problem it
solved is at most `1e-4`, and `gap` is that bound, rounded up in `T`. The problem it solved is
`‖ỹ - Zθ‖ + λps ‖θ‖₁` for the stored `Z` and `ỹ` in `T`, at the level `λps = √n λ` and the point `θ`
(`fit.result.minimizer`). `show` reads "duality certificate of the transformed problem: met" or "not
met".

**Accuracy in the original coordinates.** This is not certified. The returned `β`, `β₀` and
`residuals` are formed from `θ`, and the objective of `sqrt_lasso` at them differs from the
transformed problem by rounding. The sizes of the terms, for `u = eps(T)/2`, hold where the quantity
being rounded is in the normal range of `T`:

- the centred and scaled data: an absolute error of about `eps(T)(|xᵢⱼ| + |x̄ⱼ|)/sⱼ` in an entry of `Z`
  (and of `eps(T)(|yᵢ| + |ȳ|)` in an entry of `ỹ`), from the mean and the centring, and a relative error
  in the scale `sⱼ` of up to about `n eps(T)` (a sum of `n` squares), usually much smaller;
- the back-map `βⱼ = θⱼ/sⱼ`: a relative error of `eps(T)/2`;
- the intercept `β₀ = ȳ - x̄ᵀβ`: an error of about `eps(T)(|ȳ| + Σⱼ|x̄ⱼβⱼ|)`, which grows with the number of
  columns;
- the residuals `y - β₀ - Xβ`: an error of about `p eps(T)(|yᵢ| + |β₀| + Σⱼ|xᵢⱼβⱼ|)`.

A result below `floatmin(T)` is rounded to a subnormal number or to zero, with an absolute error of up to
`nextfloat(zero(T))/2`, which is a relative error of up to 1, and the sizes above do not apply (an example
follows the checker below). With both keywords `false` the stored problem is the problem that was given, and the
first three terms do not arise. The tests of this package compare the duality gap below with `1e-4` plus a
perturbation term derived from these sizes, for fits whose quantities are in the normal range.

Nothing in `sqrt_lasso` bounds the accuracy of `β` and `β₀` in the original coordinates, and the
package states no accuracy claim of its own. To measure it, compute a duality gap in the original
coordinates, independently of the solver. Let `Λ = √n λ`, `r = y - β₀1 - Xβ` (with `β₀ = 0` without an
intercept) and `x̃ⱼ = xⱼ - x̄ⱼ`. With an intercept let `r_c` be `r` centred at its mean; without one,
`r_c = r`. The centring is needed with an intercept: with a rounded `β₀` the vector `1ᵀr` is not zero.
For `r_c ≠ 0` let `gⱼ = |x̃ⱼᵀr_c|/(sⱼ‖r_c‖)` over the columns with `sⱼ > 0` (`g = 0` if there is none),
`t = min(1, Λ/maxⱼ gⱼ)` and `u = t r_c/‖r_c‖`. Then `‖u‖ ≤ 1` and `|x̃ⱼᵀu| ≤ Λ sⱼ`, and with an
intercept also `1ᵀu = 0`. For every `b`, and with an intercept every `b₀` (without one `b₀ = 0` and `x̃ⱼ = xⱼ`),
`‖y - b₀1 - Xb‖ + Λ Σⱼ sⱼ|bⱼ| ≥ uᵀ(y - b₀1 - Xb) + Λ Σⱼ sⱼ|bⱼ| ≥ uᵀy`, by Cauchy–Schwarz, the bound on
`|x̃ⱼᵀu|` and, with an intercept, `1ᵀu = 0`. So `D = uᵀy` is a lower bound of the minimum `P*`, and
`P = ‖r‖ + Λ Σⱼ sⱼ|βⱼ|` is an upper bound. When `D > 0`, `0 < D ≤ P* ≤ P` and `(P - D)/D` bounds
`(P - P*)/P*` for the returned point. When `D ≤ 0`, which can happen at a point that is far from the minimum
(for example after one pass), `(P - D)/D` is not a bound of anything and can be negative, so the function below
returns `Inf`; it also returns `Inf` when `r_c = 0`, where `u` is undefined. It computes the quantities in
256-bit arithmetic from the raw `X` and `y`. The level defaults to `fit.λ`, which is `T(λ)`; pass the `λ` given
to `sqrt_lasso` as `λ` when it is not a number of `T`.

```@example sqrt_lasso
function dual_gap(X, y, fit; λ = fit.λ, intercept = fit.hasintercept, standardize_x = true)
  setprecision(BigFloat, 256) do
    n, p = size(X)
    Xb, yb, β = BigFloat.(X), BigFloat.(y), BigFloat.(fit.coefficients)
    x̄ = intercept ? [sum(Xb[:, j]) / n for j in 1:p] : zeros(BigFloat, p)
    Xt = Xb .- transpose(x̄)
    s = standardize_x ? [sqrt(sum(abs2, Xt[:, j]) / n) for j in 1:p] : ones(BigFloat, p)
    Λ = sqrt(BigFloat(n)) * BigFloat(λ)
    r = (yb .- BigFloat(fit.intercept)) .- Xb * β
    rc = intercept ? r .- sum(r) / n : r
    N = sqrt(sum(abs2, rc))
    iszero(N) && return Inf                       # u is undefined
    P = sqrt(sum(abs2, r)) + Λ * sum(s .* abs.(β))
    g = maximum((abs(sum(Xt[:, j] .* rc)) / s[j] for j in 1:p if s[j] > 0); init = zero(BigFloat)) / N
    t = g > Λ ? Λ / g : one(BigFloat)
    D = t * sum(rc .* yb) / N
    D > 0 ? Float64((P - D) / D) : Inf            # a bound of (P - P*)/P* only where D > 0
  end
end

dual_gap(X, y, fit)
```

The value of `dual_gap` is a different number from `fit.result.gap` in general: that one bounds the suboptimality of the
transformed problem through ProximalSolvers' own dual point, and this one is computed in the original
coordinates from the exact residual of the returned point, so that it includes the rounding of the
preprocessing, the back-map and the intercept listed above.

The same function on three other cases, which the documentation build checks: a zero design (the fit is
`β = 0`, `g = 0` and the gap is zero), a zero response with a nonzero design (`r_c = 0`) and a point with
`D < 0` (the coefficients that a run stopped after one pass returned, on a `3 × 2` design with
`λ = 1e-5` and `standardize_x = false`).

```@example sqrt_lasso
X0, y0 = zeros(4, 1), [-1.0, 1.0, -1.0, 1.0]
fit0 = sqrt_lasso(X0, y0, 0.1, CDSolver(order = :cyclic, momentum = false))
Xe = reshape([-1.0, 1.0, -1.0, 1.0], 4, 1)
fite = sqrt_lasso(Xe, zeros(4), 0.1, CDSolver(order = :cyclic, momentum = false))
X3 = [-0.6570368881174425 1.2438338374014866; -0.8408751875192206 -0.3930342766757361;
      -0.4228327963595001 0.5047040065705481]
y3 = [-3.347462601798711, -0.633491262245945, -0.808003710231016]
point = (coefficients = [2.3328834555906037, -1.3745674525556815], intercept = 0.0, hasintercept = false, λ = 1e-5)
checks = (zero_design = dual_gap(X0, y0, fit0), zero_response = dual_gap(Xe, zeros(4), fite),
          negative_dual = dual_gap(X3, y3, point; standardize_x = false))
@assert checks.zero_design < 1e-60 && isinf(checks.zero_response) && isinf(checks.negative_dual)
checks
```

The sizes above need the normal range. With `X` columns of `±1e300` and a response of size `1e-100`, the
standardized coefficient `θ` is about `7e-101`, and the coefficient `θ/s` is about `7e-401`, below the range of
`Float64`: it is stored as `0` and the fit has `β = 0`, although ProximalSolvers certifies `θ` for the
transformed problem. The returned point is then not near the minimum in the original coordinates, and the
certificate does not say so.

```@example sqrt_lasso
Xu = reshape([-1e300, 1e300, -1e300, 1e300], 4, 1)
yu = 1e-100 .* [-1.0, 1.0, -0.5, 0.5]
fitu = sqrt_lasso(Xu, yu, 0.1, CDSolver(order = :cyclic, momentum = false))
@assert iszero(fitu.coefficients) && dual_gap(Xu, yu, fitu) > 1e-4
(θ = fitu.result.minimizer, β = fitu.coefficients, converged = fitu.result.converged,
 reported_gap = fitu.result.gap, gap_in_original_coordinates = dual_gap(Xu, yu, fitu))
```

## Where the certificate is defined

ProximalSolvers' certificate runs at the first full pass whose largest coefficient change is below the
threshold `result.tolerance` (`tol‖ỹ‖` with the default `relative = true`), and when it does not pass
the threshold is divided by 100, at most twice. The run stops without a certificate, with
`converged = false` and `gap = NaN`, in these cases (see the `CDSolver` documentation of ProximalSolvers
for the complete rules):

- the recomputed residual of the transformed problem is zero while some `θⱼ ≠ 0`, so that the dual
  direction `r/‖r‖` is undefined: the usual outcome at an exact fit `y = Xβ` with a positive level, where a
  coordinatewise minimum need not be a minimum;
- the sum of squares of that residual is below `n 2⁻⁹⁶⁹` (a root mean square below about `1.4e-146`
  on the scale of the transformed problem), outside the range of the proof;
- a pass changed nothing, or the threshold cannot be lowered below its floor.

A stop at `maxpasses` also gives `converged = false` and `gap = NaN`. The two are told apart by `passes`
and by the warning that ProximalSolvers logs (the text of the first is "stopped without a duality
certificate", of the second "did not converge in maxpasses"). `sqrt_lasso` adds no certificate, stopping
rule or recovery path.

For `ỹ = 0`, which a zero response gives and a constant response with an intercept gives (it centres to zeros),
the minimizer is `θ = 0`, so `β = 0`, `β₀ = ȳ` and `σ = 0`. ProximalSolvers certifies it, with
`converged = true` and `gap = 0`, at its first full pass if the settings let that pass meet the change test
(`maxpasses ≥ 1` and `tol > 0`); with `maxpasses = 0`, or with `tol = 0`, the run ends with `converged = false`
at the same point. With a zero design the minimizer is `β = 0` with `β₀ = ȳ` (`β₀ = 0` without an intercept).
Whether ProximalSolvers certifies it depends on the settings and on the range of the certificate: when the
residual (here `ỹ`) has a root mean square below about `1.4e-146`, the certificate is undefined and the run ends
with `converged = false` (for example a zero design with `y = 2.0^-500 .* [-1, 1, -1, 1]`).

```@example sqrt_lasso
exact = sqrt_lasso(Float64[1 2; 2 0; 3 1; 6 1], 5 .+ [1, 2, 3, 6] ./ 2, 0.5,
                   CDSolver(order = :cyclic, momentum = false); intercept = true, standardize_x = false)
(coefficients = exact.coefficients, intercept = exact.intercept, σ = exact.σ,
 converged = exact.result.converged, gap = exact.result.gap, passes = exact.result.passes)
```

This fit is exact (`y = 5 + x₁/2`), the residuals are zero, and ProximalSolvers stops without a
certificate: with a zero residual the dual direction is undefined. It logs its warning.

### Float32

At the default `tol = 1e-7` the solver does not tighten the threshold of the certificate below the floor
`16 eps(Float32)‖ỹ‖ ≈ 1.9e-6‖ỹ‖` for `Float32` data (the candidate `tol/100` is refused below it), so a
run whose first certificate attempt fails ends without a certificate instead of continuing at a smaller
threshold. The rounding of `Float32` coefficients can also keep the largest change of each full pass
above the threshold, and the run then ends at `maxpasses`. In both cases `converged` is `false`. The
tests of this package set `tol = 1e-6` and `maxpasses = 2000` for `Float32` data, and accept a certified
result, checked independently, or a run that reports itself not converged. Read `fit.result.converged`:
a `Float32` fit that is not converged is not certified, and a converged one is certified for the
transformed problem, not for the original coordinates.

## The noise estimate

`σ = ‖r‖/√n` for the residuals `r = y - β₀ - Xβ` on the original data, computed as the root mean
square of the entries divided by a power of two near the largest, so that it does not overflow, with
`σ = 0` when all residuals are zero. It is the noise estimate of the scaled lasso at the
returned coefficients, a residual-based estimate whose finite-sample bias can have either sign. It tends to
fall below the noise level where the fit adapts to `y`, because the residuals of a fit that follows the data are
smaller than the errors. It can exceed the noise level where the fit leaves signal in the residuals: for example
at `β = 0` (a level above `λmax`) with a signal present, `σ` is the root mean square of `y - β₀` and includes the
signal. No degrees-of-freedom correction is applied. The earlier function returned the sample standard
deviation of the residuals.

## A pivotal level

A penalty level of the square-root lasso can be chosen without the noise level. For Gaussian errors,
Belloni, Chernozhukov and Wang take, for columns scaled to `(1/n)Σᵢ xᵢⱼ² = 1`,

```math
\lambda = c\,\Phi^{-1}\!\left(1 - \frac{\alpha}{2p}\right)\Big/\sqrt{n}
```

in the normalized objective of this page, with `c > 1` (for example `c = 1.1`) and a small `α` (for
example `α = 0.05`); `Φ⁻¹` is the standard normal quantile function. With `standardize_x = true` and
without an intercept the columns are scaled to that mean square, and with an intercept to unit
variance. For `α = 0.05` and `p = 50`, `1 - α/(2p) = 0.9995`, and `Φ⁻¹(0.9995) ≈ 3.2905267`: the
standard normal tail beyond `3.2905267` is about `0.0005`, so the two-sided tail is about `0.001 = α/p`. The value
is hard-coded here, with no dependency on a package for the quantile.

```@example sqrt_lasso
rng = Xoshiro(2)
n, p = 100, 50
X = randn(rng, n, p)
σ_true = 0.5
y = X[:, 1:3] * [2.0, -1.5, 1.0] + σ_true * randn(rng, n)

quantile = 3.2905267                          # Φ⁻¹(1 - 0.05 / (2 * 50)) = Φ⁻¹(0.9995)
λ = 1.1 * quantile / sqrt(n)
fit = sqrt_lasso(X, y, λ, CDSolver(order = :cyclic, momentum = false); intercept = true)
(λ = λ, selected = findall(!iszero, fit.coefficients), σ_hat = fit.σ, σ_true = σ_true,
 converged = fit.result.converged)
```

The level `λ` was chosen without `σ_true`; `σ_hat = fit.σ` is the estimate of the noise level from the
fit.

## Reference

```@docs
sqrt_lasso
SqrtLassoSolution
dof(::SqrtLassoSolution)
deviance(::SqrtLassoSolution)
```
