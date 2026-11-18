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

## Inputs

- `X::Matrix{T}` and `y::Vector{T}` with `T` equal to `Float32` or `Float64`; the fit has the
  same `T`. Other array types and mixed element types throw an `ArgumentError` that names the
  supported types.
- `λ::Real`, finite, nonnegative and representable in `T` (not overflowing to `Inf` and not
  underflowing to `0` from a nonzero value). It reaches the solver in its own type, so an `Int`,
  a `Rational` or a `BigFloat` level that `T` represents is accepted, and the stored `fit.λ` is
  `T(λ)`.
- `size(X, 1) ≥ 1` is checked by this package. The remaining checks (a response of length `n`,
  `p ≥ 1`, finite entries of `X` and `y`, the level `λ`) are those of ProximalBase and
  ProximalSolvers; their errors are passed on unchanged.
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

[`LassoSolution`](@ref) has the fields `coefficients`, `residuals` (`y - X * coefficients`,
computed once), `λ` and `result`, the ProximalSolvers result unchanged. `coef(fit)`,
`predict(fit, Xnew)` (`Xnew * coef(fit)`), `residuals(fit)` and `nobs(fit)` are the StatsAPI
methods.

## Convergence and optimality

The result of the solver is `fit.result`, a `ProximalSolvers.CDResults`, with `converged`,
`passes`, `maxchange`, `tolerance` and `minimum`. For this problem `converged = true` means that
the largest change of a coefficient in a full pass, in the solver's scaling of the changes, was
below `tolerance` (the `CDSolver` documentation of ProximalSolvers defines both); it is not an
optimality certificate, and this package adds no convergence flag of its own. A run that reaches
`maxpasses` without that test being met returns `converged = false` and ProximalSolvers warns;
a run that meets the test within the cap, even after one pass, returns `converged = true`.

To check a fit without the solver's stopping rule, compare the correlations of the residuals
with the penalty level: at a minimizer, `|Xⱼ'r|/n ≤ λ` for every column and `Xⱼ'r/n = λ sign(βⱼ)`
where `βⱼ ≠ 0`, with `r = y - Xβ`. Rounding limits the accuracy that can be attained, but what a
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
```
