# HighDimensionalModels.jl

HighDimensionalModels has one estimator, the Gaussian lasso at a single penalty level:

```math
\min_{\beta \in \mathbb{R}^p}\ \frac{1}{2n}\lVert y - X\beta\rVert_2^2 + \lambda \lVert \beta \rVert_1 .
```

It is a thin layer. [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) owns the
mathematics (the loss, the penalty and their numerical contracts) and
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) owns the solver
(coordinate descent and its stopping rule). Both check their arguments. This package forms the
problem, calls the solver and returns the coefficients, the residuals and the solver's result.

```@example index
using HighDimensionalModels, Random

rng = Xoshiro(1)
n, p = 100, 20
X = randn(rng, n, p)
y = X[:, 1:3] * [2.0, -1.5, 1.0] + 0.5 * randn(rng, n)

fit = lasso(X, y, 0.1, CDSolver(order = :cyclic, momentum = false))
println(count(!iszero, coef(fit)), " nonzero coefficients of ", p)
println(round.(coef(fit)[1:5]; digits = 2))
```

The keyword `alpha`, with `0 < alpha ≤ 1`, gives the elastic net. The [lasso page](lasso.md) gives
the inputs, the result, how convergence is reported and how to check a fit independently.

## What is not supported

- Inputs other than a dense `Matrix{Float32}` or `Matrix{Float64}` with a `Vector` of the same
  element type: integer, sparse and view designs, matrix responses and mixed element types are
  refused with an `ArgumentError`; nothing is converted.
- An intercept, standardization and observation weights.
- Penalty paths, screening, continuation and cross-validation.
- Weighted, adaptive, feasible, scaled and square-root lasso; nonconvex penalties;
  generalized linear models.
- Group, sparse-group and multitask regression; quantile regression; SLOPE and knockoffs;
  structured and fused penalties; low-rank and sparse-plus-low-rank models;
  varying-coefficient models; post-selection inference.
- General coordinate-descent interfaces, the compatibility API of earlier versions, and
  extensions for HiGHS, Knockoffs, ProximalAlgorithms and StatsModels.

The complete earlier package, with its documentation, examples and changelog, is preserved on the
branch [`legacy/full`](https://github.com/mlakolar/HighDimensionalModels.jl/tree/legacy/full) and
the tag `archive/pre-minimal-core/master`.

## Reference

```@docs
HighDimensionalModels
```

The docstrings of `lasso` and `LassoSolution` are on the [lasso page](lasso.md).
`CDSolver` is that of ProximalSolvers, re-exported. `coef`, `predict`, `residuals` and `nobs` are
those of [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl), with methods for
`LassoSolution`.
