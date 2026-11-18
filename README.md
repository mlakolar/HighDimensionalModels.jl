# HighDimensionalModels.jl

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/HighDimensionalModels.jl/dev/)
[![CI](https://github.com/mlakolar/HighDimensionalModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/HighDimensionalModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

The package has one estimator, the Gaussian lasso at a single penalty level:

```
minimize over β:   ‖y - Xβ‖² / (2n) + λ ‖β‖₁ ,      n = size(X, 1).
```

It is a thin layer. [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) owns the
mathematics (the loss and the penalty) and
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) owns the solver
(coordinate descent and its stopping rule). Both check their arguments. This package forms
the problem, calls the solver and returns the coefficients, the residuals and the solver's result.
It contains no optimization code.

This is version `0.2.0-DEV`, a breaking reset of the earlier package (see [NEWS.md](NEWS.md)).

## Example

```julia
using HighDimensionalModels, Random

rng = Xoshiro(1)
n, p = 100, 20
X = randn(rng, n, p)
y = X[:, 1:3] * [2.0, -1.5, 1.0] + 0.5 * randn(rng, n)

fit = lasso(X, y, 0.1, CDSolver(order = :cyclic, momentum = false))
```

```
LassoSolution: 20 coefficients, 5 nonzero, λ = 0.1, 100 observations;
  coefficient-change test of the solver (not an optimality certificate): met after 10 passes
```

```julia
round.(coef(fit)[1:5]; digits = 2)    # [1.9, -1.35, 0.94, 0.0, 0.0]
predict(fit, X)                       # X * coef(fit)
residuals(fit), nobs(fit)             # y - X * coef(fit), and n
fit.result                            # the ProximalSolvers result (CDResults)
```

`CDSolver()` without arguments visits the coordinates in random order, drawn from the default
random number generator of the task, which advances when the ordering draws from it; two such
calls need not return identical coefficients. `CDSolver(order = :cyclic, momentum = false)`
visits them in a fixed order.
(The printed values are from Julia 1.10; the last digits can differ between Julia versions.)

`fit.result.converged` is the solver's test of the change of the coefficients in a pass. For this
problem it is not an optimality certificate. To check a fit, compare `X' * residuals(fit) / n`
with `λ`: at a minimizer its entries are at most `λ` in absolute value, and the entries with
`βⱼ ≠ 0` equal `λ * sign(βⱼ)`. How close a run gets depends on the tolerance, the pass limit, the
conditioning and the rounding of the working type, so the deviations have to be measured. The
[documentation](https://mlakolar.github.io/HighDimensionalModels.jl/dev/lasso/) has the details.

## Supported inputs

`lasso(X, y, λ, solver = CDSolver())` takes `X::Matrix{T}` and `y::Vector{T}` with `T` equal to
`Float32` or `Float64`, a real `λ`, and an optional `CDSolver` from ProximalSolvers (re-exported).
Other argument types throw an `ArgumentError`, and `X` and `y` are not converted. Invalid inputs
throw an `ArgumentError` or a `DimensionMismatch`; the checks are described in the docstring of
`lasso`. The keyword `alpha`, with `0 < alpha ≤ 1`, gives the elastic net (see the lasso page of
the documentation).

Not supported: an intercept, standardization, observation weights; penalty paths, screening and
cross-validation; weighted, adaptive, feasible, scaled and square-root lasso; nonconvex penalties;
generalized linear models; group, sparse-group and multitask regression;
quantile regression; SLOPE and knockoffs; structured, fused, low-rank and varying-coefficient
models; post-selection inference; integer, sparse and view designs.

## The earlier package

The complete earlier package (all of the above, with its tests, benchmarks, documentation and
examples) is preserved on the branch
[`legacy/full`](https://github.com/mlakolar/HighDimensionalModels.jl/tree/legacy/full) and the tag
`archive/pre-minimal-core/master`.

## Installation for development

The package needs the development `master` branches of ProximalBase (0.6 series) and
ProximalSolvers, and does not resolve against the registry. `julia scripts/test.jl` prepares
checkouts of the two packages, builds a test environment that develops them together with this
package, and runs the tests. The package requires Julia 1.10.8 or later.
