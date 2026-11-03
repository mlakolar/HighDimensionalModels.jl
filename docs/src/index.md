# HighDimensionalModels.jl

Lasso estimators computed by coordinate descent. For a response
``y \in \mathbb{R}^n`` and ``n`` observations of ``p`` predictors in the rows of
``X``, the lasso is

```math
\underset{\beta_0,\,\beta}{\text{minimize}} \quad \frac{1}{2n}\,\|y - \beta_0 - X\beta\|^2 + \lambda \sum_{j=1}^p \omega_j |\beta_j|,
```

with an unpenalized intercept ``\beta_0`` and penalty loadings ``\omega_j \ge 0``.
HighDimensionalModels solves it at one penalty level ``\lambda`` and along a path of
levels, and solves the estimators built on it:

- the lasso, with an intercept, standardized columns, penalty loadings,
  observation weights and offsets;
- the square-root and the scaled lasso, whose ``\lambda`` does not depend on the
  noise level, and the heteroscedasticity-robust (feasible) lasso;
- lasso paths, with screening, refitting and the functions of StatsAPI;
- the group lasso;
- varying-coefficient models by local polynomial regression, with the kernels
  of KernelSmoothing.jl, and sparse ones with a group lasso penalty.

The coordinate descent itself (the passes over the coordinates and the active
set, the penalty paths with screening, the reweighting of the scaled and
feasible lasso) is that of
[ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl), on the
losses and penalties of
[ProximalBase.jl](https://mlakolar.github.io/ProximalBase.jl/dev/).
[`coordinate_descent!`](@ref) runs it on other losses and on any penalty that
separates across coordinates, including your own.

## Installation

Before the first registration, use a checkout with registered dependencies:

```sh
git clone --branch master https://github.com/mlakolar/HighDimensionalModels.jl.git
cd HighDimensionalModels.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=.
```

HighDimensionalModels requires Julia 1.10 or later.

## A first example

The lasso with an intercept at ``\lambda = 0.1``, on 200 observations of 50
predictors of which the first five matter:

```@example index
using HighDimensionalModels, Random

rng = Xoshiro(1)
n, p = 200, 50
X = randn(rng, n, p)
y = 1.0 .+ X[:, 1:5] * ones(5) + randn(rng, n)

sol = lasso(X, y, 0.1; intercept = true)
findall(!iszero, sol.x)
```

The coefficients, intercept first, and the fitted values of new observations
come from the functions of StatsAPI:

```@example index
coef(sol)
```

```@example index
predict(sol, randn(rng, 3, p))
```

The lasso along 50 values of ``\lambda``, from the smallest one whose solution is
zero:

```@example index
path = LassoPath(X, y; nλ = 50, intercept = true)
[count(!iszero, β) for β in path.βpath[1:10:end]]
```

## The guide

- [The lasso and its tuning-free variants](lasso.md): the mathematics of
  [`lasso`](@ref), [`sqrt_lasso`](@ref), [`scaled_lasso`](@ref) and
  [`feasible_lasso!`](@ref), their conventions, and observation weights.
- [Paths, screening and solver options](paths.md): [`LassoPath`](@ref), the
  options of coordinate descent and when each pays off, and the Gram form.
- [The group lasso](group.md).
- [Variable selection with error control](selection.md): SLOPE, knockoff+, and complementary-pairs stability selection.
- [Writing a loss or a penalty](extending.md) of your own.
- [Varying-coefficient models](varying.md).
- The examples, which go through the package and its options with their output.

## The family

HighDimensionalModels is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|---|---|
| **HighDimensionalModels.jl** (this package) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Development tests

From the repository root, run `julia scripts/test_registered.jl` for the full
suite with registered dependencies on Julia 1.10 and newer. In a separate
checkout, `julia scripts/test_master.jl` checks integration with dependency
master checkouts. The repository's
`CONTRIBUTING.md` describes the naming convention and bootstrap scripts.

<!-- CI scenario: documentation only -->
