# The group lasso

## Objective

For disjoint, nonempty groups `G₁, …, G_K` of the columns of `X` that together contain all the columns,
weights `w₁, …, w_K ≥ 0` and `n = size(X, 1)`, `group_lasso(X, y, λ, groups)` minimizes, over `β₀`
and `β`, the objective of its keyword `standardize`:

```math
\frac{1}{2n}\lVert y - \beta_0\mathbf{1} - X\beta\rVert_2^2
+ \lambda \sum_{k=1}^{K} w_k \frac{\lVert \tilde X_{G_k}\beta_{G_k}\rVert_2}{\sqrt n}
\qquad (\texttt{standardize = :orthonormal}),
```

```math
\frac{1}{2n}\lVert y - \beta_0\mathbf{1} - X\beta\rVert_2^2
+ \lambda \sum_{k=1}^{K} w_k \lVert \beta_{G_k}\rVert_2
\qquad (\texttt{standardize = :none}).
```

Here `X̃` is `X` with each column centred at its mean when `intercept = true` (the default), and
`X̃ = X` otherwise; `X̃_G` is its columns in the group `G`. The intercept `β₀` is not penalized: it is
`0` without `intercept = true`, and `β₀ = ȳ - x̄ᵀβ` with it. With the intercept, a column (or `y`)
whose entries are all equal has that value as its mean, so that it centres to exact zeros, as in
[`lasso`](@ref). The weights default to `wₖ = √|Gₖ|`, with the number of columns of the group as
given.

The factor `1/(2n)` is ProximalBase's `LeastSquaresLoss` scaled by `1/n`, the penalty is
ProximalBase's `GroupNormL2` on a `BlockIterate`, and the solver is ProximalSolvers' block
coordinate descent (`CDSolver`), unchanged. The groups need not be contiguous or sorted.

## The orthonormal interpretation

With `standardize = :orthonormal` (the default) the penalty is on the contribution of each group to
the fitted values, `‖X̃_G β_G‖/√n`, and not on the norm of its coefficients, so it depends on a group through
its column span. If the columns `X̃_G` are replaced by `X̃_G M` for an invertible matrix `M` with as many
columns (a rescaling of a column, or a recombination of the columns of the group), the span is the same, and
`β_G ↦ M⁻¹β_G` gives the same fitted values and the same penalty: in exact arithmetic the minimum and the
fitted values do not change.

In floating point the fit uses the span of the `r` columns that the rank rule (below) retains, and the fitted
values are the same, up to rounding and the stopping tolerance of the solver, when the retained spans of `X̃_G` and `X̃_G M` are the same subspace. That
holds when the rule retains all the columns of the group before and after the transformation and `M` is well
conditioned, and when the directions it drops are at rounding level in both. Equal ranks below the number of
columns do not suffice: when a group has a direction near the tolerance, the transformation can change which
directions are retained, and the fits differ. For example, three centred orthogonal columns with norms `1`,
`1.2τ` and `0.8τ` (`τ` the tolerance below, for `|R₁₁| = 1`) retain the second column, and after rescaling the
columns by `1`, `0.5` and `2` (condition number 4) they retain the third; both have rank 2, and at `λ = 0` the
fitted values for `y` equal to the second column are `y` and `0`. The tests assert invariance after checking that
the computed ranks agree, that `M` is well conditioned and that the projectors on the retained spans agree.
The weights are part of the objective: with the default weights, adding a
redundant column to a group changes its size and so its weight `wₖ = √|Gₖ|`, and with it the fit; to compare
a group with and without a column, give the same weights.

The fit is computed on an orthonormal basis of each group. The columns `X̃_G` (`X_G` without an
intercept) are factored by a pivoted QR in the element type `T`, `X̃_G[:, P] = Q R`, and `r` is the
rank (below). The matrix `Z_G = √n Q[:, 1:r]` has `Z_G'Z_G/n = I` up to rounding, so with
`θ_G = R̃β_G[P[1:r]]` and `R̃ = R[1:r, 1:r]/√n` the penalty is `λ wₖ‖θ_G‖`, and the objective is the group lasso
on `Z` with the response `ỹ` (`y` centred with an intercept):
`‖ỹ - Zθ‖²/(2n) + λ Σₖ wₖ‖θ_{Gₖ}‖`. The columns of a block of `Z` are orthonormal, so the block problem has the
same curvature in all directions. ProximalSolvers' block step uses a bound on the curvature of the block, and
the bound exceeds the curvature by a margin that depends on `n` and `eps(T)`, so a pass moves a block toward the
minimizer of its problem and need not solve it, and the solver iterates until its change test is met or `maxpasses`
is exhausted. For a
one-column group of ones with `n = 10⁶` in `Float32`, `λ = 0.25` and no intercept, whose minimizer is `0.75`:

```@example group
using HighDimensionalModels
n1 = 10^6
x1, y1 = ones(Float32, n1, 1), ones(Float32, n1)
one_pass = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
    group_lasso(x1, y1, 0.25f0, [1:1], CDSolver(order = :cyclic, tol = 1e-6, maxpasses = 1); intercept = false)
end
converged = group_lasso(x1, y1, 0.25f0, [1:1], CDSolver(order = :cyclic, tol = 1e-6); intercept = false)
(one_pass = one_pass.coefficients[1], converged = converged.coefficients[1], passes = converged.result.passes)
```

The coefficients are mapped back by `β_G[P[1:r]] = R̃ \ θ_G`, with `β = 0` on the columns `P[r+1:end]` of the
group, and `β₀ = ȳ - x̄ᵀβ`. With `standardize = :none` the solver sees the (centred) columns as given and
`β = θ`; scaling the columns of a group, or recombining them by a matrix that is not orthogonal, can then change the
estimate.

## Groups and weights

`groups` is a vector of vectors (or ranges) of column indices, for example `[[1, 4], [2, 3], [5]]` for
five columns, copied into `fit.groups` as `Vector{Vector{Int}}`. The call throws an `ArgumentError`, with
a message of its own, for an empty group, an entry that is not an integer, an index outside `1:p`, an
index repeated within a group, a column in two groups, and a column in no group (give it a group of its
own; a weight of zero leaves a group unpenalized). `groupweights` is `nothing` or one weight per group
(`DimensionMismatch` for another length, `ArgumentError` for a negative, `NaN` or infinite weight or one
that is not representable in `T`); the weights used are stored in `fit.groupweights`, converted to `T`.
`λ` must be finite, nonnegative and representable in `T`, and is converted to `T`
for the penalty: a `BigFloat`, `Rational` or `Int` level works, and `1e300` with `Float32` data throws.

Let `λmax` be the following value. With the groups of weight zero profiled out (`U` their span,
`r₀ = ỹ - P_U ỹ`; `r₀ = ỹ` if there is none), it is the largest value over the groups of positive weight of
`‖P_{Gₖ} r₀‖/(√n wₖ)` for `:orthonormal`, with `P_G` the projector on the span of the columns `X̃_G`, and of
`‖X̃_{Gₖ}ᵀ r₀‖/(n wₖ)` for `:none`. In exact arithmetic the groups of positive weight are all zero in a minimizer
when `λ ≥ λmax`, and some group of positive weight is nonzero in each minimizer when `λ < λmax`; without a group of
weight zero the first case is `β = 0`, `β₀ = ȳ`.

## Rank, the tolerance and rank deficiency

With `:orthonormal`, the rank `r` of a group of `m` columns is the number of diagonal entries `|Rᵢᵢ|` of the
triangular factor of its pivoted QR above the tolerance

```math
\tau_G = c\,\sqrt{\max(n, m)}\;\texttt{eps}(T)\,|R_{11}|,
```

where `|R₁₁|` is the largest column norm (the first diagonal entry of a pivoted QR) and `c` is the constant
`HighDimensionalModels.RANK_TOLERANCE_C`, a provisional `10`. The tolerance is one function of the
package (`rank_tolerance`). A group in which a column is a combination of the others to the rounding of the data
has computed rank below `m`: the columns `P[r+1:end]` get the coefficient `0`, and the others carry the
fit of the group. Pivoted QR does not decide which column of a redundant set is dropped, so a dependent column
added to a group need not be the one with the zero coefficient; when the dependence holds to rounding, the fitted
values and the contribution `X̃_G β_G` of the group do not depend on that choice up to rounding, and the number of
nonzero coefficients of the group is at most its rank.

A group of rank `0` (zero columns, or constant columns with an intercept) stays in the fit as one column
of zeros, which the solver keeps at zero, so the groups of the fit are the groups given, and
its coefficients are `0`. A design in which all the groups have rank `0` returns the zero fit, with
`β₀ = ȳ` (or `0` without an intercept).

## Example: a categorical predictor coded by dummy columns

A categorical predictor with four levels, coded by one dummy column per level, is one group. With an
intercept the four centred dummy columns sum to zero up to rounding, so the group has computed rank 3 and the fit
drops one coefficient.

```@example group
using HighDimensionalModels, Random

rng = Xoshiro(1)
n = 200
level = rand(rng, 1:4, n)
D = Float64[level[i] == l for i in 1:n, l in 1:4]     # one dummy column per level
z = randn(rng, n, 3)                                  # three numeric predictors
X = [D z]
y = [0.0, 1.0, 1.0, -1.0][level] .+ 0.8 .* z[:, 1] .+ 0.5 .* randn(rng, n)
groups = [1:4, [5], [6], [7]]

solver = CDSolver(order = :cyclic)
fit = group_lasso(X, y, 0.05, groups, solver)
fit
```

```@example group
(groupweights = fit.groupweights, activegroups = fit.activegroups,
 dummies = fit.coefficients[1:4], zeros_in_dummies = count(iszero, fit.coefficients[1:4]))
```

The default weight of the dummy group is `√4 = 2`, from its four columns. Which dummy gets the zero
coefficient depends on the pivoting. The fitted values do not depend on the choice of the reference level, up to rounding and
the stopping tolerance, so leaving one dummy out (three columns) gives the same fitted values when the weight of the
group is the same:

```@example group
fit3 = group_lasso([D[:, 2:4] z], y, 0.05, [1:3, [4], [5], [6]], solver; groupweights = [2.0, 1, 1, 1])
maximum(abs, predict(fit, X) - predict(fit3, [D[:, 2:4] z]))
```

## Convergence and optimality

The result of the solver is `fit.result`, a `ProximalSolvers.CDResults`: `converged`, `passes`,
`maxchange`, `tolerance` and `minimum`. `converged = true` means that the largest change of a block in a full
pass, in the solver's scaling, was below `tolerance`; it is not an optimality certificate, and
a run that exhausts `maxpasses` without meeting that test returns `converged = false` and ProximalSolvers warns.
`fit.result.minimizer` is a `BlockIterate` in the coordinates of `Z` (with `:none` it holds the values of
`fit.coefficients`), and `fit.result.minimum` is the objective of the solver's problem there. In exact arithmetic it
equals the objective above at the mathematical map back from the minimizer to the columns of `X` and the
intercept `ȳ - x̄ᵀβ` of that map.

**Limit of the map back.** The stored `fit.coefficients` and `fit.intercept` are that map computed in `T`, which
rounds it. A coefficient whose exact value is below the range of `T` underflows to a subnormal number or to `0`, and
the residuals, which use the stored values, then differ from those of the solver's problem, as for
[`lasso`](@ref); the result of the solver is left as it returned it. For `X = reshape([-1e300, 1e300], 2, 1)`,
`y = [-1e-100, 1e-100]`, `λ = 0` and `intercept = false`, the solver returns `θ = 1e-100` with minimum `0`, the stored
coefficient is `0` (the exact value `1e-400` underflows) and `sum(abs2, residuals(fit))` is `2e-200`.

To check a fit independently, compute from the returned `β₀`, `β` and the residuals `r = y - β₀ - Xβ`
(`fit.residuals`) the gradient `g = X̃_Gᵀr/n` of each group, with `X̃` centred if `fit.hasintercept`:

- `:orthonormal`, group with `β_G ≠ 0`: `g = λ w X̃_GᵀX̃_Gβ_G / (√n‖X̃_Gβ_G‖)`; group with `β_G = 0`: `‖P_G r‖ ≤ √n λ w`.
- `:none`, group with `β_G ≠ 0`: `g = λ w β_G/‖β_G‖`; group with `β_G = 0`: `‖g‖ ≤ λ w`.

The tests check these equations in 256-bit arithmetic. Rounding, the stopping tolerance, and, for
`:orthonormal`, the condition number of the group in the map back from `θ` to `β`, limit how closely they hold.
`dof(fit)` counts the nonzero coefficients plus one with an intercept; it is a count and not an unbiased
estimate of the degrees of freedom.

## Reference

```@docs
group_lasso
GroupLassoSolution
dof(::GroupLassoSolution)
deviance(::GroupLassoSolution)
```
