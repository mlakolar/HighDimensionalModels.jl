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
the fitted values, `‖X̃_G β_G‖/√n`, and not on the norm of its coefficients. The fit then does not
depend on the basis of a group. If the columns `X̃_G` are replaced by `X̃_G M` for an invertible matrix `M`
with as many columns (a rescaling of a column, or a recombination of the columns of the group), then
`β_G ↦ M⁻¹β_G` gives the same fitted values and the same penalty, so the estimate of the fitted values is
unchanged, in exact arithmetic. In floating point the statement holds where the computed rank of the group
(below) is the same before and after the transformation and `M` is well conditioned; the tests assert it
only under those two conditions. The weights are part of the objective: with the default weights, adding a
redundant column to a group changes its size and so its weight `wₖ = √|Gₖ|`, and with it the fit; to compare
a group with and without a column, give the same weights.

The fit is computed on an orthonormal basis of each group. The columns `X̃_G` (`X_G` without an
intercept) are factored by a pivoted QR in the element type `T`, `X̃_G[:, P] = Q R`, and `r` is the
rank (below). The matrix `Z_G = √n Q[:, 1:r]` has `Z_G'Z_G/n = I` up to rounding, so with
`θ_G = R̃β_G[P[1:r]]` and `R̃ = R[1:r, 1:r]/√n` the penalty is `λ wₖ‖θ_G‖`, and the objective is the group lasso
on `Z` with the response `ỹ` (`y` centred with an intercept):
`‖ỹ - Zθ‖²/(2n) + λ Σₖ wₖ‖θ_{Gₖ}‖`. The block steps of the solver are exact on `Z` up to rounding.
The coefficients are mapped back by `β_G[P[1:r]] = R̃ \ θ_G`, with `β = 0` on the columns `P[r+1:end]` of the
group, and `β₀ = ȳ - x̄ᵀβ`. With `standardize = :none` the solver sees the (centred) columns as given and
`β = θ`; scaling or recombining the columns of a group then changes the estimate.

## Groups and weights

`groups` is a vector of vectors (or ranges) of column indices, for example `[[1, 4], [2, 3], [5]]` for
five columns, copied into `fit.groups` as `Vector{Vector{Int}}`. The call throws an `ArgumentError`, with
a message of its own, for an empty group, an entry that is not an integer, an index outside `1:p`, an
index repeated within a group, a column in two groups, and a column in no group (give it a group of its
own; a weight of zero leaves a group unpenalized). `groupweights` is `nothing` or one weight per group
(`DimensionMismatch` for another length, `ArgumentError` for a negative, `NaN` or infinite weight or one
that `T` cannot represent); the weights used are stored in `fit.groupweights`, converted to `T`.
`λ` must be finite, nonnegative and representable in `T`, and is converted to `T`
for the penalty: a `BigFloat`, `Rational` or `Int` level works, and `1e300` with `Float32` data throws.

Let `λmax` be the following value. With the groups of weight zero profiled out (`U` their span,
`r₀ = ỹ - P_U ỹ`; `r₀ = ỹ` if there is none), it is the largest value over the groups of positive weight of
`‖P_{Gₖ} r₀‖/(√n wₖ)` for `:orthonormal`, with `P_G` the projector on the span of the columns `X̃_G`, and of
`‖X̃_{Gₖ}ᵀ r₀‖/(n wₖ)` for `:none`. In exact arithmetic, every group of positive weight is zero in a minimizer
if and only if `λ ≥ λmax`; without a group of weight zero this is `β = 0`, `β₀ = ȳ`.

## Rank, the tolerance and rank deficiency

With `:orthonormal`, the rank `r` of a group of `m` columns is the number of diagonal entries `|Rᵢᵢ|` of the
triangular factor of its pivoted QR above the tolerance

```math
\tau_G = c\,\sqrt{\max(n, m)}\;\texttt{eps}(T)\,|R_{11}|,
```

where `|R₁₁|` is the largest column norm (the first diagonal entry of a pivoted QR) and `c` is the constant
`HighDimensionalModels.RANK_TOLERANCE_C`, a provisional `10`. The tolerance is one function of the
package (`rank_tolerance`). A group of columns that are linear combinations of one another has rank
below `m`: the columns `P[r+1:end]` get the coefficient `0`, and the others carry the
fit of the group. Pivoted QR does not decide which column of a redundant set is dropped, so a dependent column
added to a group need not be the one with the zero coefficient; the fitted values and the contribution
`X̃_G β_G` of the group do not depend on that choice, and the number of nonzero coefficients of the group is at
most its rank.

A group of rank `0` (zero columns, or constant columns with an intercept) stays in the fit as one column
of zeros, which the solver keeps at zero, so the groups of the fit are the groups given, and
its coefficients are `0`. A design in which every group has rank `0` returns the zero fit, with
`β₀ = ȳ` (or `0` without an intercept).

## Example: a categorical predictor coded by dummy columns

A categorical predictor with four levels, coded by one dummy column per level, is one group. With an
intercept the four centred dummy columns sum to zero, so the group has rank 3 and the fit drops one
coefficient.

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
coefficient depends on the pivoting. The fit does not depend on the choice of the reference level, so
leaving one dummy out (three columns) gives the same fitted values when the weight of the group is the same:

```@example group
fit3 = group_lasso([D[:, 2:4] z], y, 0.05, [1:3, [4], [5], [6]], solver; groupweights = [2.0, 1, 1, 1])
maximum(abs, predict(fit, X) - predict(fit3, [D[:, 2:4] z]))
```

## Convergence and optimality

The result of the solver is `fit.result`, a `ProximalSolvers.CDResults`: `converged`, `passes`,
`maxchange`, `tolerance` and `minimum`. `converged = true` means that the largest change of a block in a full
pass, in the solver's scaling, was below `tolerance`; it is not an optimality certificate, and
a run that reaches `maxpasses` returns `converged = false` and ProximalSolvers warns. `fit.result.minimizer` is
a `BlockIterate` in the coordinates of `Z` (with `:none` it holds the values of `fit.coefficients`), and
`fit.result.minimum` is the objective of the solver's problem there, which equals the objective above
at `(β₀, β)` in exact arithmetic.

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
