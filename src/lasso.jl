using ProximalBase: ElasticNet

"""
    LassoSolution{T, R}

The fit returned by [`lasso`](@ref), with fields

- `coefficients::Vector{T}`: the coefficients `β`, one per column of the design;
- `residuals::Vector{T}`: `y - β₀ - X * coefficients` on the data given to `lasso`, computed once
  at the returned point (`β₀` is `0` without an intercept);
- `λ::T`: the penalty level, converted to `T`;
- `alpha::T`: the mixing parameter of the elastic net, converted to `T`, with `one(T)` for a lasso
  fit. For `alpha < 1` the penalty is `ElasticNet(λ * alpha, λ * (1 - alpha))`, with the two levels
  computed in `T` from the stored `λ` and `alpha`;
- `intercept::T`: the intercept `β₀`, which is `0` when `hasintercept` is `false`;
- `hasintercept::Bool`: whether `lasso` was called with `intercept = true`;
- `result::R`: the ProximalSolvers result, unchanged. For `lasso` it is a
  `ProximalSolvers.CDResults`, whose documented field `minimizer` is the same vector as
  `coefficients` unless `standardize_x = true`, where it is the vector `θ` of the solver's own
  coordinates.

For this problem `result.converged` reports ProximalSolvers' coefficient-change test, not an
optimality certificate; see [`lasso`](@ref).

`result` is the result of ProximalSolvers on the problem it was given,
`‖ỹ - Zθ‖²/(2n) + λ(α‖θ‖₁ + (1 - α)‖θ‖²/2)` (the lasso `‖ỹ - Zθ‖²/(2n) + λ‖θ‖₁` when `alpha` is
`1`), where `Z` and `ỹ` are the centred and scaled data of [`lasso`](@ref) (`X` and `y` themselves
with both keywords false). With `standardize_x = true`, `result.minimizer` is `θ`; in exact arithmetic
`θⱼ = sⱼ βⱼ` for the mathematical back-map `βⱼ = θⱼ/sⱼ`, but the stored slopes are the rounded
divisions `θⱼ/sⱼ` in `T` with `0` for a column of scale `0` (see the limit of the back-map in
[`lasso`](@ref)), so `θⱼ = sⱼ βⱼ` need not hold for them. `result.minimum` is
the value of the objective above at `θ`, which equals the objective of `lasso` at the mathematical
`(β₀, β)` (the one with the weights `sⱼ`, not the plain lasso objective of `β`) in exact arithmetic. With
`intercept = true` and `standardize_x = false`, `θ = β`.

`result.maxchange` and `result.tolerance` are the solver's. With `relative = true` (the default
of `CDSolver`) and `ỹ ≠ 0`, `maxchange` is the largest change of a coefficient `|Δθⱼ|‖Zⱼ‖/√n` in
a pass and `tolerance` is `tol‖ỹ‖/√n`, where `Zⱼ` is column `j` of the matrix the solver sees:
`Z` with `standardize_x = true` (`‖Zⱼ‖ = √n` up to rounding for a column that is not zero, so the
changes are in units of `θ`), and `X`, or `X` with centred columns, otherwise (so the changes are
multiplied by `‖xⱼ‖/√n` or `‖x̃ⱼ‖/√n`). With `relative = false`, or `ỹ = 0`, the threshold is the
absolute `tol` and the changes of `θⱼ` are not multiplied.

The positional constructor `LassoSolution(coefficients, residuals, λ, alpha, intercept,
hasintercept, result)` is internal; build a fit with [`lasso`](@ref).

`coef`, `predict`, `residuals`, `nobs`, `dof` and `deviance` from StatsAPI are defined for it.
"""
struct LassoSolution{T, R}
  coefficients::Vector{T}
  residuals::Vector{T}
  λ::T
  alpha::T
  intercept::T
  hasintercept::Bool
  result::R
end

"""
    lasso(X, y, λ, solver = CDSolver(); alpha = 1, intercept = false, standardize_x = false) -> LassoSolution

Minimize `‖y - β₀1 - Xβ‖² / (2n) + λ Σⱼ (α sⱼ|βⱼ| + (1 - α) sⱼ²βⱼ²/2)` over `β₀` and `β`, with
`n = size(X, 1)`, by coordinate descent. `X` must be a `Matrix{T}` and `y` a `Vector{T}`, with `T`
equal to `Float32` or `Float64`, and `λ` a `Real`; `solver` is a `CDSolver`. Other argument types
throw an `ArgumentError`, and nothing is converted. `X` and `y` are not modified, and the solution
keeps no reference to them.

With the defaults `alpha = 1, intercept = false, standardize_x = false`, `β₀ = 0` and `sⱼ = 1` for
all `j`: the objective is `‖y - Xβ‖² / (2n) + λ‖β‖₁`, and the descent starts at `β = 0` on `X` and
`y` as given. The intercept is not penalized. With `θⱼ = sⱼβⱼ` the penalty is
`λ(α‖θ‖₁ + (1 - α)‖θ‖²/2)`, the elastic net on `θ`.

- `alpha` is the weight of the ℓ₁ part: a finite `Real` with `0 < alpha ≤ 1`, and `alpha = 1`
  is the lasso. Any other value, including a value that is not a `Real`, throws an `ArgumentError`
  that names the range. The check is made on `alpha` as given. Then `α = T(alpha)` and
  `λ_T = T(λ)` are formed, and these are the stored `fit.alpha` and `fit.λ`:
  - if `α` is `1` in `T` (for example `alpha = 1 - 1e-9` with `Float32` data), the fit is the lasso,
    solved as with `alpha = 1`, and `fit.alpha` is `1`;
  - if `α` is `0` in `T` (for example `alpha = 1e-50` with `Float32` data), or `λ > 0` and
    `λ_T * α` is `0`, the penalty would be a ridge alone, which this function does not fit, and
    an `ArgumentError` says so. With `α < 1` in `T`, a `λ > 0` with `λ_T = 0` is refused in the same
    way;
  - otherwise the penalty is ProximalBase's `ElasticNet(λ₁, λ₂)` with `λ₁ = λ_T * α` and
    `λ₂ = λ_T * (1 - α)` computed in `T`, so that `fit.λ * fit.alpha` is `λ₁` and
    `fit.λ * (1 - fit.alpha)` is `λ₂`. With `λ = 0` this is least squares.

  No agreement with glmnet is claimed: glmnet scales `y` internally, which changes the ridge part
  unless `sd(y) = 1`.
- `intercept = true` fits `β₀`. The columns of `X` and the response are centred at their means,
  and `β₀ = ȳ - x̄ᵀβ`. A column (or `y`) whose entries are all equal has that value as its mean, so
  that it centres to exact zeros. Any other mean is computed from the entries divided by a power
  of two not above the largest `|xᵢ|`, so that the sum of at most `n` terms below 2 in absolute
  value does not overflow.
- `standardize_x = true` sets `sⱼ` to the root mean square of the column `xⱼ - x̄ⱼ` (the centred
  column with an intercept, the column itself without one), computed from the entries divided by
  a power of two not above the largest `|x̃ᵢⱼ|`, so that the squares in the sum are at most 4. A
  column whose centred entries are all zero (a zero column, or a constant column with an
  intercept) has scale `0`: the solver keeps it at zero and its coefficient is `0`. No tolerance
  decides a zero scale, only exact zeros after centring. For any other column, `sⱼ` is the root
  mean square of its centred entries and, for `α = 1`, the penalty weight on `βⱼ` is `λsⱼ`. A
  constant column without an intercept is an ordinary column.

The problem is solved on the centred and scaled data: `Z` has the columns `(xⱼ - x̄ⱼ)/sⱼ`, `ỹ` is
`y - ȳ` with an intercept and `y` without one, and the solver minimizes
`‖ỹ - Zθ‖²/(2n) + λ(α‖θ‖₁ + (1 - α)‖θ‖²/2)`, the unweighted lasso when `α = 1`; then
`βⱼ = θⱼ/sⱼ` and `β₀ = ȳ - x̄ᵀβ`. With both keywords false `X` and `y`
are not copied and the solver works on them. A keyword that is not `true` or `false`
throws an `ArgumentError`. If a mean, a scale, an entry of `Z`, `β` or `β₀` is not finite (for
example, entries of opposite signs near `floatmax(T)` that overflow when centred), an
`ArgumentError` names `X` or `y` and the step. Standardization lets a design whose squares overflow
(for example `Float32` columns near `1e20`) be fit, which the default call refuses, provided that
the centring, the scales, the solve and the back-map are representable. Centring is computed in `T`,
and an `ArgumentError` is thrown when a centred entry is not finite; for example,
`Float32[floatmax(Float32), floatmax(Float32), -floatmax(Float32)]` as a column with
`intercept = true` throws it, with either value of `standardize_x`. The back-map `θⱼ/sⱼ` has a limit of representation: a coefficient whose exact value is
below the range of `T` underflows (to a subnormal number or `0`), and the residuals on the original
data, which use the stored `β`, then differ from those of the solver's problem. For example, with
`X = reshape([-1e300, 1e300], 2, 1)`, `y = [-1e-100, 1e-100]`, `λ = 0` and `standardize_x = true`,
the solver returns `θ = 1e-100` with minimum `0`, the stored coefficient is `0` (the exact value
`1e-400` underflows) and `sum(abs2, residuals(fit))` is `2e-200`. The result of the solver is not
altered.

`n ≥ 1` is checked here. With a keyword set, so are `length(y) == n` and `p ≥ 1`, which
ProximalSolvers checks on the default path, and the entries of `X` (and of `y` with an intercept)
must be finite. The other checks (finite data on the default path, `λ` finite, nonnegative and
representable in `T`) are those of ProximalBase and ProximalSolvers, and their errors are not
altered. `λ` reaches the penalty as given, in its own type.

The loss is ProximalBase's `LeastSquaresLoss`, scaled by `1/n`; the penalty is `NormL1(λ)`; the
solve is `ProximalSolvers.solve!`.

The convergence report is ProximalSolvers' (`fit.result.converged`, `passes`, `maxchange`,
`tolerance`, `minimum`); with a keyword set it is that of the solver's problem, in the coordinates
`θ` (see [`LassoSolution`](@ref) for the units of `maxchange` and `tolerance`). For this problem `converged` reports the
coefficient-change test of the solver (the largest change of a coefficient in a full pass, in the
solver's scaling of the changes, against `tolerance`), not a bound on the distance to the minimum
or on the KKT conditions. To check a fit, compute `x̃ⱼ'r/n` for the columns `x̃ⱼ` (`xⱼ - x̄ⱼ` with an
intercept, `xⱼ` without) and the residuals `r = residuals(fit)` against `λ sⱼ` (and, on the support, against `λ sⱼ sign(βⱼ)`);
with the defaults this is `maximum(abs, X' * residuals(fit)) / n` against `λ`. For `α < 1` the
conditions are `|x̃ⱼ'r/n| ≤ λα sⱼ` where `βⱼ = 0` and `x̃ⱼ'r/n - λ(1 - α)sⱼ²βⱼ = λα sⱼ sign(βⱼ)`
where `βⱼ ≠ 0`. Rounding limits the
accuracy that can be attained; the accuracy a run achieved depends also on `tol`, `maxpasses` and
the conditioning of the design, and has to be measured.

With `CDSolver()` the coordinate order is random and is drawn from the solver's random number
generator, by default the one of the task, which is borrowed and advances when the ordering draws
from it (a design with one column needs no draw). `CDSolver(order = :cyclic, momentum = false)`
gives a fixed coordinate order. `maxpasses` caps the passes; a run that reaches the cap without
meeting the change test returns `converged == false` and ProximalSolvers warns.
"""
function lasso(X::Matrix{T}, y::Vector{T}, λ::Real, solver::CDSolver = CDSolver();
               alpha = 1, intercept = false, standardize_x = false) where {T<:Union{Float32, Float64}}
  check_flag(:intercept, intercept)
  check_flag(:standardize_x, standardize_x)
  check_alpha(alpha)
  n = size(X, 1)
  n ≥ 1 || throw(DimensionMismatch("X needs at least one row, but it has $n"))
  α = T(alpha)
  isone(α) && return lasso_solve(X, y, λ, nothing, T(λ), α, solver, intercept, standardize_x)
  iszero(α) && throw(ArgumentError(
    "lasso: alpha = $alpha is 0 in $T, which would be a ridge penalty alone; ridge alone is not supported"))
  λₜ = T(λ)
  λ₁, λ₂ = λₜ * α, λₜ * (one(T) - α)
  λ > 0 && (iszero(λₜ) || iszero(λ₁)) && throw(ArgumentError(
    "lasso: with alpha = $alpha, the ℓ₁ level λ * alpha of λ = $λ is 0 in $T, which would be a ridge penalty alone; " *
    "ridge alone is not supported"))
  lasso_solve(X, y, λ₁, λ₂, λₜ, α, solver, intercept, standardize_x)
end

# The penalty of `lasso`: NormL1(λ) with `λ` in its own type (λ₂ = nothing, the lasso), or
# ElasticNet(λ, λ₂)
penalty(λ, ::Nothing) = NormL1(λ)
penalty(λ₁, λ₂) = ElasticNet(λ₁, λ₂)

# The solve of `lasso` for the penalty of (λ, λ₂), the stored `λₜ` and `α`. As in the lasso
# without alpha, the loss is built before the penalty.
function lasso_solve(X::Matrix{T}, y::Vector{T}, λ, λ₂, λₜ::T, α::T, solver::CDSolver,
                     intercept::Bool, standardize_x::Bool) where {T<:Union{Float32, Float64}}
  n = size(X, 1)
  if !intercept && !standardize_x
    f = (one(T) / T(n)) * LeastSquaresLoss(y, X)
    g = penalty(λ, λ₂)
    β = zeros(T, size(X, 2))
    result = solve!(β, f, g, solver)
    return LassoSolution(β, y - X * β, λₜ, α, zero(T), false, result)
  end
  length(y) == n || throw(DimensionMismatch("y has $(length(y)) entries but X has $n rows"))
  size(X, 2) ≥ 1 || throw(DimensionMismatch("X must have at least one column"))
  d = preprocess(X, y, intercept, standardize_x)
  f = (one(T) / T(n)) * LeastSquaresLoss(d.ỹ, d.Z)
  g = penalty(λ, λ₂)
  θ = zeros(T, size(X, 2))
  result = solve!(θ, f, g, solver)
  β = d.s === nothing ? θ : coefficients_from_scaled(θ, d.s)
  β₀ = intercept ? intercept_from_means(d.ȳ, d.x̄, β) : zero(T)
  LassoSolution(β, y .- β₀ .- X * β, λₜ, α, β₀, intercept, result)
end

lasso(X, y, λ, solver = CDSolver(); alpha = 1, intercept = false, standardize_x = false) = throw(ArgumentError(
  "lasso needs X::Matrix{T}, y::Vector{T} with T Float32 or Float64, λ::Real, a CDSolver, and the " *
  "keywords alpha (a real number with 0 < alpha ≤ 1), intercept and standardize_x (each true or false); " *
  "got $(typeof(X)), $(typeof(y)), $(typeof(λ)), $(typeof(solver)), alpha of type $(typeof(alpha)), " *
  "intercept of type $(typeof(intercept)) and standardize_x of type $(typeof(standardize_x))"))

check_flag(name::Symbol, v) = v isa Bool ||
  throw(ArgumentError("lasso: the keyword $name must be true or false, but it is of type $(typeof(v))"))

check_alpha(alpha) = alpha isa Real && isfinite(alpha) && 0 < alpha ≤ 1 ||
  throw(ArgumentError("lasso: the keyword alpha must be a finite real number with 0 < alpha ≤ 1, but it is " *
                      (alpha isa Real ? "$alpha" : "of type $(typeof(alpha))")))

# β = θ ./ s, with β[j] = 0 where s[j] = 0 (a column of zero scale, whose θ[j] is 0); an
# ArgumentError if an entry of β is not finite
function coefficients_from_scaled(θ::Vector{T}, s::Vector{T}) where T
  β = similar(θ)
  for j in eachindex(θ)
    β[j] = iszero(s[j]) ? zero(T) : θ[j] / s[j]
    isfinite(β[j]) || throw(ArgumentError(
      "lasso: the coefficient of column $j of X overflowed when the fit was scaled back by the scale of the column"))
  end
  β
end

# β₀ = ȳ - x̄ᵀβ, an ArgumentError if it is not finite
function intercept_from_means(ȳ::T, x̄::Vector{T}, β::Vector{T}) where T
  a = zero(T)
  for j in eachindex(β)
    a += x̄[j] * β[j]
  end
  β₀ = ȳ - a
  isfinite(β₀) || throw(ArgumentError("lasso: the intercept ȳ - x̄ᵀβ overflowed for the means of y and of the columns of X"))
  β₀
end

coef(fit::LassoSolution) = fit.hasintercept ? [fit.intercept; fit.coefficients] : fit.coefficients

function predict(fit::LassoSolution, X::AbstractMatrix)
  size(X, 2) == length(fit.coefficients) || throw(DimensionMismatch(
    "X has $(size(X, 2)) columns but the fit has $(length(fit.coefficients)) coefficients"))
  fitted = X * fit.coefficients
  fit.hasintercept && (fitted .+= fit.intercept)
  fitted
end

residuals(fit::LassoSolution) = fit.residuals
nobs(fit::LassoSolution) = length(fit.residuals)

"""
    dof(fit::LassoSolution) -> Int

The number of nonzero entries of `fit.coefficients`, plus one when the fit has an intercept. For
`fit.alpha < 1` this is the count that glmnet reports as its degrees of freedom; it is not an
unbiased estimate of the degrees of freedom of the fit.
"""
dof(fit::LassoSolution) = count(!iszero, fit.coefficients) + fit.hasintercept

"""
    deviance(fit::LassoSolution)

The residual sum of squares `sum(abs2, residuals(fit))`, on the data given to [`lasso`](@ref).
"""
deviance(fit::LassoSolution) = sum(abs2, fit.residuals)

function Base.show(io::IO, fit::LassoSolution)
  r = fit.result
  print(io, "LassoSolution: ", length(fit.coefficients), " coefficients, ",
        count(!iszero, fit.coefficients), " nonzero, λ = ", fit.λ,
        fit.alpha < 1 ? ", α = " * string(fit.alpha) : "",
        fit.hasintercept ? ", intercept = " * string(fit.intercept) : "",
        ", ", nobs(fit), " observations;\n",
        "  coefficient-change test of the solver (not an optimality certificate): ",
        r.converged ? "met" : "not met", " after ", r.passes, " passes")
end
