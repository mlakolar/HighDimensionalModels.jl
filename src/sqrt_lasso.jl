"""
    SqrtLassoSolution{T, R}

The fit returned by [`sqrt_lasso`](@ref), with fields

- `coefficients::Vector{T}`: the coefficients `β`, one per column of the design;
- `intercept::T`: the intercept `β₀`, which is `0` when `hasintercept` is `false`;
- `hasintercept::Bool`: whether `sqrt_lasso` was called with `intercept = true`;
- `residuals::Vector{T}`: `y - β₀ - X * coefficients` on the data given to `sqrt_lasso`, computed once
  at the returned point;
- `λ::T`: the penalty level of [`sqrt_lasso`](@ref) (the normalized one), converted to `T`;
- `σ::T`: the noise estimate `‖r‖/√n` for the residuals `r` above, the root mean square of the
  residuals (`0` when every residual is zero), computed from the entries divided by a power of two
  not above the largest `|rᵢ|`, so that the squares in the sum are at most 4;
- `result::R`: the ProximalSolvers result, unchanged. For `sqrt_lasso` it is a
  `ProximalSolvers.CDResults`.

Two statements are kept apart, because they have different strengths.

**Certification of the transformed problem.** `result.converged` and `result.gap` are those of
ProximalSolvers and are not recomputed or changed here. `result.converged` is `true` only when
ProximalSolvers' duality certificate proved that the relative gap of the problem it solved is at most
`1e-4`, and `result.gap` is that bound (rounded up in `T`). The problem it solved is
`‖ỹ - Zθ‖ + √n λ ‖θ‖₁` for the stored matrix `Z` and vector `ỹ` in `T` (the centred and scaled data of
[`sqrt_lasso`](@ref), or `X` and `y` themselves with both keywords false), at the point `θ`. In exact
arithmetic its objective is `√n` times the objective of `sqrt_lasso`. With `result.converged == false`
the point is not certified, whether the run ended at `maxpasses` or at a point where the certificate is
undefined; `result.passes` and the warning that ProximalSolvers logs tell the two apart.

**Accuracy in the original coordinates.** This is not certified. The returned `β`, `β₀` and
`residuals` are formed from `θ` by the back-transformation, and they differ from the transformed
problem by the rounding of the preprocessing (an absolute error of about `eps(T)(|xᵢⱼ| + |x̄ⱼ|)/sⱼ` in an
entry of `Z`, and a relative error of up to about `n eps(T)` in a scale `sⱼ`), of `βⱼ = θⱼ/sⱼ` (a
relative error of `eps(T)/2`), of `β₀` (an error of about `eps(T)(|ȳ| + Σⱼ|x̄ⱼβⱼ|)`, which grows with
the number of columns) and of the residuals. The page on the square-root lasso gives a duality gap
computed in the original coordinates, which a user can run on a fit.

`result` is documented on the scale of ProximalSolvers: `result.minimizer` is `θ` (the same vector as
`coefficients` unless `standardize_x = true`), `result.minimum` is `√n` times the objective of
[`sqrt_lasso`](@ref) in exact arithmetic, `result.tolerance` is the change threshold, in units of
`tol‖ỹ‖` with the default `relative = true` and `ỹ ≠ 0`, at the end of the run (it is divided by 100
when the certificate fails and the run continues), and `result.gap` is a relative gap, which has no
scale.

The positional constructor `SqrtLassoSolution(coefficients, intercept, hasintercept, residuals, λ, σ,
result)` is internal; build a fit with [`sqrt_lasso`](@ref).

`coef`, `predict`, `residuals`, `nobs`, `dof` and `deviance` from StatsAPI are defined for it.
"""
struct SqrtLassoSolution{T, R}
  coefficients::Vector{T}
  intercept::T
  hasintercept::Bool
  residuals::Vector{T}
  λ::T
  σ::T
  result::R
end

"""
    sqrt_lasso(X, y, λ, solver = CDSolver(); intercept = false, standardize_x = true) -> SqrtLassoSolution

**Breaking.** The earlier `sqrt_lasso` minimized `‖y - β₀ - Xβ‖ + λ Σⱼ ωⱼ|βⱼ|`. This one minimizes
`‖y - β₀1 - Xβ‖/√n + λ‖θ‖₁`, so the same estimate needs `λ = λ_earlier/√n`: an earlier
`sqrt_lasso(X, y, 20.0)` on `n = 400` rows is now `sqrt_lasso(X, y, 1.0)`. The noise estimate `σ` of
the result is now `‖r‖/√n` for the residuals `r`, not their sample standard deviation. The defaults
`standardize_x = true` and `intercept = false` are those of the earlier function.

Minimize `‖y - β₀1 - Xβ‖/√n + λ Σⱼ sⱼ|βⱼ|` over `β₀` and `β`, with `n = size(X, 1)` and `‖⋅‖` the
Euclidean norm, by coordinate descent. `X` must be a `Matrix{T}` and `y` a `Vector{T}`, with `T`
equal to `Float32` or `Float64`, and `λ` a `Real`; `solver` is a `CDSolver`. Other argument types
throw an `ArgumentError`, and nothing is converted. `X` and `y` are not modified, and the solution
keeps no reference to them. `β₀` is `0` and unpenalized unless `intercept = true`. The weights `sⱼ`
are those of [`lasso`](@ref): `sⱼ = 1` with `standardize_x = false`, and with `standardize_x = true`
the root mean square of the column `xⱼ - x̄ⱼ` (the centred column with an intercept, the column itself
without one), so that `Σⱼ sⱼ|βⱼ| = ‖θ‖₁` for the standardized coefficients `θⱼ = sⱼβⱼ`. Without an
intercept the scale is the uncentred root mean square. A column whose centred entries are all zero (a
zero column, or a constant column with an intercept) has scale `0`, and its coefficient is `0`.

`λ` must be positive and finite: ProximalSolvers' square-root lasso needs a positive level, and a
`λ` that is zero, negative, `NaN` or infinite throws an `ArgumentError` (for `λ = 0`, `lasso` fits the
least squares problem). Because `ProximalBase.SqrtLeastSquaresLoss` is the unnormalized `‖y - Xβ‖` and
ProximalSolvers refuses a scaled square-root loss, the solve is
`solve!(θ, SqrtLeastSquaresLoss(ỹ, Z), NormL1(λps), solver)` at the level
`λps = sqrt(Float64(n)) * Float64(λ)`, a `Float64` for either element type, on the data of
[`lasso`](@ref): `Z` has the columns `(xⱼ - x̄ⱼ)/sⱼ` (`xⱼ - x̄ⱼ` without standardization) and `ỹ` is `y - ȳ`
with an intercept (`X` and `y` themselves, not copied, with both keywords false). Then `βⱼ = θⱼ/sⱼ`
and `β₀ = ȳ - x̄ᵀβ`, with the helpers and the error messages of `lasso` for a design or response that
does not preprocess (a message that starts with `lasso:` and names the column). The level
`λps` must be representable in `T` without overflow or underflow, a limit on `√n λ`, not on `λ`:
otherwise ProximalSolvers' error is passed on unchanged. A constant `y` with an intercept centres to
`ỹ = 0`, which ProximalSolvers certifies at `θ = 0`: `β = 0`, `β₀ = ȳ`, `σ = 0`.

`n ≥ 1`, `length(y) == n` and `size(X, 2) ≥ 1` are checked here (a `DimensionMismatch`); the other
checks (finite data, the level) are those of ProximalBase and ProximalSolvers or of the preprocessing
of `lasso`. A keyword that is not `true` or `false` throws an `ArgumentError`.

The convergence report is ProximalSolvers' (`fit.result.converged`, `gap`, `passes`, `tolerance`,
`minimum`), for the problem in the coordinates `θ`; see [`SqrtLassoSolution`](@ref) for what it
certifies and for what it does not. An uncertified stop (for example at an exact fit, where the
residual is zero) and a stop at `maxpasses` both give `converged == false` and `gap == NaN`, with a
warning each; `passes` and the text of the warning tell them apart. This package adds no
certificate, stopping rule or recovery to the solver's. With `CDSolver()` the coordinate order is
random; `CDSolver(order = :cyclic, momentum = false)` fixes it.

`σ = ‖r‖/√n` is the root mean square of the residuals on the original data. It is the noise estimate of
the scaled lasso at the returned coefficients, and it is biased downward in finite samples because the
fit adapts to `y`.
"""
function sqrt_lasso(X::Matrix{T}, y::Vector{T}, λ::Real, solver::CDSolver = CDSolver();
                    intercept = false, standardize_x = true) where {T<:Union{Float32, Float64}}
  check_flag(:intercept, intercept)
  check_flag(:standardize_x, standardize_x)
  n, p = size(X)
  n ≥ 1 || throw(DimensionMismatch("X needs at least one row, but it has $n"))
  length(y) == n || throw(DimensionMismatch("y has $(length(y)) entries but X has $n rows"))
  p ≥ 1 || throw(DimensionMismatch("X must have at least one column"))
  λps = sqrt_lasso_level(λ, n)
  θ = zeros(T, p)
  if !intercept && !standardize_x
    result = solve!(θ, SqrtLeastSquaresLoss(y, X), NormL1(λps), solver)
    β, β₀ = θ, zero(T)
  else
    d = preprocess(X, y, intercept, standardize_x)
    result = solve!(θ, SqrtLeastSquaresLoss(d.ỹ, d.Z), NormL1(λps), solver)
    β = d.s === nothing ? θ : coefficients_from_scaled(θ, d.s)
    β₀ = intercept ? intercept_from_means(d.ȳ, d.x̄, β) : zero(T)
  end
  r = X * β
  r .= y .- β₀ .- r
  SqrtLassoSolution(β, β₀, intercept, r, T(λ), noise_scale(r), result)
end

sqrt_lasso(X, y, λ, solver = CDSolver(); intercept = false, standardize_x = true) = throw(ArgumentError(
  "sqrt_lasso needs X::Matrix{T}, y::Vector{T} with T Float32 or Float64, λ::Real, a CDSolver, and the " *
  "keywords intercept and standardize_x, each true or false; " *
  "got $(typeof(X)), $(typeof(y)), $(typeof(λ)), $(typeof(solver)), " *
  "intercept of type $(typeof(intercept)) and standardize_x of type $(typeof(standardize_x))"))

# The level of the solver, √n λ in Float64, for a λ that is positive and finite. An ArgumentError
# otherwise, also when the Float64 level is not finite and positive (a λ of another Real type outside
# the Float64 range). Whether the level is representable in T is checked by ProximalSolvers.
function sqrt_lasso_level(λ::Real, n::Integer)
  (isfinite(λ) && λ > 0) || throw(ArgumentError(
    "sqrt_lasso: λ must be positive and finite, but it is $λ; the square-root lasso of ProximalSolvers " *
    "needs a positive level (for λ = 0 use `lasso`, the least squares problem)"))
  λps = sqrt(Float64(n)) * Float64(λ)
  (isfinite(λps) && λps > 0) || throw(ArgumentError(
    "sqrt_lasso: the level √n λ for n = $n and λ = $λ is not a finite positive Float64"))
  λps
end

# σ = ‖r‖/√n, the root mean square of the residuals r, which must be finite; 0 when every entry of r
# is zero (`root_mean_square` needs a nonzero entry)
function noise_scale(r::Vector{T}) where T
  all(isfinite, r) || throw(ArgumentError(
    "sqrt_lasso: a residual on the original data is not finite in $T, so the noise estimate is undefined"))
  iszero(maximum(abs, r)) ? zero(T) : root_mean_square(r)
end

coef(fit::SqrtLassoSolution) = fit.hasintercept ? [fit.intercept; fit.coefficients] : fit.coefficients

function predict(fit::SqrtLassoSolution, X::AbstractMatrix)
  size(X, 2) == length(fit.coefficients) || throw(DimensionMismatch(
    "X has $(size(X, 2)) columns but the fit has $(length(fit.coefficients)) coefficients"))
  fitted = X * fit.coefficients
  fit.hasintercept && (fitted .+= fit.intercept)
  fitted
end

residuals(fit::SqrtLassoSolution) = fit.residuals
nobs(fit::SqrtLassoSolution) = length(fit.residuals)

"""
    dof(fit::SqrtLassoSolution) -> Int

The number of nonzero entries of `fit.coefficients`, plus one when the fit has an intercept.
"""
dof(fit::SqrtLassoSolution) = count(!iszero, fit.coefficients) + fit.hasintercept

"""
    deviance(fit::SqrtLassoSolution)

The residual sum of squares `sum(abs2, residuals(fit))`, on the data given to [`sqrt_lasso`](@ref).
"""
deviance(fit::SqrtLassoSolution) = sum(abs2, fit.residuals)

function Base.show(io::IO, fit::SqrtLassoSolution)
  r = fit.result
  print(io, "SqrtLassoSolution: ", length(fit.coefficients), " coefficients, ",
        count(!iszero, fit.coefficients), " nonzero, λ = ", fit.λ,
        fit.hasintercept ? ", intercept = " * string(fit.intercept) : "",
        ", σ = ", fit.σ, ", ", nobs(fit), " observations;\n",
        "  duality certificate of the transformed problem: ",
        r.converged ? "met (relative gap at most $(r.gap))" : "not met", " after ", r.passes, " passes")
end
