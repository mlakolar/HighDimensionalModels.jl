"""
    LassoSolution{T, R}

The fit returned by [`lasso`](@ref), with fields

- `coefficients::Vector{T}`: the coefficients, one per column of the design;
- `residuals::Vector{T}`: `y - X * coefficients`, computed once at the returned point;
- `λ::T`: the penalty level, converted to `T`;
- `result::R`: the ProximalSolvers result, unchanged. For `lasso` it is a
  `ProximalSolvers.CDResults`, whose documented field `minimizer` is the same
  vector as `coefficients`.

For this problem `result.converged` reports ProximalSolvers' coefficient-change
test, not an optimality certificate; see [`lasso`](@ref).

`coef`, `predict`, `residuals` and `nobs` from StatsAPI are defined for it.
"""
struct LassoSolution{T, R}
  coefficients::Vector{T}
  residuals::Vector{T}
  λ::T
  result::R
end

"""
    lasso(X, y, λ, solver = CDSolver()) -> LassoSolution

Minimize `‖y - Xβ‖² / (2n) + λ‖β‖₁` over `β`, with `n = size(X, 1)`, by coordinate
descent from `β = 0`. `X` must be a `Matrix{T}` and `y` a `Vector{T}`, with `T` equal
to `Float32` or `Float64`, and `λ` a `Real`; `solver` is a `CDSolver`. Other argument
types throw an `ArgumentError`, and nothing is converted. `X` and `y` are not modified,
and the solution keeps no reference to them.

`n ≥ 1` is checked here. The other checks (matching dimensions, `p ≥ 1`, finite data,
`λ` finite, nonnegative and representable in `T`) are those of ProximalSolvers, and
its errors are not altered. `λ` reaches the penalty as given, in its own type.

The loss is ProximalBase's `LeastSquaresLoss`, scaled by `1/n`; the penalty is
`NormL1(λ)`; the solve is `ProximalSolvers.solve!`.

The convergence report is ProximalSolvers' (`fit.result.converged`, `passes`,
`maxchange`, `tolerance`, `minimum`). For this problem `converged` reports the
coefficient-change test of the solver (the largest change of a coefficient in a full
pass, in the solver's scaling of the changes, against `tolerance`), not a bound on the
distance to the minimum or on the KKT conditions. To check a fit, compute `maximum(abs, X' * residuals(fit)) / n` against `λ`
and, on the support, `X[:, j]' * residuals(fit) / n` against `λ * sign(β[j])`.

With `CDSolver()` the coordinate order is random and the default random number
generator of the task advances. `CDSolver(order = :cyclic, momentum = false)` gives a
deterministic run. `CDSolver(maxpasses = 1)` stops after one pass with a warning and
`converged == false`.
"""
function lasso(X::Matrix{T}, y::Vector{T}, λ::Real, solver::CDSolver = CDSolver()) where {T<:Union{Float32, Float64}}
  n = size(X, 1)
  n ≥ 1 || throw(DimensionMismatch("X needs at least one row, but it has $n"))
  f = (one(T) / T(n)) * LeastSquaresLoss(y, X)
  g = NormL1(λ)
  β = zeros(T, size(X, 2))
  result = solve!(β, f, g, solver)
  LassoSolution(β, y - X * β, T(λ), result)
end

lasso(X, y, λ, solver = CDSolver()) = throw(ArgumentError(
  "lasso needs X::Matrix{T}, y::Vector{T} with T Float32 or Float64, λ::Real and a CDSolver; " *
  "got $(typeof(X)), $(typeof(y)), $(typeof(λ)) and $(typeof(solver))"))

coef(fit::LassoSolution) = fit.coefficients
predict(fit::LassoSolution, X::AbstractMatrix) = X * fit.coefficients
residuals(fit::LassoSolution) = fit.residuals
nobs(fit::LassoSolution) = length(fit.residuals)

function Base.show(io::IO, fit::LassoSolution)
  r = fit.result
  print(io, "LassoSolution: ", length(fit.coefficients), " coefficients, ",
        count(!iszero, fit.coefficients), " nonzero, λ = ", fit.λ, ", ", nobs(fit), " observations;\n",
        "  coefficient-change test of the solver (not an optimality certificate): ",
        r.converged ? "met" : "not met", " after ", r.passes, " passes")
end
