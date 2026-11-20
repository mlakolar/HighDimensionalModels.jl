"""
    GroupLassoSolution{T, R}

The fit returned by [`group_lasso`](@ref), with fields

- `coefficients::Vector{T}`: the coefficients `β`, one per column of the design, in the order of
  the columns of `X` (also for noncontiguous groups);
- `intercept::T`: the intercept `β₀`, which is `0` when `hasintercept` is `false`;
- `hasintercept::Bool`: whether `group_lasso` was called with `intercept = true`;
- `residuals::Vector{T}`: `y - β₀ - X * coefficients` on the data given to `group_lasso`, computed
  once at the returned point;
- `λ::T`: the penalty level, converted to `T`;
- `groups::Vector{Vector{Int}}`: the groups given to `group_lasso`, copied;
- `groupweights::Vector{T}`: the weights `wₖ` of the objective (the default `√|Gₖ|` or the ones
  given, converted to `T`), one per group;
- `activegroups::Vector{Int}`: the numbers of the groups with a nonzero coefficient in
  `coefficients`, in increasing order;
- `standardize::Symbol`: `:orthonormal` or `:none`, as given to `group_lasso`;
- `result::R`: the ProximalSolvers result, unchanged. Its documented field `minimizer` is a
  ProximalBase `BlockIterate` in the coordinates of the matrix the solver saw (see
  [`group_lasso`](@ref)); with `standardize = :none` it holds the same values as `coefficients`.

`result.converged` reports ProximalSolvers' coefficient-change test, not an optimality
certificate. `result.minimum` is the objective of the solver's problem at `result.minimizer`. In
exact arithmetic it equals the objective of [`group_lasso`](@ref) at the mathematical map back from
`result.minimizer` to the columns of `X` (`β_G[P[1:r]] = R̃ \\ θ_G` with `standardize = :orthonormal`, `β = θ`
with `:none`, and `β₀ = ȳ - x̄ᵀβ`). The stored `coefficients` and `intercept` are that map computed in
`T`, which rounds it: a coefficient whose exact value is below the range of `T` underflows to a subnormal
number or to `0`, and the residuals, which use the stored values, then differ from those of the solver's
problem (as for [`lasso`](@ref)); the result of the solver is not altered. For `X = reshape([-1e300, 1e300], 2, 1)`,
`y = [-1e-100, 1e-100]`, `λ = 0` and `intercept = false` the solver returns `θ = 1e-100` with minimum `0`, the
stored coefficient is `0` (the exact value `1e-400` underflows) and `sum(abs2, residuals(fit))` is `2e-200`.

The positional constructor `GroupLassoSolution(coefficients, intercept, hasintercept, residuals, λ,
groups, groupweights, activegroups, standardize, result)` is internal; build a fit with
[`group_lasso`](@ref).

`coef`, `predict`, `residuals`, `nobs`, `dof` and `deviance` from StatsAPI are defined for it.
"""
struct GroupLassoSolution{T, R}
  coefficients::Vector{T}
  intercept::T
  hasintercept::Bool
  residuals::Vector{T}
  λ::T
  groups::Vector{Vector{Int}}
  groupweights::Vector{T}
  activegroups::Vector{Int}
  standardize::Symbol
  result::R
end

# The constant c of the rank tolerance of a group, `rank_tolerance` below. PROVISIONAL: the value
# is a placeholder until the calibration of the tolerance is complete, and it may change, as may
# the formula of `rank_tolerance`.
const RANK_TOLERANCE_C = 10.0

# The tolerance τ_G = c √max(n, m) eps(T) |R₁₁| of the rank decision of a group of m columns of n
# rows, where |R₁₁| is the first diagonal entry of the triangular factor of the pivoted QR of
# its (centred) columns, which is the largest column norm up to rounding. A diagonal entry |Rᵢᵢ| above τ_G
# counts toward the rank (see `group_basis`). The rank decision takes the tolerance from this function.
# The value is a Float64, so that a finite Float32 factor does not overflow it.
rank_tolerance(::Type{T}, n::Integer, m::Integer, r11::Real) where {T<:AbstractFloat} =
  RANK_TOLERANCE_C * sqrt(max(n, m)) * Float64(eps(T)) * Float64(r11)

# The orthonormal basis of one group, from the matrix `A` of its columns (n×m, finite, with the
# centred columns when the fit has an intercept), which is overwritten. A pivoted QR in T,
# `A[:, P] = Q R`, gives the rank r, the number of diagonal entries |Rᵢᵢ| above
# `rank_tolerance`. Returns `(; Z, R, pivots)`:
#
# - `Z`: the n×r matrix `√n Q[:, 1:r]`, whose columns have `Z'Z/n = I` up to rounding;
# - `R`: the upper triangular `R̃ = R[1:r, 1:r]/√n`, so that `A[:, pivots] ≈ Z R̃` with `pivots =
#   P[1:r]` (the first r columns of the QR, the columns that the rule retains); the columns
#   `P[r+1:m]` of the group get the coefficient 0;
# - `pivots`: those column numbers within the group.
#
# For r = 0, `Z` has no column. An ArgumentError if a diagonal entry of R is not finite.
function group_basis(A::Matrix{T}) where {T<:AbstractFloat}
  n = size(A, 1)
  F = qr!(A, ColumnNorm())
  R = Matrix(F.R)
  d = [abs(R[i, i]) for i in axes(R, 1)]
  all(isfinite, d) || throw(ArgumentError("group_lasso: the triangular factor of the QR of a group is not finite (overflow)"))
  τ = rank_tolerance(T, n, size(A, 2), d[1])
  r = count(>(τ), d)
  r == 0 && return (; Z = zeros(T, n, 0), R = UpperTriangular(zeros(T, 0, 0)), pivots = Int[])
  rootn = sqrt(T(n))
  Z = F.Q * Matrix{T}(I, n, r)                 # the first r columns of Q
  Z .*= rootn
  (; Z, R = UpperTriangular(R[1:r, 1:r] ./ rootn), pivots = F.p[1:r])
end

# The groups as lists of columns of a design with p columns: disjoint, nonempty, with each
# column in one group. Returns copies as Vector{Vector{Int}}; each failure has its own message.
function check_groups(groups, p::Int)
  owner = zeros(Int, p)
  copies = Vector{Vector{Int}}(undef, length(groups))
  for (k, G) in enumerate(groups)
    G isa AbstractVector || throw(ArgumentError(
      "group_lasso: group $k must be a vector of column indices, but it is of type $(typeof(G))"))
    isempty(G) && throw(ArgumentError("group_lasso: group $k is empty"))
    all(j -> j isa Integer && !(j isa Bool), G) || throw(ArgumentError(
      "group_lasso: group $k has an entry that is not an integer column index"))
    for j in G
      1 ≤ j ≤ p || throw(ArgumentError(
        "group_lasso: group $k has the index $j, which is not a column of X (the columns are 1:$p)"))
      owner[j] == k && throw(ArgumentError("group_lasso: group $k repeats the column $j"))
      owner[j] != 0 && throw(ArgumentError(
        "group_lasso: column $j is in the groups $(owner[j]) and $k; the groups must be disjoint"))
      owner[j] = k
    end
    copies[k] = collect(Int, G)
  end
  uncovered = findall(iszero, owner)
  isempty(uncovered) || throw(ArgumentError(
    "group_lasso: the columns $uncovered are in no group; give each its own group (a zero group weight leaves it unpenalized)"))
  copies
end

# The weights of the K groups in T: `√|Gₖ|` (the size of the group given) when `groupweights` is
# nothing, otherwise the given ones, a copy.
function group_weights(groupweights, groups, ::Type{T}) where {T}
  groupweights === nothing && return T[sqrt(T(length(G))) for G in groups]
  groupweights isa AbstractVector{<:Real} || throw(ArgumentError(
    "group_lasso: groupweights must be nothing or a vector of real numbers, but it is of type $(typeof(groupweights))"))
  length(groupweights) == length(groups) || throw(DimensionMismatch(
    "group_lasso: there are $(length(groups)) groups and $(length(groupweights)) group weights"))
  w = Vector{T}(undef, length(groups))
  for k in eachindex(w)
    wk = groupweights[k]
    isfinite(wk) && wk ≥ 0 || throw(ArgumentError(
      "group_lasso: the weight of group $k is $wk; the weights must be finite and nonnegative"))
    w[k] = T(wk)
    isfinite(w[k]) && iszero(w[k]) == iszero(wk) || throw(ArgumentError(
      "group_lasso: the weight of group $k is not representable in $T (a finite nonzero number that converts to $(w[k]))"))
  end
  w
end

# λ in T, for a λ that is finite, nonnegative and representable in T (a nonzero λ that converts to
# zero or to infinity is not)
function group_level(λ::Real, ::Type{T}) where {T}
  isfinite(λ) && λ ≥ 0 || throw(ArgumentError("group_lasso: λ must be finite and nonnegative, but it is $λ"))
  λT = T(λ)
  isfinite(λT) && iszero(λT) == iszero(λ) || throw(ArgumentError(
    "group_lasso: λ is not representable in $T (a finite nonzero number that converts to $λT)"))
  λT
end

# The map from the solver's coordinates of one group of rank r > 0 to the columns of X:
# `β[pivots] = R \ θ_G` (the other columns of the group get 0). A group of rank 0 has
# `R` of size 0×0 and no pivots.
struct GroupMap{T}
  R::UpperTriangular{T, Matrix{T}}
  pivots::Vector{Int}
end

# The data of the problem solved by the CDSolver: `Z`, `ỹ`, the groups of the columns of `Z`, and
# the map back to the columns of X. Returns `(; Z, ỹ, x̄, ȳ, zgroups, maps)`, where `x̄` and `ȳ` are the
# means (zeros without an intercept) and `maps` is empty for `:none` and otherwise holds the
# `GroupMap` of each group.
function group_design(X::Matrix{T}, y::Vector{T}, groups, intercept::Bool, standardize::Symbol) where {T}
  n, p = size(X)
  d = intercept ? preprocess(X, y, true, false) : (; Z = X, ỹ = y, x̄ = zeros(T, p), ȳ = zero(T))
  standardize === :none && return (; d.Z, d.ỹ, d.x̄, d.ȳ, zgroups = groups, maps = GroupMap{T}[])
  blocks = Matrix{T}[]
  zgroups = UnitRange{Int}[]
  maps = GroupMap{T}[]
  off = 0
  for G in groups
    if !intercept
      for j in G
        all(isfinite, view(X, :, j)) || nonfinite_error("column $j of X", "has a non-finite entry")
      end
    end
    b = group_basis(d.Z[:, G])
    r = size(b.Z, 2)
    if r == 0
      # a group of rank 0 stays in the fit as one column of zeros, which the solver keeps at 0
      push!(blocks, zeros(T, n, 1))
      push!(maps, GroupMap(UpperTriangular(zeros(T, 0, 0)), Int[]))
    else
      push!(blocks, b.Z)
      push!(maps, GroupMap(b.R, G[b.pivots]))
    end
    push!(zgroups, off+1:off+max(r, 1))
    off += max(r, 1)
  end
  (; Z = reduce(hcat, blocks), d.ỹ, d.x̄, d.ȳ, zgroups, maps)
end

# β from the solver's coordinates θ: θ itself for :none, and otherwise `β[pivots] = R̃ \ θ_G` for
# each group whose θ_G is not zero. An ArgumentError if an entry of β is not finite.
function group_coefficients(θ, d, standardize::Symbol, p::Int, ::Type{T}) where {T}
  standardize === :none && return T[θ[j] for j in 1:p]
  β = zeros(T, p)
  for (k, zg) in enumerate(d.zgroups)
    θk = T[θ[j] for j in zg]
    all(iszero, θk) && continue
    b = d.maps[k]
    βk = b.R \ θk
    all(isfinite, βk) || throw(ArgumentError(
      "group_lasso: the coefficients of group $k overflowed when the fit was mapped back from the orthonormal basis of the group"))
    β[b.pivots] .= βk
  end
  β
end

"""
    group_lasso(X, y, λ, groups, solver = CDSolver(); intercept = true, standardize = :orthonormal,
                groupweights = nothing) -> GroupLassoSolution

The group lasso by block coordinate descent. For disjoint, nonempty `groups` `G₁, …, G_K` of the
columns of `X` that together contain all the columns, weights `wₖ` and `n = size(X, 1)`, it minimizes
over `β₀` and `β`

    ‖y - β₀1 - Xβ‖² / (2n) + λ Σₖ wₖ ‖X̃_{Gₖ} β_{Gₖ}‖₂ / √n     (standardize = :orthonormal, the default)
    ‖y - β₀1 - Xβ‖² / (2n) + λ Σₖ wₖ ‖β_{Gₖ}‖₂                 (standardize = :none)

where `X̃` is `X` with its columns centred at their means when `intercept = true`, and `X̃ = X`
otherwise, and `X̃_G` is the columns of `X̃` in the group `G`. The intercept `β₀` is not penalized;
it is `0` unless `intercept = true`, and then `β₀ = ȳ - x̄ᵀβ` for the means `ȳ` of `y` and `x̄` of the
columns. `X` must be a `Matrix{T}` and `y` a `Vector{T}`, with `T` equal to `Float32` or `Float64`,
`λ` a `Real` and `solver` a `CDSolver`. Other argument types throw an `ArgumentError`, and the data
are not converted. `X` and `y` are not modified, and the solution keeps no reference to them or to
`groups` or `groupweights`.

**Groups and weights.** `groups` is a vector of vectors (or ranges) of integer column indices, and
need not be contiguous or sorted: `[[1, 4], [2, 3], [5]]` is a valid partition of five columns. An
`ArgumentError` with its own message is thrown for an empty group, an entry that is not an integer, an
index outside `1:size(X, 2)`, an index repeated within a group, a column in two groups, and a column in no
group (give such a column its own group; a weight of zero leaves a group unpenalized).
`groupweights` is `nothing` or a vector of `K` nonnegative finite real numbers (`DimensionMismatch` for another
length, `ArgumentError` for a negative, `NaN` or infinite weight, or one that is not representable in
`T`); the default is `wₖ = √|Gₖ|`, with the number of columns of the group as given. The weights are
the ones in the objective and are stored in the solution, converted to `T`. `λ` must be finite,
nonnegative and representable in `T` (a `BigFloat` or `Int` `λ` is converted to `T`; a nonzero `λ` that
converts to zero or to infinity is refused).

**`standardize = :none`.** The solver minimizes the objective on the columns as they are (centred
with an intercept), with the groups as given. Scaling the columns of a group, or recombining them by a
matrix that is not orthogonal, can change the estimate.

**`standardize = :orthonormal`.** The penalty is on the contribution of each group to the fitted
values, `‖X̃_G β_G‖₂/√n`, so it depends on the group through its column span. For an invertible `M` with as
many columns, `X̃_G M` has the same span, so in exact arithmetic the minimum and the fitted values do not change
under `X̃_G → X̃_G M`: if `β_G` is a solution for `X̃_G`, then `M⁻¹β_G` is a solution for `X̃_G M`. When the columns
of the group are linearly independent and the rank rule retains all of them for both `X̃_G` and `X̃_G M`, the
solution is unique, and the returned coefficients are `M⁻¹β_G` up to rounding and the solver's tolerance. When a column is an exact
combination of the others (a group that is rank deficient) the solution is not unique, and the pivoted QR can
return another solution with the same contribution. When the rule drops a direction of a group whose columns are
linearly independent, the fit truncates that direction: the coefficients of the dropped columns are `0` and the
contribution lies in the span of the retained columns. In floating point the fit uses the span of
the `r` columns that the rank rule below retains, and the fitted values are the same, up to rounding, when the
retained spans of `X̃_G` and `X̃_G M` are the same subspace (and the solver has stopped at its tolerance). This holds when the rule retains all the columns of
the group before and after the transformation (with `M` well conditioned), and when the dropped directions are
at rounding level in both. Equal ranks below the number of columns do not suffice: the transformation can change
which directions are retained, and then the fits differ. With the default weights, adding a column to a group
changes its weight, so a comparison needs the weights given. The fit is computed on an orthonormal basis of each
group: the centred columns `X̃_G` (with an intercept; `X_G` without one) are factored by a pivoted QR in `T`,
`X̃_G[:, P] = Q R`, with `r` the number of diagonal entries `|Rᵢᵢ|` of `R` above the tolerance
`τ_G = c √max(n, |G|) eps(T) |R₁₁|` of the rank decision. The constant `c` is documented on the page of
the group lasso. The matrix `Z_G = √n Q[:, 1:r]` has `Z_G'Z_G/n = I` up to rounding, and the solver
minimizes `‖ỹ - Zθ‖²/(2n) + λ Σₖ wₖ‖θ_{Gₖ}‖₂` over the blocks `θ_{Gₖ}`. The blocks of `Z` have, up to rounding, the same curvature
in all directions, but ProximalSolvers' block step uses a bound on the curvature, which exceeds it by a margin
that depends on `n` and `eps(T)`, so a pass need not solve a block problem, and the solver iterates until its change
test is met or `maxpasses` is exhausted (see the page of the group lasso for an example). The coefficients of the group are
`β_G[P[1:r]] = R̃ \\ θ_G` with `R̃ = R[1:r, 1:r]/√n`, and
`0` on the columns `P[r+1:end]`, which the decision treats as combinations of the others. A group of
rank `0` (zero columns, or constant columns with an intercept) stays in the fit as one column of zeros,
so that the groups of the fit are the groups given; its coefficients are `0`. The solution is then
computed on the data `Z` and `ỹ` (`y` centred with an intercept); `Z` has, for each group, as many columns as its
rank, or one column for a group of rank `0`.

**Preprocessing.** With `intercept = true` the columns and the response are centred as in
[`lasso`](@ref): a column (or `y`) whose entries are all equal has that value as its mean, so that
it centres to exact zeros. Non-finite entries of `X` and of `y` are refused with an `ArgumentError`: by
this package for `X` with `standardize = :orthonormal` and for `X` and `y` with an intercept, with a message that names
the column or `y`, and otherwise by ProximalSolvers.

**Solving.** The loss is ProximalBase's `LeastSquaresLoss` scaled by `1/n`, the penalty is
`GroupNormL2(λ, groups, weights)` on a `BlockIterate`, and the solve is `ProximalSolvers.solve!`,
unchanged. The coordinates are the groups and a pass visits whole groups; `CDSolver` refuses
`momentum = true`, and its error is passed on. `fit.result.converged` is the solver's
coefficient-change test and not a bound on the distance to the minimum or on the optimality
conditions. A solve that exhausts `maxpasses` without meeting the change test returns `converged == false` and
ProximalSolvers warns.
To check a fit, compute the group optimality conditions from the returned `coefficients`,
`intercept` and `residuals` (see the page of the group lasso).

`dof(fit)` counts the nonzero coefficients plus one with an intercept; it is a count, not an
unbiased estimate of the degrees of freedom.
"""
function group_lasso(X::Matrix{T}, y::Vector{T}, λ::Real, groups::AbstractVector, solver::CDSolver = CDSolver();
                     intercept = true, standardize = :orthonormal,
                     groupweights = nothing) where {T<:Union{Float32, Float64}}
  check_flag(:intercept, intercept)
  standardize isa Symbol && (standardize === :orthonormal || standardize === :none) || throw(ArgumentError(
    "group_lasso: standardize must be :orthonormal or :none, but it is $(repr(standardize))"))
  n, p = size(X)
  n ≥ 1 || throw(DimensionMismatch("X needs at least one row, but it has $n"))
  p ≥ 1 || throw(DimensionMismatch("X must have at least one column"))
  length(y) == n || throw(DimensionMismatch("y has $(length(y)) entries but X has $n rows"))
  λT = group_level(λ, T)
  gs = check_groups(groups, p)
  w = group_weights(groupweights, gs, T)
  d = group_design(X, y, gs, intercept, standardize)
  f = (one(T) / T(n)) * LeastSquaresLoss(d.ỹ, d.Z)
  g = GroupNormL2(λT, d.zgroups, w)
  θ = BlockIterate(T, size(d.Z, 2), d.zgroups)
  result = solve!(θ, f, g, solver)
  β = group_coefficients(θ, d, standardize, p, T)
  β₀ = intercept ? intercept_from_means(d.ȳ, d.x̄, β) : zero(T)
  r = y .- β₀ .- X * β
  all(isfinite, r) || throw(ArgumentError("group_lasso: the residuals y - β₀ - Xβ are not finite (overflow)"))
  active = [k for (k, G) in enumerate(gs) if any(j -> !iszero(β[j]), G)]
  GroupLassoSolution(β, β₀, intercept, r, λT, gs, w, active, standardize, result)
end

group_lasso(X, y, λ, groups, solver = CDSolver(); intercept = true, standardize = :orthonormal,
            groupweights = nothing) = throw(ArgumentError(
  "group_lasso needs X::Matrix{T}, y::Vector{T} with T Float32 or Float64, λ::Real, groups::AbstractVector, a CDSolver, " *
  "and the keywords intercept (true or false), standardize (:orthonormal or :none) and groupweights; " *
  "got $(typeof(X)), $(typeof(y)), $(typeof(λ)), $(typeof(groups)) and $(typeof(solver))"))

coef(fit::GroupLassoSolution) = fit.hasintercept ? [fit.intercept; fit.coefficients] : fit.coefficients

function predict(fit::GroupLassoSolution, X::AbstractMatrix)
  size(X, 2) == length(fit.coefficients) || throw(DimensionMismatch(
    "X has $(size(X, 2)) columns but the fit has $(length(fit.coefficients)) coefficients"))
  fitted = X * fit.coefficients
  fit.hasintercept && (fitted .+= fit.intercept)
  fitted
end

residuals(fit::GroupLassoSolution) = fit.residuals
nobs(fit::GroupLassoSolution) = length(fit.residuals)

"""
    dof(fit::GroupLassoSolution) -> Int

The number of nonzero entries of `fit.coefficients`, plus one when the fit has an intercept. This is a
count, not an unbiased estimate of the degrees of freedom of the group lasso.
"""
dof(fit::GroupLassoSolution) = count(!iszero, fit.coefficients) + fit.hasintercept

"""
    deviance(fit::GroupLassoSolution)

The residual sum of squares `sum(abs2, residuals(fit))`, on the data given to [`group_lasso`](@ref).
"""
deviance(fit::GroupLassoSolution) = sum(abs2, fit.residuals)

function Base.show(io::IO, fit::GroupLassoSolution)
  r = fit.result
  print(io, "GroupLassoSolution: ", length(fit.groups), " groups, ", length(fit.activegroups), " active, ",
        length(fit.coefficients), " coefficients, ", count(!iszero, fit.coefficients), " nonzero, λ = ", fit.λ,
        fit.hasintercept ? ", intercept = " * string(fit.intercept) : "",
        ", standardize = ", repr(fit.standardize), ", ", nobs(fit), " observations;\n",
        "  coefficient-change test of the solver (not an optimality certificate): ",
        r.converged ? "met" : "not met", " after ", r.passes, " passes")
end
