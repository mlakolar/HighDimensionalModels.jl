# Regression with penalties on differences. These penalties are not separable
# in beta, so the coordinate-descent engine is not applicable.

# A signal fit must never materialize the n by n identity design.
struct _SignalDesign{T} <: AbstractMatrix{T}
  n::Int
end
Base.size(X::_SignalDesign) = (X.n, X.n)
function Base.getindex(X::_SignalDesign{T}, i::Int, j::Int) where {T}
  @boundscheck checkbounds(X, i, j)
  T(i == j)
end
_structured_product(::_SignalDesign, b) = copy(b)
_structured_product(A, b) = A * b
_structured_tproduct(::_SignalDesign, b) = copy(b)
_structured_tproduct(A, b) = transpose(A) * b
_structured_gram(X::_SignalDesign{T}) where {T} = Diagonal(ones(T, X.n))
_structured_gram(X::AbstractMatrix) = transpose(X) * X
_structured_curvature(X::_SignalDesign{T}) where {T} = one(T)
_structured_curvature(X::AbstractMatrix) = sum(abs2, X) / size(X, 2)

# The signal rule rho ~ n*lambda can be far too small for a regression design
# with column squared norms of order n. Scale against that curvature as well.
_generalized_rho(curvature, n, λ, ::Type{T}) where {T} = T(max(n*λ, curvature, eps(T)))
function _trend_rho(A, curvature, n, λ, spacing, order, ::Type{T}) where {T}
  paper = n * λ * spacing^order
  T(max(paper, A isa _SignalDesign ? eps(T) : curvature * spacing^(2order)))
end

"""
    difference_matrix(n, order = 1; x = nothing)

Sparse difference matrix with `n - order` rows. `order = 0` gives the
identity; `order = 1` gives `beta[j+1] - beta[j]`. For strictly increasing
locations `x`, higher differences use the divided-difference convention of
trend filtering: `D[k+1] = D[1] * diag(k / (x[j+k]-x[j])) * D[k]`.
Unit-spaced locations give ordinary successive differences.
"""
function difference_matrix(n::Integer, order::Integer = 1; x = nothing)
  n > 0 || throw(ArgumentError("n must be positive."))
  0 <= order < n || throw(ArgumentError("order must be between 0 and n - 1."))
  T = x === nothing ? Float64 : float(eltype(x))
  if x !== nothing
    length(x) == n || throw(DimensionMismatch("x must have n locations."))
    all(isfinite, x) && all(>(0), diff(x)) ||
      throw(ArgumentError("x must be finite and strictly increasing."))
  end
  D = spdiagm(0 => ones(T, n))
  for k in 1:order
    if k > 1 && x !== nothing
      w = T[(k - 1) / (x[j+k-1] - x[j]) for j in 1:n-k+1]
      D = Diagonal(w) * D
    end
    m = n - k
    A = sparse([1:m; 1:m], [1:m; 2:m+1], [-ones(T, m); ones(T, m)], m, m+1)
    D = A * D
  end
  D
end

# Keep the sparse matrix for factorization and public use. Iterative trend
# products only need its short row stencil. Read the actual matrix entries,
# including divided-difference weights, rather than recomputing coefficients.
struct _TrendProducts{T, W, C}
  coefficients::C
  n::Int
  m::Int
end

function _trend_products(K::SparseMatrixCSC, order)
  order isa Integer && order in 0:2 || return K
  _trend_products(K, Val(order + 1))
end

function _trend_products(K::SparseMatrixCSC{T}, ::Val{W}) where {T, W}
  m, n = size(K)
  m > 0 && n == m + W - 1 && nnz(K) == m * W || return K
  coefficients = ntuple(_ -> zero(T), Val(W))
  regular = true
  for j in axes(K, 2), k in nzrange(K, j)
    i, value = rowvals(K)[k], nonzeros(K)[k]
    offset = j - i + 1
    1 <= offset <= W && isfinite(value) && !iszero(value) || return K
    if iszero(coefficients[offset])
      coefficients = Base.setindex(coefficients, value, offset)
    elseif coefficients[offset] != value
      regular = false
    end
  end
  if regular
    return _TrendProducts{T, W, typeof(coefficients)}(coefficients, n, m)
  end
  rows = ntuple(_ -> zeros(T, m), Val(W))
  for j in axes(K, 2), k in nzrange(K, j)
    i = rowvals(K)[k]
    rows[j-i+1][i] = nonzeros(K)[k]
  end
  _TrendProducts{T, W, typeof(rows)}(rows, n, m)
end
_trend_products(K, order) = K

@inline _trend_coefficient(c::Number, i) = c
@inline _trend_coefficient(c::AbstractVector, i) = @inbounds c[i]
_split_product!(y, K, x) = mul!(y, K, x)
_split_tproduct!(y, K, x) = mul!(y, transpose(K), x)

# Coverage reports the end of every @simd loop as unexecuted; those ends are excluded.
function _split_product!(y, K::_TrendProducts{T, W}, x) where {T, W}
  length(y) == K.m && length(x) == K.n || throw(DimensionMismatch("trend product dimensions"))
  @inbounds @simd for i in 1:K.m
    value = zero(T)
    for offset in 1:W
      value += _trend_coefficient(K.coefficients[offset], i) * x[i+offset-1]
    end
    y[i] = value
  end # COV_EXCL_LINE
  y
end

function _split_tproduct!(y, K::_TrendProducts{T, W}, x) where {T, W}
  length(y) == K.n && length(x) == K.m || throw(DimensionMismatch("trend transpose dimensions"))
  @inbounds @simd for j in 1:K.n
    value = zero(T)
    for i in max(1, j-W+1):min(j, K.m)
      value += _trend_coefficient(K.coefficients[j-i+1], i) * x[i]
    end
    y[j] = value
  end # COV_EXCL_LINE
  y
end

# Keep boundary handling outside the interior loop. Short dynamic ranges and
# tuple indices otherwise account for much of the transpose-product work.
function _split_tproduct!(y, K::_TrendProducts{T, 2}, x) where {T}
  length(y) == K.n && length(x) == K.m || throw(DimensionMismatch("trend transpose dimensions"))
  a, b = K.coefficients
  @inbounds begin
    y[1] = zero(T) + _trend_coefficient(a, 1) * x[1]
    @simd for j in 2:K.m
      y[j] = (zero(T) + _trend_coefficient(b, j-1) * x[j-1]) + _trend_coefficient(a, j) * x[j]
    end # COV_EXCL_LINE
    y[K.n] = zero(T) + _trend_coefficient(b, K.m) * x[K.m]
  end
  y
end

function _split_tproduct!(y, K::_TrendProducts{T, 3}, x) where {T}
  length(y) == K.n && length(x) == K.m || throw(DimensionMismatch("trend transpose dimensions"))
  a, b, c = K.coefficients
  @inbounds begin
    y[1] = zero(T) + _trend_coefficient(a, 1) * x[1]
    if K.m == 1
      y[2] = zero(T) + _trend_coefficient(b, 1) * x[1]
    else
      y[2] = (zero(T) + _trend_coefficient(b, 1) * x[1]) + _trend_coefficient(a, 2) * x[2]
      @simd for j in 3:K.m
        y[j] = ((zero(T) + _trend_coefficient(c, j-2) * x[j-2]) +
          _trend_coefficient(b, j-1) * x[j-1]) + _trend_coefficient(a, j) * x[j]
      end # COV_EXCL_LINE
      y[K.m+1] = (zero(T) + _trend_coefficient(c, K.m-1) * x[K.m-1]) +
        _trend_coefficient(b, K.m) * x[K.m]
    end
    y[K.n] = zero(T) + _trend_coefficient(c, K.m) * x[K.m]
  end
  y
end

"""
    graph_difference(p, edges; weights = nothing)

Sparse oriented incidence matrix of a graph on vertices `1:p`. Each pair
`(i, j)` in `edges` contributes the row `weight * (beta[j] - beta[i])`.
Weights must be positive and finite; orientation does not affect the penalty.
"""
function graph_difference(p::Integer, edges; weights = nothing)
  p > 0 || throw(ArgumentError("p must be positive."))
  e = collect(edges)
  w = weights === nothing ? ones(length(e)) : float.(collect(weights))
  length(w) == length(e) || throw(DimensionMismatch("one weight is required per edge."))
  all(v -> isfinite(v) && v > 0, w) || throw(ArgumentError("edge weights must be positive and finite."))
  rows, cols, vals = Int[], Int[], eltype(w)[]
  for (k, edge) in enumerate(e)
    length(edge) == 2 || throw(ArgumentError("each edge must have two vertices."))
    i, j = edge
    i isa Integer && j isa Integer && 1 <= i <= p && 1 <= j <= p && i != j ||
      throw(ArgumentError("edges must join distinct vertices in 1:p."))
    append!(rows, (k, k)); append!(cols, (i, j)); append!(vals, (-w[k], w[k]))
  end
  sparse(rows, cols, vals, length(e), p)
end

"""
    StructuredLassoSolution

Fit from [`fused_lasso`](@ref), [`generalized_lasso`](@ref) or
[`trend_filter`](@ref). `x` contains the coefficients, `intercept` is the
unpenalized intercept, `residuals` are on the original data, and `objective`
is `sum(abs2, residuals)/(2n) + penalty`. `iterations`, `converged`,
`primal_residual` and `dual_residual` describe the solver termination.
`df` estimates the generalized-lasso degrees of freedom. Use `coef`,
`predict`, `fitted`, `residuals`, `nobs`, `deviance` and `dof` as for other fits.
"""
struct StructuredLassoSolution{T<:AbstractFloat}
  x::Vector{T}
  intercept::T
  hasintercept::Bool
  residuals::Vector{T}
  fitted::Vector{T}
  objective::T
  iterations::Int
  converged::Bool
  primal_residual::T
  dual_residual::T
  df::Int
end

function _structured_options(max_iter, abstol, reltol, df_tol)
  max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative."))
  all(v -> isfinite(v) && v >= 0, (abstol, reltol, df_tol)) ||
    throw(ArgumentError("tolerances must be finite and nonnegative."))
  nothing
end

# Calibrated separately by solver family. Residual thresholds do not represent
# the same objective or coefficient accuracy in ADMM and proximal gradient.
# Keep the independently calibrated linear-signal rule and non-Float64 defaults.
function _structured_tolerances(X, y, family, order = nothing)
  T = float(promote_type(eltype(X), eltype(y)))
  if T !== Float64 || (family == :trend && X isa _SignalDesign && order == 1)
    return (abstol = 1e-7, reltol = 1e-6)
  elseif family == :generalized
    return (abstol = 1e-9, reltol = 1e-8)
  end
  (abstol = 1e-8, reltol = 1e-7)
end

function _structured_data(X, y, intercept)
  T, A, b = _float_data(X, y)
  n, p = size(A)
  n > 0 && p > 0 || throw(ArgumentError("X must have positive dimensions."))
  length(b) == n || throw(DimensionMismatch("X and y must have the same number of observations."))
  all(isfinite, A) && all(isfinite, b) || throw(ArgumentError("X and y must be finite."))
  xm = intercept ? vec(mean(A; dims = 1)) : zeros(T, p)
  ym = intercept ? mean(b) : zero(T)
  Ac = intercept ? Matrix(A) .- transpose(xm) : A
  Ac, Vector{T}(b .- ym), xm, ym
end

function _structured_data(X::_SignalDesign{T}, y, intercept) where {T}
  intercept && throw(ArgumentError("a signal fit already includes its constant component; use intercept = false."))
  length(y) > 0 && all(isfinite, y) || throw(ArgumentError("y must be nonempty and finite."))
  X, Vector{T}(y), zeros(T, length(y)), zero(T)
end

function _structured_penalty(D, p, ::Type{T}) where {T}
  size(D, 2) == p || throw(DimensionMismatch("D must have one column per coefficient."))
  all(isfinite, D) || throw(ArgumentError("D must be finite."))
  convert(AbstractMatrix{T}, D)
end

function _structured_lambda(λ)
  isfinite(λ) && λ >= 0 || throw(ArgumentError("penalty levels must be finite and nonnegative."))
  λ
end

function _structured_finish(A, b, xm, ym, β, intercept, penalty, it, ok, pr, dr, df)
  T = eltype(β)
  r = b - _structured_product(A, β)
  StructuredLassoSolution{T}(copy(β), T(ym - dot(xm, β)), intercept, Vector{T}(r), Vector{T}(b .+ ym .- r),
    T(sum(abs2, r) / (2length(b)) + penalty), it, ok, T(pr), T(dr), df + intercept)
end

# For a graph, nullity of inactive rows is the number of connected components.
# Single-coordinate constraints (the l1 part of sparse fusion) pin components
# to zero. This also handles redundant edges without dense rank computations.
function _incidence_components(R::SparseMatrixCSC, inactive; check_rank = false)
  p = size(R, 1)
  parents = collect(1:p)
  function root(i)
    while parents[i] != i
      parents[i] = parents[parents[i]]
      i = parents[i]
    end
    i
  end
  pins = Int[]
  T = typeof(float(real(zero(eltype(R)))))
  smallest, frobenius = T(Inf), zero(T)
  for j in inactive
    nz = nzrange(R, j)
    countnz, i1, i2 = 0, 0, 0
    v1, v2 = zero(eltype(R)), zero(eltype(R))
    for i in nz
      v = nonzeros(R)[i]
      iszero(v) && continue
      smallest = min(smallest, abs(v))
      frobenius = hypot(frobenius, v)
      countnz += 1
      if countnz == 1
        i1, v1 = rowvals(R)[i], v
      elseif countnz == 2
        i2, v2 = rowvals(R)[i], v
      else
        return nothing
      end
    end
    if countnz == 1
      push!(pins, i1)
    elseif countnz == 2 && v1 == -v2
      parents[root(i1)] = root(i2)
    elseif countnz > 0
      return nothing
    end
  end
  pinned = falses(p)
  for i in pins
    pinned[root(i)] = true
  end
  # A weighted incidence matrix (including pinned vertices) has nonzero
  # singular values bounded below by min_weight/(p+1). Use this conservative
  # bound to retain the dense SVD convention when numerical rank is ambiguous.
  if check_rank && smallest / (p + 1) <=
      16max(length(inactive), p) * eps(real(float(one(eltype(R))))) * frobenius
    return nothing
  end
  labels, counts = zeros(Int, p), Int[]
  for i in 1:p
    r = root(i)
    pinned[r] && continue
    if labels[r] == 0
      push!(counts, 0)
      labels[r] = length(counts)
    end
    labels[i] = labels[r]
    counts[labels[i]] += 1
  end
  labels, counts
end

function _incidence_df(D, inactive)
  components = _incidence_components(sparse(transpose(D)), inactive)
  components === nothing ? nothing : length(components[2])
end

# The cache belongs to one path with a fixed centered design, penalty and
# df_tol. Keep only the last pattern: storage is O(rows(D)+nnz(D)), independent
# of path length. There is no shared/global state or caller-owned buffer.
mutable struct _StructuredDFWorkspace{R}
  rows::R
  inactive::Vector{Int}
  value::Int
  valid::Bool
  design_rank::Int
  full_column_rank::Bool
end

function _StructuredDFWorkspace(A, D)
  p = size(A, 2)
  if A isa _SignalDesign
    design_rank, full_column_rank = p, true
  else
    s = svdvals(Matrix(A))
    threshold = isempty(s) ? zero(eltype(A)) : min(size(A)...) * eps(eltype(s)) * first(s)
    design_rank = count(>(threshold), s)
    full_column_rank = length(s) == p && last(s) > 16threshold
  end
  _StructuredDFWorkspace(sparse(transpose(D)), Int[], 0, false, design_rank, full_column_rank)
end

function _structured_df(A, D, β, tol; trend_order = nothing, workspace = nothing)
  δ = D * β
  inactive = findall(v -> abs(v) <= tol, δ)
  if A isa _SignalDesign && trend_order !== nothing
    return min(length(β), trend_order + 1 + length(δ) - length(inactive))
  end
  if A isa _SignalDesign
    d = _incidence_df(D, inactive)
    d === nothing || return d
  end
  if workspace !== nothing && workspace.valid && inactive == workspace.inactive
    return workspace.value
  end
  df = _structured_df_rank(A, D, β, inactive, workspace)
  if workspace !== nothing
    resize!(workspace.inactive, length(inactive))
    copyto!(workspace.inactive, inactive)
    workspace.value, workspace.valid = df, true
  end
  df
end

function _structured_df_rank(A, D, β, inactive, workspace)
  isempty(inactive) && return workspace === nothing ? rank(Matrix(A)) : workspace.design_rank
  R = workspace === nothing ? sparse(transpose(D)) : workspace.rows
  components = _incidence_components(R, inactive; check_rank = true)
  if components !== nothing
    labels, counts = components
    k = length(counts)
    k == 0 && return 0
    workspace !== nothing && workspace.full_column_rank && return k
    # Unit-length component columns are orthonormal, just as the SVD basis
    # below. Normalization preserves rank's relative singular-value tolerance.
    rows = findall(!iszero, labels)
    values = [inv(sqrt(eltype(β)(counts[labels[i]]))) for i in rows]
    C = sparse(rows, labels[rows], values, length(β), k)
    AC = Matrix(A * C)
    s = svdvals(AC)
    threshold = min(size(AC)...) * eps(eltype(s)) * first(s)
    # Cancellation or a singular value close to the rank threshold can make
    # the two orthonormal bases disagree numerically. Preserve the old dense
    # calculation in those cases, including a design annihilating a component.
    ambiguous = first(s) <= 16max(size(A)...) * eps(eltype(s)) * norm(A) ||
      any(v -> threshold / 16 <= v <= 16threshold, s)
    ambiguous || return count(>(threshold), s)
  end
  # rank(X N), N spanning null(D_inactive), remains valid for a rank-deficient
  # design. No ridge is silently added to change the estimator.
  N = nullspace(Matrix(D[inactive, :]))
  workspace !== nothing && workspace.full_column_rank && return size(N, 2)
  size(N, 2) == 0 ? 0 : rank(_structured_product(A, N))
end

# The standard D beta = z split, or the specialized trend-filtering split
# K beta = z with TV(z). Factors are reused by a path at a fixed rho.
# Signal trend systems have at most three subdiagonals for the usual orders.
# Store unit-lower LDL' multipliers by subdiagonal and keep the pivots separate.
struct _StructuredBandFactor{T, W}
  lower::Matrix{T}
  pivots::Vector{T}
end

function _structured_band_factor(H, width)
  T, n = eltype(H), size(H, 1)
  L, pivots = zeros(T, width, n), zeros(T, n)
  for j in 1:n, idx in nzrange(H, j)
    i = rowvals(H)[idx]
    if i == j
      pivots[j] = nonzeros(H)[idx]
    elseif j < i <= min(n, j + width)
      L[i-j, j] = nonzeros(H)[idx]
    end
  end
  for j in 1:n
    d = pivots[j]
    for k in max(1, j-width):j-1
      d -= abs2(L[j-k, k]) * pivots[k]
    end
    isfinite(d) && d > 0 || throw(ArgumentError("trend-filtering normal equations must be positive definite."))
    pivots[j] = d
    for i in j+1:min(n, j+width)
      v = L[i-j, j]
      for k in max(1, i-width):j-1
        v -= L[i-k, k] * pivots[k] * L[j-k, k]
      end
      l = v / d
      isfinite(l) || throw(ArgumentError("trend-filtering normal equations must be positive definite."))
      L[i-j, j] = l
    end
  end
  _StructuredBandFactor{T, width}(L, pivots)
end

_structured_solve!(x, F::_StructuredBandFactor{T, W}, b) where {T, W} =
  _structured_band_solve!(x, F.lower, F.pivots, b, Val(W))

function _structured_band_solve!(x, L, d, b, ::Val{0})
  @inbounds @simd for j in eachindex(x)
    x[j] = b[j] / d[j]
  end # COV_EXCL_LINE
  x
end

# Carry each recurrence in registers. Division precedes the backward updates;
# storing inverse pivots would overflow for positive subnormal diagonals.
function _structured_band_solve!(x, L, d, b, ::Val{1})
  n = length(x)
  @inbounds begin
    previous = b[1]
    x[1] = previous
    for j in 2:n
      previous = muladd(-L[1, j-1], previous, b[j])
      x[j] = previous
    end
    previous /= d[n]
    x[n] = previous
    for j in n-1:-1:1
      previous = muladd(-L[1, j], previous, x[j] / d[j])
      x[j] = previous
    end
  end
  x
end

function _structured_band_solve!(x, L, d, b, ::Val{2})
  n = length(x)
  n == 1 && return _structured_band_solve!(x, L, d, b, Val(1))
  @inbounds begin
    y1 = b[1]
    y2 = muladd(-L[1, 1], y1, b[2])
    x[1], x[2] = y1, y2
    for j in 3:n
      v = muladd(-L[1, j-1], y2, muladd(-L[2, j-2], y1, b[j]))
      x[j] = v
      y1, y2 = y2, v
    end
    x1 = y2 / d[n]
    x2 = muladd(-L[1, n-1], x1, y1 / d[n-1])
    x[n], x[n-1] = x1, x2
    for j in n-2:-1:1
      v = muladd(-L[1, j], x2, muladd(-L[2, j], x1, x[j] / d[j]))
      x[j] = v
      x1, x2 = x2, v
    end
  end
  x
end

function _structured_band_solve!(x, L, d, b, ::Val{3})
  n = length(x)
  n <= 2 && return _structured_band_solve!(x, L, d, b, Val(2))
  @inbounds begin
    y1 = b[1]
    y2 = muladd(-L[1, 1], y1, b[2])
    y3 = muladd(-L[1, 2], y2, muladd(-L[2, 1], y1, b[3]))
    x[1], x[2], x[3] = y1, y2, y3
    for j in 4:n
      v = muladd(-L[1, j-1], y3, muladd(-L[2, j-2], y2, muladd(-L[3, j-3], y1, b[j])))
      x[j] = v
      y1, y2, y3 = y2, y3, v
    end
    x1 = y3 / d[n]
    x2 = muladd(-L[1, n-1], x1, y2 / d[n-1])
    x3 = muladd(-L[1, n-2], x2, muladd(-L[2, n-2], x1, y1 / d[n-2]))
    x[n], x[n-1], x[n-2] = x1, x2, x3
    for j in n-3:-1:1
      v = muladd(-L[1, j], x3, muladd(-L[2, j], x2, muladd(-L[3, j], x1, x[j] / d[j])))
      x[j] = v
      x1, x2, x3 = x2, x3, v
    end
  end
  x
end

function _structured_solve!(x, F, b)
  ldiv!(x, F, b)
  x
end

# The generic three-argument ldiv! is applicable on Julia 1.10 even though its
# two-argument CHOLMOD implementation is missing. Use the compatible solve for
# general sparse systems; signal trend systems use the banded method above.
_structured_solve!(x, F::SparseArrays.CHOLMOD.Factor, b) = copyto!(x, F \ b)

function _split_factor(gram, K, ρ; bandwidth = nothing)
  H = gram + ρ * (transpose(K) * K)
  if bandwidth !== nothing && 0 <= bandwidth <= 3 && gram isa Diagonal
    return _structured_band_factor(sparse(H), bandwidth)
  end
  F = cholesky(Symmetric(H); check = false)
  issuccess(F) || throw(ArgumentError("the stacked design [X; D] must have full column rank for ADMM."))
  F
end

# A feasible dual for 1/2||beta-b||^2 + alpha TV(K*beta) is given by
# q in [-alpha,alpha], with v = K'D1'q. Clipping cumulative ADMM multipliers
# supplies q even before convergence. Evaluate the gap as stationarity plus
# nonnegative Fenchel terms, without subtracting two nearly equal objectives.
# The two work vectors are no longer needed by the current ADMM iteration.
function _structured_signal_gap!(stationarity, w, K, beta, b, kb, u, rho, alpha)
  T = eltype(beta)
  cumulative, previous = zero(T), zero(T)
  slack, penalty = zero(T), zero(T)
  @inbounds for j in 1:length(w)-1
    cumulative += rho * u[j]
    q = clamp(-cumulative, -alpha, alpha)
    w[j] = previous - q
    delta = kb[j+1] - kb[j]
    penalty += alpha * abs(delta)
    slack += alpha * abs(delta) - q * delta
    previous = q
  end
  w[end] = previous
  _split_tproduct!(stationarity, K, w)
  residual_square, stationarity_square = zero(T), zero(T)
  @inbounds for j in eachindex(beta)
    residual = beta[j] - b[j]
    residual_square += abs2(residual)
    stationarity_square += abs2(residual + stationarity[j])
  end
  stationarity_square / 2 + slack, residual_square / 2 + penalty
end

@inline _trend_entry(x, i) = @inbounds x[i]
@inline _trend_entry(x::Tuple, i) = @inbounds x[1][i] - x[2][i]

@inline function _trend_tinterior(K::_TrendProducts{T, W}, x, j) where {T, W}
  value = zero(T)
  @inbounds for offset in 1:W
    i = j - W + offset
    value += _trend_coefficient(K.coefficients[W-offset+1], i) * _trend_entry(x, i)
  end
  value
end

@inline function _trend_tinterior(K::_TrendProducts{T, 2}, x, j) where {T}
  a, b = K.coefficients
  (zero(T) + _trend_coefficient(b, j-1) * _trend_entry(x, j-1)) +
    _trend_coefficient(a, j) * _trend_entry(x, j)
end

@inline function _trend_tinterior(K::_TrendProducts{T, 3}, x, j) where {T}
  a, b, c = K.coefficients
  ((zero(T) + _trend_coefficient(c, j-2) * _trend_entry(x, j-2)) +
    _trend_coefficient(b, j-1) * _trend_entry(x, j-1)) +
    _trend_coefficient(a, j) * _trend_entry(x, j)
end

@inline function _trend_tvalue(K::_TrendProducts{T, W}, x, j) where {T, W}
  value = zero(T)
  @inbounds for i in max(1, j-W+1):min(j, K.m)
    value += _trend_coefficient(K.coefficients[j-i+1], i) * _trend_entry(x, i)
  end
  value
end

@inline function _trend_tvalue(K::_TrendProducts{T, 2}, x, j) where {T}
  a, b = K.coefficients
  j == 1 && return zero(T) + _trend_coefficient(a, 1) * _trend_entry(x, 1)
  j == K.n && return zero(T) + _trend_coefficient(b, K.m) * _trend_entry(x, K.m)
  _trend_tinterior(K, x, j)
end

@inline function _trend_tvalue(K::_TrendProducts{T, 3}, x, j) where {T}
  a, b, c = K.coefficients
  j == 1 && return zero(T) + _trend_coefficient(a, 1) * _trend_entry(x, 1)
  j == K.n && return zero(T) + _trend_coefficient(c, K.m) * _trend_entry(x, K.m)
  if j == 2
    value = zero(T) + _trend_coefficient(b, 1) * _trend_entry(x, 1)
    return K.m == 1 ? value : value + _trend_coefficient(a, 2) * _trend_entry(x, 2)
  elseif j == K.m + 1
    return (zero(T) + _trend_coefficient(c, j-2) * _trend_entry(x, j-2)) +
      _trend_coefficient(b, j-1) * _trend_entry(x, j-1)
  end
  _trend_tinterior(K, x, j)
end

function _split_rhs!(rhs, products, xy, ρ, z, u, v)
  @. v = z - u
  _split_tproduct!(rhs, products, v)
  @. rhs = xy + ρ * rhs
end

function _split_rhs!(rhs, products::_TrendProducts{T, W}, xy, ρ, z, u, v) where {T, W}
  @inbounds for j in 1:W-1
    rhs[j] = xy[j] + ρ * _trend_tvalue(products, (z, u), j)
  end
  @inbounds @simd for j in W:products.m
    rhs[j] = xy[j] + ρ * _trend_tinterior(products, (z, u), j)
  end # COV_EXCL_LINE
  @inbounds for j in max(W, products.m+1):products.n
    rhs[j] = xy[j] + ρ * _trend_tvalue(products, (z, u), j)
  end
  rhs
end

function _split_primal!(kb, v, products, β, u)
  _split_product!(kb, products, β)
  @. v = kb + u
end

function _split_primal!(kb, v, products::_TrendProducts{T, W}, β, u) where {T, W}
  @inbounds @simd for i in eachindex(kb)
    value = zero(T)
    for offset in 1:W
      value += _trend_coefficient(products.coefficients[offset], i) * β[i+offset-1]
    end
    kb[i] = value
    v[i] = value + u[i]
  end # COV_EXCL_LINE
  kb
end

@inline _structured_square(x::Union{Float32, Float64}) = abs2(Float64(x))
@inline _structured_square(x) = 0.0
# Use the same fixed reductions in fused and materialized passes. Independent
# partials expose vector additions; compensated block totals limit roundoff.
@inline _structured_sumadd(s, x) = ntuple(k -> s[k] + _structured_square(x[k]), Val(8))
@inline _structured_sum(s) = ((s[1] + s[2]) + (s[3] + s[4])) + ((s[5] + s[6]) + (s[7] + s[8]))
@inline function _structured_sumfinish(s, correction, value)
  y = value - correction
  total = s + y
  total, (total - s) - y
end
_structured_sumsq(x) = 0.0
function _structured_sumsq(x::AbstractVector{<:Union{Float32, Float64}})
  s, correction = 0.0, 0.0
  n = length(x)
  @inbounds for first in 1:256:n-7
    partial = ntuple(_ -> 0.0, Val(8))
    for i in first:8:min(first+255, n-7)
      partial = _structured_sumadd(partial, ntuple(k -> x[i+k-1], Val(8)))
    end
    s, correction = _structured_sumfinish(s, correction, _structured_sum(partial))
  end
  @inbounds for i in n - n % 8 + 1:n
    s, correction = _structured_sumfinish(s, correction, _structured_square(x[i]))
  end
  s
end

# Scaled norm handles underflow, overflow and nonfinite values as before.
# Materialize only on that path; ordinary Float32/64 sums use no scratch pass.
@inline function _structured_norm(s, x::AbstractVector{T}, materialize!::F = identity, args...) where {T, F}
  if T <: Union{Float32, Float64} && isfinite(s) && !(T === Float64 && s < 0x1p-900)
    return T(sqrt(s))
  end
  norm(materialize!(x, args...))
end

function _split_residuals!(u, kb, z)
  pr, kbs, zs, pc, kc, zc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  n = length(u)
  @inbounds for first in 1:256:n-7
    primal = kb_square = z_square = ntuple(_ -> 0.0, Val(8))
    for i in first:8:min(first+255, n-7)
      residual = ntuple(Val(8)) do k
        j = i + k - 1
        u[j] = (u[j] + kb[j]) - z[j]
        kb[j] - z[j]
      end
      primal = _structured_sumadd(primal, residual)
      kb_square = _structured_sumadd(kb_square, ntuple(k -> kb[i+k-1], Val(8)))
      z_square = _structured_sumadd(z_square, ntuple(k -> z[i+k-1], Val(8)))
    end
    pr, pc = _structured_sumfinish(pr, pc, _structured_sum(primal))
    kbs, kc = _structured_sumfinish(kbs, kc, _structured_sum(kb_square))
    zs, zc = _structured_sumfinish(zs, zc, _structured_sum(z_square))
  end
  @inbounds for i in n - n % 8 + 1:n
    u[i] = (u[i] + kb[i]) - z[i]
    pr, pc = _structured_sumfinish(pr, pc, _structured_square(kb[i] - z[i]))
    kbs, kc = _structured_sumfinish(kbs, kc, _structured_square(kb[i]))
    zs, zc = _structured_sumfinish(zs, zc, _structured_square(z[i]))
  end
  pr, kbs, zs
end

function _split_difference!(v, x, y)
  @. v = x - y
end

function _split_dual!(dual, products, z, zold, v)
  @. v = z - zold
  _split_tproduct!(dual, products, v)
end

function _split_dual_sums!(dual, ktu, v, products, z, zold, u)
  _split_dual!(dual, products, z, zold, v)
  _split_tproduct!(ktu, products, u)
  _structured_sumsq(dual), _structured_sumsq(ktu)
end

@inline function _split_dual_group(ds, us, products, z, zold, u, j, stencil::F) where {F}
  delta = ntuple(k -> stencil(products, (z, zold), j+k-1), Val(8))
  ku = ntuple(k -> stencil(products, u, j+k-1), Val(8))
  _structured_sumadd(ds, delta), _structured_sumadd(us, ku)
end

function _split_dual_sums!(dual, ktu, v, products::_TrendProducts{T, W}, z, zold, u) where {T, W}
  ds, us, dc, uc = 0.0, 0.0, 0.0, 0.0
  n = length(dual)
  # Peel containing groups so each column retains its materialized sum lane.
  interior_first = 1 + 8cld(W-1, 8)
  interior_last = 8fld(products.m, 8) - 7
  @inbounds for first in 1:256:n-7
    dual_square = ktu_square = ntuple(_ -> 0.0, Val(8))
    last = min(first+255, n-7)
    for j in first:8:min(last, interior_first-8)
      dual_square, ktu_square = _split_dual_group(dual_square, ktu_square,
        products, z, zold, u, j, _trend_tvalue)
    end
    for j in max(first, interior_first):8:min(last, interior_last)
      dual_square, ktu_square = _split_dual_group(dual_square, ktu_square,
        products, z, zold, u, j, _trend_tinterior)
    end
    for j in max(first, interior_first, interior_last+8):8:last
      dual_square, ktu_square = _split_dual_group(dual_square, ktu_square,
        products, z, zold, u, j, _trend_tvalue)
    end
    ds, dc = _structured_sumfinish(ds, dc, _structured_sum(dual_square))
    us, uc = _structured_sumfinish(us, uc, _structured_sum(ktu_square))
  end
  @inbounds for j in n - n % 8 + 1:n
    ds, dc = _structured_sumfinish(ds, dc, _structured_square(_trend_tvalue(products, (z, zold), j)))
    us, uc = _structured_sumfinish(us, uc, _structured_square(_trend_tvalue(products, u, j)))
  end
  ds, us
end

function _split_fit(A, b, K, g, ρ; max_iter, abstol, reltol, start = nothing,
                    gram = _structured_gram(A), xy = _structured_tproduct(A, b), factor = nothing,
                    bandwidth = nothing, relative_gap = nothing, products = K)
  T, p, m = eltype(b), size(A, 2), size(K, 1)
  F = factor === nothing ? _split_factor(gram, K, ρ; bandwidth) : factor
  β = start === nothing ? zeros(T, p) : copy(start.β)
  z = start === nothing ? _split_product!(zeros(T, m), products, β) : copy(start.z)
  u = start === nothing ? zeros(T, m) : (start.ρ / ρ) .* start.u
  kb, v, zold = zeros(T, m), zeros(T, m), zeros(T, m)
  rhs, dual, ktu = zeros(T, p), zeros(T, p), zeros(T, p)
  energy = relative_gap === nothing ? zero(T) : sum(abs2, b)
  ok, it, pr, dr = false, 0, T(Inf), T(Inf)
  for iter in 1:max_iter
    _split_rhs!(rhs, products, xy, ρ, z, u, v)
    _structured_solve!(β, F, rhs)
    _split_primal!(kb, v, products, β, u)
    z, zold = zold, z
    prox!(z, g, v, inv(ρ))
    primal_square, kb_square, z_square = _split_residuals!(u, kb, z)
    dual_square, ktu_square = _split_dual_sums!(dual, ktu, v, products, z, zold, u)
    pr = _structured_norm(primal_square, v, _split_difference!, kb, z)
    dr = ρ * _structured_norm(dual_square, dual, _split_dual!, products, z, zold, v)
    kb_norm, z_norm = _structured_norm(kb_square, kb), _structured_norm(z_square, z)
    it = iter
    if relative_gap === nothing
      if pr <= sqrt(m) * abstol + reltol * max(kb_norm, z_norm) &&
         dr <= sqrt(p) * abstol + reltol * ρ * _structured_norm(ktu_square, ktu, _split_tproduct!, products, u)
        ok = true
        break
      end
    else
      ep = sqrt(m) * abstol + reltol * max(kb_norm, z_norm)
      ed = sqrt(p) * abstol + reltol * ρ * _structured_norm(ktu_square, ktu, _split_tproduct!, products, u)
      if pr <= 5ep && dr <= 5ed && (iter == 1 || iter % 10 == 0 || iter == max_iter)
        gap, objective = _structured_signal_gap!(dual, v, products, β, b, kb, u, ρ, g.λ)
        # Relative objective accuracy is not meaningful near a zero optimum.
        # Preserve the original residual test below its arithmetic scale.
        nearzero = objective <= 100eps(T) * energy && pr <= ep && dr <= ed
        if isfinite(gap) && isfinite(objective) && isfinite(energy) &&
           (gap <= relative_gap * objective || nearzero)
          ok = true
          break
        end
      end
    end
  end
  β, it, ok, pr, dr, (; β, z, u, ρ)
end

"""
    generalized_lasso(X, y, D, λ; intercept = false, rho = nothing,
                      max_iter = 20_000, abstol = 1e-9, reltol = 1e-8,
                      df_tol = 1e-4)
    generalized_lasso(y, D, λ; kwargs...)

Minimize `norm(y - intercept - X*beta)^2/(2n) + λ*norm(D*beta, 1)`
by ADMM. `D` can be dense or sparse, for example [`graph_difference`](@ref).
The signal form uses the identity design without allocating it. ADMM factors
`X'X + rho*D'D` once; `[X; D]` must have full column rank. `rho` is on the
unnormalized-loss scale and defaults to the larger of `n*λ` and the average
column squared norm of the centered design (one for a signal fit).

No standardization is applied, since it would change the difference penalty.
At zero penalty the minimum-norm least-squares solution is returned. The
result reports nonconvergence if the residual stopping criteria are unmet.
`dof` is `rank(X * null(D[inactive, :]))`, including an intercept when fitted;
inactive rows satisfy `abs(D*beta) <= df_tol`. At transition points the
degrees-of-freedom estimate is sensitive to this numerical tolerance.
Compared with R's `genlasso`, use `lambda_R = n*λ`.
The default residual tolerances above apply to Float64 data; other floating
types retain `abstol = 1e-7, reltol = 1e-6`. Residual convergence does not
certify a uniform relative coefficient error on ill-conditioned designs.
"""
function generalized_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                           D::AbstractMatrix{<:Real}, λ::Real;
                           intercept::Bool = false, rho = nothing, max_iter::Integer = 20_000,
                           abstol::Real = _structured_tolerances(X, y, :generalized).abstol,
                           reltol::Real = _structured_tolerances(X, y, :generalized).reltol,
                           df_tol::Real = 1e-4)
  _structured_lambda(λ)
  _structured_options(max_iter, abstol, reltol, df_tol)
  A, b, xm, ym = _structured_data(X, y, intercept)
  T = eltype(b)
  M = _structured_penalty(D, size(A, 2), T)
  ρ = rho === nothing ? _generalized_rho(_structured_curvature(A), length(b), λ, T) : T(rho)
  isfinite(ρ) && ρ > 0 || throw(ArgumentError("rho must be positive and finite."))
  if iszero(λ) || size(M, 1) == 0
    β = A isa _SignalDesign ? copy(b) : pinv(Matrix(A)) * b
    df = A isa _SignalDesign ? length(b) : rank(A)
    return _structured_finish(A, b, xm, ym, β, intercept, zero(T), 0, true, 0, 0, df)
  end
  β, it, ok, pr, dr, _ = _split_fit(A, b, M, NormL1(T(length(b) * λ)), ρ;
                                    max_iter, abstol, reltol)
  df = _structured_df(A, M, β, df_tol)
  _structured_finish(A, b, xm, ym, β, intercept, λ * sum(abs, M * β), it, ok, pr, dr, df)
end

generalized_lasso(y::AbstractVector{<:Real}, D::AbstractMatrix{<:Real}, λ::Real; kwargs...) =
  generalized_lasso(_SignalDesign{float(eltype(y))}(length(y)), y, D, λ; kwargs...)

"""
    fused_lasso(X, y, λ1, λ2; intercept = false, max_iter = 20_000,
                abstol = 1e-8, reltol = 1e-7, df_tol = 1e-4)
    fused_lasso(y, λ1, λ2; kwargs...)

Minimize `norm(y - intercept - X*beta)^2/(2n) + λ1*norm(beta,1) +
λ2*sum(abs, diff(beta))`. Regression uses accelerated proximal gradient with backtracking
and ProximalBase's exact sparse fused-lasso proximal operator. The signal form
is a single proximal step. Columns retain their original scales and order.
The stopping test is the proximal-gradient residual in coefficient units,
`norm(beta - prox(beta - step*gradient))/step`, against
`sqrt(p)*abstol + reltol*max(norm(X'y/n), 1)`. Before reporting convergence,
the residual is checked at the returned coefficients using the original design;
`dual_residual` reports that value. The defaults above apply to Float64 data;
other floating types retain `abstol = 1e-7, reltol = 1e-6`. This is an
optimality-residual test, not a uniform relative coefficient-error guarantee.
"""
function fused_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ1::Real, λ2::Real;
                     intercept::Bool = false, max_iter::Integer = 20_000,
                     abstol::Real = _structured_tolerances(X, y, :fused).abstol,
                     reltol::Real = _structured_tolerances(X, y, :fused).reltol,
                     df_tol::Real = 1e-4,
                     _start = nothing, _data = nothing, _gram = nothing, _xy = nothing,
                     _df_workspace = nothing, _df_constraints = nothing,
                     _prediction_only::Bool = false)
  _structured_lambda(λ1); _structured_lambda(λ2)
  _structured_options(max_iter, abstol, reltol, df_tol)
  A, b, xm, ym = _data === nothing ? _structured_data(X, y, intercept) : _data
  T, n, p = eltype(b), length(b), size(A, 2)
  g = FusedLasso(T(λ1), T(λ2))
  β = _start === nothing ? zeros(T, p) : Vector{T}(_start)
  it, ok, dr = 0, false, T(Inf)
  if A isa _SignalDesign
    prox!(β, g, b, T(n))
    it, ok, dr = 1, true, zero(T)
  else
    # Estimate the smooth curvature with a few power iterations; backtracking
    # checks every accepted step, including designs with a missed eigenspace.
    use_gram = _gram !== nothing
    L = use_gram ? sum(diag(_gram)) : sum(abs2, A) / n
    grad, candidate, v = zeros(T, p), zeros(T, p), fill(inv(sqrt(T(p))), p)
    r, rn = zeros(T, n), zeros(T, n)
    if L > 0
      for _ in 1:8
        if use_gram
          mul!(grad, _gram, v)
        else
          mul!(r, A, v)
          mul!(grad, transpose(A), r)
        end
        magnitude = norm(grad)
        magnitude == 0 && break
        L = use_gram ? dot(v, grad) : dot(v, grad) / n
        @. v = grad / magnitude
      end
    end
    step = L <= 0 ? one(T) : inv(T(L))
    extrapolated, delta = copy(β), zeros(T, p)
    momentum = one(T)
    threshold = sqrt(p) * abstol + reltol * max(use_gram ? norm(_xy) : norm(transpose(A) * b) / n, 1)
    for iter in 1:max_iter
      if use_gram
        mul!(grad, _gram, extrapolated); grad .-= _xy
      else
        mul!(r, A, extrapolated); r .-= b
        mul!(grad, transpose(A), r); grad ./= n
      end
      for backtrack in 1:100
        @. v = extrapolated - step * grad
        prox!(candidate, g, v, step)
        @. v = candidate - extrapolated
        # For a quadratic loss this is the majorization test exactly, without
        # subtracting two almost equal objective values near convergence.
        curvature = if use_gram
          mul!(delta, _gram, v)
          dot(v, delta)
        else
          mul!(rn, A, v)
          sum(abs2, rn) / n
        end
        if curvature <= sum(abs2, v) / step
          break
        end
        backtrack == 100 && error("proximal-gradient backtracking failed.")
        step /= 2
      end
      dr = norm(v) / step
      @. delta = candidate - β
      # Gradient restart removes momentum when extrapolation turns uphill.
      next_momentum = dot(v, delta) < 0 ? one(T) : (one(T) + sqrt(one(T) + 4momentum^2)) / 2
      @. extrapolated = candidate + ((momentum - 1) / next_momentum) * delta
      if next_momentum == 1
        copyto!(extrapolated, candidate)
      end
      momentum = next_momentum
      copyto!(β, candidate)
      it = iter
      if dr <= threshold
        # Check the returned point, not only its extrapolated predecessor.
        # Original-design products also protect against normal-matrix error.
        mul!(r, A, β); r .-= b
        mul!(grad, transpose(A), r); grad ./= n
        @. v = β - step * grad
        prox!(delta, g, v, step)
        @. v = β - delta
        dr = norm(v) / step
        if !(dr <= threshold)
          use_gram = false
          momentum = one(T)
          copyto!(extrapolated, β)
          continue
        end
        ok = true
        break
      end
    end
  end
  _prediction_only && return (β, it, ok, dr)
  if A isa _SignalDesign
    df = 0
    left = 1
    for j in 1:p
      if j == p || λ2 == 0 || abs(β[j+1] - β[j]) > df_tol
        df += λ1 == 0 || any(v -> abs(v) > df_tol, @view β[left:j])
        left = j + 1
      end
    end
  else
    if iszero(λ1) && iszero(λ2)
      df = _df_workspace === nothing ? rank(Matrix(A)) : _df_workspace.design_rank
    else
      constraints = _df_constraints
      if constraints === nothing
        constraints = λ2 > 0 && p > 1 ? difference_matrix(p) : spzeros(T, 0, p)
        λ1 > 0 && (constraints = [constraints; spdiagm(0 => ones(T, p))])
      end
      df = _structured_df(A, constraints, β, df_tol; workspace = _df_workspace)
    end
  end
  _structured_finish(A, b, xm, ym, β, intercept, g(β), it, ok, 0, dr, df)
end

fused_lasso(y::AbstractVector{<:Real}, λ1::Real, λ2::Real; kwargs...) =
  fused_lasso(_SignalDesign{float(eltype(y))}(length(y)), y, λ1, λ2; kwargs...)

"""
    trend_filter(y, λ; order = 1, x = nothing, kwargs...)
    trend_filter(X, y, λ; order = 1, x = nothing, kwargs...)

Trend filtering with `λ * norm(difference_matrix(p, order+1; x)*beta, 1)`
and loss `norm(y-X*beta)^2/(2n)`. `order = k` gives piecewise polynomials of
degree `k` (order zero is total-variation denoising). `x` gives the ordered
coefficient locations, not the rows of the design.

For `order > 0`, the specialized ADMM of Ramdas and Tibshirani (2016) splits
the order-`k` divided differences and applies exact TV denoising in each
iteration. Sparse banded normal equations avoid a dense identity in signal
fits. Keywords and termination diagnostics follow [`generalized_lasso`](@ref).
For a signal, default `rho = n*λ*spacing^order` follows the paper's spacing
rule. With a design, it is at least the average column squared norm times
`spacing^(2order)`. R's `glmgen::trendfilter` uses `lambda = n*λ`.

Float64 linear signal fits (`order = 1`, positive `reltol`) stop using a
feasible primal-dual gap at most `0.3reltol` times the primal objective,
with primal and dual residuals at most five times their usual bounds. The
gap is checked every ten iterations and at the first and final iteration.
Near a zero objective the original residual test is used. Other fits and
absolute-only tolerances retain residual stopping.
Other Float64 trend fits default to `abstol = 1e-8, reltol = 1e-7`.
Linear signals and other floating types retain `1e-7, 1e-6`. Explicit
tolerances override these defaults, including on paths and CV folds.
"""
function trend_filter(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real;
                      order::Integer = 1, x = nothing, intercept::Bool = false,
                      rho = nothing, max_iter::Integer = 20_000,
                      abstol::Real = _structured_tolerances(X, y, :trend, order).abstol,
                      reltol::Real = _structured_tolerances(X, y, :trend, order).reltol,
                      df_tol::Real = 1e-4)
  _structured_lambda(λ)
  _structured_options(max_iter, abstol, reltol, df_tol)
  p = size(X, 2)
  rho === nothing || (isfinite(rho) && rho > 0) || throw(ArgumentError("rho must be positive and finite."))
  0 <= order < p - 1 || throw(ArgumentError("order must be between 0 and p - 2."))
  D = difference_matrix(p, order + 1; x)
  if order == 0
    return fused_lasso(X, y, 0, λ; intercept, max_iter, abstol, reltol, df_tol)
  elseif λ == 0
    return generalized_lasso(X, y, D, λ; intercept, rho, max_iter, abstol, reltol, df_tol)
  end
  A, b, xm, ym = _structured_data(X, y, intercept)
  T = eltype(b)
  # D[k+1] = D[1] * K also for irregular locations.
  K = difference_matrix(p, order; x)
  if x !== nothing
    K = Diagonal(T[order / (x[j+order] - x[j]) for j in 1:p-order]) * K
  end
  spacing = x === nothing ? one(T) : T((x[end] - x[1]) / (p - 1))
  ρ = rho === nothing ? _trend_rho(A, _structured_curvature(A), length(b), λ, spacing, order, T) : T(rho)
  isfinite(ρ) && ρ > 0 || throw(ArgumentError("rho must be positive and finite."))
  K = convert(SparseMatrixCSC{T, Int}, K)
  gap_tol = A isa _SignalDesign{Float64} && order == 1 && reltol > 0 ? T(3reltol / 10) : nothing
  β, it, ok, pr, dr, _ = _split_fit(A, b, K, TotalVariation1D(T(length(b) * λ)), ρ;
                                    max_iter, abstol, reltol, bandwidth = A isa _SignalDesign ? order : nothing,
                                    relative_gap = gap_tol, products = _trend_products(K, order))
  df = _structured_df(A, D, β, df_tol; trend_order = order)
  _structured_finish(A, b, xm, ym, β, intercept, λ * sum(abs, D * β), it, ok, pr, dr, df)
end

trend_filter(y::AbstractVector{<:Real}, λ::Real; kwargs...) =
  trend_filter(_SignalDesign{float(eltype(y))}(length(y)), y, λ; kwargs...)

StatsAPI.coef(s::StructuredLassoSolution) = s.hasintercept ? [s.intercept; s.x] : copy(s.x)
function StatsAPI.predict(s::StructuredLassoSolution, X::AbstractMatrix{<:Real})
  _checkcolumns(X, length(s.x))
  X * s.x .+ s.intercept
end
StatsAPI.residuals(s::StructuredLassoSolution) = copy(s.residuals)
StatsAPI.fitted(s::StructuredLassoSolution) = copy(s.fitted)
StatsAPI.nobs(s::StructuredLassoSolution) = length(s.residuals)
StatsAPI.dof(s::StructuredLassoSolution) = s.df
StatsAPI.deviance(s::StructuredLassoSolution) = sum(abs2, s.residuals)

"""
    GeneralizedLassoPath(X, y, D, levels; kwargs...)
    GeneralizedLassoPath(y, D, levels; kwargs...)

Generalized-lasso fits at the supplied decreasing nonnegative penalty levels.
ADMM warm starts its primal and scaled dual variables and reuses the normal
equations. Consecutive fits with the same `rho` also reuse the factorization,
including an automatically chosen value. The keywords
are those of [`generalized_lasso`](@ref). The field `solutions` holds each fit;
`λpath` holds its penalty level. `coef`, `predict`, `dof`, `deviance`, `aic` and
`bic` return one column or value per level. `coef(path, λ)` requires an exact
grid level: this ADMM path does not compute or interpolate homotopy knots.
Information criteria return `NaN` at unconverged fits or when `df >= n`.
"""
struct GeneralizedLassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  solutions::Vector{StructuredLassoSolution{T}}
  family::Symbol
end

function _structured_levels(levels, ::Type{T}) where {T}
  λs = collect(T, levels)
  !isempty(λs) && all(v -> isfinite(v) && v >= 0, λs) &&
    all(<(0), diff(λs)) ||
    throw(ArgumentError("levels must be nonempty, finite, nonnegative and strictly decreasing."))
  λs
end

function _admm_path(X, y, D, K, levels; family = :generalized, trend_order = nothing,
                    intercept::Bool = false, rho = nothing, max_iter::Integer = 20_000,
                    abstol::Real = _structured_tolerances(X, y, family, trend_order).abstol,
                    reltol::Real = _structured_tolerances(X, y, family, trend_order).reltol,
                    df_tol::Real = 1e-4,
                    spacing = 1)
  _structured_options(max_iter, abstol, reltol, df_tol)
  A, b, xm, ym = _structured_data(X, y, intercept)
  T, n = eltype(b), length(b)
  M = _structured_penalty(D, size(A, 2), T)
  K = _structured_penalty(K, size(A, 2), T)
  products = _trend_products(K, trend_order)
  λs = _structured_levels(levels, T)
  gram, xy = _structured_gram(A), _structured_tproduct(A, b)
  curvature = sum(diag(gram)) / size(A, 2)
  solutions = StructuredLassoSolution{T}[]
  df_workspace = A isa _SignalDesign ? nothing : _StructuredDFWorkspace(A, M)
  state, fixed_factor, last_rho = nothing, nothing, nothing
  bandwidth = A isa _SignalDesign ? trend_order : nothing
  gap_tol = A isa _SignalDesign{Float64} && trend_order == 1 && reltol > 0 ? T(3reltol / 10) : nothing
  if rho !== nothing
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be positive and finite."))
    if any(>(0), λs) && size(K, 1) > 0
      fixed_factor = _split_factor(gram, K, T(rho); bandwidth)
      last_rho = T(rho)
    end
  end
  for λ in λs
    if λ == 0 || size(M, 1) == 0
      push!(solutions, generalized_lasso(X, y, M, λ; intercept, rho, max_iter, abstol, reltol, df_tol))
      continue
    end
    ρ = rho === nothing ? (trend_order === nothing ? _generalized_rho(curvature, n, λ, T) :
        _trend_rho(A, curvature, n, λ, spacing, trend_order, T)) : T(rho)
    if last_rho != ρ
      fixed_factor = _split_factor(gram, K, ρ; bandwidth)
      last_rho = ρ
    end
    g = trend_order === nothing ? NormL1(T(n*λ)) : TotalVariation1D(T(n*λ))
    β, it, ok, pr, dr, state = _split_fit(A, b, K, g, ρ;
      max_iter, abstol, reltol, start = state, gram, xy, factor = fixed_factor, relative_gap = gap_tol, products)
    df = _structured_df(A, M, β, df_tol; trend_order, workspace = df_workspace)
    push!(solutions, _structured_finish(A, b, xm, ym, β, intercept,
      λ * sum(abs, M * β), it, ok, pr, dr, df))
  end
  GeneralizedLassoPath{T}(λs, solutions, family)
end

GeneralizedLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                    D::AbstractMatrix{<:Real}, levels; kwargs...) =
  _admm_path(X, y, D, D, levels; kwargs...)
GeneralizedLassoPath(y::AbstractVector{<:Real}, D::AbstractMatrix{<:Real}, levels; kwargs...) =
  GeneralizedLassoPath(_SignalDesign{float(eltype(y))}(length(y)), y, D, levels; kwargs...)

"""
    fused_lasso_path(X, y, levels; l1_ratio = 0.5, kwargs...)
    fused_lasso_path(y, levels; kwargs...)

Warm-started fused regression at `λ1 = l1_ratio*λ`,
`λ2 = (1-l1_ratio)*λ` for each decreasing value in `levels`. Keywords follow
[`fused_lasso`](@ref). Returns a [`GeneralizedLassoPath`](@ref).
"""
function fused_lasso_path(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, levels;
                          l1_ratio::Real = 0.5, kwargs...)
  isfinite(l1_ratio) && 0 <= l1_ratio <= 1 || throw(ArgumentError("l1_ratio must be in [0, 1]."))
  T = float(promote_type(eltype(X), eltype(y)))
  λs = _structured_levels(levels, T)
  data = _structured_data(X, y, get(kwargs, :intercept, false))
  A, b = data[1], data[2]
  n, p = size(A)
  gram, xy = nothing, nothing
  # Cap the extra normal-matrix storage at 32 MiB, and amortize its formation
  # over a path only for tall double-precision dense designs.
  if A isa StridedMatrix{Float64} && n >= 4p && p <= 2048
    H = transpose(A) * A / n
    if all(isfinite, H) && issuccess(cholesky(Symmetric(H); check=false))
      gram, xy = H, transpose(A) * b / n
    end
  end
  fits = StructuredLassoSolution{T}[]
  df_constraints, df_workspace = nothing, nothing
  if !(A isa _SignalDesign)
    df_constraints = l1_ratio < 1 && p > 1 ? difference_matrix(p) : spzeros(T, 0, p)
    l1_ratio > 0 && (df_constraints = [df_constraints; spdiagm(0 => ones(T, p))])
    df_workspace = _StructuredDFWorkspace(A, df_constraints)
  end
  start = nothing
  for λ in λs
    s = fused_lasso(X, y, l1_ratio * λ, (1-l1_ratio) * λ;
      _start = start, _data = data, _gram = gram, _xy = xy,
      _df_workspace = df_workspace, _df_constraints = df_constraints, kwargs...)
    push!(fits, s)
    start = s.x
  end
  GeneralizedLassoPath{T}(λs, fits, :fused)
end
fused_lasso_path(y::AbstractVector{<:Real}, levels; kwargs...) =
  fused_lasso_path(_SignalDesign{float(eltype(y))}(length(y)), y, levels; kwargs...)

"""
    trend_filter_path(X, y, levels; order = 1, x = nothing, kwargs...)
    trend_filter_path(y, levels; kwargs...)

Warm-started specialized ADMM path for [`trend_filter`](@ref), at supplied
decreasing levels. Returns a [`GeneralizedLassoPath`](@ref). For order zero,
each signal fit is a single exact proximal step.
"""
function trend_filter_path(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, levels;
                           order::Integer = 1, x = nothing, rho = nothing, kwargs...)
  p = size(X, 2)
  0 <= order < p - 1 || throw(ArgumentError("order must be between 0 and p - 2."))
  D = difference_matrix(p, order + 1; x)
  rho === nothing || (isfinite(rho) && rho > 0) || throw(ArgumentError("rho must be positive and finite."))
  if order == 0
    path = fused_lasso_path(X, y, levels; l1_ratio = 0, kwargs...)
    return GeneralizedLassoPath(path.λpath, path.solutions, :trend)
  end
  K = difference_matrix(p, order; x)
  if x !== nothing
    K = Diagonal([order / (x[j+order] - x[j]) for j in 1:p-order]) * K
  end
  spacing = x === nothing ? 1 : (x[end] - x[1]) / (p - 1)
  _admm_path(X, y, D, K, levels; family = :trend, trend_order = order, spacing, rho, kwargs...)
end
trend_filter_path(y::AbstractVector{<:Real}, levels; kwargs...) =
  trend_filter_path(_SignalDesign{float(eltype(y))}(length(y)), y, levels; kwargs...)

StatsAPI.coef(p::GeneralizedLassoPath) = reduce(hcat, coef.(p.solutions))
function _structured_index(p, λ)
  j = findfirst(==(λ), p.λpath)
  j === nothing && throw(ArgumentError("λ must equal a fitted grid level; refit to evaluate a new level."))
  j
end
StatsAPI.coef(p::GeneralizedLassoPath, λ::Real) = coef(p.solutions[_structured_index(p, λ)])
StatsAPI.predict(p::GeneralizedLassoPath, X::AbstractMatrix{<:Real}) =
  reduce(hcat, [predict(s, X) for s in p.solutions])
StatsAPI.predict(p::GeneralizedLassoPath, X::AbstractMatrix{<:Real}, λ::Real) =
  predict(p.solutions[_structured_index(p, λ)], X)
StatsAPI.nobs(p::GeneralizedLassoPath) = nobs(first(p.solutions))
StatsAPI.dof(p::GeneralizedLassoPath) = dof.(p.solutions)
StatsAPI.deviance(p::GeneralizedLassoPath) = deviance.(p.solutions)
StatsAPI.fitted(p::GeneralizedLassoPath) = reduce(hcat, fitted.(p.solutions))
StatsAPI.residuals(p::GeneralizedLassoPath) = reduce(hcat, residuals.(p.solutions))
StatsAPI.fit(::Type{GeneralizedLassoPath}, args...; kwargs...) = GeneralizedLassoPath(args...; kwargs...)

function _structured_ic(p::GeneralizedLassoPath, multiplier)
  n = nobs(p)
  [s.converged && dof(s) < n ? n * log(deviance(s) / n) + multiplier * dof(s) : NaN for s in p.solutions]
end
"""
    aic(path::GeneralizedLassoPath)
    bic(path::GeneralizedLassoPath)

Gaussian information criteria `n*log(RSS/n) + c*df`, where `c = 2` for AIC
and `c = log(n)` for BIC. `df = rank(X*null(D_inactive))`, rather than a
count of nonzero coefficients. Values are `NaN` for unconverged fits or
`df >= n`. Omitted Gaussian constants are common to all levels.
"""
StatsAPI.aic(p::GeneralizedLassoPath) = _structured_ic(p, 2)
StatsAPI.bic(p::GeneralizedLassoPath) = _structured_ic(p, log(nobs(p)))

"""
    CVGeneralizedLassoPath(X, y, D, levels; nfolds = 10, folds = nothing,
                           rng = Random.default_rng(), threads = true, kwargs...)

Prediction-error cross-validation on a common explicit penalty grid, using
[`GeneralizedLassoPath`](@ref). The same fixed coefficient penalty is used in
each training fold; centering is learned only on its training observations.
User fold labels allow blocked or grouped validation. Fields `path`, `λpath`,
`cvm`, `cvse`, `λmin`, `λ1se` and `folds` follow [`CVLassoPath`](@ref).
Nonconverged fold fits raise an error instead of entering the CV averages.
"""
struct CVGeneralizedLassoPath{T<:AbstractFloat}
  path::GeneralizedLassoPath{T}
  λpath::Vector{T}
  cvm::Vector{T}
  cvse::Vector{T}
  λmin::T
  λ1se::T
  folds::Vector{Int}
end

function _structured_cv(makepath, X, y, levels; nfolds::Integer = 10, folds = nothing,
                         rng::AbstractRNG = default_rng(), threads::Bool = true)
  n = length(y)
  size(X, 1) == n || throw(DimensionMismatch("X and y must have the same number of observations."))
  n >= 2 || throw(ArgumentError("cross-validation needs at least two observations."))
  fl, K = _foldlabels(rng, n, nfolds, folds)
  all(k -> 1 <= k <= K, fl) || throw(ArgumentError("fold labels must be positive."))
  path = makepath(X, y, levels)
  all(s -> s.converged, path.solutions) || error("the full-data path did not converge.")
  T = eltype(path.λpath)
  errs = zeros(T, K, length(path.λpath))
  function fold!(k)
    train, test = findall(!=(k), fl), findall(==(k), fl)
    fp = makepath(X[train, :], y[train], path.λpath)
    all(s -> s.converged, fp.solutions) || error("the path in fold $k did not converge.")
    preds = predict(fp, X[test, :])
    for j in axes(preds, 2)
      errs[k, j] = mean(abs2, y[test] - preds[:, j])
    end
  end
  _runfolds(fold!, K, threads)
  cvm, cvse, jmin, j1se = _cvsummary(errs, T[count(==(k), fl) for k in 1:K])
  CVGeneralizedLassoPath{T}(path, path.λpath, cvm, cvse, path.λpath[jmin], path.λpath[j1se], fl)
end

function CVGeneralizedLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                               D::AbstractMatrix{<:Real}, levels;
                               nfolds::Integer = 10, folds = nothing,
                               rng::AbstractRNG = default_rng(), threads::Bool = true, kwargs...)
  _structured_cv((A, b, ls) -> GeneralizedLassoPath(A, b, D, ls; kwargs...), X, y, levels;
                  nfolds, folds, rng, threads)
end
function CVGeneralizedLassoPath(y::AbstractVector{<:Real}, D::AbstractMatrix{<:Real}, levels;
                               nfolds::Integer=10, folds=nothing, rng::AbstractRNG=default_rng(),
                               threads::Bool=true, max_concurrent_folds::Integer=2, kwargs...)
  T=float(eltype(y));p=length(y);M=_structured_penalty(D,p,T)
  makepath=(ls)->GeneralizedLassoPath(y,M,ls;kwargs...)
  foldpath=(train,ls,emit)->_signal_admm_errors(_MaskedSignalDesign{T}(p,train),T.(y[train]),M,M,ls,emit;kwargs...)
  _signal_cv(makepath,foldpath,y,levels;nfolds,folds,rng,threads,max_concurrent_folds)
end

"""
    cv_fused_lasso(X, y, levels; l1_ratio = 0.5, nfolds = 10, folds = nothing,
                   rng = Random.default_rng(), threads = true, kwargs...)
    cv_trend_filter(X, y, levels; order = 1, x = nothing, kwargs...)
    cv_trend_filter(y, levels; kwargs...)

Cross-validation using [`fused_lasso_path`](@ref) or [`trend_filter_path`](@ref).
For signals, masked training observations keep the complete location grid.
Fold work streams prediction errors without computing degrees of freedom or
retaining a training path. The full-data returned path uses direct signal
dispatch. `max_concurrent_folds = 2` bounds simultaneous signal-fold work when
`threads = true`; use one for serial fold memory. No held-out response enters
a training fit or changes its difference penalty.
`order` is the polynomial degree, and `x` must be strictly increasing.
Returns a [`CVGeneralizedLassoPath`](@ref).
"""
function cv_fused_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, levels;
                         nfolds::Integer = 10, folds = nothing, rng::AbstractRNG = default_rng(),
                         threads::Bool = true, kwargs...)
  _structured_cv((A, b, ls) -> fused_lasso_path(A, b, ls; kwargs...), X, y, levels;
                  nfolds, folds, rng, threads)
end
function cv_fused_lasso(y::AbstractVector{<:Real}, levels;
                        nfolds::Integer=10, folds=nothing, rng::AbstractRNG=default_rng(),
                        threads::Bool=true, max_concurrent_folds::Integer=2, l1_ratio::Real=.5, kwargs...)
  isfinite(l1_ratio) && 0<=l1_ratio<=1 || throw(ArgumentError("l1_ratio must be in [0, 1]."))
  T=float(eltype(y));p=length(y)
  makepath=(ls)->fused_lasso_path(y,ls;l1_ratio,kwargs...)
  foldpath=(train,ls,emit)->_signal_fused_errors(_MaskedSignalDesign{T}(p,train),T.(y[train]),ls,emit;l1_ratio,kwargs...)
  _signal_cv(makepath,foldpath,y,levels;nfolds,folds,rng,threads,max_concurrent_folds)
end
"""
    cv_trend_filter(X, y, levels; order = 1, x = nothing, kwargs...)
    cv_trend_filter(y, levels; kwargs...)

Cross-validate a trend-filtering path. Signal folds retain the full location
grid and use diagonal observation Grams and eligible banded systems. Positive
penalties require enough training locations to identify the degree-`order`
polynomial component; unidentified folds fail without adding a ridge. See
[`cv_fused_lasso`](@ref) for the fold, grid and tuning conventions.
"""
function cv_trend_filter(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, levels;
                          nfolds::Integer = 10, folds = nothing, rng::AbstractRNG = default_rng(),
                          threads::Bool = true, kwargs...)
  _structured_cv((A, b, ls) -> trend_filter_path(A, b, ls; kwargs...), X, y, levels;
                  nfolds, folds, rng, threads)
end
function cv_trend_filter(y::AbstractVector{<:Real}, levels;
                         nfolds::Integer=10, folds=nothing, rng::AbstractRNG=default_rng(),
                         threads::Bool=true, max_concurrent_folds::Integer=2,
                         order::Integer=1, x=nothing, rho=nothing, kwargs...)
  T=float(eltype(y));p=length(y)
  0<=order<p-1 || throw(ArgumentError("order must be between 0 and p - 2."))
  D=difference_matrix(p,order+1;x)
  K=order==0 ? nothing : difference_matrix(p,order;x)
  if order>0 && x!==nothing
    K=Diagonal(T[order/(x[j+order]-x[j]) for j in 1:p-order])*K
  end
  spacing=x===nothing ? one(T) : T((x[end]-x[1])/(p-1))
  makepath=(ls)->trend_filter_path(y,ls;order,x,rho,kwargs...)
  function foldpath(train,ls,emit)
    # Distinct ordered training locations identify the polynomial null space.
    any(>(0),ls) && length(train)<order+1 && throw(ArgumentError("training observations do not identify the degree-$order polynomial component."))
    A=_MaskedSignalDesign{T}(p,train);b=T.(y[train])
    order==0 ? _signal_fused_errors(A,b,ls,emit;l1_ratio=0,kwargs...) :
      _signal_admm_errors(A,b,D,K,ls,emit;family=:trend,trend_order=order,spacing,rho,kwargs...)
  end
  _signal_cv(makepath,foldpath,y,levels;nfolds,folds,rng,threads,max_concurrent_folds)
end

StatsAPI.coef(cv::CVGeneralizedLassoPath; λ = :λ1se) = coef(cv.path, _cvlevel(cv, λ))
StatsAPI.predict(cv::CVGeneralizedLassoPath, X::AbstractMatrix; λ = :λ1se) =
  predict(cv.path, X, _cvlevel(cv, λ))
StatsAPI.fit(::Type{CVGeneralizedLassoPath}, args...; kwargs...) = CVGeneralizedLassoPath(args...; kwargs...)
# CI scenario: a comment-only change to a focused source file.
