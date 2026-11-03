######################################################################
#
#   Cross-validation and information criteria along a lasso path
#
######################################################################

"""
    CVLassoPath

Result returned by [`CVLassoPath`](@ref)`(X, y)`.

# Fields
- `path::LassoPath`: the path on all the data
- `λpath::Vector`: its levels, the grid of every fold
- `cvm::Vector`, `cvse::Vector`: the cross-validated mean squared prediction
  error at each level, and its standard error (`cvm` is `NaN` at the levels that
  no fold reaches, `cvse` where fewer than two do)
- `nnz::Vector{Int}`: the number of nonzero coefficients of `path` at each level
- `λmin`, `λ1se`: among the levels that every fold reaches, the level with the
  smallest `cvm`, and the largest level whose `cvm` is within one standard error
  of it
- `folds::Vector{Int}`: the fold of each observation
"""
struct CVLassoPath{T<:AbstractFloat}
  path::LassoPath{T}
  λpath::Vector{T}
  cvm::Vector{T}
  cvse::Vector{T}
  nnz::Vector{Int}
  λmin::T
  λ1se::T
  folds::Vector{Int}
end

# a balanced random assignment of n observations to K folds
function _folds(rng::AbstractRNG, n::Integer, K::Integer)
  2 <= K <= n || throw(ArgumentError("nfolds must be between 2 and the number of observations $n, but it is $K."))
  f = Vector{Int}(undef, n)
  for (i, j) in enumerate(randperm(rng, n))
    f[j] = (i - 1) % K + 1
  end
  f
end

"""
    CVLassoPath(X, y, options = CDOptions(); nfolds = 10, folds = nothing, rng = Random.default_rng(),
                threads = true, nλ = 100, λminratio, intercept = false, standardize_x = true, kwargs...)

K-fold cross-validation of the lasso path of [`LassoPath`](@ref) (with the
same options and keywords): the path on all the data gives the levels, each
fold's path is computed on the other folds at those levels, and the mean
squared error on the held-out fold estimates the prediction error at each
level. Standardization and the intercept are computed on each training fold,
so nothing leaks from the held-out one.

`cvm` is the average of the fold errors weighted by the fold sizes, and `cvse`
its standard error, `sqrt(∑ₖ nₖ (eₖ - cvm)² / n / (K - 1))`, as in glmnet.
`λmin` minimizes `cvm`, and `λ1se` is the largest level whose `cvm` is within
`cvse` of the minimum (the one-standard-error rule). A fold path that stops
early (`max_hat_s`) leaves the remaining levels out of the averages: `cvm` is
`NaN` at the levels that no fold reaches and `cvse` where fewer than two do,
and `λmin` and `λ1se` are chosen among the levels that every fold reaches.

The folds are a balanced random assignment drawn from `rng`, or the vector
`folds` of fold labels `1, …, K` given, for example grouped or blocked folds for
dependent data. With `threads`, the folds run as tasks on the available
threads; each fold draws its random coordinate orders from a generator seeded
from `rng`, so that the result does not depend on the number of threads.

Observation `weights` and an `offset` (see [`lasso`](@ref)) follow the
observations into the folds. The fold errors are then the weighted mean squared
errors of `y - offset`, and the folds are weighted by their sums of weights.
With `gram = :auto`, the choice for the full data is used for every training
fold, so that holding out observations does not switch a nearly square design
from the Gram form to the residual form. `gram = true` or `false` selects the
form explicitly for all fits.

Returns a [`CVLassoPath`](@ref); `coef(cv)` and `predict(cv, X)` use the
solution at `λ1se`, or at `λmin` with `λ = :λmin`.
"""
function CVLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, options::CDOptions = CDOptions();
                     nfolds::Integer = 10, folds::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                     rng::AbstractRNG = default_rng(), threads::Bool = true, nλ::Integer = 100,
                     λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                     weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                     offset::Union{Nothing, AbstractVector{<:Real}} = nothing,
                     gram::Union{Bool, Symbol} = :auto,
                     kwargs...) where {T<:AbstractFloat}
  n = size(X, 1)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  fl, K = _foldlabels(rng, n, nfolds, folds)
  w, o = _weights(T, weights, n), _offset(T, offset, n)
  usegram = _usegram(gram, X, true)

  seeds = rand(rng, UInt64, K + 1)
  path = LassoPath(X, y, options; nλ, λminratio, gram = usegram, rng = Xoshiro(seeds[end]), _wo(w, o)..., kwargs...)
  λs = path.λpath
  errs = fill(T(NaN), K, length(λs))
  function fold!(k)
    train, test = findall(!=(k), fl), findall(==(k), fl)
    fp = LassoPath(X[train, :], y[train], λs, options; gram = usegram, rng = Xoshiro(seeds[k]),
                   _wo(_rows(w, train), _rows(o, train))..., kwargs...)
    Ŷ = predict(fp, X[test, :])
    for j in axes(Ŷ, 2)
      errs[k, j] = _wmean(i -> abs2(y[test[i]] - _entry(o, test[i]) - Ŷ[i, j]), _rows(w, test), length(test))
    end
  end
  _runfolds(fold!, K, threads)
  cvm, cvse, jmin, j1se = _cvsummary(errs, _foldweights(T, fl, K, w))
  CVLassoPath{T}(path, λs, cvm, cvse, [nnz(β) for β in path.βpath], λs[jmin], λs[j1se], fl)
end

# the fold of each of n observations, drawn from rng or the labels given, and
# the number of folds K
function _foldlabels(rng::AbstractRNG, n::Integer, nfolds::Integer, folds)
  fl = folds === nothing ? _folds(rng, n, nfolds) : collect(Int, folds)
  length(fl) == n || throw(DimensionMismatch("folds must have one label per observation"))
  K = maximum(fl)
  K >= 2 && all(k -> any(==(k), fl), 1:K) ||
    throw(ArgumentError("the fold labels must be 1, …, K for some K ≥ 2, each used at least once"))
  fl, K
end

# the rows idx of the observation weights or the offset (nothing for none), the
# entry i (zero for no offset), and the keywords that pass them on
_rows(::Nothing, idx) = nothing
_rows(v::AbstractVector, idx) = v[idx]
_entry(::Nothing, i::Integer) = false
_entry(v::AbstractVector, i::Integer) = v[i]
_wo(w, o) = (; (w === nothing ? () : (weights = w,))..., (o === nothing ? () : (offset = o,))...)

# the mean of f(i), i = 1, …, m, weighted by w (nothing for equal weights)
_wmean(f, ::Nothing, m::Integer) = sum(f, 1:m) / m
_wmean(f, w::AbstractVector, m::Integer) = sum(i -> w[i] * f(i), 1:m) / sum(w)

# the weight of each fold in the averages: its number of observations, or the
# sum of their weights
_foldweights(::Type{T}, fl, K, ::Nothing) where {T} = T[count(==(k), fl) for k in 1:K]
_foldweights(::Type{T}, fl, K, w::AbstractVector) where {T} = T[sum(w[i] for i in eachindex(fl) if fl[i] == k) for k in 1:K]

function _runfolds(fold!, K::Integer, threads::Bool)
  if threads
    foreach(wait, [Threads.@spawn fold!(k) for k in 1:K])
  else
    foreach(fold!, 1:K)
  end
end

# The weighted mean cvm of the fold errors errs (K × L, NaN at the levels a fold
# path did not reach) over the folds that reached each level, and its standard
# error cvse, as in glmnet: cvm is NaN at the levels that no fold reached, and
# cvse where fewer than two did. The level jmin minimizes cvm among the levels
# that every fold reached, and j1se is the first (largest) of those whose cvm is
# within cvse of the minimum, so that both compare the same folds.
function _cvsummary(errs::Matrix{T}, wk::Vector{T}) where {T}
  K, L = size(errs)
  cvm, cvse = fill(T(NaN), L), fill(T(NaN), L)
  for j in 1:L
    ok = findall(k -> !isnan(errs[k, j]), 1:K)
    isempty(ok) && continue
    w = wk[ok]
    m = sum(w .* errs[ok, j]) / sum(w)
    cvm[j] = m
    length(ok) > 1 && (cvse[j] = sqrt(sum(w .* abs2.(errs[ok, j] .- m)) / sum(w) / (length(ok) - 1)))
  end
  every = [!isnan(cvm[j]) && !any(isnan, view(errs, :, j)) for j in 1:L]
  any(every) || throw(ArgumentError("no level of the path has an error from every fold; check the fold fits."))
  candidates = findall(every)
  jmin = candidates[argmin(view(cvm, candidates))]
  bound = cvm[jmin] + (isnan(cvse[jmin]) ? zero(T) : cvse[jmin])
  cvm, cvse, jmin, findfirst(j -> every[j] && cvm[j] <= bound, 1:jmin)
end

function CVLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  CVLassoPath(Xf, yf, options; kwargs...)
end

_cvlevel(cv, λ::Symbol) = λ === :λ1se ? cv.λ1se : λ === :λmin ? cv.λmin :
  throw(ArgumentError("λ must be :λ1se, :λmin or a number, but it is :$λ."))
_cvlevel(cv, λ::Real) = λ

"""
    coef(cv::CVLassoPath; λ = :λ1se)
    predict(cv::CVLassoPath, X; λ = :λ1se)

The coefficients (`[β₀; β]` with an intercept), and the predictions at the rows
of `X`, of the path on all the data at `cv.λ1se`, at `cv.λmin` with
`λ = :λmin`, or at a level `λ` within the path.
"""
StatsAPI.coef(cv::CVLassoPath; λ::Union{Symbol, Real} = :λ1se) = coef(cv.path, _cvlevel(cv, λ))

function StatsAPI.predict(cv::CVLassoPath, X::AbstractMatrix{<:Real}; λ::Union{Symbol, Real} = :λ1se)
  b = coef(cv; λ)
  cv.path.intercept ? Vector(X * b[2:end]) .+ b[1] : Vector(X * b)
end

# the residual sums of squares and the numbers of nonzero coefficients along the path
function _rss_df(path::LassoPath, X::AbstractMatrix, y::AbstractVector)
  size(X, 1) == length(y) || throw(DimensionMismatch("X has $(size(X, 1)) rows and y has length $(length(y))"))
  Ŷ = predict(path, X)
  [sum(abs2, y .- view(Ŷ, :, j)) for j in axes(Ŷ, 2)], [nnz(β) for β in path.βpath]
end

function _criterion(path::LassoPath{T}, X, y, penalty) where {T}
  n = length(y)
  rss, s = _rss_df(path, X, y)
  map(rss, s) do r, sk
    df = sk + path.intercept
    df < n ? n * log(r / n) + penalty(df, sk) : T(NaN)
  end
end

"""
    ebic(path::LassoPath, X, y; γ = 0.5)
    aic(path::LassoPath, X, y)
    bic(path::LassoPath, X, y)

Information criteria along the lasso `path` fitted to `X` and `y`, one per
level of `path.λpath`:

- `aic = n log(RSS/n) + 2 df`;
- `bic = n log(RSS/n) + log(n) df`;
- `ebic = bic + 2γ log C(p, s)`, the extended BIC of Chen and Chen (2008) for
  large model spaces, with `s` nonzero coefficients among `p`.

`RSS` is the residual sum of squares and `df` the number of nonzero
coefficients plus one for an intercept, an unbiased estimate of the degrees of
freedom of the lasso (Zou, Hastie and Tibshirani, 2007). A level with `df ≥ n`
gets `NaN`. The level that minimizes a criterion `c` is
`path.λpath[argmin(c)]`, with the `NaN`s left out.
"""
function ebic(path::LassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; γ::Real = 0.5)
  0 <= γ <= 1 || throw(ArgumentError("γ must be in [0, 1], but it is $γ."))
  p = size(X, 2)
  logbinom(s) = s == 0 ? 0.0 : sum(log(p - s + i) - log(i) for i in 1:s)
  _criterion(path, X, y, (df, s) -> log(length(y)) * df + 2γ * logbinom(s))
end

"""
    aic(path::LassoPath, X, y)

The Akaike criterion `n log(RSS/n) + 2 df` at each level of the lasso path; see
[`ebic`](@ref).
"""
StatsAPI.aic(path::LassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}) =
  _criterion(path, X, y, (df, s) -> 2df)

"""
    bic(path::LassoPath, X, y)

The Bayesian criterion `n log(RSS/n) + log(n) df` at each level of the lasso
path; see [`ebic`](@ref).
"""
StatsAPI.bic(path::LassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}) =
  _criterion(path, X, y, (df, s) -> log(length(y)) * df)

######################################################################
#
#   Cross-validation of penalized generalized linear models
#
######################################################################

"""
    CVGLMLassoPath

Result returned by [`CVGLMLassoPath`](@ref)`(X, y, loss)`.

# Fields
- `path::GLMLassoPath`: the path on all the data
- `λpath::Vector`: its levels, the grid of every fold
- `cvm::Vector`, `cvse::Vector`: the cross-validated measure at each level, and
  its standard error (`cvm` is `NaN` at the levels that no fold reaches, `cvse`
  where fewer than two do)
- `nnz::Vector{Int}`: the number of nonzero coefficients of `path` at each level
- `λmin`, `λ1se`: among the levels that every fold reaches, the level with the
  smallest `cvm`, and the largest level whose `cvm` is within one standard error
  of it
- `folds::Vector{Int}`: the fold of each observation
- `measure::Symbol`: the measure of the held-out error
"""
struct CVGLMLassoPath{T<:AbstractFloat, L}
  path::GLMLassoPath{T, L}
  λpath::Vector{T}
  cvm::Vector{T}
  cvse::Vector{T}
  nnz::Vector{Int}
  λmin::T
  λ1se::T
  folds::Vector{Int}
  measure::Symbol
end

"""
    CVGLMLassoPath(X, y, loss = ProximalBase.Logistic(), options = CDOptions(); measure = :deviance,
                   nfolds = 10, folds = nothing, rng = Random.default_rng(), threads = true, nλ = 100,
                   λminratio, weights = nothing, offset = nothing, kwargs...)

K-fold cross-validation of the path of [`GLMLassoPath`](@ref) (with the same
loss, options and keywords, such as `intercept`, `standardize_x`, `alpha` or
`penalty`), as [`CVLassoPath`](@ref) does for least squares, with glmnet's
measures of the held-out error (`type.measure` of `cv.glmnet`):

- `:deviance`, the default: `2 ∑ᵢ wᵢ (ℓ(ηᵢ, yᵢ) - minᵤ ℓ(u, yᵢ)) / ∑ᵢ wᵢ`, for
  the logistic and Poisson losses;
- `:mse` and `:mae`: the mean squared and absolute errors of the predicted
  means `μᵢ` (probabilities for logistic regression);
- `:class`: the misclassification rate of `μᵢ > 1/2`, for logistic regression.

The path on all the data gives the levels; each fold's path is computed on the
other folds at those levels, with their observation `weights` and `offset`,
and the measure is computed on the held-out fold, weighted by its weights and
with its offset in the linear predictor. `cvm` averages the folds, weighted by
their sums of weights (their sizes without weights), and `cvse` is its standard
error, `λmin` minimizes `cvm` and `λ1se` is the largest level within `cvse` of
the minimum. A fold path that stops early (`max_hat_s`, `devmax`) leaves its
remaining levels out of the averages: `cvm` is `NaN` at the levels that no fold
reaches and `cvse` where fewer than two do, and `λmin` and `λ1se` are chosen
among the levels that every fold reaches. The folds and threads are as for
[`CVLassoPath`](@ref): the result does not depend on the number of threads.

Returns a [`CVGLMLassoPath`](@ref); `coef(cv)` and `predict(cv, X)` use the
solution at `λ1se`, or at `λmin` with `λ = :λmin`.
"""
function CVGLMLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T},
                        loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions();
                        measure::Symbol = :deviance, nfolds::Integer = 10,
                        folds::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                        rng::AbstractRNG = default_rng(), threads::Bool = true, nλ::Integer = 100,
                        λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                        offset::Union{Nothing, AbstractVector{<:Real}} = nothing,
                        kwargs...) where {T<:AbstractFloat}
  _checkmeasure(loss, measure, T)
  n = size(X, 1)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  fl, K = _foldlabels(rng, n, nfolds, folds)
  w = weights === nothing ? nothing : convert(Vector{T}, weights)
  o = offset === nothing ? nothing : convert(Vector{T}, offset)
  w === nothing || length(w) == n || throw(DimensionMismatch("weights has length $(length(w)), but X has $n rows"))
  o === nothing || length(o) == n || throw(DimensionMismatch("offset has length $(length(o)), but X has $n rows"))

  seeds = rand(rng, UInt64, K + 1)
  path = GLMLassoPath(X, y, loss, options; nλ, λminratio, weights = w, offset = o, rng = Xoshiro(seeds[end]),
                      kwargs...)
  λs = path.λpath
  errs = fill(T(NaN), K, length(λs))
  function fold!(k)
    train, test = findall(!=(k), fl), findall(==(k), fl)
    fp = GLMLassoPath(X[train, :], y[train], λs, loss, options; weights = _rows(w, train), offset = _rows(o, train),
                      rng = Xoshiro(seeds[k]), kwargs...)
    H = predict(fp, X[test, :]; type = :link, offset = _rows(o, test))
    for j in axes(H, 2)
      errs[k, j] = _wmean(i -> _cvmeasure(loss, measure, H[i, j], y[test[i]]), _rows(w, test), length(test))
    end
  end
  _runfolds(fold!, K, threads)
  cvm, cvse, jmin, j1se = _cvsummary(errs, _foldweights(T, fl, K, w))
  CVGLMLassoPath{T, typeof(loss)}(path, λs, cvm, cvse, [nnz(β) for β in path.βpath], λs[jmin], λs[j1se], fl,
                                  measure)
end

function CVGLMLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                        loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions();
                        kwargs...)
  T, Xf, yf = _float_data(X, y)
  CVGLMLassoPath(Xf, yf, loss, options; kwargs...)
end

function _checkmeasure(loss, measure::Symbol, ::Type{T}) where {T}
  measure in (:deviance, :mse, :mae, :class) ||
    throw(ArgumentError("measure must be :deviance, :mse, :mae or :class, but it is :$measure."))
  measure === :class && !(loss isa ProximalBase.Logistic) &&
    throw(ArgumentError("measure = :class needs logistic regression."))
  measure === :deviance && isnan(ProximalSolvers._saturated(loss, one(T))) &&
    throw(ArgumentError("the deviance of the loss $(typeof(loss)) is not known; use measure = :mse or :mae."))
  measure in (:mse, :mae) && _inverselink(loss, zero(T))      # throws when the mean of the response is not known
  nothing
end

# the held-out error of one observation with linear predictor η and response y
function _cvmeasure(loss, measure::Symbol, η::T, y::T) where {T}
  measure === :deviance && return 2 * (T(loss(η, y)) - ProximalSolvers._saturated(loss, y))
  μ = _inverselink(loss, η)
  measure === :mse && return abs2(y - μ)
  measure === :mae && return abs(y - μ)
  T((μ > 1 / 2) != (y > 1 / 2))                              # :class
end

"""
    coef(cv::CVGLMLassoPath; λ = :λ1se)
    predict(cv::CVGLMLassoPath, X; λ = :λ1se, type = :response, offset = nothing)

The coefficients (`[β₀; β]` with an intercept), and the predictions at the rows
of `X` on the scale of the response or of the linear predictor (see
[`glm_lasso`](@ref)), of the path on all the data at `cv.λ1se`, at `cv.λmin` with
`λ = :λmin`, or at a level `λ` within the path.
"""
StatsAPI.coef(cv::CVGLMLassoPath; λ::Union{Symbol, Real} = :λ1se) = coef(cv.path, _cvlevel(cv, λ))

function StatsAPI.predict(cv::CVGLMLassoPath, X::AbstractMatrix{<:Real}; λ::Union{Symbol, Real} = :λ1se,
                          type::Symbol = :response, offset = nothing)
  b = coef(cv; λ)
  p = length(b) - cv.path.intercept
  _checkcolumns(X, p)
  η = cv.path.intercept ? Vector(X * b[2:end]) .+ b[1] : Vector(X * b)
  _glmscale(cv.path.loss, _addoffset!(η, offset), type)
end
# CI scenario: a comment-only change to shared cross-validation code.
