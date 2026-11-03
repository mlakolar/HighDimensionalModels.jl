######################################################################
#
#   Penalized generalized linear models
#
######################################################################

"""
    GLMLassoSolution

Result returned by [`glm_lasso`](@ref).

# Fields
- `x::SparseIterate`: the estimated coefficients `β`
- `intercept`: the unpenalized intercept `β₀`, zero unless the solver was
  called with `intercept = true`
- `penalty`: the penalty at which `x` was computed: `NormL1(λ, ω)`, the penalty
  of `alpha` or `penalty` at the level `λ`, or either on the box of `lower` and
  `upper`, `Constrained(g, IndBox(l, u))`
- `loss`: the scalar loss of the model, such as `ProximalBase.Logistic()`
- `linearpredictor::Vector`: `η = β₀ + Xβ + offset`
- `deviance`: `2 ∑ᵢ wᵢ (ℓ(ηᵢ, yᵢ) - minᵤ ℓ(u, yᵢ))`, twice the negative
  log-likelihood ratio against the saturated model for the logistic and Poisson
  losses, with the observation weights `wᵢ`
- `nulldeviance`: the deviance of the fit with `β = 0` (the intercept and the
  offset only); `1 - deviance / nulldeviance` is the fraction of the deviance
  that the fit explains
- `iterations::Int`: passes of coordinate descent, summed over the proximal
  Newton steps
- `converged::Bool`: whether proximal Newton, and coordinate descent at each of
  its steps, met their stopping rules
- `hasintercept::Bool`: whether the solver fitted an intercept
- `scales`: `nothing` when `penalty` applies to `x`, or the scales `s` of the
  columns when it applies to `s .* x` (`standardize_x` with a penalty other
  than the ℓ1 norm), as for [`LassoSolution`](@ref)

The functions of StatsAPI apply to it: [`coef`](@ref), [`predict`](@ref) (on
the scale of the response or of the linear predictor), `fitted`,
[`nobs`](@ref), [`dof`](@ref), [`deviance`](@ref) and `nulldeviance`.
"""
struct GLMLassoSolution{T, L, S}
  x::SparseIterate{T, 1}
  intercept::T
  penalty::S
  loss::L
  linearpredictor::Vector{T}
  deviance::T
  nulldeviance::T
  iterations::Int
  converged::Bool
  hasintercept::Bool
  scales::Union{Nothing, Vector{T}}
end

# the responses that the built-in scalar losses accept
_check_response(::ProximalBase.Logistic, y) =
  all(t -> 0 <= t <= 1, y) || throw(ArgumentError("logistic regression needs responses in [0, 1]"))
_check_response(::ProximalBase.Poisson, y) =
  all(t -> t >= 0, y) || throw(ArgumentError("Poisson regression needs nonnegative responses"))
_check_response(::ProximalBase.Squared, y) =
  throw(ArgumentError("the squared loss is least squares, which proximal Newton does not take: use lasso"))
_check_response(loss, y) = true

# The model of ∑ᵢ wᵢ ℓ(β₀ + xᵢ'β + oᵢ, yᵢ) / ∑ᵢ wᵢ for proximal Newton, and the
# penalty loadings
function _glm_model(X::AbstractMatrix{T}, y::AbstractVector{T}, loss, weights, offset, intercept::Bool,
                   standardize_x::Bool) where {T}
  n, p = size(X)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  _check_finite(X, "X"); _check_finite(y, "y")
  _check_response(loss, y)
  w = weights === nothing ? nothing : convert(Vector{T}, weights)
  o = offset === nothing ? nothing : convert(Vector{T}, offset)
  w === nothing || (_check_finite(w, "weights"); all(>=(0), w) && sum(w) > 0 ||
                    throw(ArgumentError("the weights must be nonnegative, with a positive sum")))
  o === nothing || _check_finite(o, "offset")
  sw = w === nothing ? T(n) : sum(w)
  f = inv(sw) * GLMLoss(loss, y, X; weights = w, offset = o)
  m = ProximalSolvers.cdmodel(f, SparseIterate(T, p); intercept)
  m, (standardize_x ? _loadings(X, intercept, w) : ones(T, p))
end

# The model, the penalty at level one and the column scales of a penalized GLM,
# as _path_problem builds them for least squares: the ℓ1 norm with the loadings
# of standardize_x as weights, or another penalty on the columns divided by
# their scales (nothing without standardize_x); either on the box of lower and
# upper.
function _glm_problem(X::AbstractMatrix{T}, y, loss, weights, offset, intercept::Bool, standardize_x::Bool,
                     template, lower, upper) where {T}
  p = size(X, 2)
  _llaonly(template) &&
    throw(ArgumentError("$(nameof(typeof(template))) needs local linear approximation, which the GLM fits do not " *
                        "offer; lasso, LassoPath and CVLassoPath take it for least squares."))
  if _isl1(template)
    m, ω = _glm_model(X, y, loss, weights, offset, intercept, standardize_x)
    return m, _constrain(_l1penalty(template, one(T), ω), _box(T, lower, upper, p)), nothing
  end
  _check_finite(X, "X")
  s = standardize_x ? _scales(X, intercept, weights === nothing ? nothing : convert(Vector{T}, weights)) : nothing
  m, _ = _glm_model(_scale_columns(X, s), y, loss, weights, offset, intercept, false)
  m, _constrain(template, _box(T, lower, upper, p, s)), s
end

_newton(options::CDOptions, newton_tol, newton_max_iter; rng::AbstractRNG = default_rng()) =
  ProxNewton(; inner = _solver(options; rng), tol = newton_tol, maxiter = newton_max_iter)

"""
    glm_lasso(X, y, λ, loss = ProximalBase.Logistic(), options = CDOptions();
             intercept = true, standardize_x = true, weights = nothing, offset = nothing,
             alpha = 1, penalty = nothing, lower = -Inf, upper = Inf, newton_tol = 1e-8, newton_max_iter = 100,
             rng = Random.default_rng())

Solve the penalized generalized linear model

    minimize  ∑ᵢ wᵢ ℓ(β₀ + xᵢ'β + oᵢ, yᵢ) / ∑ᵢ wᵢ + λ ∑ⱼ ωⱼ |βⱼ|

for a scalar loss `ℓ` of ProximalBase: `ProximalBase.Logistic()`, the negative
log-likelihood of logistic regression (`yᵢ ∈ [0, 1]`, usually `{0, 1}`);
`ProximalBase.Poisson()`, Poisson regression with the log link (`yᵢ ≥ 0`);
`ProximalBase.Huber(δ)`; or a loss of your own. The observation weights `wᵢ`
(nonnegative, one when omitted) and the offset `oᵢ` (zero when omitted) are
those of `weights` and `offset`. The intercept `β₀` is not penalized, and it is
zero unless `intercept` is true. With `standardize_x` the loadings `ωⱼ` are the
standard deviations of the columns, weighted by `w` and centered at their
weighted means with an intercept, so that the coefficients are on the scale of
the data; otherwise `ωⱼ = 1`. These are glmnet's conventions and defaults.
`lower` and `upper` bound the coefficients, `lower ≤ β ≤ upper`, as for
[`lasso`](@ref).

`alpha` and `penalty` give the elastic net and the other penalties, as for
[`lasso`](@ref): `alpha < 1` is glmnet's elastic net
`λ (α ∑ⱼ |βⱼ| + (1 - α) ‖β‖² / 2)`, and `penalty` a penalty of ProximalBase
whose level `λ` replaces (`ElasticNet(1, λ2)`, `MCP(1, a)`, `SCAD(1, a)`,
`NormL1Nonneg(1)`, or `NormL1(1, w)` for weights). With `standardize_x`, such a
penalty applies to the coefficients of the standardized columns, as in glmnet;
the coefficients returned are those of the columns of `X`, and the scales are
stored in the solution's `scales`. The elastic net equals glmnet's `binomial`
and `poisson` fits. A nonconvex penalty (MCP, SCAD) is computed along
`options.num_steps` levels from λmax whatever `options.warm_start`, as ncvreg
does; `num_steps = 0` solves at `λ` from zero. The solutions are stationary
points of the objective above. ncvreg's `binomial` and `poisson` families do
not solve it: their coordinate steps scale the concavity of the penalty by the
weighted curvature of each column at the current fit (they divide by
`v (1 - 1/a)` where the exact step divides by `v - 1/a`), so their solutions
differ (in its gaussian family `v = 1` and the two agree).

The problem is solved by proximal Newton (ProximalSolvers' `ProxNewton`, the
method of glmnet): each step minimizes the weighted least-squares model of the
loss at the current fit by coordinate descent with `options`, then backtracks,
until the objective changes by less than `newton_tol` relative to its value or
after `newton_max_iter` steps. With `options.warm_start` false (or a nonconvex
penalty), the solution is computed along `options.num_steps` levels from λmax (see
[`GLMLassoPath`](@ref)), each warm started from the previous one; otherwise it
is computed directly, starting from `β = 0` and the intercept of the fit with
`β = 0`. The random order of coordinate descent is drawn from `rng`. Returns a
[`GLMLassoSolution`](@ref).

```julia
y = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-X * β)))
sol = glm_lasso(X, y, 0.05)                               # logistic
sol = glm_lasso(X, counts, 0.05, ProximalBase.Poisson(); offset = log.(exposure))
```
"""
function glm_lasso(X::AbstractMatrix{T}, y::AbstractVector{T}, λ::T,
                  loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions();
                  intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing,
                  weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                  offset::Union{Nothing, AbstractVector{<:Real}} = nothing,
                  alpha::Real = 1, penalty::Union{Nothing, AbstractFunction} = nothing,
                  lower::Union{Real, AbstractVector{<:Real}} = -Inf, upper::Union{Real, AbstractVector{<:Real}} = Inf,
                  newton_tol::Union{Nothing, Real}=nothing, newton_max_iter::Union{Nothing, Integer}=nothing,
                  rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  newton_tol, kwargs = _legacy_keyword(newton_tol, kwargs, Val(:newtonTol), Val(:newton_tol), 1e-8)
  newton_tol::Real
  newton_max_iter, kwargs = _legacy_keyword(newton_max_iter, kwargs, Val(:newtonMaxIter), Val(:newton_max_iter), 100)
  newton_max_iter::Integer
  _check_keywords(CDOptions, (), kwargs)
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  p = size(X, 2)
  m, g1, s = _glm_problem(X, y, loss, weights, offset, intercept, standardize_x, _template(T, penalty, alpha),
                         lower, upper)
  method = _newton(options, newton_tol, newton_max_iter; rng)
  x = SparseIterate(T, p)
  g = _atlevel(g1, λ)
  # a path from λmax when warm_start is false, and always for a nonconvex penalty
  usepath = options.num_steps > 0 && !(isconvex(g) && options.warm_start)
  λmax = usepath ? lambda_max(m, g1, x) : zero(T)
  if !usepath || λ >= λmax
    ProximalSolvers.initialize!(m, x)            # the intercept of the fit with β = 0
    r = solve!(x, m, g, method)
    iterations, converged = r.passes, r.converged
  else
    λs = [λmax * (λ / λmax)^(T(k) / options.num_steps) for k in 0:options.num_steps]
    λs[end] = λ
    path = solve_path(m, g1, λs, method; x, devmax = 1, _gapkw(options)...)
    iterations, converged = sum(path.passes), last(path.converged)
  end
  β0 = ProximalSolvers.intercept(m, x)
  s === nothing || _unscale!(x, s)
  GLMLassoSolution{T, typeof(loss), typeof(g)}(x, β0, g, loss, copy(ProximalSolvers.linear_predictor(m)),
                                               ProximalSolvers.deviance(m), ProximalSolvers.nulldeviance(m),
                                               iterations, converged, intercept, s)
end

function glm_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
                  loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  glm_lasso(Xf, yf, T(λ), loss, options; kwargs...)
end

"""
    GLMLassoPath(X, y, loss = ProximalBase.Logistic(), options = CDOptions(); nλ = 100, λminratio, kwargs...)
    GLMLassoPath(X, y, λpath, loss = ProximalBase.Logistic(), options = CDOptions();
                 intercept = true, standardize_x = true, weights = nothing, offset = nothing,
                 alpha = 1, penalty = nothing, lower = -Inf, upper = Inf, screening = true, max_hat_s = Inf,
                 devmax = 0.999, newton_tol = 1e-8, newton_max_iter = 100, rng = Random.default_rng())

Solutions `β0path[k]` and `βpath[k]` of the penalized generalized linear model
of [`glm_lasso`](@ref) along a decreasing sequence `λpath`, each one warm
started from the previous solution and its intercept, by proximal Newton with
the inner coordinate descent of `options` (ProximalSolvers' `solve_path` with
`ProxNewton`).

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero (at the intercept of the fit
with `β = 0`), down to `λminratio ⋅ λmax`; `λminratio` defaults to `0.01` when
`X` has more columns than rows and to `1e-4` otherwise. The first value is
`λmax` times `1 + √eps`, so that rounding cannot make the first solution
nonzero.

With `screening`, each λ is solved first over the coordinates kept by the
sequential strong rule of Tibshirani et al. (2012), and the optimality
conditions of the others are checked afterwards (`kkt_checks[k]` times at
`λpath[k]`), as in [`LassoPath`](@ref). The path stops early after the first
solution with more than `max_hat_s` nonzero coefficients, or once the fit
explains a fraction `devmax` of the null deviance, as glmnet's does: along a
logistic path on separable data the coefficients grow without bound, and the
explained fraction `devratio[k] = 1 - deviance / nulldeviance` tends to one.
`devmax = 1` turns this off. The returned `λpath` then contains only the values
that were solved for. `iterations[k]` and `converged[k]` are the passes of
coordinate descent at `λpath[k]`, summed over the proximal Newton steps, and
whether the solve met its stopping rules. `nobs` is the number of observations.

`alpha`, `penalty`, `lower` and `upper` are as for [`glm_lasso`](@ref); along a
path of a nonconvex penalty, each solution is the local one reached from the
previous solution. `penalty` holds the penalty at level one (`NormL1(1, ω)`
for the lasso), applied to `scales .* β` when `scales` is not `nothing`.
`coef`, `predict` and `nobs` apply to the path, as for [`LassoPath`](@ref), and
`fit(GLMLassoPath, X, y, ...)` is `GLMLassoPath(X, y, ...)`.
"""
struct GLMLassoPath{T<:AbstractFloat, L}
  λpath::Vector{T}
  βpath::Vector{SparseVector{T, Int}}
  β0path::Vector{T}
  devratio::Vector{T}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kkt_checks::Vector{Int}
  intercept::Bool
  loss::L
  nulldeviance::T
  nobs::Int
  penalty::AbstractFunction
  scales::Union{Nothing, Vector{T}}
end

function GLMLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, λpath::Vector{T},
                      loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions();
                      intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing,
                      weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      offset::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      alpha::Real = 1, penalty::Union{Nothing, AbstractFunction} = nothing,
                      lower::Union{Real, AbstractVector{<:Real}} = -Inf, upper::Union{Real, AbstractVector{<:Real}} = Inf,
                      screening::Bool = true, max_hat_s = Inf, devmax::Real = 0.999,
                      newton_tol::Union{Nothing, Real}=nothing, newton_max_iter::Union{Nothing, Integer}=nothing,
                      rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  p = size(X, 2)
  m, g, s = _glm_problem(X, y, loss, weights, offset, intercept, standardize_x, _template(T, penalty, alpha),
                        lower, upper)
  _glmlassopath(m, g, s, p, λpath, loss, options, size(X, 1), intercept;
                screening, max_hat_s, devmax, newton_tol, newton_max_iter, rng, kwargs...)
end

# Both constructors solve the problem they prepared; the automatic grid also
# uses this model to find λmax, without constructing and scaling it again.
function _glmlassopath(m, g, s, p::Int, λpath::Vector{T}, loss, options::CDOptions, n::Int, intercept::Bool;
                       screening::Bool = true, max_hat_s = Inf, devmax::Real = 0.999,
                       newton_tol::Union{Nothing, Real} = nothing,
                       newton_max_iter::Union{Nothing, Integer} = nothing,
                       rng::AbstractRNG = default_rng(), kwargs...) where {T}
  newton_tol, kwargs = _legacy_keyword(newton_tol, kwargs, Val(:newtonTol), Val(:newton_tol), 1e-8)
  newton_tol::Real
  newton_max_iter, kwargs = _legacy_keyword(newton_max_iter, kwargs, Val(:newtonMaxIter), Val(:newton_max_iter), 100)
  newton_max_iter::Integer
  _check_keywords(CDOptions, (), kwargs)
  D0 = ProximalSolvers.nulldeviance(m)
  β0path, devratio = T[], T[]
  record(k, λ, x, m) = (push!(β0path, ProximalSolvers.intercept(m, x));
                        push!(devratio, 1 - ProximalSolvers.deviance(m) / D0); false)
  path = solve_path(m, g, λpath, _newton(options, newton_tol, newton_max_iter; rng);
                    x = SparseIterate(T, p), screening, maxnnz = max_hat_s, devmax, callback = record,
                    _gapkw(options)...)
  βpath = s === nothing ? path.solutions : [_unscale(β, s) for β in path.solutions]
  GLMLassoPath{T, typeof(loss)}(path.λ, βpath, β0path, devratio, path.passes, path.converged,
                                path.kkt_checks, intercept, loss, D0, n, g, s)
end

function GLMLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T},
                      loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions();
                      nλ::Integer = 100, λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                      intercept::Bool = true, standardize_x::Union{Nothing, Bool}=nothing,
                      weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      offset::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      alpha::Real = 1, penalty::Union{Nothing, AbstractFunction} = nothing,
                      lower::Union{Real, AbstractVector{<:Real}} = -Inf, upper::Union{Real, AbstractVector{<:Real}} = Inf,
                      kwargs...) where {T<:AbstractFloat}
  standardize_x, kwargs = _legacy_keyword(standardize_x, kwargs, Val(:standardizeX), Val(:standardize_x), true)
  standardize_x::Bool
  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  p = size(X, 2)
  m, g, s = _glm_problem(X, y, loss, weights, offset, intercept, standardize_x, _template(T, penalty, alpha),
                        lower, upper)
  λmax = lambda_max(m, g, SparseIterate(T, p)) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  _glmlassopath(m, g, s, p, λpath, loss, options, size(X, 1), intercept; kwargs...)
end

function GLMLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λpath::AbstractVector{<:Real},
                      loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  GLMLassoPath(Xf, yf, convert(Vector{T}, λpath), loss, options; kwargs...)
end

function GLMLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                      loss::ProximalBase.ScalarLoss = ProximalBase.Logistic(), options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  GLMLassoPath(Xf, yf, loss, options; kwargs...)
end
# CI scenario: a comment-only change to the GLM source.
