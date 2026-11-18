# Accuracy certificate of the benchmark cases, computed from the inputs and the
# returned coefficients. It states the objective of `lasso`,
#
#   ‖y - Xβ‖²/(2n) + λ‖β‖₁,
#
# evaluates it at the returned coefficients, and bounds the distance to the
# minimum by a duality gap, independently of the solver's stopping rule.
# Nothing here calls HighDimensionalModels, ProximalBase or ProximalSolvers.
#
# A certificate returns (; value, measure, status, note), with status :ok or
# :uncertifiable (value NaN, and a note saying why). Nonfinite inputs are
# uncertifiable, not a zero gap. All arithmetic is in Float64.
#
# Relative gap (measure :relative_gap): (P - D)/max(|P|, floatmin) for the
# primal value P at the returned point and a dual (lower) bound D. With the
# rounding bound tol = 16 γₘ (|P| + |D| + scale), γₘ = m eps/(1 - m eps) for
# sums of m terms and scale the sum of the magnitudes of their terms,
# P - D < -tol is a weak-duality violation, which makes the case
# uncertifiable, and -tol ≤ P - D < 0 counts as a zero gap.
#
# With λ = 0 the certificate does not use a duality gap: it reports the KKT
# residual ‖X'r/n‖∞ (measure :kkt_residual), relative to max(1, ‖X'y‖∞/n).
#
# `lasso_gap_standardized` is the certificate for the fit with an intercept and
# column scales (λ > 0), at the stored intercept β₀ and coefficients β:
#
#   ‖y - β₀1 - Xβ‖²/(2n) + λ Σⱼ sⱼ|βⱼ|,
#
# with sⱼ the root mean square of the centred column xⱼ - x̄ⱼ (0 for a column of
# exact zeros after centring, which the penalty ignores). Its primal value is
# evaluated on the raw data at (β₀, β); its dual point comes from the centred and
# scaled problem, computed here with sums in a fixed order. The relative gap is
# as above. The profiled minimum over β₀ is the minimum of the centred problem,
# so the dual value bounds it from below, and a perturbed β₀ raises the primal
# value and with it the gap.

using LinearAlgebra

const CERTIFICATE_MEASURES = (:relative_gap, :kkt_residual)

cert_ok(value, measure; note = "") = (; value = Float64(value), measure, status = :ok, note = String(note))
uncertifiable(note, measure = :relative_gap) = (; value = NaN, measure, status = :uncertifiable, note = String(note))

cert_finite(x::Number) = isfinite(x)
cert_finite(x::AbstractArray) = all(isfinite, x)
cert_finite(xs...) = all(cert_finite, xs)

f64(x::Number) = Float64(x)
f64(x::AbstractArray) = Array{Float64}(x)

cert_maxabs(x) = maximum(abs, x; init = 0.0)

function relative_gap(P, D; scale = 0.0, terms::Integer = 1)
  cert_finite(P, D, scale) && scale >= 0 || return uncertifiable("nonfinite primal or dual value")
  γ = terms * eps() / (1 - terms * eps())
  tol = 16γ * (abs(P) + abs(D) + scale)
  P - D < -tol && return uncertifiable("weak duality violation: P - D = $(P - D) < -$tol")
  cert_ok(max(P - D, 0.0) / max(abs(P), floatmin(Float64)), :relative_gap)
end

# a residual relative to max(1, g0), g0 = ‖X'y‖∞/n
relative_residual(value, g0, measure) =
  cert_finite(value, g0) ? cert_ok(value / max(1.0, g0), measure) : uncertifiable("nonfinite residual", measure)

# the largest t ≤ 1 with |t zⱼ| ≤ c for every j (c ≥ 0)
function box_scale(z, c)
  t = 1.0
  for j in eachindex(z)
    abs(z[j]) > 0 && (t = min(t, c / abs(z[j])))
  end
  t
end

# The distance of -g from the subdifferential of λ‖·‖₁ at β, the largest over
# the coordinates
function l1_kkt(g, β, λ)
  maximum((β[j] == 0 ? max(abs(g[j]) - λ, 0.0) : abs(g[j] + λ * sign(β[j])) for j in eachindex(g, β));
          init = 0.0)
end

# The inputs of the certificate in Float64, or a note saying why there is none
function cert_regression(X, y, β)
  X isa AbstractMatrix && y isa AbstractVector && β isa AbstractVector ||
    return "X must be a matrix, y and β vectors"
  size(X) == (length(y), length(β)) || return "the dimensions of X, y and β disagree"
  size(X, 1) >= 1 || return "X has no rows"
  cert_finite(X, y, β) || return "nonfinite input"
  (; X = f64(X), y = f64(y), β = f64(β), n = length(y), p = length(β))
end

# P = ‖r‖²/(2n) + λ‖β‖₁ for r = y - Xβ. With θ = r/n, the dual point is tθ,
# t = min(1, λ/‖X'θ‖∞), so that ‖X'(tθ)‖∞ ≤ λ, and the dual value is
# D = t y'θ - t² n‖θ‖²/2, a lower bound on the minimum by weak duality; the
# returned θ is tθ. For X = I (n = 2), y = (2, -1), β = 0 and λ = 0.5: θ = (1, -0.5),
# t = 0.5, D = 0.5·2.5 - 0.25·1.25 = 0.9375.
function lasso_parts(X, y, β, λ)
  d = cert_regression(X, y, β)
  d isa String && return d
  cert_finite(λ) && λ >= 0 || return "λ must be finite and nonnegative"
  (; X, y, β, n, p) = d
  λ = f64(λ)
  fit = X * β
  r = y .- fit
  penalty = λ * sum(abs, β)
  P = dot(r, r) / (2n) + penalty
  if λ == 0
    g = -(transpose(X) * r) ./ n
    return (; P, kkt = l1_kkt(g, β, 0.0), g0 = cert_maxabs(transpose(X) * y) / n)
  end
  θ = r ./ n
  t = box_scale(transpose(X) * θ, λ)
  D = t * dot(y, θ) - t^2 * dot(r, r) / (2n)
  magnitude = abs.(y) .+ abs.(fit)
  scale = dot(magnitude, magnitude) / (2n) + penalty + t * dot(abs.(y), abs.(θ)) + t^2 * dot(r, r) / (2n)
  (; P, D, θ = t .* θ, t, scale, terms = max(n, p), kkt = nothing)
end

# The mean of v with the terms added in order
function fixed_mean(v)
  s = 0.0
  for t in v
    s += t
  end
  s / length(v)
end

# The columns of X centred, divided by their root mean squares (columns that are
# exact zeros after centring stay zero, with scale 0), and the scales; the sums
# add in a fixed order
function centered_scaled(X)
  Z = similar(X)
  s = zeros(size(X, 2))
  for j in axes(X, 2)
    x = view(X, :, j)
    c = fixed_mean(x)
    z = x .- c
    s[j] = sqrt(fixed_mean(abs2.(z)))
    Z[:, j] = s[j] > 0 ? z ./ s[j] : zero(z)
  end
  Z, s
end

# P = ‖r‖²/(2n) + λ Σⱼ sⱼ|βⱼ| for r = y - β₀ - Xβ, from the raw data. The dual point is
# built from the centred residual u = r - r̄ (which sums to 0): with θ = u/n and
# t = min(1, λ/‖Z'θ‖∞) for the scaled centred columns Z, it is tθ, and with the centred
# response ỹ the dual value is D = t ỹ'θ - t² n‖θ‖²/2, a lower bound on the minimum of the
# centred problem by weak duality.
function lasso_standardized_parts(X, y, β₀, β, λ)
  d = cert_regression(X, y, β)
  d isa String && return d
  cert_finite(β₀) || return "nonfinite intercept"
  cert_finite(λ) && λ > 0 || return "λ must be finite and positive"
  (; X, y, β, n, p) = d
  β₀, λ = f64(β₀), f64(λ)
  Z, s = centered_scaled(X)
  fit = X * β
  r = y .- β₀ .- fit
  penalty = λ * dot(s, abs.(β))
  P = dot(r, r) / (2n) + penalty
  u = r .- fixed_mean(r)
  θ = u ./ n
  ỹ = y .- fixed_mean(y)
  t = box_scale(transpose(Z) * θ, λ)
  D = t * dot(ỹ, θ) - t^2 * dot(u, u) / (2n)
  magnitude = abs.(y) .+ abs(β₀) .+ abs.(fit)
  scale = dot(magnitude, magnitude) / (2n) + penalty + t * dot(abs.(ỹ), abs.(θ)) + t^2 * dot(u, u) / (2n)
  (; P, D, θ = t .* θ, t, scale, terms = max(n, p))
end

function lasso_gap_standardized(X, y, β₀, β, λ)
  a = lasso_standardized_parts(X, y, β₀, β, λ)
  a isa String && return uncertifiable(a)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

function lasso_gap(X, y, β, λ)
  a = lasso_parts(X, y, β, λ)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end
