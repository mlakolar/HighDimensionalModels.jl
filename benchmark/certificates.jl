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
# uncertifiable, never a zero gap. All arithmetic is in Float64.
#
# Relative gap (measure :relative_gap): (P - D)/max(|P|, floatmin) for the
# primal value P at the returned point and a dual (lower) bound D. With the
# rounding bound tol = 16 γₘ (|P| + |D| + scale), γₘ = m eps/(1 - m eps) for
# sums of m terms and scale the sum of the magnitudes of their terms,
# P - D < -tol is a weak-duality violation, which makes the case
# uncertifiable, and -tol ≤ P - D < 0 counts as a zero gap.
#
# With λ = 0 the dual has equality constraints that no scaled point meets, so
# the certificate reports the KKT residual (measure :kkt_residual), relative to
# max(1, ‖X'y‖∞/n), instead of a gap.

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

# P = ‖r‖²/(2n) + λ‖β‖₁ for r = y - Xβ. The dual point is θ = t r/n with
# t = min(1, λ/‖X'r/n‖∞), so that ‖X'θ‖∞ ≤ λ, and the dual value is
# t y'θ - t²‖r‖²/(2n), a lower bound on the minimum by weak duality.
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

function lasso_gap(X, y, β, λ)
  a = lasso_parts(X, y, β, λ)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end
