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
# uncertifiable, not a zero gap. The arithmetic is in Float64, except in
# `lasso_gap_standardized` (below).
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
# with sⱼ the root mean square of the centred column xⱼ - x̄ⱼ. A column whose
# entries are all equal has that value as its mean, so that it centres to exact
# zeros and has scale 0 (the penalty ignores it); a constant response is centred
# in the same way. Means, centred columns, scales and the dual point are computed
# here in 256-bit BigFloat from the stored Float64 entries, with the terms added
# in a fixed order, so that no square underflows or overflows. Its primal value is
# evaluated on the raw data at (β₀, β) in Float64. The dual point comes from the
# centred and scaled problem; it is scaled down (rounding the scale down to a
# Float64) until |x̃ⱼ'θ| ≤ λ sⱼ holds for every column in the original coordinates,
# which is checked in BigFloat, and the certificate is :ok only if it holds. A
# column that is not all zeros after centring and whose scale is not a normal
# Float64 (it rounds to zero, is subnormal, or is not finite) makes the
# certificate :uncertifiable. The relative gap is as above. In exact arithmetic
# the profiled minimum over β₀ is the minimum of the centred problem, so the dual
# value bounds it from below; a perturbation of β₀ away from the profiled
# optimum for the fixed slopes raises the primal value and with it the gap
# (a perturbation toward it lowers it).

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

# The mean of v in BigFloat with the terms added in order; the value of a vector whose
# entries are all equal
function exact_mean(v)
  v1 = first(v)
  all(==(v1), v) && return BigFloat(v1)
  acc = zero(BigFloat)
  for t in v
    acc += BigFloat(t)
  end
  acc / length(v)
end

# The columns of X centred (in BigFloat, with the exact mean of a constant column) and
# their root mean squares sⱼ = sqrt(Σᵢ x̃ᵢⱼ²/n), which are 0 for a column of exact zeros
# (BigFloat has no underflow of squares at these magnitudes). Call inside
# `setprecision(BigFloat, 256)`.
function centered_columns(X)
  n, p = size(X)
  Xt = Matrix{BigFloat}(undef, n, p)
  s = zeros(BigFloat, p)
  for j in 1:p
    x = view(X, :, j)
    c = exact_mean(x)
    for i in 1:n
      Xt[i, j] = BigFloat(x[i]) - c
    end
    if any(!iszero, view(Xt, :, j))
      acc = zero(BigFloat)
      for i in 1:n
        acc += Xt[i, j]^2
      end
      s[j] = sqrt(acc / n)
    end
  end
  Xt, s
end

# A scale that is not a normal Float64 for a column that is not all zeros: its index, or 0
function bad_scale(Xt, s)
  for j in eachindex(s)
    (iszero(s[j]) && any(!iszero, view(Xt, :, j))) && return j     # cannot happen in BigFloat
    iszero(s[j]) && continue
    sf = Float64(s[j])
    (isfinite(sf) && sf >= floatmin(Float64)) || return j
  end
  0
end

# The columns of X centred and divided by their scales (zero for a column of scale 0) and the
# scales, as Float64, from centered_columns; an error if a scale is not a normal Float64
function centered_scaled(X)
  setprecision(BigFloat, 256) do
    Xt, s = centered_columns(X)
    j = bad_scale(Xt, s)
    j == 0 || error("the scale of column $j is not a normal Float64")
    Z = Float64.(Xt ./ transpose(ifelse.(iszero.(s), one(BigFloat), s)))
    Z, Float64.(s)
  end
end

# P = ‖r‖²/(2n) + λ Σⱼ sⱼ|βⱼ| for r = y - β₀ - Xβ, from the raw data in Float64. The dual
# point is built from the centred residual u = r - r̄ (which sums to 0 in exact arithmetic):
# θ = u/n, scaled to tθ with the largest t ≤ 1 (rounded down to a Float64) such that
# |x̃ⱼ'(tθ)| ≤ λ sⱼ for every column j, the constraint in the original coordinates (the same as
# |Zⱼ'(tθ)| ≤ λ for the scaled columns Zⱼ = x̃ⱼ/sⱼ). With the centred response ỹ the dual value
# is D = t ỹ'θ - t² n‖θ‖²/2, a lower bound on the minimum of the centred problem by weak
# duality. The sums for D, t and the feasibility `feasibility` = maxⱼ |x̃ⱼ'(tθ)|/(λ sⱼ) are
# formed in BigFloat; `feasibility` ≤ 1 for a dual feasible point. Returns a String with the
# reason when there is no certificate.
function lasso_standardized_parts(X, y, β₀, β, λ)
  d = cert_regression(X, y, β)
  d isa String && return d
  cert_finite(β₀) || return "nonfinite intercept"
  cert_finite(λ) && λ > 0 || return "λ must be finite and positive"
  (; X, y, β, n, p) = d
  β₀, λ = f64(β₀), f64(λ)
  setprecision(BigFloat, 256) do
    Xt, sb = centered_columns(X)
    j = bad_scale(Xt, sb)
    j == 0 || return "the scale of column $j of X is not a normal Float64 (zero, subnormal or not finite)"
    s = Float64.(sb)
    fit = X * β
    r = y .- β₀ .- fit
    penalty = λ * dot(s, abs.(β))
    P = dot(r, r) / (2n) + penalty
    cert_finite(P, r) || return "nonfinite primal value"
    ỹ = BigFloat.(y) .- exact_mean(y)
    rb = BigFloat.(r)
    θ = (rb .- exact_mean(r)) ./ n
    g = [sum(Xt[i, j] * θ[i] for i in 1:n) for j in 1:p]      # x̃ⱼ'θ, in the order of i
    t = one(BigFloat)
    for j in 1:p
      iszero(sb[j]) && continue
      abs(g[j]) > λ * sb[j] && (t = min(t, BigFloat(λ) * sb[j] / abs(g[j])))
    end
    tf = Float64(t, RoundDown)
    feasibility = maximum(j -> iszero(sb[j]) ? zero(BigFloat) : tf * abs(g[j]) / (BigFloat(λ) * sb[j]), 1:p)
    θt = tf .* θ
    D = tf * sum(ỹ[i] * θ[i] for i in 1:n) - tf^2 * n * sum(abs2, θ) / 2
    scale = Float64(sum((abs(BigFloat(y[i])) + abs(β₀) + abs(fit[i]))^2 for i in 1:n) / (2n)) + penalty +
            Float64(tf * sum(abs(ỹ[i]) * abs(θ[i]) for i in 1:n) + tf^2 * n * sum(abs2, θ) / 2)
    (; P, D = Float64(D), θ = Float64.(θt), t = tf, feasibility = Float64(feasibility), scale, terms = max(n, p))
  end
end

function lasso_gap_standardized(X, y, β₀, β, λ)
  a = lasso_standardized_parts(X, y, β₀, β, λ)
  a isa String && return uncertifiable(a)
  a.feasibility <= 1 + 1e-12 || return uncertifiable("the dual point is not feasible: maxⱼ |x̃ⱼ'θ|/(λ sⱼ) = $(a.feasibility)")
  relative_gap(a.P, a.D; a.scale, a.terms)
end

function lasso_gap(X, y, β, λ)
  a = lasso_parts(X, y, β, λ)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end
