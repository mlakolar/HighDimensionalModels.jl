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
# `lasso_gap_standardized` (below), which uses 256-bit BigFloat with error bounds.
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
# in the same way. The primal and dual core, from the stored Float64 inputs (converted
# exactly) to the verdict, is arithmetic in 256-bit BigFloat with a hand-derived
# absolute error bound for each computed quantity; the error model is written above
# `lasso_standardized_parts`. It takes the target of the case as a required keyword and
# computes an upper bound value_hi and a lower bound value_lo of the relative gap
# (P - D)/max(P, floatmin) from the bounds of P and D. It is :uncertifiable when
# value_lo ≤ target < value_hi (the rounding uncertainty could change pass/fail), and
# otherwise returns value_hi, so that the harness verdict is the true one. It is also
# :uncertifiable when the dual point cannot be shown feasible in the original coordinates
# (|x̃ⱼ'θ| ≤ λ sⱼ in exact arithmetic), when the error bound of the gap relative to the
# denominator exceeds 1e-8, and, for a column that is not constant, when its scale is not
# a normal Float64 or is not above its error bound. The Float64 steps outside the core are
# the validation of the scales, the report of P, D, θ and t, and the rounding of the two
# bounds (up for value_hi, down for value_lo). In exact arithmetic the profiled minimum over
# β₀ is the minimum of the centred problem, so the dual value bounds it from below; a
# perturbation of β₀ away from the profiled optimum for the fixed slopes raises the primal
# value and with it the absolute duality gap, and a perturbation toward it lowers them (the
# relative gap need not change in the same direction).

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

# ---------------------------------------------------------------------------------------------
# The standardized fit with an intercept: arithmetic and error model
# ---------------------------------------------------------------------------------------------
#
# All of `lasso_standardized_parts` after the stored Float64 inputs are read (they are converted
# exactly) is arithmetic in 256-bit BigFloat, round to nearest, with unit roundoff u = 2⁻²⁵⁶. Each
# computed quantity carries an absolute error bound, in a plain variable beside its value (a hat
# marks a computed value, a plain symbol the exact one defined by the stored inputs). The bounds
# use G(k) = 8 k u, which is at least γ_k (1 + γ_k)² for γ_k = k u/(1 - k u) and k u ≤ 1/8: this
# also absorbs the relative rounding of the computed magnitudes and of the bound arithmetic.
#   * Sum of k terms tᵢ in any order: |fl(Σ tᵢ) - Σ tᵢ| ≤ G(k) Σ|tᵢ|; a dot product of k pairs:
#     |fl(aᵀb) - aᵀb| ≤ G(k) Σ|aᵢ||bᵢ|. The magnitudes Σ|tᵢ|, Σ|aᵢ||bᵢ|, not the computed sum,
#     carry the cancellation.
#   * Operands with errors |âᵢ - aᵢ| ≤ eaᵢ, |b̂ᵢ - bᵢ| ≤ ebᵢ: |âᵀb̂ - aᵀb| ≤
#     Σ(|âᵢ| ebᵢ + |b̂ᵢ| eaᵢ + eaᵢ ebᵢ); a square: |â² - a²| ≤ ea (2|â| + ea).
#   * One rounded operation x ∘ y: relative error u, covered by G(1).
# The chain, with n rows and p columns:
#   means x̄ⱼ (the stored entries of a column that are all equal give that value, exactly):
#     ex̄ⱼ = G(n+1) Σᵢ|xᵢⱼ|/n; centred entries x̂ᵢⱼ = fl(xᵢⱼ - x̄̂ⱼ): |x̂ᵢⱼ - x̃ᵢⱼ| ≤ εxⱼ =
#     ex̄ⱼ + G(1) maxᵢ|x̂ᵢⱼ|;
#   scales sⱼ = ‖x̃ⱼ‖/√n (0 for a constant column): |‖x̂ⱼ‖ - ‖x̃ⱼ‖| ≤ √n εxⱼ by the triangle
#     inequality, and the computed fl(sqrt(fl(Σ x̂²)/n)) is within a relative G(n+2) of ‖x̂ⱼ‖/√n, so
#     esⱼ = εxⱼ + G(n+2) ŝⱼ; every column that is not constant needs ŝⱼ > esⱼ;
#   fitted values f̂ᵢ = fl(Σⱼ Xᵢⱼ βⱼ): efᵢ = G(p) Σⱼ|Xᵢⱼ||βⱼ|; residuals r̂ᵢ = fl(fl(yᵢ - β₀) - f̂ᵢ):
#     erᵢ = efᵢ + G(2)(|yᵢ| + |β₀| + |f̂ᵢ|);
#   primal P = ‖r‖²/(2n) + λ Σⱼ sⱼ|βⱼ|: eQ = Σᵢ erᵢ(2|r̂ᵢ| + erᵢ) + G(n+1) Σr̂ᵢ², eΠ =
#     λ(Σⱼ esⱼ|βⱼ| + G(p+2) Σⱼ ŝⱼ|βⱼ|), eP = eQ/(2n) + eΠ + G(3) P̂;
#   dual point θ = (r - r̄)/n: mean error ecr = Σᵢ erᵢ/n + G(n+1) Σᵢ|r̂ᵢ|/n, ûᵢ = fl(r̂ᵢ - r̄̂),
#     euᵢ = erᵢ + ecr + G(1)|ûᵢ|, θ̂ᵢ = fl(ûᵢ/n), eθᵢ = euᵢ/n + G(1)|θ̂ᵢ|; (θ sums to 0 exactly, as
#     the exact residuals r - r̄ do);
#   correlations gⱼ = x̃ⱼᵀθ: egⱼ = G(n) Σᵢ|x̂ᵢⱼ||θ̂ᵢ| + Σᵢ(|x̂ᵢⱼ| eθᵢ + |θ̂ᵢ| εxⱼ + εxⱼ eθᵢ);
#   dual point tθ with the number t = min(1, minⱼ λ(ŝⱼ - esⱼ)/(|ĝⱼ| + egⱼ)) over the columns with
#     sⱼ > 0, times 1 - 2⁻¹⁰⁰ when below 1 (no loop), and the check t (|ĝⱼ| + egⱼ) ≤ λ(ŝⱼ - esⱼ)
#     for those columns, which gives |x̃ⱼᵀ(tθ)| ≤ λ sⱼ in exact arithmetic (a constant column has
#     x̃ⱼ = 0);
#   dual value D = t A - n t² B/2 with A = ỹᵀθ, B = ‖θ‖²: ỹ = y - ȳ with ȳ exact for a constant y,
#     else eȳ = G(n+1) Σ|yᵢ|/n, eỹᵢ = eȳ + G(1)|ỹ̂ᵢ|; eA = G(n) Σ|ỹ̂ᵢ||θ̂ᵢ| + Σ(|ỹ̂ᵢ| eθᵢ + |θ̂ᵢ| eỹᵢ +
#     eỹᵢ eθᵢ); eB = Σ eθᵢ(2|θ̂ᵢ| + eθᵢ) + G(n+1) B̂; eD = t eA + n t² eB/2 + G(6)(t|Â| + n t² B̂/2).
# Weak duality: for the exact dual point tθ, which is feasible by the check above and sums to 0,
# ‖r‖²/(2n) ≥ (tθ)ᵀr - n‖tθ‖²/2 and λ sⱼ|βⱼ| ≥ |βⱼ||x̃ⱼᵀ(tθ)|, so that P(β₀, β) ≥ D, where D is
# the exact dual value, and D is below the minimum of the problem.
# Verdict. P_hi = P̂ + eP, P_lo = P̂ - eP, D_hi = D̂ + eD, D_lo = D̂ - eD. The relative gap of the
# harness is max(P - D, 0)/max(P, floatmin(Float64)) (P ≥ 0), and the harness records a miss when
# the returned value exceeds the target. Let value_hi = max(P_hi - D_lo, 0)/max(P_lo, floatmin) and
# value_lo = max(P_lo - D_hi, 0)/max(P_hi, floatmin) in BigFloat. Since P_lo ≤ P ≤ P_hi and
# D_lo ≤ D ≤ D_hi, the numerator is between the numerators and the denominator between the
# denominators, so value_lo ≤ the true relative gap ≤ value_hi. They are converted to Float64,
# value_hi rounded up and value_lo rounded down, which keeps value_lo ≤ true ≤ value_hi in Float64.
# Rule: if value_lo ≤ target < value_hi, the rounding uncertainty could change pass/fail and the
# certificate is uncertifiable (the note gives both bounds and the target). Otherwise value_hi is
# returned with `cert_ok`, and the harness verdict is the true one: if value_hi ≤ target the true
# gap is at most the target (ok), and if target < value_lo the true gap exceeds the target (a miss,
# and value_hi ≥ value_lo > target is a miss too). The shared `relative_gap` is bypassed: its
# rounding allowance and Float64 clamp would break this argument. The certificate is also
# uncertifiable when the error bounds contradict weak duality (P_hi < D_lo), when the dual point
# cannot be shown feasible, when (eP + eD)/max(P_lo, floatmin) > 1e-8, when P or D is not a normal
# Float64, and, for a column that is not constant (positive scale), when its scale is not a normal
# Float64, is 0, or is not above its error bound (constant columns have scale 0 and no bound). The
# 1e-8 cap is not needed for the verdict, which the rule above settles; it keeps the recorded value
# within a stated distance of the true gap: value_hi exceeds the true gap by at most
# 2 (1 + gap) 1e-8 before the final rounding (num_hi ≤ num + 2(eP + eD) and den_lo ≥ den - 2eP),
# and the rounding up adds at most a relative 2⁻⁵² of the value. The Float64 steps are: the exact
# conversion of the inputs, the validation of the scales in `bad_scale`, the report of P̂, D̂, θ̂, t
# and the two verdict bounds (rounded up and down as stated).

# The mean of v in BigFloat with the terms added in order; the value of a vector whose
# entries are all equal (decided from the stored entries, exactly)
function exact_mean(v)
  v1 = first(v)
  all(==(v1), v) && return BigFloat(v1)
  acc = zero(BigFloat)
  for t in v
    acc += BigFloat(t)
  end
  acc / length(v)
end

# The error bound G(k) = 8 k 2⁻²⁵⁶ of the model above
errG(k) = 8 * BigFloat(k) * BigFloat(2)^-256

# The columns of X centred in BigFloat and their root mean squares ŝⱼ, with the error bounds
# εx (centred entries) and es (scales) of the model above. A column whose stored entries are all
# equal has the exact mean, x̂ = 0, s = 0 and zero bounds. Call inside `setprecision(BigFloat, 256)`.
function centered_columns(X)
  n, p = size(X)
  Xt = Matrix{BigFloat}(undef, n, p)
  s = zeros(BigFloat, p)
  εx = zeros(BigFloat, p)
  es = zeros(BigFloat, p)
  for j in 1:p
    x = view(X, :, j)
    c = exact_mean(x)
    for i in 1:n
      Xt[i, j] = BigFloat(x[i]) - c
    end
    all(==(first(x)), x) && continue
    exc = errG(n + 1) * sum(abs ∘ BigFloat, x) / n
    εx[j] = exc + errG(1) * maximum(abs, view(Xt, :, j))
    acc = zero(BigFloat)
    for i in 1:n
      acc += Xt[i, j]^2
    end
    s[j] = sqrt(acc / n)
    es[j] = εx[j] + errG(n + 2) * s[j]
  end
  Xt, s, εx, es
end

# A scale that is not usable: the index of a column whose scale is not a normal Float64, or is 0
# although the stored entries are not all equal; 0 if there is none
function bad_scale(X, s)
  for j in eachindex(s)
    if iszero(s[j])
      all(==(first(view(X, :, j))), view(X, :, j)) || return j
      continue
    end
    sf = Float64(s[j])
    (isfinite(sf) && sf >= floatmin(Float64)) || return j
  end
  0
end

# The columns of X centred and divided by their scales (zero for a column of scale 0) and the
# scales, as Float64, from centered_columns; an error if a scale is not a normal Float64
function centered_scaled(X)
  setprecision(BigFloat, 256) do
    Xt, s, = centered_columns(X)
    j = bad_scale(X, s)
    j == 0 || error("the scale of column $j is not a normal Float64")
    Z = Float64.(Xt ./ transpose(ifelse.(iszero.(s), one(BigFloat), s)))
    Z, Float64.(s)
  end
end

# A BigFloat value as a Float64 for reporting: nothing if the value is not zero and is not a
# normal Float64 (it rounds to zero or a subnormal number, or is not finite)
function report64(x::BigFloat)
  v = Float64(x)
  (iszero(x) || (isfinite(v) && abs(v) >= floatmin(Float64))) ? v : nothing
end

# The computed primal value P̂ and dual value D̂ with their error bounds, the dual point and the
# upper bound `gap` of the relative gap, as described in the error model above. Returns a String
# with the reason when there is no certificate.
function lasso_standardized_parts(X, y, β₀, β, λ)
  d = cert_regression(X, y, β)
  d isa String && return d
  cert_finite(β₀) || return "nonfinite intercept"
  cert_finite(λ) && λ > 0 || return "λ must be finite and positive"
  (; X, y, β, n, p) = d
  setprecision(BigFloat, 256) do
    Xt, s, εx, es = centered_columns(X)
    j = bad_scale(X, s)
    j == 0 || return "the scale of column $j of X is not a normal Float64 (zero, subnormal or not finite) or is 0 for a column that is not constant"
    positive = [j for j in 1:p if !iszero(s[j])]
    all(j -> es[j] < s[j], positive) || return "the error bound of a scale is not below the scale (column $(first(j for j in positive if !(es[j] < s[j]))))"
    λb, β₀b, yb, βb, Xb = BigFloat(λ), BigFloat(β₀), BigFloat.(y), BigFloat.(β), BigFloat.(X)
    # primal
    f = Xb * βb
    ef = errG(p) .* (abs.(Xb) * abs.(βb))
    r = (yb .- β₀b) .- f
    er = ef .+ errG(2) .* (abs.(yb) .+ abs(β₀b) .+ abs.(f))
    πs = sum(s[j] * abs(βb[j]) for j in 1:p)
    Π = λb * πs
    eΠ = λb * (sum(es[j] * abs(βb[j]) for j in 1:p) + errG(p + 2) * πs)
    Q = sum(abs2, r)
    eQ = sum(er[i] * (2 * abs(r[i]) + er[i]) for i in 1:n) + errG(n + 1) * Q
    P = Q / (2n) + Π
    eP = eQ / (2n) + eΠ + errG(3) * P
    # dual point
    r̄ = sum(r) / n
    ecr = sum(er) / n + errG(n + 1) * sum(abs, r) / n
    u = r .- r̄
    eu = er .+ ecr .+ errG(1) .* abs.(u)
    θ = u ./ n
    eθ = eu ./ n .+ errG(1) .* abs.(θ)
    g = [sum(Xt[i, j] * θ[i] for i in 1:n) for j in 1:p]
    eg = [errG(n) * sum(abs(Xt[i, j]) * abs(θ[i]) for i in 1:n) +
          sum(abs(Xt[i, j]) * eθ[i] + abs(θ[i]) * εx[j] + εx[j] * eθ[i] for i in 1:n) for j in 1:p]
    room = [λb * (s[j] - es[j]) for j in 1:p]
    need = [abs(g[j]) + eg[j] for j in 1:p]
    t = one(BigFloat)
    for j in positive
      need[j] > room[j] && (t = min(t, room[j] / need[j]))
    end
    t < 1 && (t *= 1 - BigFloat(2)^-100)
    all(j -> t * need[j] <= room[j], positive) || return "the dual point cannot be shown feasible in the original coordinates"
    feasibility = isempty(positive) ? zero(BigFloat) : maximum(j -> t * need[j] / room[j], positive)
    # dual value
    cy = exact_mean(y)
    ecy = all(==(first(y)), y) ? zero(BigFloat) : errG(n + 1) * sum(abs, yb) / n
    ỹ = yb .- cy
    eỹ = ecy .+ errG(1) .* abs.(ỹ)
    A = sum(ỹ[i] * θ[i] for i in 1:n)
    eA = errG(n) * sum(abs(ỹ[i]) * abs(θ[i]) for i in 1:n) + sum(abs(ỹ[i]) * eθ[i] + abs(θ[i]) * eỹ[i] + eỹ[i] * eθ[i] for i in 1:n)
    B = sum(abs2, θ)
    eB = sum(eθ[i] * (2 * abs(θ[i]) + eθ[i]) for i in 1:n) + errG(n + 1) * B
    t2 = t * t
    D = t * A - n * t2 * B / 2
    eD = t * eA + n * t2 * eB / 2 + errG(6) * (t * abs(A) + n * t2 * B / 2)
    # verdict
    P_hi, D_lo = P + eP, D - eD
    P_hi < D_lo && return "the error bounds contradict weak duality: P + eP = $(Float64(P_hi)) < D - eD = $(Float64(D_lo))"
    den_lo = max(P - eP, BigFloat(floatmin(Float64)))
    width = (eP + eD) / den_lo
    width > 1e-8 && return "the error bound of the gap, relative to the primal value, is $(Float64(width)) > 1e-8: the arithmetic cannot support a certificate"
    gap = Float64(max(P_hi - D_lo, zero(BigFloat)) / den_lo, RoundUp)
    isfinite(gap) || return "the upper bound of the relative gap is not finite"
    den_hi = max(P_hi, BigFloat(floatmin(Float64)))
    gap_lo = Float64(max((P - eP) - (D + eD), zero(BigFloat)) / den_hi, RoundDown)
    P64, D64 = report64(P), report64(D)
    (P64 === nothing || D64 === nothing) && return "the primal or dual value is not a normal Float64"
    (; P = P64, D = D64, eP = Float64(eP, RoundUp), eD = Float64(eD, RoundUp), θ = Float64.(t .* θ), t = Float64(t),
       feasibility = Float64(feasibility, RoundUp), width = Float64(width, RoundUp), gap, gap_lo,
       Pb = P, ePb = eP, Db = D, eDb = eD)
  end
end

function lasso_gap_standardized(X, y, β₀, β, λ; target::Float64)
  isfinite(target) && target > 0 || return uncertifiable("the target must be finite and positive")
  a = lasso_standardized_parts(X, y, β₀, β, λ)
  a isa String && return uncertifiable(a)
  a.gap_lo <= target < a.gap && return uncertifiable(
    "the rounding uncertainty could change the verdict: the relative gap is in [$(a.gap_lo), $(a.gap)] and the target is $target")
  cert_ok(a.gap, :relative_gap; note = "upper bound; the relative width of the error bound is $(a.width)")
end

function lasso_gap(X, y, β, λ)
  a = lasso_parts(X, y, β, λ)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end
