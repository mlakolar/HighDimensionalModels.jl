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
# cannot be shown feasible, when (eP + eD)/max(P_lo, floatmin) > 1e-8, when a nonzero P or D would
# be reported as zero, a subnormal number or a non-finite number (a zero value is accepted), and, for a column that is not constant (positive scale), when its scale is not a normal
# Float64, is 0, or is not above its error bound (constant columns have scale 0 and no bound). The
# 1e-8 cap is not needed for the verdict, which the rule above settles; it keeps the recorded value
# within a stated distance of the true gap: value_hi exceeds the true gap by at most
# 2 (1 + gap) 1e-8 before the final rounding (num_hi ≤ num + 2(eP + eD) and den_lo ≥ den - 2eP),
# and the rounding up adds at most a relative 2⁻⁵² of the value when the Float64 result is a positive
# normal number, and at most nextfloat(0.0) in absolute terms when it is subnormal. The Float64 steps are: the exact
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

# ---- elastic net (#222) ----
#
# `elastic_net_gap` is the certificate for the elastic net with an intercept and column scales
# (λ > 0, 0 < α ≤ 1), at the stored intercept β₀ and coefficients β:
#
#   ‖y - β₀1 - Xβ‖²/(2n) + λα Σⱼ sⱼ|βⱼ| + (λ(1 - α)/2) Σⱼ sⱼ²βⱼ²,
#
# with sⱼ as for `lasso_gap_standardized`. With θⱼ = sⱼβⱼ, ν = λ(1 - α) and the augmented design
# Zₐ = [Z; √(nν) I], yₐ = [ỹ; 0] (Z the centred columns over their scales, ỹ = y - ȳ),
#
#   ‖yₐ - Zₐθ‖²/(2n) = ‖ỹ - Zθ‖²/(2n) + (ν/2)‖θ‖²,
#
# so that the objective in θ is the lasso ‖yₐ - Zₐθ‖²/(2n) + λα‖θ‖₁ with the divisor n of the
# original problem, not n + p (the number of rows of Zₐ). The primal value P is the objective on the
# raw data at the stored (β₀, β). The dual point is built from the residual r = y - β₀ - Xβ of the
# raw data, with θᵣ = (r - r̄)/n (so that θᵣ = (ỹ - Zθ)/n) and the point of the augmented problem
# t·rₐ/n = t[θᵣ; -√(ν/n) θ]:
#
#   ‖Zₐᵀ(t rₐ/n)‖∞ ≤ λα  is  |x̃ⱼᵀθᵣ - ν sⱼ²βⱼ| t ≤ λα sⱼ  for every column with sⱼ > 0
#   (a column with sⱼ = 0 has x̃ⱼ = 0 and θⱼ = 0: its entry of the product is 0),
#
#   D = t ỹᵀθᵣ - n t²‖θᵣ‖²/2 - t² ν Σⱼ sⱼ²βⱼ²/2 = (‖yₐ‖² - ‖yₐ - t rₐ‖²)/(2n).
#
# Weak duality for the lasso on (Zₐ, yₐ) with the divisor n gives D ≤ ‖yₐ - Zₐθ'‖²/(2n) + λα‖θ'‖₁
# for every θ', and in exact arithmetic min over β₀ of the objective is the objective of the centred
# problem, so D is below the minimum of the problem and P ≥ D. For α = 1 (ν = 0) this is the
# certificate of `lasso_gap_standardized`.
#
# The arithmetic and the verdict are those of `lasso_standardized_parts`: 256-bit BigFloat from the
# stored Float64 inputs, G(k) = 8ku, an absolute error bound for each computed quantity, the interval
# [value_lo, value_hi] of the relative gap, and the status uncertifiable when the target lies in
# [value_lo, value_hi). The quantities of the model above that are new, with hats for computed values:
#   * a = λα is exact (the product of two Float64 numbers has at most 106 bits); ν̂ = fl(λ fl(1 - α)),
#     eν = G(2) ν̂ (zero for α = 1);
#   * ĉⱼ = fl(ŝⱼ βⱼ) for θⱼ = sⱼβⱼ: ecⱼ = esⱼ|βⱼ| + G(1)|ĉⱼ|; Ĉ = Σ ĉⱼ² for C = Σ θⱼ²:
#     eC = Σ ecⱼ(2|ĉⱼ| + ecⱼ) + G(p+1) Ĉ;
#   * m̂ⱼ = fl(ŝⱼ ĉⱼ) for sⱼθⱼ = sⱼ²βⱼ: emⱼ = |ŝⱼ| ecⱼ + |ĉⱼ| esⱼ + esⱼ ecⱼ + G(1)|ŝⱼ ĉⱼ|;
#     q̂ⱼ = fl(ν̂ m̂ⱼ) for ν sⱼ²βⱼ: eqⱼ = ν̂ emⱼ + |m̂ⱼ| eν + eν emⱼ + G(1) ν̂ |m̂ⱼ|;
#   * ĥⱼ = fl(ĝⱼ - q̂ⱼ), for hⱼ = x̃ⱼᵀθᵣ - ν sⱼ²βⱼ with ĝⱼ and egⱼ of the model of
#     `lasso_standardized_parts`: ehⱼ = egⱼ + eqⱼ + G(1)|ĥⱼ|;
#   * the dual point tθᵣ with the number t = min(1, minⱼ a(ŝⱼ - esⱼ)/(|ĥⱼ| + ehⱼ)) over the columns with
#     sⱼ > 0, times 1 - 2⁻¹⁰⁰ when below 1, and the check t(|ĥⱼ| + ehⱼ) ≤ a(ŝⱼ - esⱼ), which gives
#     |t hⱼ| ≤ a sⱼ in exact arithmetic;
#   * primal P = ‖r‖²/(2n) + a Σⱼ sⱼ|βⱼ| + νC/2: eΠ = a(Σⱼ esⱼ|βⱼ| + G(p+2) Σⱼ ŝⱼ|βⱼ|),
#     eR = (ν̂ eC + Ĉ eν + eν eC)/2 + G(2) ν̂ Ĉ/2, eP = eQ/(2n) + eΠ + eR + G(5) P̂ (eQ as before);
#   * dual value D = t Â - n t² B̂/2 - t² ν̂ Ĉ/2: eD = t eA + n t² eB/2 + t² (ν̂ eC + Ĉ eν + eν eC)/2 +
#     G(9)(t|Â| + n t² B̂/2 + t² ν̂ Ĉ/2).
# The divisor of the quadratic terms and of the dual point is the original n = size(X, 1); no divisor is n + p.

# The computed primal and dual values with their error bounds, the dual point and the bounds of the
# relative gap, as described above. Returns a String with the reason when there is no certificate.
function elastic_net_parts(X, y, β₀, β, λ, α)
  d = cert_regression(X, y, β)
  d isa String && return d
  cert_finite(β₀) || return "nonfinite intercept"
  cert_finite(λ) && λ > 0 || return "λ must be finite and positive"
  cert_finite(α) && 0 < α <= 1 || return "α must be finite with 0 < α ≤ 1"
  (; X, y, β, n, p) = d
  setprecision(BigFloat, 256) do
    Xt, s, εx, es = centered_columns(X)
    j = bad_scale(X, s)
    j == 0 || return "the scale of column $j of X is not a normal Float64 (zero, subnormal or not finite) or is 0 for a column that is not constant"
    positive = [j for j in 1:p if !iszero(s[j])]
    all(j -> es[j] < s[j], positive) || return "the error bound of a scale is not below the scale (column $(first(j for j in positive if !(es[j] < s[j]))))"
    λb, αb, β₀b, yb, βb, Xb = BigFloat(λ), BigFloat(α), BigFloat(β₀), BigFloat.(y), BigFloat.(β), BigFloat.(X)
    a = λb * αb                                          # exact: at most 106 bits
    ν = λb * (1 - αb)
    eν = isone(αb) ? zero(BigFloat) : errG(2) * ν
    # the quantities of the ridge term
    c = [s[j] * βb[j] for j in 1:p]
    ec = [es[j] * abs(βb[j]) + errG(1) * abs(c[j]) for j in 1:p]
    C = sum(abs2, c)
    eC = sum(ec[j] * (2 * abs(c[j]) + ec[j]) for j in 1:p) + errG(p + 1) * C
    m = [s[j] * c[j] for j in 1:p]
    em = [s[j] * ec[j] + abs(c[j]) * es[j] + es[j] * ec[j] + errG(1) * abs(s[j] * c[j]) for j in 1:p]
    q = [ν * m[j] for j in 1:p]
    eq = [ν * em[j] + abs(m[j]) * eν + eν * em[j] + errG(1) * ν * abs(m[j]) for j in 1:p]
    ridge = ν * C / 2
    eridge = (ν * eC + C * eν + eν * eC) / 2 + errG(2) * ν * C / 2
    # primal
    f = Xb * βb
    ef = errG(p) .* (abs.(Xb) * abs.(βb))
    r = (yb .- β₀b) .- f
    er = ef .+ errG(2) .* (abs.(yb) .+ abs(β₀b) .+ abs.(f))
    πs = sum(s[j] * abs(βb[j]) for j in 1:p)
    Π = a * πs
    eΠ = a * (sum(es[j] * abs(βb[j]) for j in 1:p) + errG(p + 2) * πs)
    Q = sum(abs2, r)
    eQ = sum(er[i] * (2 * abs(r[i]) + er[i]) for i in 1:n) + errG(n + 1) * Q
    P = Q / (2n) + Π + ridge
    eP = eQ / (2n) + eΠ + eridge + errG(5) * P
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
    h = g .- q
    eh = eg .+ eq .+ errG(1) .* abs.(h)
    room = [a * (s[j] - es[j]) for j in 1:p]
    need = [abs(h[j]) + eh[j] for j in 1:p]
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
    D = t * A - n * t2 * B / 2 - t2 * ridge
    eD = t * eA + n * t2 * eB / 2 + t2 * eridge + errG(9) * (t * abs(A) + n * t2 * B / 2 + t2 * ridge)
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

function elastic_net_gap(X, y, β₀, β, λ, α; target::Float64)
  isfinite(target) && target > 0 || return uncertifiable("the target must be finite and positive")
  a = elastic_net_parts(X, y, β₀, β, λ, α)
  a isa String && return uncertifiable(a)
  a.gap_lo <= target < a.gap && return uncertifiable(
    "the rounding uncertainty could change the verdict: the relative gap is in [$(a.gap_lo), $(a.gap)] and the target is $target")
  cert_ok(a.gap, :relative_gap; note = "upper bound; the relative width of the error bound is $(a.width)")
end

# ---- square-root lasso (#223) ----
#
# `sqrt_lasso_gap_standardized` is the certificate for `sqrt_lasso` with an intercept and column scales
# (λ > 0), at the stored intercept β₀ and coefficients β, from the raw inputs:
#
#   P = ‖y - β₀1 - Xβ‖/√n + λ Σⱼ sⱼ|βⱼ|,
#
# with sⱼ the root mean square of the centred column xⱼ - x̄ⱼ (0 for a column whose entries are all equal,
# as for `lasso_gap_standardized`, whose helpers `centered_columns`, `bad_scale`, `errG` and `report64` it
# uses). With r = y - β₀1 - Xβ and r_c = r - mean(r)1 (the exact residual of the stored point; the centring
# is needed because 1ᵀr is not 0 for a rounded β₀), the dual point is u = t r_c/‖r_c‖ with
# gⱼ = |x̃ⱼᵀr_c|/‖r_c‖ for x̃ⱼ = xⱼ - x̄ⱼ and t = min(1, minⱼ λ√n sⱼ/gⱼ) over the columns with sⱼ > 0.
# Then ‖u‖ = t ≤ 1, 1ᵀu = 0 and |x̃ⱼᵀu| ≤ λ√n sⱼ, so that for every (b₀, b)
#
#   ‖y - b₀1 - Xb‖/√n + λ Σⱼ sⱼ|bⱼ| ≥ uᵀ(y - b₀1 - Xb)/√n + λ Σⱼ sⱼ|bⱼ| ≥ uᵀy/√n =: D
#
# (Cauchy–Schwarz, then 1ᵀu = 0 and the bound on |x̃ⱼᵀu|), and D is at most the minimum, which is at most P.
# D = t r_cᵀy/(‖r_c‖√n) uses the raw y, since 1ᵀr_c = 0.
#
# The arithmetic is in 256-bit BigFloat with an absolute error bound for each computed quantity, in the
# model of `lasso_standardized_parts` (G(k) = 8ku, u = 2⁻²⁵⁶; a hat marks a computed value). In addition to
# the quantities of that chain (the centred columns x̂ with the bounds εx, the scales ŝ with es, the fitted
# values and residuals r̂ with er), for the exact residual r:
#   ‖r‖ = √Q for Q = Σ r̂ᵢ²: |√Q - √Q̂| ≤ eQ/√Q̂ for |Q - Q̂| ≤ eQ (|√a - √b| ≤ |a - b|/(√a + √b)), and the
#     square root rounds by G(1); enr = eQ/√Q̂ + G(1)√Q̂; the division by the computed √n adds G(2) and √n has
#     the lower bound √n(1 - G(1)):
#     P̂ = ‖r̂‖/√n + λΣŝⱼ|βⱼ|, eP = (enr + G(2)√Q̂)/(√n(1 - G(1))) + λ(Σ esⱼ|βⱼ| + G(p+2)Σŝⱼ|βⱼ|) + G(3)P̂;
#   centred residual û = r̂ - fl(mean r̂) with the error euᵢ = erᵢ + ecr + G(1)|ûᵢ| as in the lasso chain, and
#     N̂ = √(Σûᵢ²) against N = ‖r_c‖: by the triangle inequality |‖û‖ - N| ≤ ‖eu‖, and the computed norm
#     is within G(n+2) of ‖û‖, so eN = ‖eu‖(1 + G(n+2)) + G(n+2)N̂; the dual point exists where N̂ - eN > 0
#     (the centred residual is not 0), which is the condition `uncertifiable` reports otherwise;
#   correlations aⱼ = x̃ⱼᵀr_c: eaⱼ = G(n)Σ|x̂ᵢⱼ||ûᵢ| + Σ(|x̂ᵢⱼ|euᵢ + |ûᵢ|εxⱼ + εxⱼ euᵢ) for the computed
#     âⱼ = x̂ⱼᵀû, and the ratios q = a/N: |a/N - â/N̂| ≤ ea/(N̂ - eN) + |â|eN/((N̂ - eN)N̂), which with the
#     rounding of the quotient G(1)|â|/N̂ gives egⱼ for ĝⱼ = |âⱼ|/N̂;
#   dual point tr_c/N: t = min(1, minⱼ λ√n(1 - G(3))(ŝⱼ - esⱼ)/(ĝⱼ + egⱼ)) over the columns with sⱼ > 0, times
#     1 - 2⁻¹⁰⁰ when below 1, and the check t(ĝⱼ + egⱼ) ≤ λ√n(1 - G(3))(ŝⱼ - esⱼ), which gives
#     |x̃ⱼᵀ(t r_c/N)| ≤ λ√n sⱼ in exact arithmetic;
#   dual value D = t A/(N√n) for A = r_cᵀy: Â = Σ ûᵢyᵢ with eA = G(n)Σ|ûᵢ||yᵢ| + Σ euᵢ|yᵢ|, so that
#     |A/N - Â/N̂| ≤ eA/(N̂ - eN) + |Â|eN/((N̂ - eN)N̂) and eD = t(that)/(√n(1 - G(1))) + G(6)|D̂|.
# The verdict, the rounding of the two bounds and the conditions that make the case `uncertifiable` are those of
# `lasso_standardized_parts` and `lasso_gap_standardized`: the relative gap of the harness is
# max(P - D, 0)/max(P, floatmin(Float64)), so value_lo ≤ the true gap ≤ value_hi, and the certificate is
# uncertifiable when value_lo ≤ target < value_hi, when P_hi < D_lo, when the dual point cannot be shown
# feasible, when (eP + eD)/max(P_lo, floatmin) > 1e-8, and for the scales and the Float64 reports as
# there. This denominator P is weaker than the D of the gap (P - D)/D that bounds (P - P*)/P*
# (P ≥ D), and it is valid: (P - D)/P ≤ target gives (P - D)/D ≤ target/(1 - target).

function sqrt_lasso_standardized_parts(X, y, β₀, β, λ)
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
    sn = sqrt(BigFloat(n))
    snlo = sn * (1 - errG(1))
    # primal
    f = Xb * βb
    ef = errG(p) .* (abs.(Xb) * abs.(βb))
    r = (yb .- β₀b) .- f
    er = ef .+ errG(2) .* (abs.(yb) .+ abs(β₀b) .+ abs.(f))
    Q = sum(abs2, r)
    iszero(Q) && return "the residual of the stored point is zero in the computed arithmetic: the dual direction is undefined"
    eQ = sum(er[i] * (2 * abs(r[i]) + er[i]) for i in 1:n) + errG(n + 1) * Q
    nr = sqrt(Q)
    enr = eQ / nr + errG(1) * nr
    πs = sum(s[j] * abs(βb[j]) for j in 1:p)
    eΠ = λb * (sum(es[j] * abs(βb[j]) for j in 1:p) + errG(p + 2) * πs)
    P = nr / sn + λb * πs
    eP = (enr + errG(2) * nr) / snlo + eΠ + errG(3) * P
    # dual point
    r̄ = sum(r) / n
    ecr = sum(er) / n + errG(n + 1) * sum(abs, r) / n
    u = r .- r̄
    eu = er .+ ecr .+ errG(1) .* abs.(u)
    N = sqrt(sum(abs2, u))
    eN = sqrt(sum(abs2, eu)) * (1 + errG(n + 2)) + errG(n + 2) * N
    Nlo = N - eN
    Nlo > 0 || return "the centred residual of the stored point is zero or not above its error bound: the dual direction is undefined"
    ratio_error(a, ea) = ea / Nlo + abs(a) * eN / (Nlo * N)
    a = [sum(Xt[i, j] * u[i] for i in 1:n) for j in 1:p]
    ea = [errG(n) * sum(abs(Xt[i, j]) * abs(u[i]) for i in 1:n) +
          sum(abs(Xt[i, j]) * eu[i] + abs(u[i]) * εx[j] + εx[j] * eu[i] for i in 1:n) for j in 1:p]
    g = abs.(a) ./ N
    eg = [ratio_error(a[j], ea[j]) + errG(1) * g[j] for j in 1:p]
    Λlo = λb * sn * (1 - errG(3))
    room = [Λlo * (s[j] - es[j]) for j in 1:p]
    need = g .+ eg
    t = one(BigFloat)
    for j in positive
      need[j] > room[j] && (t = min(t, room[j] / need[j]))
    end
    t < 1 && (t *= 1 - BigFloat(2)^-100)
    all(j -> t * need[j] <= room[j], positive) || return "the dual point cannot be shown feasible in the original coordinates"
    feasibility = isempty(positive) ? zero(BigFloat) : maximum(j -> t * need[j] / room[j], positive)
    # dual value
    A = sum(u[i] * yb[i] for i in 1:n)
    eA = errG(n) * sum(abs(u[i]) * abs(yb[i]) for i in 1:n) + sum(eu[i] * abs(yb[i]) for i in 1:n)
    D = t * A / (N * sn)
    eD = t * ratio_error(A, eA) / snlo + errG(6) * abs(D)
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
    (; P = P64, D = D64, eP = Float64(eP, RoundUp), eD = Float64(eD, RoundUp), u = Float64.(t .* u ./ N), t = Float64(t),
       feasibility = Float64(feasibility, RoundUp), width = Float64(width, RoundUp), gap, gap_lo,
       Pb = P, ePb = eP, Db = D, eDb = eD)
  end
end

function sqrt_lasso_gap_standardized(X, y, β₀, β, λ; target::Float64)
  isfinite(target) && target > 0 || return uncertifiable("the target must be finite and positive")
  a = sqrt_lasso_standardized_parts(X, y, β₀, β, λ)
  a isa String && return uncertifiable(a)
  a.gap_lo <= target < a.gap && return uncertifiable(
    "the rounding uncertainty could change the verdict: the relative gap is in [$(a.gap_lo), $(a.gap)] and the target is $target")
  cert_ok(a.gap, :relative_gap; note = "upper bound; the relative width of the error bound is $(a.width)")
end

# ---- group lasso (#224) ----
#
# Certificate of `group_lasso` (intercept = true), at the returned intercept β₀ and coefficients β. The
# objective, with X̃ the columns of X centred at their exact means (a column whose entries are all equal
# has that value as its mean in exact arithmetic too, so it centres to zeros) and the weights wₖ of the
# fit (positive), is
#
#   P = ‖y - β₀1 - Xβ‖²/(2n) + λ Σₖ wₖ ‖X̃_Gₖ β_Gₖ‖/√n      (orthonormal = true)
#   P = ‖y - β₀1 - Xβ‖²/(2n) + λ Σₖ wₖ ‖β_Gₖ‖               (orthonormal = false).
#
# The dual point is u = t r_c with r_c = r - r̄1 the centred residual of r = y - β₀1 - Xβ, and θ = u/n. For
# any θ that sums to 0, ‖r‖²/(2n) ≥ θ'r - n‖θ‖²/2, and θ'r = ỹ'θ - Σₖ θ'X̃_Gₖ β_Gₖ with ỹ = y - ȳ1 (θ sums to 0),
# so that P(β₀, β) ≥ D = ỹ'θ - n‖θ‖²/2 = t ỹ'r_c/n - t² ‖r_c‖²/(2n) if |θ'X̃_G β_G| ≤ λ w ‖X̃_G β_G‖/√n
# (orthonormal) or ≤ λ w ‖β_G‖ (not) for each group. These inequalities hold for all β_G if
#
#   orthonormal:  ‖P_G u‖ ≤ √n λ w,  P_G the projector on the FULL column span of X̃_G,
#                 that is u'X̃_G (X̃_G'X̃_G)⁻¹ X̃_G'u ≤ n λ² w²;
#   not:          ‖X̃_G'u‖/n ≤ λ w.
#
# D is at most the minimum over β₀ and β, so P - D bounds P - P* from above for a feasible u. The projector is
# built from all the columns of the group, and not from a truncated set of directions (which can underestimate
# ‖P_G u‖ and make u infeasible): when the Gram matrix of X̃_G is not positive definite in rational arithmetic
# the certificate is `uncertifiable`. The projector on the span of dependent columns exists; this routine
# computes it from the normal equations and does not construct it for a singular Gram matrix.
#
# Arithmetic. Each stored Float64 input is converted without rounding to a rational number (BigInt numerator and
# denominator), and r, r̄, r_c, the centred columns, the Gram matrices, X̃_G'u and the quantities that enter P and
# D (apart from the square roots) are computed in rational arithmetic: there is no rounding to bound, unlike the
# error model of `lasso_standardized_parts`. The factor t is a rational number (a 256-bit BigFloat, which is
# a rational) with t ≤ 1 and t² ≤ nλ²w²/(r_c'X̃_G(X̃_G'X̃_G)⁻¹X̃_G'r_c) (orthonormal) or t² ≤ n²λ²w²/‖X̃_G'r_c‖²
# (not) for each group, so that the dual point is feasible; the inequality is checked in rational arithmetic.
# The square roots of the penalty are bracketed by 256-bit BigFloat bounds computed with directed rounding
# (the operands are nonnegative), and D is converted to BigFloat with directed rounding. So P_lo ≤ P ≤ P_hi and
# D_lo ≤ D ≤ D_hi hold in exact arithmetic, and value_hi = max(P_hi - D_lo, 0)/max(P_lo, floatmin) and
# value_lo = max(P_lo - D_hi, 0)/max(P_hi, floatmin) (the first computed rounding up and the second rounding
# down, then converted to Float64 in the same direction) satisfy value_lo ≤ the relative gap
# max(P - D, 0)/max(P, floatmin) ≤ value_hi. As in `lasso_gap_standardized`, the certificate is
# `uncertifiable` when value_lo ≤ target < value_hi, and otherwise returns value_hi, so that the verdict of the
# harness is the true one.
#
# Domain. The routine is for λ > 0 and positive weights, and returns `uncertifiable` for λ = 0 or a weight of
# zero. That is a limit of the routine, not a statement that no dual point exists there (t = 0 gives u = 0 and
# D = 0, a feasible point with a valid, possibly loose gap, and a positive t can be feasible too). A KKT residual is
# not reported in its place, since it is not a duality gap.

# the exact rational numbers of the certificate
const GroupRational = Rational{BigInt}

# A 256-bit BigFloat bound of the rational q: the numerator and the denominator are converted without
# rounding (at a precision above their bit lengths) and the quotient is rounded in the direction `mode`
# (RoundDown: at most q; RoundUp: at least q)
function rational_bound(q::GroupRational, mode)
  bits = max(ndigits(abs(numerator(q)); base = 2), ndigits(denominator(q); base = 2), 256) + 2
  num, den = setprecision(() -> (BigFloat(numerator(q)), BigFloat(denominator(q))), BigFloat, bits)
  setprecision(BigFloat, 256) do
    setrounding(() -> num / den, BigFloat, mode)
  end
end

# gᵀG⁻¹g for a symmetric matrix G and a vector g of rational numbers, in rational arithmetic by elimination
# (LDLᵀ); nothing if a pivot is not positive, that is if G is not positive definite
function posdef_quadratic(G::AbstractMatrix{GroupRational}, g::AbstractVector{GroupRational})
  m = length(g)
  A, b = copy(G), copy(g)
  value = zero(GroupRational)
  for i in 1:m
    pivot = A[i, i]
    pivot > 0 || return nothing
    value += b[i]^2 / pivot
    for j in i+1:m
      f = A[j, i] / pivot
      iszero(f) && continue
      for l in i:m
        A[j, l] -= f * A[i, l]
      end
      b[j] -= f * b[i]
    end
  end
  value
end

# A rational t ≥ 0 with t² ≤ x (a rational x ≥ 0), close to √x: the square root of a 256-bit lower bound of x,
# rounded down, lowered until the inequality holds in rational arithmetic
function rational_sqrt_below(x::GroupRational)
  x == 1 && return one(GroupRational)
  b = setprecision(BigFloat, 256) do
    setrounding(() -> sqrt(rational_bound(x, RoundDown)), BigFloat, RoundDown)
  end
  t = GroupRational(b)
  while t^2 > x
    b = prevfloat(b)
    t = GroupRational(b)
  end
  t
end

# The reason why the groups, the weights and λ do not define a group lasso with a gap, or nothing
function group_problem(groups, weights, λ, p::Integer)
  groups isa AbstractVector || return "the groups must be a vector of vectors of column indices"
  seen = falses(p)
  for G in groups
    (G isa AbstractVector && !isempty(G)) || return "the groups must be nonempty vectors of column indices"
    for j in G
      (j isa Integer && 1 <= j <= p && !seen[j]) || return "the groups must be disjoint and contain only columns of X"
      seen[j] = true
    end
  end
  all(seen) || return "the groups must contain every column of X"
  (length(weights) == length(groups) && cert_finite(λ, weights) && all(>=(0), weights) && λ >= 0) ||
    return "λ and the group weights must be finite and nonnegative, with one weight per group"
  (λ > 0 && all(>(0), weights)) ||
    return "λ and the group weights must be positive: λ = 0 and a weight of zero are outside the domain of this certificate (a KKT residual is not reported as a duality gap)"
  nothing
end

# The primal and dual values of the certificate at (β₀, β), as described above: a String with the reason
# if there is none, otherwise (; P, D, t, u, gap, gap_lo, P_lo, P_hi, D_lo, D_hi), with P and D Float64
# (P rounded from the middle of its bounds), u = t r_c as Float64, and the bounds in BigFloat.
function group_parts(X, y, β₀, β, λ, groups, weights; orthonormal::Bool)
  d = cert_regression(X, y, β)
  d isa String && return d
  cert_finite(β₀) || return "nonfinite intercept"
  (; X, y, β, n, p) = d
  reason = group_problem(groups, weights, λ, p)
  reason === nothing || return reason
  Q = GroupRational
  Xq, yq, βq = Q.(X), Q.(y), Q.(β)
  λq, wq = Q(f64(λ)), Q.(f64(weights))
  r = yq .- Q(f64(β₀)) .- Xq * βq
  rc = r .- sum(r) / n                                  # sums to 0
  ỹ = yq .- sum(yq) / n
  K = length(groups)
  q = Vector{Q}(undef, K)                               # the argument of the square root of the penalty of group k
  t_sq = one(Q)
  for (k, G) in enumerate(groups)
    A, bG = Xq[:, G], βq[G]
    g = transpose(A) * rc                               # X̃_G'r_c, since r_c sums to 0
    if orthonormal
      Ac = A .- sum(A; dims = 1) ./ n
      q[k] = sum(abs2, Ac * bG) / n
      s = posdef_quadratic(transpose(Ac) * Ac, g)
      s === nothing && return "the Gram matrix of the centred columns of group $k is singular in rational arithmetic: this certificate computes the projector on their span from the normal equations, which needs it positive definite"
      cap = n * (λq * wq[k])^2
    else
      q[k] = sum(abs2, bG)
      s = sum(abs2, g)
      cap = Q(n)^2 * (λq * wq[k])^2
    end
    s > 0 && (t_sq = min(t_sq, cap / s))
  end
  t = rational_sqrt_below(t_sq)
  D = t * sum(ỹ .* rc) / n - t^2 * sum(abs2, rc) / (2n)
  quad = sum(abs2, r) / (2n)
  lw = λq .* wq
  P_lo, P_hi = map((RoundDown, RoundUp)) do mode
    setprecision(BigFloat, 256) do
      setrounding(BigFloat, mode) do
        acc = rational_bound(quad, mode)
        for k in 1:K
          acc += rational_bound(lw[k], mode) * sqrt(rational_bound(q[k], mode))
        end
        acc
      end
    end
  end
  D_lo, D_hi = rational_bound(D, RoundDown), rational_bound(D, RoundUp)
  P_hi < D_lo && return "the bounds contradict weak duality: P_hi = $(Float64(P_hi)) < D_lo = $(Float64(D_lo))"
  floor_ = BigFloat(floatmin(Float64))
  gap, gap_lo = setprecision(BigFloat, 256) do
    num_hi = setrounding(() -> max(P_hi - D_lo, zero(BigFloat)), BigFloat, RoundUp)
    gap = setrounding(() -> Float64(num_hi / max(P_lo, floor_), RoundUp), BigFloat, RoundUp)
    num_lo = setrounding(() -> max(P_lo - D_hi, zero(BigFloat)), BigFloat, RoundDown)
    gap_lo = setrounding(() -> Float64(num_lo / max(P_hi, floor_), RoundDown), BigFloat, RoundDown)
    gap, gap_lo
  end
  isfinite(gap) || return "the upper bound of the relative gap is not finite"
  (; P = Float64((P_lo + P_hi) / 2), D = Float64((D_lo + D_hi) / 2), t = Float64(t), u = Float64.(t .* rc),
     gap, gap_lo, P_lo, P_hi, D_lo, D_hi)
end

function group_gap(X, y, β₀, β, λ, groups, weights; orthonormal::Bool, target::Float64)
  isfinite(target) && target > 0 || return uncertifiable("the target must be finite and positive")
  a = group_parts(X, y, β₀, β, λ, groups, weights; orthonormal)
  a isa String && return uncertifiable(a)
  a.gap_lo <= target < a.gap && return uncertifiable(
    "the rounding of the bounds could change the verdict: the relative gap is in [$(a.gap_lo), $(a.gap)] and the target is $target")
  cert_ok(a.gap, :relative_gap; note = "upper bound; exact rational arithmetic for the data, the dual point and the quadratic forms")
end
