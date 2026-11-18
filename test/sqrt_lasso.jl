module SqrtLassoTest
# `sqrt_lasso`: the minimizer of the normalized objective
#   ‖y - β₀1 - Xβ‖/√n + λ Σⱼ sⱼ|βⱼ|,
# with β₀ = 0 unless `intercept` and sⱼ = 1 unless `standardize_x` (the root mean square of the centred
# column, of the column itself without an intercept). The references share nothing with the
# preprocessing or with coordinate descent: a duality gap, the objective and the weights in 256-bit
# BigFloat from the stored Float inputs, the exact solution of constructed problems, and invariances
# that hold bit for bit on data where the sums are exact.
#
# Two claims are tested apart (the docstring of `SqrtLassoSolution` states them):
#   (a) the certification of the transformed problem: ProximalSolvers' `converged` and `gap` describe the
#       problem it solved, the stored `(Z, ỹ)` in T at the level √n λ and the point θ. The dual gap of
#       that problem is computed here in BigFloat from the stored data;
#   (b) the accuracy in the original coordinates: the dual gap of the stored (β₀, β) on the raw X and y,
#       which no field of the fit certifies, against 1e-4 plus a perturbation term derived below.
#
# The dual gap. Let x̄ and s be the column means (zero without an intercept) and weights, r = y - β₀1 - Xβ,
# r_c = r - mean(r)1 with an intercept (r_c = r without one; the centring is needed, because with a
# rounded β₀ the vector 1ᵀr is not zero), and Λ = √n λ the level of the objective ‖r‖ + Λ Σⱼ sⱼ|βⱼ|, which
# is √n times the one above. With gⱼ = |(xⱼ - x̄ⱼ)ᵀr_c|/(sⱼ‖r_c‖) over the columns with sⱼ > 0, t = min(1,
# Λ/maxⱼ gⱼ) and u = t r_c/‖r_c‖: ‖u‖ ≤ 1, 1ᵀu = 0 and |(xⱼ - x̄ⱼ)ᵀu| ≤ Λ sⱼ. By Cauchy-Schwarz, for every
# (b₀, b), ‖y - b₀1 - Xb‖ + Λ Σⱼ sⱼ|bⱼ| ≥ uᵀ(y - b₀1 - Xb) + Λ Σⱼ sⱼ|bⱼ| ≥ uᵀy, so D = uᵀy is a lower
# bound of the minimum, P = ‖r‖ + Λ Σⱼ sⱼ|βⱼ| an upper bound, and (P - D)/D bounds the relative
# suboptimality (P - P*)/P* of the stored point. ProximalSolvers' certificate is a bound of the same
# ratio, on its own problem.
#
# The perturbation term ω of (b). The solver sees Ẑ and ỹ̂, the rounded centred and scaled data in T, and
# returns θ; the stored β̂ⱼ = fl(θⱼ/ŝⱼ) and β̂₀ = fl(ȳ̂ - x̄̂ᵀβ̂) are formed from them. Let Zⱼ, ỹ be the exact
# centred and scaled data, φ = Sβ̂ (exact weights s), δ₀ = ȳ - x̄ᵀβ̂ - β̂₀, r̂ = ỹ̂ - Ẑθ and E = Ẑ - Z. Then
#   r_c = ỹ - Zφ = r̂ - (ỹ̂ - ỹ) + Eθ + Z(φ - θ)   and   r = r_c + δ₀1,
# so ‖r_c - r̂‖ ≤ Δ. Write u = eps(T)/2, G = (n + p + 8)u (the factor of a sum of n terms, with room for the
# scalings and the final operations), c̄ = maxⱼ ‖Zⱼ‖/√n (1 with standardization), m̄ = maxⱼ |x̄ⱼ|/rmsⱼ and
# ‖ỹ‖ the norm of the exact centred response. First-order bounds for the rounding of the preprocessing, with
# a factor 4 that absorbs the products of errors, are
#   ‖Eⱼ‖ ≤ ζ_Z = 4G√n c̄(1 + m̄),   ‖ỹ̂ - ỹ‖ ≤ ζ_y = 4G(‖ỹ‖ + 2√n|ȳ|),
#   √n|δ₀| ≤ Δ₀ = 2G√n(2|ȳ| + ‖ỹ‖/√n + c̄(1 + 3m̄)‖φ‖₁),   ‖φ - θ‖₁ ≤ 4G(1 + m̄)‖φ‖₁,
# where the error of a mean enters through |x̄ⱼ| + rmsⱼ, and with both keywords false every one of these is
# zero (the stored problem is the raw problem). Hence ‖r_c - r̂‖ ≤ Δ = ζ_y + 2ζ_Z‖φ‖₁. The relative
# difference of the level, from the rounding of λps = √n λ in Float64, is η_Λ = 4·2⁻⁵³. For the stored
# point and the exact quantities N = ‖r_c‖, d = r_c/N, g = maxⱼ gⱼ, and the hatted ones of the transformed
# problem (the same formulas with x̄ = 0, s = 1 at Ẑ, ỹ̂, θ and the level Λ' = λps):
#   |P - P̂| ≤ Δ + Δ₀ + (η_Λ + 4G(1 + m̄))Λ‖φ‖₁                      (‖r‖ ≤ ‖r_c‖ + √n|δ₀|),
#   ‖d - d̂‖ ≤ 2Δ/N, |ĝ - g| ≤ δg = ζ_Z + (√n c̄ + ζ_Z)2Δ/N,
#   |t - t̂| ≤ τ = (δg + η_ΛΛ)/((1 - η_Λ)Λ - δg)     (t = min(1, Λ/g): each case of the minimum gives the
#                                                      bound (|ĝ - g| + |Λ - Λ'|)/min(g, ĝ), and min(g, ĝ) is at
#                                                      least Λ' - δg wherever t ≠ t̂),
#   |D - D̂| ≤ ΔD = ‖ỹ‖(2Δ/N + τ) + ζ_y             (D = t dᵀỹ, ‖d - d̂‖ and |t - t̂| as above).
# Since P/D - P̂/D̂ = (ΔP D̂ - P̂ ΔD)/(D D̂) and P̂/D̂ = 1 + gap of the transformed problem ≤ 1 + 1e-4,
#   (P - D)/D ≤ gap_T + ω,   ω = (|P - P̂| bound + (1 + 1e-4) ΔD)/(D - ΔD),
# which is what (b) tests with gap_T ≤ 1e-4 from (a). The bound is a worst case: the observed difference of the
# two gaps is printed at the end of this file (a ratio below 1 is the margin), and the data of the tests keep
# ω far below 1.

using HighDimensionalModels, Test, LinearAlgebra, Random
using ProximalBase: NormL1, SqrtLeastSquaresLoss
using ProximalSolvers: solve!

const HDM = HighDimensionalModels

# a CoreLogging logger that drops the warnings of ProximalSolvers (an uncertified stop or maxpasses)
quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# ---------------------------------------------------------------------------------
# settings and data
# ---------------------------------------------------------------------------------

tolof(T) = T === Float32 ? 1e-6 : 1e-10
cyc(T; kw...) = CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 2000; kw...)
const COMBOS = ((false, false), (true, false), (false, true), (true, true))
label(ic, st) = "intercept = $ic, standardize_x = $st"
const UNCERTIFIED = r"stopped without a duality certificate"
const CAPPED = r"did not converge in maxpasses"

# columns of different scales and means, a response with an intercept
function problem(T, seed, n, p; shift = 1.0, k = 3, noise = 0.5, offset = 1.0)
  rng = Xoshiro(seed)
  scales = 0.5 .+ 1.5 .* rand(rng, p)
  X = randn(rng, n, p) .* transpose(scales) .+ transpose(shift .* randn(rng, p))
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  T.(X), T.(X * β .+ offset .+ noise .* randn(rng, n))
end

# data on a grid of 1/16, with n a power of two: the sums of a column are exact in Float32 and Float64
grid(x) = round(x * 16) / 16
function grid_problem(T, seed, n, p)
  rng = Xoshiro(seed)
  scales = [1.0, 2.0, 0.5, 4.0, 1.5, 3.0, 1.0, 2.5][1:p]
  X = grid.(randn(rng, n, p) .* transpose(scales))
  β = zeros(p); β[1:3] .= [2.0, -1.5, 1.0]
  T.(X), T.(grid.(X * β .+ 1.0 .+ 0.5 .* randn(rng, n)))
end

# ---------------------------------------------------------------------------------
# References in 256-bit BigFloat from the stored Float inputs
# ---------------------------------------------------------------------------------

dotb(a, b) = sum(a .* b)

# the centred data and the weights of the objective: x̄, ȳ (zero without an intercept), the root mean
# squares of the centred columns, and the weights s (those, or ones). The mean of a column whose entries
# are all equal is that entry exactly (the sum of equal Float entries and the division are exact here).
function reference(X, y, intercept::Bool, standardize::Bool)
  n, p = size(X)
  Xb, yb = BigFloat.(X), BigFloat.(y)
  x̄ = intercept ? [sum(Xb[:, j]) / n for j in 1:p] : zeros(BigFloat, p)
  ȳ = intercept ? sum(yb) / n : zero(BigFloat)
  Xt, yt = Xb .- transpose(x̄), yb .- ȳ
  rms = [sqrt(sum(abs2, Xt[:, j]) / n) for j in 1:p]
  (; Xb, yb, Xt, yt, x̄, ȳ, rms, s = standardize ? rms : ones(BigFloat, p), n, p)
end

# λmax of the normalized problem: the least λ with β = 0 optimal, maxⱼ |zⱼᵀỹ|/(√n‖ỹ‖) over sⱼ > 0, as a
# Float64 (rounded from BigFloat)
function lambda_max(X, y, intercept, standardize)
  setprecision(BigFloat, 256) do
    ref = reference(X, y, intercept, standardize)
    top = maximum((abs(dotb(ref.Xt[:, j], ref.yt)) / ref.s[j] for j in 1:ref.p if !iszero(ref.s[j])); init = zero(BigFloat))
    Float64(top / (sqrt(BigFloat(ref.n)) * sqrt(dotb(ref.yt, ref.yt))))
  end
end

# The dual gap of the stored point (β₀, β) on the data X, y at the level Λ of ‖r‖ + Λ Σⱼ sⱼ|βⱼ|, as in the
# header, with (P, D, gap, N) as Float64; `nothing` where r_c = 0 (the dual point is undefined). With
# `eps_T` (the machine epsilon of the data's type) it also returns the term ω of the header, for a point
# that comes from a solve in the transformed coordinates; `pre` says whether there was preprocessing.
function dual_gap(X, y, β₀, β, Λ; intercept::Bool, standardize::Bool, eps_T = nothing, pre::Bool = true)
  setprecision(BigFloat, 256) do
    ref = reference(X, y, intercept, standardize)
    (; Xb, yb, Xt, yt, x̄, ȳ, rms, s, n, p) = ref
    βb, Λb = BigFloat.(β), BigFloat(Λ)
    r = (yb .- BigFloat(β₀)) .- Xb * βb
    rc = intercept ? r .- sum(r) / n : r
    N = sqrt(dotb(rc, rc))
    iszero(N) && return nothing
    φ1 = sum(s[j] * abs(βb[j]) for j in 1:p)
    P = sqrt(dotb(r, r)) + Λb * φ1
    g = maximum((abs(dotb(Xt[:, j], rc)) / s[j] for j in 1:p if !iszero(s[j])); init = zero(BigFloat)) / N
    t = g > Λb ? Λb / g : one(BigFloat)
    D = t * dotb(rc, yb) / N
    gap = D > 0 ? (P - D) / D : BigFloat(Inf)
    out = (; P = Float64(P), D = Float64(D), gap = Float64(gap), N = Float64(N), weak = P - D >= -BigFloat(2)^-200 * P)
    eps_T === nothing && return out
    u = BigFloat(eps_T) / 2
    G = (n + p + 8) * u
    sn = sqrt(BigFloat(n))
    ηΛ = 4 * BigFloat(2)^-53
    yn = sqrt(dotb(yt, yt))
    cbar = maximum((rms[j] / s[j] for j in 1:p if !iszero(s[j])); init = zero(BigFloat))
    mbar = maximum((abs(x̄[j]) / rms[j] for j in 1:p if !iszero(rms[j])); init = zero(BigFloat))
    ζZ = pre ? 4G * sn * cbar * (1 + mbar) : zero(BigFloat)
    ζy = pre && intercept ? 4G * (yn + 2sn * abs(ȳ)) : zero(BigFloat)
    Δ₀ = pre && intercept ? 2G * sn * (2abs(ȳ) + yn / sn + cbar * (1 + 3mbar) * φ1) : zero(BigFloat)
    Δ = ζy + 2ζZ * φ1
    ΔP = Δ + Δ₀ + (ηΛ + (pre ? 4G * (1 + mbar) : zero(BigFloat))) * Λb * φ1
    δg = ζZ + (sn * cbar + ζZ) * 2Δ / N
    den = (1 - ηΛ) * Λb - δg
    ΔD = den > 0 ? yn * (2Δ / N + (δg + ηΛ * Λb) / den) + ζy : BigFloat(Inf)
    ω = D > ΔD ? (ΔP + (1 + BigFloat(10)^-4) * ΔD) / (D - ΔD) : BigFloat(Inf)
    merge(out, (; ω = Float64(ω), ΔP = Float64(ΔP)))
  end
end

# the problem that ProximalSolvers solved: the stored Z and ỹ (the data themselves with both keywords false)
transformed(X, y, ic, st) = ic || st ? HDM.preprocess(X, y, ic, st) : (; Z = X, ỹ = y)
level(λ, n) = sqrt(Float64(n)) * Float64(λ)

# The normalized KKT violation of the stored transformed problem, relative to the level Λps. With
# r = ỹ - Zθ and cⱼ = zⱼᵀr/‖r‖, off the support the violation is (|cⱼ| - Λps)₊ and on it |cⱼ - Λps sign θⱼ|.
function kkt(Z, ỹ, θ, Λps)
  setprecision(BigFloat, 256) do
    Zb, yb, θb = BigFloat.(Z), BigFloat.(ỹ), BigFloat.(θ)
    r = yb .- Zb * θb
    N = sqrt(dotb(r, r))
    Λ = BigFloat(Λps)
    v = zero(BigFloat)
    for j in axes(Zb, 2)
      c = dotb(Zb[:, j], r) / N
      v = max(v, iszero(θb[j]) ? max(abs(c) - Λ, 0) : abs(c - Λ * sign(θb[j])))
    end
    (; viol = Float64(v / Λ), N = Float64(N), zmax = Float64(maximum(sqrt(dotb(Zb[:, j], Zb[:, j])) for j in axes(Zb, 2))))
  end
end

# ---------------------------------------------------------------------------------
# the checks of one fit
# ---------------------------------------------------------------------------------

# claims (a) and (b) for a fit of `sqrt_lasso(X, y, λ, ...; intercept = ic, standardize_x = st)`, when it
# reports `converged`; the observed ratio of the two gaps' difference to ω is recorded in `margins`
const margins = Float64[]
function check_certified(fit, X::Matrix{T}, y, λ, ic, st; exact_fit = false, tight = true) where T
  n = size(X, 1)
  @test fit.result.converged
  d = transformed(X, y, ic, st)
  a = dual_gap(d.Z, d.ỹ, 0, fit.result.minimizer, level(λ, n); intercept = false, standardize = false)
  b = dual_gap(X, y, fit.intercept, fit.coefficients, sqrt(BigFloat(n)) * BigFloat(λ);
               intercept = ic, standardize = st, eps_T = eps(T), pre = ic || st)
  if a === nothing || b === nothing
    # a zero residual: the dual point is undefined, so the gap formula is silent; allowed only at an exact fit
    @test exact_fit
    return nothing
  end
  # (a) certification of the transformed problem: the independent gap is below the certified bound
  @test a.weak && a.gap <= 1e-4
  # and, where the residual is not tiny against the data, no larger than the reported bound up to the rounding of the
  # dual direction (ProximalSolvers' dual point is along its own recomputed residual, this one along the exact one)
  tight && @test a.gap <= Float64(fit.result.gap) * (1 + 1e-8) + 1e-12
  # (b) accuracy in the original coordinates: 1e-4 plus the perturbation term
  @test b.weak && b.gap <= 1e-4 + b.ω
  isfinite(b.ω) && b.ω > 0 && push!(margins, max(b.gap - a.gap, 0) / b.ω)
  (a, b)
end

# the stored fields of a fit at the point it returns
function check_fields(fit, X::Matrix{T}, y, λ, ic) where T
  n, p = size(X)
  @test fit isa SqrtLassoSolution{T, <:Any}
  @test fit.hasintercept == ic && fit.λ == T(λ) && (ic || iszero(fit.intercept))
  @test length(fit.coefficients) == p && length(fit.residuals) == n
  @test all(isfinite, fit.coefficients) && isfinite(fit.intercept) && isfinite(fit.σ)
  # the residuals are those of y - β₀ - Xβ, and σ their root mean square
  setprecision(BigFloat, 256) do
    r = (BigFloat.(y) .- BigFloat(fit.intercept)) .- BigFloat.(X) * BigFloat.(fit.coefficients)
    scale = maximum(abs, BigFloat.(y)) + abs(BigFloat(fit.intercept)) + maximum(abs.(BigFloat.(X)) * abs.(BigFloat.(fit.coefficients)); init = BigFloat(0))
    @test all(i -> abs(BigFloat(fit.residuals[i]) - r[i]) <= (p + 4) * BigFloat(eps(T)) * scale, 1:n)
    σ = sqrt(dotb(BigFloat.(fit.residuals), BigFloat.(fit.residuals)) / n)
    @test abs(BigFloat(fit.σ) - σ) <= 4 * BigFloat(eps(T)) * (n + 4) * σ
  end
end

# same fields, including every field of the ProximalSolvers result (NaN equals NaN)
function same_fit(a, b)
  fields(f) = (f.coefficients, f.intercept, f.hasintercept, f.residuals, f.λ, f.σ, f.result.minimizer,
               f.result.minimum, f.result.passes, f.result.converged, f.result.maxchange, f.result.tolerance,
               f.result.kkt_checks, f.result.gap)
  all(isequal(x, y) for (x, y) in zip(fields(a), fields(b))) && typeof(a) === typeof(b)
end

@testset "sqrt_lasso" begin

  @testset "dual gap: transformed problem (a) and original coordinates (b), $T" for T in (Float32, Float64)
    for (k, (ic, st)) in enumerate(COMBOS)
      @testset "$(label(ic, st))" begin
        n, p = 60, 8
        X, y = problem(T, 10k, n, p)
        λ = T(0.3 * lambda_max(X, y, ic, st))
        fit = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
        check_fields(fit, X, y, λ, ic)
        T === Float64 && @test fit.result.converged
        @test count(!iszero, fit.coefficients) >= 1
        # a Float32 run either certifies (checked independently) or reports that it did not
        fit.result.converged ? check_certified(fit, X, y, λ, ic, st) : @test isnan(fit.result.gap)
        T === Float64 && @test fit.result.gap <= 1e-4
      end
    end
  end

  @testset "claim (b) is strict at zero means in Float64" begin
    # columns and response with exact zero means (each row followed by its negative), on a grid: the
    # centring changes nothing, so with standardize_x = false the stored problem is the raw one
    n, p = 64, 6
    A, a = grid_problem(Float64, 5, n ÷ 2, p)
    X, y = [A; -A], [a; -a]
    for st in (false, true)
      λ = 0.3 * lambda_max(X, y, true, st)
      fit = sqrt_lasso(X, y, λ, cyc(Float64); intercept = true, standardize_x = st)
      @test fit.result.converged && iszero(fit.intercept)
      a_, b_ = check_certified(fit, X, y, λ, true, st)
      st || @test b_.gap <= 1e-4                      # no allowance for the preprocessing
      @test b_.gap <= 1e-6 && a_.gap <= 1e-6
    end
  end

  @testset "level relation: the direct ProximalSolvers call at √n λ, $T" for T in (Float32, Float64)
    for (k, (ic, st)) in enumerate(COMBOS)
      n, p = 60, 8
      X, y = problem(T, 20k, n, p)
      λ = T(0.25 * lambda_max(X, y, ic, st))
      mk() = CDSolver(tol = tolof(T), maxpasses = 2000, rng = Xoshiro(7))
      fit = quiet(() -> sqrt_lasso(X, y, λ, mk(); intercept = ic, standardize_x = st))
      d = transformed(X, y, ic, st)
      θ = zeros(T, p)
      res = quiet(() -> solve!(θ, SqrtLeastSquaresLoss(d.ỹ, d.Z), NormL1(sqrt(Float64(n)) * Float64(λ)), mk()))
      # bit for bit: the solution of the solve at λps is the fit's result, field by field
      r = fit.result
      @test r.minimizer == θ == res.minimizer
      @test all(isequal(getfield(r, f), getfield(res, f)) for f in fieldnames(typeof(r)))
      # the level is a Float64 for either element type
      @test sqrt(Float64(n)) * Float64(λ) isa Float64
      # result.minimum / √n is the objective of sqrt_lasso on the stored problem, to the rounding of
      # the solver's own sums, and on the original data to the perturbation of claim (b)
      # (where the run stopped at a certificate or without one, the minimum is formed from the residual
      # that the certificate recomputed; at maxpasses it is the solver's running residual)
      r.converged && setprecision(BigFloat, 256) do
        Zb, yb, θb = BigFloat.(d.Z), BigFloat.(d.ỹ), BigFloat.(θ)
        e = yb .- Zb * θb
        primal = sqrt(dotb(e, e) / n) + BigFloat(λ) * sum(abs, θb)
        @test abs(BigFloat(r.minimum) / sqrt(BigFloat(n)) - primal) <= (n + p + 8) * BigFloat(eps(T)) * primal
      end
      b = dual_gap(X, y, fit.intercept, fit.coefficients, sqrt(BigFloat(n)) * BigFloat(λ);
                   intercept = ic, standardize = st, eps_T = eps(T), pre = ic || st)
      b === nothing || !r.converged || @test abs(b.P - Float64(r.minimum)) <= 2 * b.ΔP + 4 * (n + p + 8) * eps(T) * b.P
    end
    # a level that Float64 rounds: λ = 1/3 in Float64 for Float32 data keeps the Float64 rounding of √n λ
    X, y = problem(T, 777, 60, 5)
    λ = 1 / 3
    fit = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); standardize_x = false))
    θ = zeros(T, 5)
    res = quiet(() -> solve!(θ, SqrtLeastSquaresLoss(y, X), NormL1(sqrt(60.0) * (1 / 3)), cyc(T)))
    @test fit.result.minimizer == θ && fit.λ == T(1 / 3)
    # a level of another Real type reaches the solver as Float64(λ)
    fit_i = quiet(() -> sqrt_lasso(X, y, 1, cyc(T); standardize_x = false))
    fit_f = quiet(() -> sqrt_lasso(X, y, 1.0, cyc(T); standardize_x = false))
    @test same_fit(fit_i, fit_f)
    fit_r = quiet(() -> sqrt_lasso(X, y, 1 // 2, cyc(T); standardize_x = false))
    fit_h = quiet(() -> sqrt_lasso(X, y, 0.5, cyc(T); standardize_x = false))
    @test same_fit(fit_r, fit_h)
  end

  @testset "the level boundary: β = 0 for λ at or above λmax of the data, $T" for T in (Float32, Float64)
    # λmax = maxⱼ |zⱼᵀỹ|/(√n‖ỹ‖) is computed from the data above (not by the solver); 1 + 1e-9 is above it
    # by far more than the rounding of its computation and 1 - 1e-3 is far above the solver's change
    # threshold. The Float32 level is rounded to Float32, so its sides are 1 ± 1e-3.
    above, below = T === Float32 ? (1 + 1e-3, 1 - 1e-2) : (1 + 1e-9, 1 - 1e-3)
    for (ic, st) in COMBOS
      X, y = problem(T, 31, 60, 8)
      λmax = lambda_max(X, y, ic, st)
      fit = quiet(() -> sqrt_lasso(X, y, T(λmax * above), cyc(T); intercept = ic, standardize_x = st))
      @test iszero(fit.coefficients) && fit.result.converged
      fit = quiet(() -> sqrt_lasso(X, y, T(λmax * below), cyc(T); intercept = ic, standardize_x = st))
      @test count(!iszero, fit.coefficients) >= 1
    end
  end

  @testset "KKT of the transformed problem within the tolerance implied by the stopping test, $T" for T in (Float32, Float64)
    # At a full pass whose largest scaled change is below tolerance = result.tolerance, each coordinate
    # was a minimizer when visited, and the residual direction moved by at most p·tolerance·‖zⱼ‖/‖r‖
    # afterwards, so |violation|/Λps ≤ 2 p tolerance zmax/(‖r‖Λps) (first order, with a factor 2); the
    # roundoff of the correlations adds 4(n + p + 8)u zmax/Λps. The conditions are a diagnostic: the gap
    # of claim (a) carries the evidence.
    for (k, (ic, st)) in enumerate(COMBOS)
      n, p = 60, 8
      X, y = problem(T, 40k, n, p)
      λ = T(0.3 * lambda_max(X, y, ic, st))
      fit = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
      fit.result.converged || continue
      d = transformed(X, y, ic, st)
      Λps = level(λ, n)
      q = kkt(d.Z, d.ỹ, fit.result.minimizer, Λps)
      tol = 2p * Float64(fit.result.tolerance) * q.zmax / (q.N * Λps) + 4 * (n + p + 8) * eps(T) * q.zmax / Λps
      @test q.viol <= tol
    end
  end

  @testset "keyword combinations and defaults, $T" for T in (Float32, Float64)
    n, p = 60, 8
    X, y = problem(T, 50, n, p)
    λ = T(0.3 * lambda_max(X, y, false, true))
    mk() = CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 2000)
    # the defaults are intercept = false, standardize_x = true
    f0 = quiet(() -> sqrt_lasso(X, y, λ, mk()))
    f1 = quiet(() -> sqrt_lasso(X, y, λ, mk(); intercept = false, standardize_x = true))
    @test same_fit(f0, f1) && !f0.hasintercept && iszero(f0.intercept) && length(coef(f0)) == p
    # without a solver argument the solver is CDSolver(), whose coordinate order is random
    @test quiet(() -> sqrt_lasso(X, y, λ)) isa SqrtLassoSolution{T}
    for (ic, st) in COMBOS
      f = quiet(() -> sqrt_lasso(X, y, λ, mk(); intercept = ic, standardize_x = st))
      @test f.hasintercept == ic && length(coef(f)) == p + ic
      ic ? @test(coef(f) == [f.intercept; f.coefficients]) : @test(coef(f) === f.coefficients)
    end
  end

  @testset "invariances, $T" for T in (Float32, Float64)
    # n a power of two and entries on a grid of 1/16 and integer shifts: the sums of a column are exact,
    # so the centred columns, the scales and the solve are bitwise those of the unshifted data
    n, p = 64, 6
    X, y = grid_problem(T, 61, n, p)
    shifts = T[3, -2, 5, 0, 7, -4]
    for st in (false, true)
      λ = T(0.3 * lambda_max(X, y, true, st))
      f = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = true, standardize_x = st))
      g = quiet(() -> sqrt_lasso(X .+ transpose(shifts), y .+ T(9), λ, cyc(T); intercept = true, standardize_x = st))
      @test f.coefficients == g.coefficients
      @test f.result.minimizer == g.result.minimizer && isequal(f.result.gap, g.result.gap)
      # the intercept and the residuals are formed in T from shifted values: equal to the rounding of those sums
      @test g.intercept ≈ f.intercept + T(9) - sum(shifts .* f.coefficients) atol = 8 * eps(T) * (p + 4) * 20
      @test isapprox(g.residuals, f.residuals; atol = 8 * eps(T) * (p + 4) * 40) && isapprox(g.σ, f.σ; atol = 8 * eps(T) * (p + 4) * 40)
    end
    # a power-of-two rescaling of one column: with standardization the weights absorb it, so the standardized
    # coefficient is unchanged and βⱼ scales by 1/c (bit for bit); without it the penalty is on βⱼ itself
    λ = T(0.3 * lambda_max(X, y, true, true))
    f = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = true, standardize_x = true))
    for c in (T(2)^10, T(2)^-10)
      Xc = copy(X); Xc[:, 2] .*= c
      g = quiet(() -> sqrt_lasso(Xc, y, λ, cyc(T); intercept = true, standardize_x = true))
      @test g.coefficients[2] == f.coefficients[2] / c
      @test g.coefficients[[1; 3:p]] == f.coefficients[[1; 3:p]] && g.result.minimizer == f.result.minimizer
      @test g.σ == f.σ && g.intercept == f.intercept
    end
  end

  @testset "extreme representable scales, $T" for (T, k) in ((Float64, 400), (Float32, 40))
    # y -> c y with c a power of two inside the range of the certificate, at a fixed λ: the fit scales by c
    # (the problem is homogeneous of degree one in y), and σ and the intercept too
    n, p = 60, 8
    for (ic, st) in ((true, true), (false, false))
      X, y = problem(T, 71, n, p)
      λ = T(0.3 * lambda_max(X, y, ic, st))
      f = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
      @test f.result.converged
      for c in (T(2)^k, T(2)^-k)
        g = quiet(() -> sqrt_lasso(X, c .* y, λ, cyc(T); intercept = ic, standardize_x = st))
        @test g.result.converged
        @test isapprox(g.coefficients, c .* f.coefficients; rtol = T === Float64 ? 1e-8 : 1e-4)
        @test isapprox(g.σ, c * f.σ; rtol = T === Float64 ? 1e-8 : 1e-4) && isapprox(g.intercept, c * f.intercept; rtol = T === Float64 ? 1e-8 : 1e-4)
        @test g.coefficients ≈ c .* f.coefficients rtol = 1e-4
        # the fit is certified on its own scale: claim (a) at the scaled data
        check_certified(g, X, c .* y, λ, ic, st)
      end
    end
    # design scale, X -> c X: with standardization β scales by 1/c and λ is unchanged; without it λ -> c λ
    X, y = problem(T, 72, n, p)
    for c in (T(2)^(k ÷ 2), T(2)^-(k ÷ 2))
      for (ic, st) in ((true, true), (true, false))
        λ = T(0.3 * lambda_max(X, y, ic, st))
        λc = st ? λ : T(c * λ)
        f = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
        g = quiet(() -> sqrt_lasso(c .* X, y, λc, cyc(T); intercept = ic, standardize_x = st))
        @test f.result.converged && g.result.converged
        @test isapprox(g.coefficients, f.coefficients ./ c; rtol = T === Float64 ? 1e-8 : 1e-4)
        @test isapprox(g.σ, f.σ; rtol = T === Float64 ? 1e-8 : 1e-4)
      end
    end
  end

  @testset "design scale where ProximalSolvers refuses or the standardization absorbs it (Float64)" begin
    # a design whose squares overflow or underflow in Float64: with standardization Z is finite and the fit
    # is that of the unscaled problem; without it ProximalSolvers refuses the curvature, with an ArgumentError
    # that is not changed here, and no fit is returned
    n, p = 60, 8
    X, y = problem(Float64, 73, n, p)
    f = sqrt_lasso(X, y, 0.1, cyc(Float64); intercept = true)
    for c in (1e200, 1e-200)
      g = sqrt_lasso(c .* X, y, 0.1, cyc(Float64); intercept = true)
      @test g.result.converged && isapprox(g.coefficients .* c, f.coefficients; rtol = 1e-8)
      err = try sqrt_lasso(c .* X, y, 0.1 * c, cyc(Float64); intercept = true, standardize_x = false); nothing catch e e end
      @test err isa ArgumentError && occursin("rescale the data", sprint(showerror, err))
    end
  end

  @testset "outside the range of the certificate: an uncertified stop, Float64" begin
    # a response of scale 2^-500 has a residual with Σr̂² below n 2⁻⁹⁶⁹: ProximalSolvers' certificate is
    # undefined there, and the run stops before maxpasses without a certificate
    n, p = 60, 8
    X, y = problem(Float64, 74, n, p)
    c = 2.0^-500
    f = @test_logs (:warn, UNCERTIFIED) sqrt_lasso(X, c .* y, 0.1, cyc(Float64); intercept = true)
    @test !f.result.converged && isnan(f.result.gap) && f.result.passes < 2000
    @test all(isfinite, f.coefficients) && count(!iszero, f.coefficients) >= 1
    f0 = sqrt_lasso(X, y, 0.1, cyc(Float64); intercept = true)
    @test isapprox(f.coefficients, c .* f0.coefficients; rtol = 1e-6)
  end

  @testset "an exact fit that reaches the uncertified stop, $T" for T in (Float32, Float64)
    # dyadic data with y = 2 X[:, 1] exactly, cyclic order, standardize_x = false: the first visit sets
    # β₁ = 2 with the residual exactly zero, and the certificate is undefined there (a zero residual with a
    # nonzero coefficient), which needs λps < ‖X[:, 1]‖ so that the first visit does not set β₁ to 0
    rng = Xoshiro(81)
    X = T.(rand(rng, -4:4, 8, 3)) ./ 4
    X[:, 1] .= T[1, 2, 3, -1, 2, 0.5, -2, 1]
    y = 2 .* X[:, 1]
    λ = T(0.25 * norm(X[:, 1]) / sqrt(8))
    @test level(λ, 8) < norm(Float64.(X[:, 1]))
    f = @test_logs (:warn, UNCERTIFIED) sqrt_lasso(X, y, λ, cyc(T); standardize_x = false)
    @test !f.result.converged && isnan(f.result.gap) && f.result.passes < 2000 && f.result.passes >= 1
    @test f.coefficients[1] == 2 && iszero(f.coefficients[2:3]) && iszero(f.residuals) && iszero(f.σ)
    # with an intercept, data with exact means: x = [1, 2, 3, 6] (mean 3), y = 5 + x/2
    x = T[1, 2, 3, 6]; x2 = T[2, 0, 1, 1]
    X = [x x2]; y = 5 .+ x ./ 2
    λ = T(0.5)
    @test level(λ, 4) < sqrt(sum(abs2, Float64.(x) .- 3))
    f = @test_logs (:warn, UNCERTIFIED) sqrt_lasso(X, y, λ, cyc(T); intercept = true, standardize_x = false)
    @test !f.result.converged && isnan(f.result.gap) && f.result.passes < 2000
    @test f.coefficients == T[0.5, 0] && f.intercept == 5 && iszero(f.residuals) && iszero(f.σ)
    # a generic exact fit from zero: certified (checked independently, where the residual is not zero) or not
    # converged, and never a converged fit that fails the check
    for (ic, st) in COMBOS
      X, _ = problem(T, 82, 60, 8)
      y = X[:, 1:2] * T[1.5, -2.0]
      λ = T(0.1)
      f = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
      f.result.converged ? check_certified(f, X, y, λ, ic, st; exact_fit = true, tight = false) : @test isnan(f.result.gap)
    end
  end

  @testset "near-zero residuals, $T" for T in (Float32, Float64)
    # noise about 1e-10 relative to the signal: certified (checked independently) or not converged
    for (ic, st) in COMBOS
      X, _ = problem(T, 91, 60, 8)
      signal = X[:, 1:3] * T[2, -1.5, 1]
      noise = randn(Xoshiro(92), T, 60)
      y = signal .+ T(1e-10) .* noise .* maximum(abs, signal)
      λ = T(0.1)
      f = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
      f.result.converged ? check_certified(f, X, y, λ, ic, st; exact_fit = T === Float32, tight = false) : @test isnan(f.result.gap)
    end
  end

  @testset "nonconvergence is reported: maxpasses = 1, $T" for T in (Float32, Float64)
    n, p = 60, 8
    X, y = problem(T, 101, n, p)
    for (ic, st) in COMBOS
      λ = T(0.1 * lambda_max(X, y, ic, st))
      f = @test_logs (:warn, CAPPED) sqrt_lasso(X, y, λ, cyc(T; maxpasses = 1); intercept = ic, standardize_x = st)
      @test !f.result.converged && f.result.passes == 1 && isnan(f.result.gap)
      @test all(isfinite, f.coefficients) && isfinite(f.intercept) && isfinite(f.σ) && all(isfinite, f.residuals)
      # the same problem with the default cap and a tight tolerance: a different, converged result
      g = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st))
      T === Float64 && @test g.result.converged && all(isfinite, g.coefficients)
    end
  end

  @testset "a constant response with an intercept, $T" for T in (Float32, Float64)
    n, p = 40, 5
    X, _ = problem(T, 111, n, p)
    for c in (T(3), T(0.1), T(-7.25)), st in (false, true)
      y = fill(c, n)
      f = sqrt_lasso(X, y, T(0.2), cyc(T); intercept = true, standardize_x = st)
      @test iszero(f.coefficients) && f.intercept == c && f.result.converged && iszero(f.σ)
      @test f.result.gap == 0 && f.result.passes == 1 && iszero(f.residuals)
    end
  end

  @testset "historical rows: a zero response and a zero design, $T" for T in (Float32, Float64)
    # (ported from the earlier tests of degenerate data, adapted to the contract of ProximalSolvers)
    n, p = 60, 8
    X, y = problem(T, 121, n, p)
    # a zero response: ỹ = 0 for either keyword (a constant response with an intercept), which ProximalSolvers
    # certifies at θ = 0 in one pass with gap 0: β = 0, β₀ = 0, σ = 0, converged
    for (ic, st) in COMBOS
      f = sqrt_lasso(X, zeros(T, n), T(0.3), cyc(T); intercept = ic, standardize_x = st)
      @test iszero(f.coefficients) && iszero(f.intercept) && f.result.converged && f.result.gap == 0
      @test f.result.passes == 1 && iszero(f.σ) && iszero(f.residuals)
    end
    # a zero design: every column is flat, the solver keeps β at 0 and certifies it; with an intercept the fit is
    # the mean of the response and σ the root mean square of y - ȳ, without one σ is that of y
    X0 = zeros(T, n, p)
    for (ic, st) in COMBOS
      f = quiet(() -> sqrt_lasso(X0, y, T(0.3), cyc(T); intercept = ic, standardize_x = st))
      @test iszero(f.coefficients) && f.result.converged && f.result.gap <= 1e-4
      setprecision(BigFloat, 256) do
        yb = BigFloat.(y)
        ȳ = ic ? sum(yb) / n : zero(BigFloat)
        @test abs(BigFloat(f.intercept) - ȳ) <= 8 * (n + 4) * BigFloat(eps(T)) * maximum(abs, yb)
        rms = sqrt(sum(abs2, yb .- ȳ) / n)
        @test abs(BigFloat(f.σ) - rms) <= 8 * (n + 4) * BigFloat(eps(T)) * rms
      end
      @test f.residuals == y .- f.intercept
    end
    # a zero response and a zero design together
    f = sqrt_lasso(X0, zeros(T, n), T(0.3), cyc(T); intercept = true)
    @test iszero(f.coefficients) && iszero(f.intercept) && f.result.converged && iszero(f.σ)
    # a zero column among others has coefficient 0 (its scale is 0), with a certified fit of the rest
    Xz = copy(X); Xz[:, 3] .= 0
    for (ic, st) in COMBOS
      λ = T(0.3 * lambda_max(Xz, y, ic, st))
      f = quiet(() -> sqrt_lasso(Xz, y, λ, cyc(T); intercept = ic, standardize_x = st))
      @test iszero(f.coefficients[3])
      f.result.converged ? check_certified(f, Xz, y, λ, ic, st) : @test isnan(f.result.gap)
    end
  end

  @testset "errors" begin
    n, p = 40, 5
    X, y = problem(Float64, 131, n, p)
    # a level that is not positive and finite: an HDM ArgumentError that names the contract and `lasso`
    for λ in (0.0, -0.0, -1.0, -1e-300, NaN, Inf, -Inf, 0, -1, 0f0, big"0.0", -1 // 2)
      err = try sqrt_lasso(X, y, λ); nothing catch e e end
      @test err isa ArgumentError
      msg = sprint(showerror, err)
      @test occursin("sqrt_lasso: λ must be positive and finite", msg) && occursin("`lasso`", msg)
    end
    X32, y32 = problem(Float32, 132, n, p)
    @test_throws ArgumentError sqrt_lasso(X32, y32, 0.0f0)
    @test_throws ArgumentError sqrt_lasso(X32, y32, NaN32)
    # a level that Float64 cannot represent: the level √n λ is not a finite positive Float64
    @test_throws ArgumentError sqrt_lasso(X, y, big"1e-400")
    err = try sqrt_lasso(X, y, big"1e-400"); nothing catch e e end
    @test occursin("not a finite positive Float64", sprint(showerror, err))
    # √n λ not representable in T: the error of ProximalSolvers, passed on (λ = 1e-47 and n = 40 gives 6e-47,
    # which rounds to 0 in Float32; 1e46 overflows)
    for T in (Float32,), λ in (1e-47, 1e46)
      Xs, ys = problem(T, 133, n, p)
      @test T(sqrt(Float64(n)) * λ) in (zero(T), T(Inf))
      for st in (false, true)
        err = try sqrt_lasso(Xs, ys, λ; standardize_x = st); nothing catch e e end
        @test err isa ArgumentError
        @test occursin("penalty lambda is not representable without overflow or underflow in Float32", sprint(showerror, err))
      end
    end
    # λ itself in Float64 range but beyond Float64 once multiplied by √n
    @test_throws ArgumentError sqrt_lasso(X, y, 1e308)
    # wrong types: nothing is converted
    for args in ((X32, y, 0.1), (X, y32, 0.1), (Int.(round.(X)), y, 0.1), (view(X, :, 1:3), y, 0.1),
                 (X, view(y, 1:n), 0.1), (X, reshape(y, n, 1), 0.1), (X, y, "0.1"), (X, y, 0.1 + 0im),
                 (X, y, 0.1, :cyclic), (X, y, nothing))
      err = try sqrt_lasso(args...); nothing catch e e end
      @test err isa ArgumentError && occursin("sqrt_lasso needs X::Matrix{T}, y::Vector{T}", sprint(showerror, err))
    end
    # keywords that are not Bool; the message of the shared check names the keyword
    for bad in (1, 0, "true", nothing, missing, 1.0)
      for kw in (:intercept, :standardize_x)
        err = try sqrt_lasso(X, y, 0.1; (kw => bad,)...); nothing catch e e end
        @test err isa ArgumentError && occursin("the keyword $kw must be true or false", sprint(showerror, err))
      end
    end
    # shapes
    @test_throws DimensionMismatch sqrt_lasso(X, y[1:end-1], 0.1)
    @test_throws DimensionMismatch sqrt_lasso(zeros(0, 3), zeros(0), 0.1)
    @test_throws DimensionMismatch sqrt_lasso(zeros(5, 0), zeros(5), 0.1)
    for ic in (false, true), st in (false, true)
      @test_throws DimensionMismatch sqrt_lasso(X, y[1:end-1], 0.1; intercept = ic, standardize_x = st)
    end
    # nonfinite data
    Xn = copy(X); Xn[3, 2] = NaN
    yn = copy(y); yn[4] = Inf
    for ic in (false, true), st in (false, true)
      @test_throws ArgumentError sqrt_lasso(Xn, y, 0.1; intercept = ic, standardize_x = st)
      @test_throws ArgumentError sqrt_lasso(X, yn, 0.1; intercept = ic, standardize_x = st)
    end
    err = try sqrt_lasso(Xn, y, 0.1; intercept = true); nothing catch e e end
    @test occursin("column 2 of X has a non-finite entry", sprint(showerror, err))
    # a design whose scale underflows to zero while it is not all zeros
    @test_throws ArgumentError sqrt_lasso(reshape([5e-324; zeros(39)], :, 1), y, 0.1)
  end

  @testset "conventions: non-mutation, no aliasing, StatsAPI, show, $T" for T in (Float32, Float64)
    n, p = 60, 8
    X, y = problem(T, 141, n, p)
    # historical row (T12): the fit does not write to its arguments and does not return them or reference them
    for (ic, st) in COMBOS
      Xm, ym = copy(X), copy(y)
      f = quiet(() -> sqrt_lasso(Xm, ym, T(0.3), cyc(T); intercept = ic, standardize_x = st))
      @test Xm == X && ym == y
      fields = (f.coefficients, f.residuals, f.result.minimizer)
      @test !any(a -> a === Xm || a === ym || Base.mightalias(a, Xm) || Base.mightalias(a, ym), fields)
      @test !Base.mightalias(f.coefficients, f.residuals)
      # the fit is not changed by later changes of the inputs
      kept = (copy(f.coefficients), copy(f.residuals), f.intercept, f.σ)
      Xm .+= 1; ym .+= 1
      @test (f.coefficients, f.residuals, f.intercept, f.σ) == kept
    end

    λ = T(0.3 * lambda_max(X, y, true, true))
    fit = quiet(() -> sqrt_lasso(X, y, λ, cyc(T); intercept = true, standardize_x = true))
    b = coef(fit)
    @test b isa Vector{T} && b == [fit.intercept; fit.coefficients] && length(b) == p + 1
    @test residuals(fit) === fit.residuals && nobs(fit) == n && nobs(fit) isa Int
    @test dof(fit) == count(!iszero, fit.coefficients) + 1 && dof(fit) isa Int
    @test deviance(fit) == sum(abs2, fit.residuals) && deviance(fit) isa T
    @test predict(fit, X) isa Vector{T} && predict(fit, X) == X * fit.coefficients .+ fit.intercept
    @test predict(fit, X) + residuals(fit) ≈ y rtol = 8 * (p + 4) * eps(T)
    Xnew = randn(Xoshiro(142), T, 7, p)
    @test predict(fit, Xnew) == Xnew * fit.coefficients .+ fit.intercept
    @test predict(fit, zeros(T, 0, p)) == T[]
    @test_throws DimensionMismatch predict(fit, randn(Xoshiro(3), T, 7, p - 1))
    @test_throws MethodError predict(fit, randn(Xoshiro(3), T, p))
    # σ is the root mean square of the residuals, 0 when they are all zero
    @test fit.σ ≈ sqrt(sum(abs2, Float64.(fit.residuals)) / n) rtol = 8 * (n + 4) * eps(T)
    # without an intercept: coef is the coefficient vector itself, dof counts no intercept
    f0 = quiet(() -> sqrt_lasso(X, y, T(0.3 * lambda_max(X, y, false, true)), cyc(T)))
    @test coef(f0) === f0.coefficients && dof(f0) == count(!iszero, f0.coefficients)
    @test predict(f0, X) == X * f0.coefficients
    # show, both certificate states
    s = sprint(show, f0)
    nz = count(!iszero, f0.coefficients)
    @test startswith(s, "SqrtLassoSolution: $p coefficients, $nz nonzero, λ = $(f0.λ), σ = $(f0.σ), $n observations;")
    @test occursin("duality certificate of the transformed problem: ", s) && count(==('\n'), s) == 1
    if f0.result.converged
      @test occursin("duality certificate of the transformed problem: met (relative gap at most $(f0.result.gap)) after $(f0.result.passes) passes", s)
      @test !occursin("not met", s)
    end
    @test s == string(f0) == repr(MIME("text/plain"), f0)
    @test occursin("intercept = $(fit.intercept), σ = ", sprint(show, fit))
    stopped = @test_logs (:warn, CAPPED) sqrt_lasso(X, y, T(0.3 * lambda_max(X, y, false, true)) / 4, cyc(T; maxpasses = 1))
    @test occursin("duality certificate of the transformed problem: not met after 1 passes", sprint(show, stopped))
  end

end

# a summary of the margin of ω in the observed differences of the two gaps (the ratio of the observed
# difference to the allowance); printed once, not asserted beyond the bound itself
isempty(margins) || println("sqrt_lasso: largest observed (gap_b - gap_a)/ω = ", maximum(margins), " over ", length(margins), " certified fits; ", count(isequal(0.0), margins), " with no difference")

end
