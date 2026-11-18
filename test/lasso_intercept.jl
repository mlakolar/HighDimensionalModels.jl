module LassoInterceptTest
# `lasso` with `intercept = true` and `standardize_x = true`: the objective
#   ‖y - β₀1 - Xβ‖² / (2n) + λ Σⱼ sⱼ|βⱼ|,
# with β₀ = 0 unless `intercept` and sⱼ = 1 unless `standardize_x`, where sⱼ is the root mean
# square of the centred column (of the column itself without an intercept). The references share
# nothing with the preprocessing or with coordinate descent:
#   * the optimality conditions and the duality gap in 256-bit BigFloat from the stored inputs;
#   * the exact solution of small problems, by enumeration of supports and signs, in rational
#     arithmetic (intercept) and in 256-bit BigFloat with the weights sⱼ (standardization);
#   * bitwise invariants of the fit on data with n a power of two and entries on a grid of 1/16,
#     where the sums are exact;
#   * least squares at λ = 0, and λmax computed from the data.
# The default path (both keywords false) is that of test/lasso.jl and test/lasso_reference.jl.

using HighDimensionalModels, Test, LinearAlgebra, Random
using ProximalBase: LeastSquaresLoss, NormL1
using ProximalSolvers: solve!

const HDM = HighDimensionalModels

# ---------------------------------------------------------------------------------
# settings and data
# ---------------------------------------------------------------------------------

tolof(T) = T === Float32 ? 1e-6 : 1e-10
cyc(T) = CDSolver(order = :cyclic, momentum = false, tol = tolof(T))
const COMBOS = ((false, false), (true, false), (false, true), (true, true))
label(ic, st) = "intercept = $ic, standardize_x = $st"

bits(a::AbstractArray{Float64}) = reinterpret(UInt64, a)
bits(a::AbstractArray{Float32}) = reinterpret(UInt32, a)
bitwise_equal(a, b) = size(a) == size(b) && bits(a) == bits(b)

# columns of different scales and means, a response with an intercept; `shared` correlates the columns
function problem(T, seed, n, p; shift = 1.5, shared = 0.0, k = min(3, p))
  rng = Xoshiro(seed)
  z = randn(rng, n)
  scales = 0.5 .+ 1.5 .* rand(rng, p)
  X = randn(rng, n, p) .* transpose(scales) .+ transpose(shift .* randn(rng, p)) .+ shared .* z
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  T.(X), T.(X * β .+ 1.0 .+ 0.5 .* randn(rng, n))
end

# data on a grid of 1/16 (exact in Float32 and Float64, and small as rationals), with columns of
# different scales and means
grid(x) = round(x * 16) / 16
function grid_problem(seed, n, p; shared = 0.0, k = min(3, p), shift = 1.0)
  rng = Xoshiro(seed)
  z = randn(rng, n)
  scales = [1.0, 2.0, 0.5, 4.0, 1.5, 3.0, 1.0, 2.5][1:p]
  shifts = round.(shift .* randn(rng, p))
  X = grid.(randn(rng, n, p) .* transpose(scales) .+ transpose(shifts) .+ shared .* z)
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  X, grid.(X * β .+ 1.0 .+ 0.5 .* randn(rng, n))
end

# ---------------------------------------------------------------------------------
# References in extended precision, from the stored Float inputs (call inside setprecision)
# ---------------------------------------------------------------------------------

# A column whose entries are all equal has that value as its mean: the rule of `lasso` for constant
# columns, which keeps them at exact zeros after centring. Other means are the exact means.
bigmean(v) = all(==(first(v)), v) ? first(v) : sum(v) / length(v)

# Xt, yt: the centred data (the data themselves without an intercept); s: the root mean squares of
# the columns of Xt with standardization (zero for a column that is all zeros), else ones;
# Zb: the columns of Xt over s (zeros where s = 0).
function reference(X::Matrix, y::Vector, ic::Bool, st::Bool)
  n, p = size(X)
  Xb, yb = BigFloat.(X), BigFloat.(y)
  x̄ = ic ? [bigmean(Xb[:, j]) for j in 1:p] : zeros(BigFloat, p)
  ȳ = ic ? bigmean(yb) : zero(BigFloat)
  Xt, yt = Xb .- transpose(x̄), yb .- ȳ
  s = st ? [sqrt(sum(abs2, Xt[:, j]) / n) for j in 1:p] : ones(BigFloat, p)
  Zb = similar(Xt)
  for j in 1:p
    Zb[:, j] = iszero(s[j]) ? zeros(BigFloat, n) : Xt[:, j] ./ s[j]
  end
  (; Xb, yb, Xt, yt, x̄, ȳ, s, Zb, Zf = Float64.(Zb), n, p)
end

# λmax of the weighted problem: max over the columns with sⱼ > 0 of |x̃ⱼ'ỹ| / (n sⱼ)
function lambda_max(ref)
  isempty(findall(>(0), ref.s)) && return zero(BigFloat)
  maximum(j -> abs(dot(view(ref.Xt, :, j), ref.yt)) / (ref.n * ref.s[j]), findall(>(0), ref.s))
end

# The violation of the optimality conditions of min ‖y - β₀ - Xβ‖²/(2n) + λΣ sⱼ|βⱼ| at the stored
# (β₀, β), in original coordinates, with g = x̃'r/n, r = y - β₀ - Xβ: for sⱼ > 0, the excess of
# |gⱼ| over λsⱼ, and where βⱼ ≠ 0 the distance of gⱼ from λsⱼ sign(βⱼ). `viol` is per column, in
# the units of the original coordinates (divide by sⱼ for those of θ); `sumr` is Σr.
function kkt_report(fit, ref)
  β, β₀, λb = fit.coefficients, fit.intercept, BigFloat(fit.λ)
  r = ref.yb .- BigFloat(β₀) .- ref.Xb * BigFloat.(β)
  g = ref.Xt' * r / ref.n
  viol = zeros(Float64, ref.p)
  for j in 1:ref.p
    iszero(ref.s[j]) && continue
    w = λb * ref.s[j]
    v = abs(g[j]) - w
    iszero(β[j]) || (v = max(v, abs(g[j] - w * sign(β[j]))))
    viol[j] = Float64(max(v, zero(BigFloat)))
  end
  weighted = dot(r, r) / (2 * ref.n) + λb * sum(j -> ref.s[j] * abs(BigFloat(β[j])), 1:ref.p)
  (; viol, sumr = Float64(sum(r)), primal = Float64(weighted))
end

# The objective and the duality gap of the problem that the solver sees, min ‖ỹ - Zθ‖²/(2n) +
# λ‖θ‖₁, at θ = sβ from the stored β: the gap is the primal value minus the dual value at the
# residual scaled into the dual feasible set (as in test/lasso_reference.jl).
function gap_report(fit, ref)
  λb, n = BigFloat(fit.λ), ref.n
  θ = BigFloat.(fit.coefficients) .* ref.s
  r = ref.yt - ref.Zb * θ
  g = ref.Zb' * r / n
  l1 = sum(abs, θ)
  fitv = dot(r, r) / (2n)
  primal = fitv + λb * l1
  m = maximum(abs, g)
  c = m <= λb ? one(BigFloat) : λb / m
  gap = primal - (dot(ref.yt, ref.yt) - dot(ref.yt - c * r, ref.yt - c * r)) / (2n)
  (; gap = Float64(gap), l1 = Float64(l1), fit = Float64(fitv), primal = Float64(primal))
end

# A bound on the violations of kkt_report after a run that ended on the coefficient-change test,
# per column in original units (κ) and in the units of θ (κθ = κ/s).
#   * The change-test term of test/lasso_reference.jl (`kkt_bound`) for the solver's problem in
#     the columns zⱼ, p · max‖z‖ · tolerance/√n, multiplied by sⱼ to be in original units;
#   * the rounding of the preprocessing and of the back-transformation, c eps(T) (‖xⱼ‖ + √n|x̄ⱼ|)
#     (‖y‖ + √n|ȳ|)/n with c = 16 (p + 4): the centred column and response carry errors of a few
#     eps(T) per entry relative to |xᵢⱼ| and |x̄ⱼ|, and ‖r‖, ‖Xβ‖ ≤ 2‖ỹ‖ ≤ 2(‖y‖ + √n|ȳ|). It also
#     covers the rounding of the solver's own correlations.
function kappa(fit, X::Matrix{T}, y::Vector{T}, ref) where T
  n, p = size(X)
  zmax = maximum(j -> iszero(ref.s[j]) ? 0.0 : Float64(norm(ref.Xt[:, j]) / ref.s[j]), 1:p)
  change = p * zmax * Float64(fit.result.tolerance) / sqrt(n)
  ysc = norm(Float64.(y)) + sqrt(n) * abs(Float64(ref.ȳ))
  κ = zeros(p); κθ = zeros(p)
  for j in 1:p
    iszero(ref.s[j]) && continue
    xsc = norm(Float64.(X[:, j])) + sqrt(n) * abs(Float64(ref.x̄[j]))
    rounding = 16 * (p + 4) * eps(T) * xsc * ysc / n
    sj = Float64(ref.s[j])
    κ[j] = sj * change + rounding
    κθ[j] = change + rounding / sj
  end
  (; κ, κθ)
end

# for the solver's β and the exact β*, both supported on A with X_A of full column rank:
# ‖β - β*‖₂ ≤ √|A| κ / μ, μ = λmin(X_A'X_A/n), when the subgradients are perturbed by at most κ
function coef_bound(Z::Matrix{Float64}, A, κ)
  isempty(A) && return 0.0
  ZA = Z[:, A]
  sqrt(length(A)) * κ / eigmin(Symmetric(ZA' * ZA / size(Z, 1)))
end

# A bound on |β₀ - β₀*| from Δβ: the means of the columns and the rounding of the means, as in `kappa`
function intercept_bound(X::Matrix{T}, y::Vector{T}, ref, Δβ, β) where T
  n, p = size(X)
  16 * (p + 4) * eps(T) * (norm(Float64.(y)) / sqrt(n) + sum(j -> abs(β[j]) * norm(Float64.(X[:, j])) / sqrt(n), 1:p)) +
    sum(j -> abs(Float64(ref.x̄[j])) * Δβ[j], 1:p)
end

# ‖θ₂ - scale θ₁‖∞ against the sum of the bounds of the two fits to the same problem (up to the
# scaling of the data), whose supports contain A: each is within coef_bound of the exact solution
function same_theta(f1, X1, y1, ref1, f2, X2, y2, ref2, A; scale = 1.0)
  b1 = coef_bound(ref1.Zf, A, maximum(kappa(f1, X1, y1, ref1).κθ))
  b2 = coef_bound(ref2.Zf, A, maximum(kappa(f2, X2, y2, ref2).κθ))
  norm(Float64.(f2.result.minimizer) - scale .* Float64.(f1.result.minimizer), Inf) <= scale * b1 + b2
end

# the checks of a fit against the references: optimality conditions, duality gap, the stored
# minimum and the sum of the residuals
function check_fit(fit, X::Matrix{T}, y::Vector{T}, ref) where T
  n, p = size(X)
  ic = fit.hasintercept
  rep = kkt_report(fit, ref)
  gp = gap_report(fit, ref)
  κ, κθ = kappa(fit, X, y, ref)
  @test all(rep.viol .<= κ)
  @test gp.gap >= -1e-12
  if fit.λ > 0
    sθ = maximum(κθ)
    @test gp.gap <= 2sθ * gp.l1 + (sθ / fit.λ)^2 * gp.fit
  end
  # the stored minimum is the weighted objective at (β₀, β) up to the rounding of the preprocessing
  ysc = norm(Float64.(y)) + sqrt(n) * abs(Float64(ref.ȳ))
  @test fit.result.minimum ≈ rep.primal rtol = 64 * (p + 4) * eps(T) atol = 16 * (p + 4) * eps(T) * ysc^2 / (2n)
  if ic
    @test abs(rep.sumr) / n <= 32 * (p + 4) * eps(T) * (norm(Float64.(y)) / sqrt(n) + sum(j -> abs(fit.coefficients[j]) * norm(Float64.(X[:, j])) / sqrt(n), 1:p))
  end
  (; rep, gp, κ, κθ)
end

# ---------------------------------------------------------------------------------
# The exact lasso solution of a small problem, by enumeration (as in test/lasso_reference.jl)
# ---------------------------------------------------------------------------------

# For each support A with X_A of full column rank (|A| ≤ n) and each sign vector s on A, the
# stationarity condition X_A'(y - X_A β_A)/n = λ s gives β_A = (X_A'X_A)⁻¹(X_A'y - nλ s). The
# pair is a solution when sign(β_A) = s and |X_j'(y - X_A β_A)|/n ≤ λ off A. When the
# inequalities hold strictly the solution is unique. Returns the passing candidates, each
# (A, β on 1:p, strict).
function exact_candidates(X::Matrix, y::Vector, λ)
  n, p = size(X)
  Q = Rational{BigInt}
  Xr, yr, λr = Q.(X), Q.(y), Q(λ)
  found = NamedTuple{(:A, :β, :strict), Tuple{Vector{Int}, Vector{Q}, Bool}}[]
  for mask in 0:(2^p - 1)
    A = [j for j in 1:p if (mask >> (j - 1)) & 1 == 1]
    k = length(A)
    k <= n || continue
    XA = Xr[:, A]
    G = XA' * XA
    if k > 0
      iszero(det(G)) && continue
      Ginv = inv(G)
      c = XA' * yr
    end
    for s in Iterators.product(ntuple(_ -> (-1, 1), k)...)
      sv = collect(Q, s)
      βA = k == 0 ? Q[] : Ginv * (c - n * λr * sv)
      all(i -> βA[i] * sv[i] > 0, 1:k) || continue
      g = Xr' * (yr - XA * βA) / n
      off = setdiff(1:p, A)
      all(j -> abs(g[j]) <= λr, off) || continue
      β = zeros(Q, p)
      β[A] = βA
      push!(found, (A = A, β = β, strict = all(j -> abs(g[j]) < λr, off)))
    end
  end
  found
end

function exact_lasso(X, y, λ)
  c = exact_candidates(X, y, λ)
  (length(c) == 1 && only(c).strict) || error("the exact reference is not unique: $(length(c)) candidates")
  only(c)
end

# ---------------------------------------------------------------------------------

setprecision(BigFloat, 256) do
@testset "lasso with an intercept and standardization" begin

  # -------------------------------------------------------------------------------
  @testset "exact solutions of small problems with an intercept, $T" for T in (Float32, Float64)
    # the data are centred in Rational{BigInt} arithmetic, which is exact; the intercept is the mean of y minus x̄'β
    cases = [
      ("independent, p = 5", grid_problem(30, 20, 5), 0.30),
      ("correlated, p = 6", grid_problem(31, 25, 6; shared = 2.0), 0.20),
      ("p > n (6 × 8)", grid_problem(32, 6, 8), 0.30),
    ]
    Q = Rational{BigInt}
    for (name, (X64, y64), frac) in cases
      X, y = T.(X64), T.(y64)
      n, p = size(X)
      Xr, yr = Q.(X), Q.(y)
      x̄ = vec(sum(Xr, dims = 1)) ./ n
      ȳ = sum(yr) / n
      Xc, yc = Xr .- transpose(x̄), yr .- ȳ
      λ = T(frac * Float64(maximum(abs, Xc' * yc) / n))
      ex = exact_lasso(Xc, yc, Float64(λ))
      fit = lasso(X, y, λ, cyc(T); intercept = true)
      β = Float64.(fit.coefficients)
      ref = reference(X, y, true, false)
      κθ = maximum(kappa(fit, X, y, ref).κθ)
      @test findall(!iszero, β) == ex.A
      @test sign.(β) == sign.(Float64.(ex.β))
      bound = coef_bound(ref.Zf, ex.A, κθ)
      @test norm(β - Float64.(ex.β), Inf) <= bound
      β₀exact = Float64(ȳ - dot(x̄, ex.β))
      @test abs(Float64(fit.intercept) - β₀exact) <= intercept_bound(X, y, ref, abs.(β - Float64.(ex.β)), β)
      @test fit.result.minimizer === fit.coefficients
    end
  end

  @testset "exact solutions of small problems with standardization, $T, intercept = $ic" for T in (Float32, Float64), ic in (false, true)
    # the weighted problem is the plain lasso on Z with the columns x̃ⱼ/sⱼ, solved by the same
    # enumeration with Z rounded to 256 bits; θ = sβ
    cases = [
      ("independent, p = 5", grid_problem(30, 20, 5), 0.30),
      ("correlated, p = 6", grid_problem(31, 25, 6; shared = 2.0), 0.20),
      ("p > n (6 × 8)", grid_problem(32, 6, 8), 0.30),
    ]
    for (name, (X64, y64), frac) in cases
      X, y = T.(X64), T.(y64)
      n, p = size(X)
      ref = reference(X, y, ic, true)
      λ = T(frac * Float64(lambda_max(ref)))
      Zr = Rational{BigInt}.(ref.Zb)
      yr = Rational{BigInt}.(ref.yt)
      ex = exact_lasso(Zr, yr, Float64(λ))
      fit = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = true)
      θ = Float64.(fit.result.minimizer)
      κθ = maximum(kappa(fit, X, y, ref).κθ)
      @test findall(!iszero, fit.coefficients) == ex.A
      @test sign.(θ) == sign.(Float64.(ex.β))
      bound = coef_bound(ref.Zf, ex.A, κθ)
      @test norm(θ - Float64.(ex.β), Inf) <= bound
      # the coefficients are θ/s; the rounding of the division is half an ulp
      s = Float64.(ref.s)
      β = Float64.(fit.coefficients)
      for j in 1:p
        @test abs(β[j] - Float64(ex.β[j] / ref.s[j])) <= (bound + 2eps(T) * abs(θ[j])) / s[j] + eps(T) * abs(β[j])
      end
      if ic
        β₀exact = Float64(ref.ȳ - dot(ref.x̄, BigFloat.(ex.β) ./ ref.s))
        Δβ = [abs(β[j] - Float64(ex.β[j] / ref.s[j])) for j in 1:p]
        @test abs(Float64(fit.intercept) - β₀exact) <= intercept_bound(X, y, ref, Δβ, β)
      else
        @test iszero(fit.intercept) && !fit.hasintercept
      end
      @test fit.result.minimizer !== fit.coefficients
    end
  end

  # -------------------------------------------------------------------------------
  @testset "bitwise invariants, n = 32, $T" for T in (Float32, Float64)
    # n is a power of two and the data lie on a grid, so that the sums for the means, the centred
    # entries and the scales are exact: the solver sees the same bits
    n, p = 32, 4
    X64, y64 = grid_problem(50, n, p)
    X, y = T.(X64), T.(y64)
    λ = T(0.15 * Float64(lambda_max(reference(X, y, true, true))))
    for st in (false, true)
      base = lasso(X, y, λ, cyc(T); intercept = true, standardize_x = st)
      @test count(!iszero, base.coefficients) >= 1
      # X .+ 8 changes the intercept and nothing else
      shifted = lasso(X .+ T(8), y, λ, cyc(T); intercept = true, standardize_x = st)
      @test bitwise_equal(shifted.coefficients, base.coefficients)
      @test shifted.intercept ≈ base.intercept - 8 * sum(base.coefficients) rtol = 64 * eps(T) atol = 64 * eps(T) * 8 * sum(abs, base.coefficients)
      # a power-of-two scaling of a column divides its coefficient by that scale, with λ unchanged for standardization
      st || continue
      for c in (T(8), T(0.25)), j in (2, 4)
        Xc = copy(X); Xc[:, j] .*= c
        f = lasso(Xc, y, λ, cyc(T); intercept = true, standardize_x = true)
        want = copy(base.coefficients); want[j] /= c
        @test bitwise_equal(f.coefficients, want)
        @test bitwise_equal(f.result.minimizer, base.result.minimizer)
        @test bitwise_equal([f.intercept], [base.intercept])
      end
    end
    # without an intercept the standardization alone gives the same invariance under scaling
    base = lasso(X, y, λ, cyc(T); standardize_x = true)
    Xc = copy(X); Xc[:, 3] .*= T(4)
    f = lasso(Xc, y, λ, cyc(T); standardize_x = true)
    want = copy(base.coefficients); want[3] /= T(4)
    @test bitwise_equal(f.coefficients, want)
  end

  @testset "balanced ±1 columns with a zero-sum response reproduce the default fit, $T" for T in (Float32, Float64)
    n, p = 32, 5
    rng = Xoshiro(51)
    X = T[(i <= 16 ? 1 : -1) for i in 1:n, j in 1:p]
    for j in 1:p
      X[:, j] = X[randperm(rng, n), j]
    end
    y64 = grid.(Float64.(X[:, 1]) .- 0.5 .* Float64.(X[:, 2]) .+ 0.4 .* randn(rng, n))
    y64[end] = -sum(y64[1:end-1])                                    # a zero sum (sums of grid values are exact)
    y = T.(y64)
    @test sum(y) == 0
    for λ in (T(0.05), T(0.2))
      d = lasso(X, y, λ, cyc(T))
      @test any(!iszero, d.coefficients)
      for (ic, st) in COMBOS[2:end]
        f = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st)
        @test bitwise_equal(f.coefficients, d.coefficients)
        @test bitwise_equal(f.residuals, d.residuals)
        @test f.result.passes == d.result.passes && f.result.minimum === d.result.minimum
        @test iszero(f.intercept)
        @test f.hasintercept == ic
      end
    end
  end

  # -------------------------------------------------------------------------------
  @testset "optimality conditions and duality gap in extended precision, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    problems = [
      ("independent columns", problem(T, 10, 40, 8), 0.15),
      ("correlated columns", problem(T, 11, 50, 10; shared = 2.0), 0.10),
      ("p > n", problem(T, 12, 15, 40), 0.25),
    ]
    # a mean shift of order 1e3 is ill conditioned without an intercept: this problem is run with an intercept
    ic && push!(problems, ("mean shift 1e3", problem(T, 13, 40, 6; shift = 1e3), 0.15))
    for (name, (X, y), frac) in problems
      ref = reference(X, y, ic, st)
      λ = T(frac * Float64(lambda_max(ref)))
      fit = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st)
      @test fit.result.converged
      @test fit.hasintercept == ic
      check_fit(fit, X, y, ref)
      @test any(!iszero, fit.coefficients) && any(iszero, fit.coefficients)
      # the returned residuals are those of the original data at the returned point
      r = y .- fit.intercept .- X * fit.coefficients
      @test fit.residuals == r
    end
  end

  # -------------------------------------------------------------------------------
  @testset "λ = 0 is least squares, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    # full column rank: the least squares solution of the centred (and scaled) data in extended precision
    X, y = problem(T, 20, 40, 5)
    n, p = size(X)
    ref = reference(X, y, ic, st)
    fit = lasso(X, y, 0, cyc(T); intercept = ic, standardize_x = st)
    θ = Float64.(fit.result.minimizer)
    θls = Float64.(ref.Zb \ ref.yt)
    check = check_fit(fit, X, y, ref)
    bound = coef_bound(ref.Zf, 1:p, maximum(check.κθ))
    @test norm(θ - θls, Inf) <= bound + 4eps(T) * maximum(abs, θls)
    if ic
      βls = θls ./ Float64.(ref.s)
      @test abs(Float64(fit.intercept) - Float64(ref.ȳ - dot(ref.x̄, BigFloat.(βls)))) <=
            intercept_bound(X, y, ref, abs.(Float64.(fit.coefficients) .- βls), βls)
    end
    # rank deficient (a column that is the sum of two others, and a duplicate) and wide designs:
    # the fitted values, which are determined, are the projection of ỹ on the columns of Z, up to
    # n √p κ/σ for the smallest nonzero singular value σ of Z (the component of the residuals in
    # the range of Z has Z'r = n g, ‖g‖ ≤ √p κ)
    Xr = [X X[:, 1] .+ X[:, 2] X[:, 3]]
    Xw, yw = problem(T, 21, 12, 30)
    for (name, Xd, yd) in (("rank deficient", Xr, y), ("wide", Xw, yw))
      refd = reference(Xd, yd, ic, st)
      f = lasso(Xd, yd, 0, cyc(T); intercept = ic, standardize_x = st)
      @test f.result.converged
      c = check_fit(f, Xd, yd, refd)
      F = svd(refd.Zf)
      tol = max(size(refd.Zf)...) * eps() * F.S[1]
      r = count(>(tol), F.S)
      σ = F.S[r]
      P = F.U[:, 1:r]
      θd = Float64.(f.result.minimizer)
      want = P * (P' * Float64.(refd.yt))
      err = norm(refd.Zf * θd - want)
      @test err <= size(Xd, 1) * sqrt(size(Xd, 2)) * maximum(c.κθ) / σ + 64 * eps() * (F.S[1] / σ) * norm(Float64.(refd.yt))
    end
  end

  # -------------------------------------------------------------------------------
  @testset "λmax: the zero solution above it and an active coordinate below it, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    X, y = problem(T, 30, 30, 6)
    ref = reference(X, y, ic, st)
    λm = Float64(lambda_max(ref))
    above = T === Float32 ? 1.001 : 1 + 1e-9
    fit = lasso(X, y, T(above * λm), cyc(T); intercept = ic, standardize_x = st)
    @test iszero(fit.coefficients) && fit.result.converged
    if ic
      @test abs(Float64(fit.intercept) - Float64(ref.ȳ)) <= 16 * 8 * eps(T) * norm(Float64.(y)) / sqrt(length(y))
    else
      @test iszero(fit.intercept)
    end
    @test fit.residuals ≈ y .- fit.intercept
    # below it the coordinate with the largest |x̃ⱼ'ỹ|/sⱼ enters with the sign of x̃ⱼ'ỹ
    below = lasso(X, y, T((1 - 1e-3) * λm), cyc(T); intercept = ic, standardize_x = st)
    j = argmax([iszero(ref.s[k]) ? 0.0 : Float64(abs(dot(ref.Xt[:, k], ref.yt)) / ref.s[k]) for k in 1:ref.p])
    @test below.coefficients[j] != 0 && sign(below.coefficients[j]) == sign(Float64(dot(ref.Xt[:, j], ref.yt)))
    # a larger level gives zero as well
    @test iszero(lasso(X, y, T(1e3 * λm), cyc(T); intercept = ic, standardize_x = st).coefficients)
  end

  # -------------------------------------------------------------------------------
  @testset "constant and zero columns, $T" for T in (Float32, Float64)
    X, y = problem(T, 40, 30, 4)
    n = 30
    λ = T(0.15)
    # columns 3, 4 and 5 are constant (0.1 has no exact mean as a sum of copies), zero and constant
    Xz = [X[:, 1:2] fill(T(0.1), n) zeros(T, n) fill(T(-3), n) X[:, 3:4]]
    keep = [1, 2, 6, 7]
    for st in (false, true)
      f = @test_logs lasso(Xz, y, λ, cyc(T); intercept = true, standardize_x = st)
      @test all(j -> f.coefficients[j] === zero(T), 3:5)
      @test f.result.converged && all(isfinite, f.coefficients) && isfinite(f.intercept)
      base = lasso(X, y, λ, cyc(T); intercept = true, standardize_x = st)
      refz, refb = reference(Xz, y, true, st), reference(X, y, true, st)
      check_fit(f, Xz, y, refz)
      check_fit(base, X, y, refb)
      # the same fit as without the columns, up to the two optimality bounds
      A = findall(!iszero, base.coefficients)
      b1 = coef_bound(refb.Zf, A, maximum(kappa(base, X, y, refb).κθ))
      b2 = coef_bound(refz.Zf[:, keep], A, maximum(kappa(f, Xz, y, refz).κθ))
      @test norm(Float64.(f.result.minimizer[keep]) - Float64.(base.result.minimizer), Inf) <= b1 + b2
      # the means of the constant columns are their values, so that they centre to exact zeros
      d = HDM.preprocess(Xz, y, true, st)
      @test d.x̄[3] === T(0.1) && d.x̄[4] === zero(T) && d.x̄[5] === T(-3)
      @test all(iszero, d.Z[:, 3:5])
      st && @test d.s[3:5] == zeros(T, 3)
    end
    # without an intercept a constant column is an ordinary column, with scale 0.1 for fill(0.1, n):
    # it absorbs the intercept of the response
    for st in (false, true)
      Xc = [X[:, 1:2] fill(T(0.1), n) X[:, 3:4]]
      yc = y .+ T(10)
      f = lasso(Xc, yc, λ, cyc(T); standardize_x = st)
      @test f.coefficients[3] != 0
      refc = reference(Xc, yc, false, st)
      check_fit(f, Xc, yc, refc)
      st && @test HDM.preprocess(Xc, yc, false, true).s[3] ≈ 0.1 rtol = 4eps(T)
    end
    # all-zero and constant columns together with a zero design
    f = lasso(zeros(T, 10, 3), y[1:10], λ, cyc(T); intercept = true, standardize_x = true)
    @test iszero(f.coefficients) && f.result.converged && isfinite(f.intercept)
    @test abs(Float64(f.intercept) - sum(Float64.(y[1:10])) / 10) <= 16 * eps(T) * maximum(abs, y[1:10])
    @test f.residuals ≈ y[1:10] .- f.intercept
    f = lasso(zeros(T, 10, 3), y[1:10], λ, cyc(T); intercept = true)
    @test iszero(f.coefficients) && abs(Float64(f.intercept) - sum(Float64.(y[1:10])) / 10) <= 16 * eps(T) * maximum(abs, y[1:10])
    f = lasso(zeros(T, 10, 3), y[1:10], λ, cyc(T); standardize_x = true)
    @test iszero(f.coefficients) && iszero(f.intercept) && f.residuals == y[1:10]
  end

  @testset "constant and zero responses, $T" for T in (Float32, Float64)
    X, _ = problem(T, 41, 25, 4)
    λ = T(0.2)
    # a constant response with an intercept: the intercept is its value and the coefficients are 0
    yc = fill(T(3), 25)
    for st in (false, true)
      f = lasso(X, yc, λ, cyc(T); intercept = true, standardize_x = st)
      @test iszero(f.coefficients) && f.intercept === T(3) && iszero(f.residuals) && f.result.converged
    end
    # without an intercept it is an ordinary response
    for st in (false, true)
      f = lasso(X, yc, λ, cyc(T); standardize_x = st)
      check_fit(f, X, yc, reference(X, yc, false, st))
    end
    # the zero response: β = 0, β₀ = 0 and zero residuals, in each combination
    for (ic, st) in COMBOS
      f = lasso(X, zeros(T, 25), λ, cyc(T); intercept = ic, standardize_x = st)
      @test iszero(f.coefficients) && iszero(f.intercept) && iszero(f.residuals) && f.result.converged
      f = lasso(X, zeros(T, 25), 0, cyc(T); intercept = ic, standardize_x = st)
      @test iszero(f.coefficients) && iszero(f.intercept) && f.result.converged
    end
  end

  @testset "one observation, $T" for T in (Float32, Float64)
    X = T[1 2 3]; y = T[2]
    for st in (false, true)
      f = lasso(X, y, T(0.1), cyc(T); intercept = true, standardize_x = st)
      @test iszero(f.coefficients) && f.intercept === T(2) && iszero(f.residuals) && f.result.converged
      @test nobs(f) == 1 && dof(f) == 1 && iszero(deviance(f))
    end
    # without an intercept and with standardization the columns have scales 1, 2, 3 and are identical
    # after scaling: the sum of the coefficients of Z is determined
    ref = reference(X, y, false, true)
    f = lasso(X, y, T(0.1), cyc(T); standardize_x = true)
    check_fit(f, X, y, ref)
    @test ref.s == BigFloat[1, 2, 3]
  end

  # -------------------------------------------------------------------------------
  @testset "scales: the root mean square, subnormal columns, overflow" begin
    # Float32 entries near 1e37 with n = 10⁴: the norm of the column overflows, the root mean square does not
    n = 10^4
    rng = Xoshiro(60)
    x = 1f37 .* randn(rng, Float32, n)
    @test isinf(norm(x)) && all(isfinite, x)
    X = [x randn(rng, Float32, n, 2)]
    y = X[:, 2] .- 0.5f0 .* X[:, 3] .+ 0.3f0 .* randn(rng, Float32, n)
    for ic in (false, true)
      d = HDM.preprocess(X, y, ic, true)
      ref = reference(X, y, ic, true)
      @test all(isfinite, d.s) && all(isfinite, d.Z)
      for j in 1:3
        @test Float64(d.s[j]) ≈ Float64(ref.s[j]) rtol = 4 * eps(Float32) * (ceil(log2(n)) + 2)
      end
      λ = Float32(0.3 * Float64(lambda_max(ref)))
      fit = lasso(X, y, λ, cyc(Float32); intercept = ic, standardize_x = true)
      @test fit.result.converged && all(isfinite, fit.coefficients) && all(isfinite, fit.residuals)
      check_fit(fit, X, y, ref)
      # the default path and the centred path without scales refuse: the squares of the column overflow
      @test_throws ArgumentError lasso(X, y, λ, cyc(Float32); intercept = ic)
    end
    @test_throws ArgumentError lasso(X, y, 0.1f0, cyc(Float32))

    # Float64: a column of 1e-300, whose squares underflow, has a representable scale; a back-transformation
    # that overflows is an error that names the column of X
    n = 20
    rng = Xoshiro(61)
    x1 = 1e-300 .* randn(rng, n)
    X = [x1 randn(rng, n)]
    d = HDM.preprocess(X, zeros(n), false, true)
    @test d.s[1] ≈ 1e-300 * sqrt(sum(abs2, x1 ./ 1e-300) / n) rtol = 1e-14
    @test all(isfinite, d.Z)
    @test sum(abs2, d.Z[:, 1]) ≈ n rtol = 1e-14
    y = 1e12 .* randn(rng, n)
    @test_throws "column 1 of X" lasso(X, y, 0.0, cyc(Float64); standardize_x = true)

    # a subnormal column: positive scale and finite Z, with and without an intercept
    sub = 5e-324 .* collect(1.0:10.0)
    @test all(x -> 0 < x < floatmin(Float64), sub)
    Xs = [sub randn(Xoshiro(62), 10)]
    ys = randn(Xoshiro(63), 10)
    for ic in (false, true)
      d = HDM.preprocess(Xs, ys, ic, true)
      @test d.s[1] > 0 && all(isfinite, d.Z)
      ref = reference(Xs, ys, ic, true)
      @test abs(d.s[1] - Float64(ref.s[1])) <= max(8 * eps() * Float64(ref.s[1]), 2 * nextfloat(0.0))
      # the coefficient of the subnormal column would be θ/s with s about 1e-323: a level above λmax keeps it 0
      λ = 1.001 * Float64(lambda_max(ref))
      fit = lasso(Xs, ys, λ, cyc(Float64); intercept = ic, standardize_x = true)
      @test fit.result.converged && iszero(fit.coefficients) && all(isfinite, fit.residuals)
    end
    # a nonzero column whose scale underflows to 0 is refused with a message that names the column of X
    tiny = [5e-324; zeros(99)]
    Xt = [tiny randn(Xoshiro(64), 100)]
    @test_throws "underflows" lasso(Xt, randn(Xoshiro(65), 100), 0.1, cyc(Float64); standardize_x = true)

    # centring overflows: entries near ±floatmax with a mean that is not zero
    big = floatmax(Float64)
    Xo = [big, big, -big, 1.0, 2.0] * [1.0 0.0] .+ [0.0 1.0] .* [1.0, 2.0, 3.0, 4.0, 5.0]
    yo = [1.0, 2.0, 3.0, 4.0, 5.0]
    @test_throws "centring column 1 of X" lasso(Xo, yo, 0.1, cyc(Float64); intercept = true)
    @test_throws "centring column 1 of X" lasso(Xo, yo, 0.1, cyc(Float64); intercept = true, standardize_x = true)
    @test_throws "centring y" lasso(Xo[:, 2:2], [big, big, -big, 1.0, 2.0], 0.1, cyc(Float64); intercept = true)
    # non-finite entries are named
    Xn = copy(Xo[:, 2:2]); Xn[2] = NaN
    @test_throws "column 1 of X" lasso(Xn, yo, 0.1, cyc(Float64); intercept = true)
    @test_throws "column 1 of X" lasso(Xn, yo, 0.1, cyc(Float64); standardize_x = true)
    yn = copy(yo); yn[3] = Inf
    @test_throws "y has a non-finite entry" lasso(Xo[:, 2:2], yn, 0.1, cyc(Float64); intercept = true)
    @test_throws ArgumentError lasso(Xo[:, 2:2], yn, 0.1, cyc(Float64); standardize_x = true)
  end

  # -------------------------------------------------------------------------------
  @testset "extreme but representable scales, Float64, $(label(ic, st))" for (ic, st) in COMBOS[2:end]
    X, y = problem(Float64, 70, 30, 6)
    λ = 0.2
    base = lasso(X, y, λ, cyc(Float64); intercept = ic, standardize_x = st)
    refb = reference(X, y, ic, st)
    check_fit(base, X, y, refb)
    A = findall(!iszero, base.coefficients)
    @test !isempty(A)
    # X → cX: with standardization λ is unchanged and β → β/c; without it β → β/c needs λ → cλ
    # (the solver's θ is the same up to the optimality bounds, or c θ for θ = β)
    for c in (st ? (1e-150, 1e150, 1e200) : (1e-150, 1e150))
      Xc = c .* X
      fit = lasso(Xc, y, st ? λ : c * λ, cyc(Float64); intercept = ic, standardize_x = st)
      @test fit.result.converged && all(isfinite, fit.coefficients)
      refc = reference(Xc, y, ic, st)
      check_fit(fit, Xc, y, refc)
      @test same_theta(base, X, y, refb, fit, Xc, y, refc, A; scale = st ? 1.0 : 1 / c)
    end
    # beyond the range of Float64 the squares of the centred design overflow and the solver refuses
    # when the columns are not scaled
    st || @test_throws ArgumentError lasso(1e200 .* X, y, 1e200 * λ, cyc(Float64); intercept = ic)
    # y → cy at the level cλ: β → cβ and θ → cθ, in extremes where the squares of y underflow (1e-150
    # and 1e-100) or are large. A run reports convergence, or says that it did not.
    for c in (1e-150, 1e-100, 1e100)
      yc = c .* y
      fit = lasso(X, yc, c * λ, cyc(Float64); intercept = ic, standardize_x = st)
      ok = !fit.result.converged || begin
        refc = reference(X, yc, ic, st)
        check_fit(fit, X, yc, refc)
        all(isfinite, fit.coefficients) && same_theta(base, X, y, refb, fit, X, yc, refc, A; scale = c)
      end
      @test ok
    end
  end

  @testset "the default path refuses a design whose squares overflow, $T" for T in (Float32, Float64)
    X, y = problem(T, 71, 30, 6)
    big = T === Float32 ? T(1e20) : T(1e200)
    @test_throws ArgumentError lasso(big .* X, y, 0.1, cyc(T))
    # with standardization and an intercept the same data are fit, and agree with the fit of X after scaling back
    f = lasso(big .* X, y, T(0.2), cyc(T); intercept = true, standardize_x = true)
    g = lasso(X, y, T(0.2), cyc(T); intercept = true, standardize_x = true)
    @test f.result.converged && all(isfinite, f.coefficients)
    reff, refg = reference(big .* X, y, true, true), reference(X, y, true, true)
    check_fit(f, big .* X, y, reff)
    @test same_theta(g, X, y, refg, f, big .* X, y, reff, findall(!iszero, g.coefficients))
  end

  # -------------------------------------------------------------------------------
  @testset "element types: Float32 in, Float32 out; $(label(ic, st))" for (ic, st) in COMBOS
    X, y = problem(Float32, 80, 30, 5)
    for λ in (0.1f0, 0.1)
      fit = lasso(X, y, λ, cyc(Float32); intercept = ic, standardize_x = st)
      @test fit isa LassoSolution{Float32}
      @test eltype(fit.coefficients) === Float32 && eltype(fit.residuals) === Float32
      @test fit.intercept isa Float32 && fit.λ isa Float32
      @test eltype(coef(fit)) === Float32 && eltype(predict(fit, X)) === Float32
      @test eltype(fit.result.minimizer) === Float32 && fit.result.converged
      @test deviance(fit) isa Float32
    end
    Xd, yd = problem(Float64, 80, 30, 5)
    fit = lasso(Xd, yd, 0.1, cyc(Float64); intercept = ic, standardize_x = st)
    @test fit isa LassoSolution{Float64} && fit.intercept isa Float64
  end

  # -------------------------------------------------------------------------------
  @testset "conventions, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    X, y = problem(T, 90, 30, 6)
    ref = reference(X, y, ic, st)
    λ = T(0.15 * Float64(lambda_max(ref)))
    fit = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st)
    n, p = size(X)
    @test fit.hasintercept === ic
    @test ic ? length(coef(fit)) == p + 1 && coef(fit) == [fit.intercept; fit.coefficients] : coef(fit) === fit.coefficients
    @test length(fit.coefficients) == p && length(fit.residuals) == n
    @test fit.residuals == y .- fit.intercept .- X * fit.coefficients
    @test predict(fit, X) isa Vector{T}
    @test predict(fit, X) == (ic ? X * fit.coefficients .+ fit.intercept : X * fit.coefficients)
    @test predict(fit, X) + fit.residuals ≈ y rtol = 8 * eps(T) atol = 8 * eps(T) * maximum(abs, y)
    Xnew = randn(Xoshiro(91), T, 7, p)
    @test predict(fit, Xnew) ≈ Xnew * fit.coefficients .+ fit.intercept
    @test predict(fit, view(Xnew, 2:4, :)) ≈ predict(fit, Xnew)[2:4]
    @test predict(fit, zeros(T, 0, p)) == T[]
    @test predict(fit, Float64.(Xnew)) isa Vector{Float64}
    @test_throws DimensionMismatch predict(fit, randn(Xoshiro(92), T, 7, p - 1))
    @test_throws DimensionMismatch predict(fit, randn(Xoshiro(92), T, 7, p + 1))
    @test_throws DimensionMismatch predict(fit, zeros(T, 3, 0))
    @test dof(fit) isa Int && dof(fit) == count(!iszero, fit.coefficients) + ic
    @test deviance(fit) == sum(abs2, residuals(fit)) && residuals(fit) === fit.residuals
    @test nobs(fit) == n && nobs(fit) isa Int
    # the documented field `minimizer`: the coefficients themselves unless the columns are scaled
    if st
      @test fit.result.minimizer !== fit.coefficients
      s = Float64.(ref.s)
      @test Float64.(fit.result.minimizer) ≈ Float64.(fit.coefficients) .* s rtol = 4eps(T) atol = 4eps(T) * maximum(abs, fit.result.minimizer)
    else
      @test fit.result.minimizer === fit.coefficients
    end
    @test propertynames(fit) == (:coefficients, :residuals, :λ, :intercept, :hasintercept, :result)
    # the printed form shows the intercept when there is one
    s = sprint(show, fit)
    @test startswith(s, "LassoSolution: $p coefficients, $(count(!iszero, fit.coefficients)) nonzero, λ = $λ")
    @test occursin("intercept = $(fit.intercept)", s) == ic
    @test count(==('\n'), s) == 1 && s == string(fit)
    # no coefficient: the stored fields are read
    f0 = lasso(X, y, T(1e3 * Float64(lambda_max(ref))), cyc(T); intercept = ic, standardize_x = st)
    @test iszero(f0.coefficients) && dof(f0) == ic && f0.residuals == y .- f0.intercept
    @test predict(f0, Xnew) == fill(f0.intercept, 7)
  end

  # -------------------------------------------------------------------------------
  @testset "the inputs are not modified, and the fit does not alias them, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    X, y = problem(T, 100, 25, 6)
    X0, y0 = copy(X), copy(y)
    λ = T(0.2)
    fit = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st)
    @test bitwise_equal(X, X0) && bitwise_equal(y, y0)
    @test fit.coefficients !== y && fit.residuals !== y
    for f in 1:fieldcount(typeof(fit.result))
      v = getfield(fit.result, f)
      v isa AbstractArray && @test v !== X && v !== y
    end
    β0, r0, m0 = copy(fit.coefficients), copy(fit.residuals), copy(fit.result.minimizer)
    fill!(X, 7); fill!(y, -3)
    @test fit.coefficients == β0 && fit.residuals == r0 && fit.result.minimizer == m0
    X, y = copy(X0), copy(y0)
    f1 = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st)
    f2 = lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st)
    @test f1.coefficients !== f2.coefficients && f1.residuals !== f2.residuals
    @test bitwise_equal(f1.coefficients, f2.coefficients) && f1.intercept === f2.intercept
    f1.coefficients .= 0; f1.residuals .= 0
    @test f2.coefficients == β0 && f2.residuals == r0
    # a failing call leaves the data unchanged
    Xn = copy(X); Xn[3, 3] = NaN
    yn = copy(y); yn[2] = Inf
    for (Xa, ya, λa) in ((Xn, y, λ), (X, yn, λ), (X, y, -λ), (X, y[1:5], λ))
      Xa0, ya0 = copy(Xa), copy(ya)
      @test_throws Exception lasso(Xa, ya, λa, cyc(T); intercept = ic, standardize_x = st)
      @test isequal(Xa, Xa0) && isequal(ya, ya0)
    end
    @test bitwise_equal(X, X0) && bitwise_equal(y, y0)
    big = floatmax(T)
    Xo = copy(X); Xo[1:2, 1] .= big; Xo[3, 1] = -big
    Xo0 = copy(Xo)
    ic && @test_throws ArgumentError lasso(Xo, y, λ, cyc(T); intercept = ic, standardize_x = st)
    @test bitwise_equal(Xo, Xo0)
  end

  # -------------------------------------------------------------------------------
  @testset "errors and the existing error paths with the keywords set, $T" for T in (Float32, Float64)
    X, y = problem(T, 110, 20, 4)
    # a keyword that is not a Bool is named
    for bad in (1, 0, "yes", nothing, missing, 1.0, :yes, [true])
      @test_throws "intercept" lasso(X, y, 0.1, cyc(T); intercept = bad)
      @test_throws "standardize_x" lasso(X, y, 0.1, cyc(T); standardize_x = bad)
      @test_throws ArgumentError lasso(X, y, 0.1; intercept = bad)
    end
    @test_throws ArgumentError lasso(X, y, 0.1, cyc(T); intercept = 1, standardize_x = 1)
    # the fallback method names the keywords
    @test_throws "standardize_x" lasso(X', y, 0.1; intercept = true)
    @test_throws "intercept" lasso(Int.(round.(X)), Int.(round.(y)), 0.1; standardize_x = true)
    for (ic, st) in COMBOS
      kw = (; intercept = ic, standardize_x = st)
      # the penalty
      @test_throws ArgumentError lasso(X, y, T(-0.1); kw...)
      @test_throws ArgumentError lasso(X, y, NaN; kw...)
      @test_throws ArgumentError lasso(X, y, Inf; kw...)
      @test_throws ArgumentError lasso(X, y, -Inf; kw...)
      # non-finite data
      for bad in (NaN, Inf, -Inf)
        Xb = copy(X); Xb[2, 2] = bad
        yb = copy(y); yb[3] = bad
        @test_throws ArgumentError lasso(Xb, y, 0.1; kw...)
        @test_throws ArgumentError lasso(X, yb, 0.1; kw...)
      end
      # dimensions
      @test_throws DimensionMismatch lasso(X, y[1:19], 0.1; kw...)
      @test_throws DimensionMismatch lasso(X, [y; y[1]], 0.1; kw...)
      @test_throws "at least one row" lasso(zeros(T, 0, 3), T[], 0.1; kw...)
      @test_throws DimensionMismatch lasso(zeros(T, 20, 0), y, 0.1; kw...)
      @test_throws DimensionMismatch lasso(zeros(T, 0, 0), T[], 0.1; kw...)
      @test_throws DimensionMismatch lasso(zeros(T, 0, 3), T[], 0.1; kw...)
      # unsupported types
      @test_throws ArgumentError lasso(view(X, :, 1:2), y, 0.1; kw...)
      @test_throws ArgumentError lasso(X, view(y, 1:20), 0.1; kw...)
      @test_throws ArgumentError lasso(BigFloat.(X), BigFloat.(y), 0.1; kw...)
      @test_throws ArgumentError lasso(X, y, 0.1, :cyclic; kw...)
      @test_throws ArgumentError lasso(X, y, "0.1"; kw...)
      other = T === Float32 ? Float64 : Float32
      @test_throws ArgumentError lasso(other.(X), y, 0.1; kw...)
      @test_throws ArgumentError lasso(X, y, 0.1, CDSolver(order = :cyclic, momentum = true); kw...)
      # a level outside the range of the element type
      if T === Float32
        @test_throws ArgumentError lasso(X, y, 1e-60; kw...)
        @test_throws ArgumentError lasso(X, y, 1e60; kw...)
      end
    end
    # a level of another type reaches the penalty as given, and is stored in T
    for λ in (1, 1 // 10, 0.1, Float16(0.5))
      f = lasso(X, y, λ, cyc(T); intercept = true, standardize_x = true)
      @test f.λ isa T && f.λ == T(λ) && f.result.converged
    end
    # maxpasses = 1 at 0.1 λmax: not converged, a warning, finite output
    for (ic, st) in COMBOS
      Xc, yc = problem(T, 111, 40, 6; shared = 2.0)
      λ = T(0.1 * Float64(lambda_max(reference(Xc, yc, ic, st))))
      fit = @test_logs (:warn,) lasso(Xc, yc, λ, CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 1); intercept = ic, standardize_x = st)
      @test !fit.result.converged && fit.result.passes == 1
      @test all(isfinite, fit.coefficients) && isfinite(fit.intercept) && all(isfinite, fit.residuals)
      @test any(!iszero, fit.coefficients)
      zero_passes = @test_logs (:warn,) lasso(Xc, yc, λ, CDSolver(order = :cyclic, momentum = false, maxpasses = 0); intercept = ic, standardize_x = st)
      @test iszero(zero_passes.coefficients) && !zero_passes.result.converged && isfinite(zero_passes.intercept)
    end
  end

  @testset "the default path is the direct solve, $T" for T in (Float32, Float64)
    # the keywords false: ProximalBase and ProximalSolvers on X and y themselves, as in test/lasso.jl
    X, y = problem(T, 120, 35, 7)
    n, p = size(X)
    for λ in (T(0.3), T(0.05), T(0))
      fit = lasso(X, y, λ, cyc(T); intercept = false, standardize_x = false)
      β = zeros(T, p)
      r = solve!(β, (one(T) / T(n)) * LeastSquaresLoss(y, X), NormL1(λ), cyc(T))
      @test bitwise_equal(fit.coefficients, β) && fit.result.minimizer === fit.coefficients
      @test fit.result.minimum === r.minimum && fit.result.passes == r.passes
      @test iszero(fit.intercept) && !fit.hasintercept && coef(fit) === fit.coefficients
      @test fit.residuals == y - X * β
    end
    # with intercept = true and standardize_x = false, the solver minimizes over the centred data and writes into the coefficient vector
    d = HDM.preprocess(X, y, true, false)
    β = zeros(T, p)
    r = solve!(β, (one(T) / T(n)) * LeastSquaresLoss(d.ỹ, d.Z), NormL1(T(0.1)), cyc(T))
    fit = lasso(X, y, T(0.1), cyc(T); intercept = true)
    @test bitwise_equal(fit.coefficients, β) && fit.result.minimizer === fit.coefficients
    @test fit.result.minimum === r.minimum && fit.result.passes == r.passes
  end
end
end

end
