module ElasticNetTest
# `lasso` with `alpha < 1`: the elastic net
#   ‖y - β₀1 - Xβ‖² / (2n) + λ (α‖θ‖₁ + (1 - α)‖θ‖²/2),   θⱼ = sⱼβⱼ,
# with β₀ = 0 unless `intercept` and sⱼ = 1 unless `standardize_x` (the lasso at `alpha = 1`). The
# references share nothing with the preprocessing or with coordinate descent:
#   * the optimality conditions of the problem on the centred and scaled data, in 256-bit BigFloat
#     from the stored coefficients;
#   * the duality gap of the lasso on the augmented design [Z; √(nν) I], ν = λ(1 - α), with the
#     response [ỹ; 0] and the original divisor n, computed from the stored λ, α, β₀ and β, with
#     controls (weak duality, a closed-form optimum, and the versions with n + p or without √n);
#   * the full objective recomputed from X, y, β₀, β and the scales;
#   * the closed form on an orthogonal design, and the exact solution of small problems by
#     enumeration of supports and signs in rational arithmetic;
#   * the effective penalty levels, recomputed in T.
# The default path (alpha omitted) is that of test/lasso.jl, test/lasso_reference.jl and
# test/lasso_intercept.jl.

using HighDimensionalModels, Test, LinearAlgebra, Random
using ProximalBase: LeastSquaresLoss, NormL1, ElasticNet
using ProximalSolvers: solve!

const HDM = HighDimensionalModels

# ---------------------------------------------------------------------------------
# settings and data
# ---------------------------------------------------------------------------------

tolof(T) = T === Float32 ? 1e-6 : 1e-10
cyc(T) = CDSolver(order = :cyclic, momentum = false, tol = tolof(T))
const COMBOS = ((false, false), (true, false), (false, true), (true, true))
const ALPHAS = (0.2, 0.5, 0.8)
label(ic, st) = "intercept = $ic, standardize_x = $st"

bits(a::AbstractArray{Float64}) = reinterpret(UInt64, a)
bits(a::AbstractArray{Float32}) = reinterpret(UInt32, a)
bitwise_equal(a, b) = size(a) == size(b) && bits(a) == bits(b)
same_result(r, s) = typeof(r) == typeof(s) && all(i -> isequal(getfield(r, i), getfield(s, i)), 1:fieldcount(typeof(r)))

# columns of different scales and means, a response with an intercept; `shared` correlates the columns
function problem(T, seed, n, p; shift = 1.5, shared = 0.0, k = min(3, p))
  rng = Xoshiro(seed)
  z = randn(rng, n)
  scales = 0.5 .+ 1.5 .* rand(rng, p)
  X = randn(rng, n, p) .* transpose(scales) .+ transpose(shift .* randn(rng, p)) .+ shared .* z
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  T.(X), T.(X * β .+ 1.0 .+ 0.5 .* randn(rng, n))
end

# data on a grid of 1/16 (exact in Float32 and Float64, and small as rationals)
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

# The effective penalty levels of `lasso` for data of type T, recomputed in T: the stored λ and
# alpha, and λ₁ = λ α, λ₂ = λ (1 - α) with the products and the difference taken in T
effective(T, λ, alpha) = (λ = T(λ), α = T(alpha), λ₁ = T(λ) * T(alpha), λ₂ = T(λ) * (one(T) - T(alpha)))

# ---------------------------------------------------------------------------------
# References in extended precision, from the stored Float inputs (call inside setprecision)
# ---------------------------------------------------------------------------------

# A column whose entries are all equal has that value as its mean: the rule of `lasso` for constant
# columns, which keeps them at exact zeros after centring
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

# λmax of the elastic net: max over the columns with sⱼ > 0 of |zⱼ'ỹ| / (n α), the level from which
# the slopes are all zero (zⱼ = x̃ⱼ/sⱼ)
function lambda_max(ref, α)
  js = findall(>(0), ref.s)
  isempty(js) && return zero(BigFloat)
  maximum(j -> abs(dot(view(ref.Zb, :, j), ref.yt)), js) / (ref.n * BigFloat(α))
end

# The duality gap of the lasso on the augmented design Zₐ = [Z; ρ I], yₐ = [ỹ; 0] at θ, with the
# divisor m (the number of rows of the loss) and ℓ₁ level a = λα:
#   P = ‖yₐ - Zₐθ‖²/(2m) + a‖θ‖₁,   g = Zₐ'(yₐ - Zₐθ)/m,   t = min(1, a/‖g‖∞),
#   D = (‖yₐ‖² - ‖yₐ - t rₐ‖²)/(2m),   gap = P - D.
# With ρ = √(nν) and m = n, ν = λ(1 - α), this is the elastic net on Z with the original
# normalization: ‖yₐ - Zₐθ‖²/(2n) = ‖ỹ - Zθ‖²/(2n) + (ν/2)‖θ‖². The keywords select the wrong
# versions the controls must tell apart: `rows = :unscaled` has ρ = √ν, and `divisor = :n_plus_p`
# has m = n + p.
function augmented_gap(Z, yt, θ, λ, α; rows = :sqrt_n, divisor = :n)
  n, p = size(Z)
  ν = BigFloat(λ) * (1 - BigFloat(α))
  ρ = rows === :sqrt_n ? sqrt(n * ν) : sqrt(ν)
  m = divisor === :n ? n : n + p
  a = BigFloat(λ) * BigFloat(α)
  Za = [Z; ρ .* Matrix{BigFloat}(I, p, p)]
  ya = [yt; zeros(BigFloat, p)]
  ra = ya - Za * θ
  P = dot(ra, ra) / (2m) + a * sum(abs, θ)
  g = Za' * ra / m
  gmax = maximum(abs, g)
  t = gmax <= a ? one(BigFloat) : a / gmax
  D = (dot(ya, ya) - dot(ya - t * ra, ya - t * ra)) / (2m)
  (; gap = P - D, P, D, t, fit = dot(ra, ra) / (2m))
end

# The objective at the stored (β₀, β), from X, y and the scales alone:
# ‖y - β₀1 - Xβ‖²/(2n) + λα Σ sⱼ|βⱼ| + (λ(1 - α)/2) Σ sⱼ²βⱼ²
function objective(fit, X::Matrix, y::Vector, s)
  n = size(X, 1)
  λ, α = BigFloat(fit.λ), BigFloat(fit.alpha)
  β = BigFloat.(fit.coefficients)
  r = BigFloat.(y) .- BigFloat(fit.intercept) .- BigFloat.(X) * β
  dot(r, r) / (2n) + λ * α * sum(s .* abs.(β)) + λ * (1 - α) / 2 * sum(abs2, s .* β), sum(r) / n
end

# The violation of the optimality conditions of min ‖ỹ - Zθ‖²/(2n) + λα‖θ‖₁ + (λ(1 - α)/2)‖θ‖² at
# θ = sβ for the stored β, by column, in the units of θ, with g = Z'(ỹ - Zθ)/n: for sⱼ > 0, the
# excess of |gⱼ| over λα where θⱼ = 0, and the distance of gⱼ - λ(1 - α)θⱼ from λα sign(θⱼ) where θⱼ ≠ 0.
function kkt_report(fit, ref)
  λb, αb = BigFloat(fit.λ), BigFloat(fit.alpha)
  a, ν = λb * αb, λb * (1 - αb)
  θ = BigFloat.(fit.coefficients) .* ref.s
  r = ref.yt - ref.Zb * θ
  g = ref.Zb' * r / ref.n
  viol = zeros(Float64, ref.p)
  for j in 1:ref.p
    iszero(ref.s[j]) && continue
    v = iszero(θ[j]) ? abs(g[j]) - a : abs(g[j] - ν * θ[j] - a * sign(θ[j]))
    viol[j] = Float64(max(v, zero(BigFloat)))
  end
  (; viol, θ, r, ν = Float64(ν), a = Float64(a))
end

# A bound on the violations of kkt_report after a run that ended on the coefficient-change test,
# per column in the units of θ.
#   * The change-test term of test/lasso_reference.jl for the columns zⱼ of the solver's problem,
#     p · max‖z‖ · tolerance/√n (the ridge term of coordinate j involves θⱼ, which changes when
#     coordinate j is updated, so it adds none);
#   * the rounding of the preprocessing and of the back-transformation, c eps(T) (‖xⱼ‖ + √n|x̄ⱼ|)
#     (‖y‖ + √n|ȳ|)/(n sⱼ) with c = 16 (p + 4), as in test/lasso_intercept.jl;
#   * the rounding of θ = sβ in the ridge term: the stored β is a rounded quotient, and the scale
#     is a computed root mean square with a relative error of at most 4 eps(T) (⌈log2 n⌉ + 2), so
#     that the ridge term λ(1 - α)θⱼ carries 16 eps(T) (⌈log2 n⌉ + 2) λ(1 - α)|θⱼ|.
function kappa(fit, X::Matrix{T}, y::Vector{T}, ref) where T
  n, p = size(X)
  ν = Float64(BigFloat(fit.λ) * (1 - BigFloat(fit.alpha)))
  zmax = maximum(j -> iszero(ref.s[j]) ? 0.0 : Float64(norm(ref.Xt[:, j]) / ref.s[j]), 1:p)
  change = p * zmax * Float64(fit.result.tolerance) / sqrt(n)
  ysc = norm(Float64.(y)) + sqrt(n) * abs(Float64(ref.ȳ))
  κθ = zeros(p)
  for j in 1:p
    iszero(ref.s[j]) && continue
    xsc = norm(Float64.(X[:, j])) + sqrt(n) * abs(Float64(ref.x̄[j]))
    rounding = 16 * (p + 4) * eps(T) * xsc * ysc / n
    θj = abs(Float64(BigFloat(fit.coefficients[j]) * ref.s[j]))
    κθ[j] = change + rounding / Float64(ref.s[j]) + 16 * eps(T) * (ceil(log2(n)) + 2) * ν * θj
  end
  κθ
end

# for the solver's θ and the exact θ*, both supported on A: the strong convexity of the loss plus
# the ridge term, μ = λmin(Z_A'Z_A/n) + ν, gives ‖θ - θ*‖₂ ≤ √|A| κ / μ when the subgradients are
# perturbed by at most κ
function coef_bound(Z::Matrix{Float64}, A, κ, ν)
  isempty(A) && return 0.0
  ZA = Z[:, A]
  sqrt(length(A)) * κ / (eigmin(Symmetric(ZA' * ZA / size(Z, 1))) + ν)
end

# the checks of a fit against the references: the optimality conditions, the augmented-design gap
# from the stored λ, α, β₀ and β, the objective recomputed from X, y, β₀, β and the scales, the
# stored minimum and the sum of the residuals
function check_fit(fit, X::Matrix{T}, y::Vector{T}, ref) where T
  n, p = size(X)
  rep = kkt_report(fit, ref)
  κθ = kappa(fit, X, y, ref)
  @test all(rep.viol .<= κθ)
  # the gap: the primal is the objective on the raw data at the stored intercept, which exceeds the
  # primal of the centred problem by r̄²/2 for the mean r̄ of the raw residuals (with an intercept)
  obj, r̄ = objective(fit, X, y, ref.s)
  ag = augmented_gap(ref.Zb, ref.yt, rep.θ, fit.λ, fit.alpha)
  gap = Float64(obj - ag.D)
  extra = fit.hasintercept ? r̄^2 / 2 : zero(BigFloat)
  @test Float64(abs(obj - ag.P - extra)) <= 1e-60 + 1e-40 * Float64(ag.P)       # two evaluations of one primal value
  @test gap >= -1e-30
  if fit.λ * fit.alpha > 0
    sθ = maximum(κθ)
    a = rep.a
    l1 = Float64(sum(abs, rep.θ))
    @test gap <= 2sθ * l1 + (sθ / a)^2 * Float64(ag.fit) + Float64(extra)
  end
  ysc = norm(Float64.(y)) + sqrt(n) * abs(Float64(ref.ȳ))
  @test fit.result.minimum ≈ Float64(obj) rtol = 64 * (p + 4) * eps(T) atol = 16 * (p + 4) * eps(T) * ysc^2 / (2n)
  if fit.hasintercept
    @test abs(Float64(r̄)) <= 32 * (p + 4) * eps(T) * (norm(Float64.(y)) / sqrt(n) + sum(j -> abs(fit.coefficients[j]) * norm(Float64.(X[:, j])) / sqrt(n), 1:p))
  end
  (; rep, κθ, gap)
end

# ---------------------------------------------------------------------------------
# The exact elastic net solution of a small problem, by enumeration, in rational arithmetic
# ---------------------------------------------------------------------------------

# For each support A and each sign vector s on A, the stationarity condition
# Z_A'(ỹ - Z_Aθ_A)/n - ν θ_A = a s gives (Z_A'Z_A/n + νI) θ_A = Z_A'ỹ/n - a s, with ν = λ(1 - α) and
# a = λα. The pair is a solution when sign(θ_A) = s and |zⱼ'(ỹ - Z_Aθ_A)|/n ≤ a off A. For ν > 0 the
# matrix is positive definite also when p > n, and the problem is strictly convex.
# When the inequalities hold strictly the solution is unique. Returns the passing candidates, each
# (A, θ on 1:p, strict).
function exact_candidates(Z::Matrix, y::Vector, λ, α)
  n, p = size(Z)
  Q = Rational{BigInt}
  Zr, yr, λr, αr = Q.(Z), Q.(y), Q(λ), Q(α)
  a, ν = λr * αr, λr * (1 - αr)
  found = NamedTuple{(:A, :θ, :strict), Tuple{Vector{Int}, Vector{Q}, Bool}}[]
  for mask in 0:(2^p - 1)
    A = [j for j in 1:p if (mask >> (j - 1)) & 1 == 1]
    k = length(A)
    ZA = Zr[:, A]
    if k > 0
      M = ZA' * ZA / n + ν * Matrix{Q}(I, k, k)
      Minv = inv(M)
      c = ZA' * yr / n
    end
    for s in Iterators.product(ntuple(_ -> (-1, 1), k)...)
      sv = collect(Q, s)
      θA = k == 0 ? Q[] : Minv * (c - a * sv)
      all(i -> θA[i] * sv[i] > 0, 1:k) || continue
      g = Zr' * (yr - ZA * θA) / n
      off = setdiff(1:p, A)
      all(j -> abs(g[j]) <= a, off) || continue
      θ = zeros(Q, p)
      θ[A] = θA
      push!(found, (A = A, θ = θ, strict = all(j -> abs(g[j]) < a, off)))
    end
  end
  found
end

function exact_elastic_net(Z, y, λ, α)
  c = exact_candidates(Z, y, λ, α)
  (length(c) == 1 && only(c).strict) || error("the exact reference is not unique: $(length(c)) candidates")
  only(c)
end

soft(c, t) = sign(c) * max(abs(c) - t, zero(c))

# ---------------------------------------------------------------------------------

setprecision(BigFloat, 256) do
@testset "lasso with alpha: the elastic net" begin

  # -------------------------------------------------------------------------------
  @testset "the controls of the augmented-design gap" begin
    # an orthogonal design, ZᵀZ = n I with n = 8: the columns of the Hadamard matrix other than the first
    H2 = [1 1; 1 -1]
    H = kron(H2, kron(H2, H2))
    Z = BigFloat.(H[:, 2:8])
    n, p = size(Z)
    yt = BigFloat.([3.0, -1.0, 0.5, 2.0, -2.5, 1.0, 0.25, -0.75])
    yt .-= sum(yt) / n
    c = Z' * yt / n
    @test maximum(abs, Z' * Z - n * I) == 0
    for (λ, α) in ((0.3, 0.5), (0.2, 0.2), (0.5, 0.8), (0.1, 0.6))
      a, ν = BigFloat(λ) * BigFloat(α), BigFloat(λ) * (1 - BigFloat(α))
      θ = soft.(c, a) ./ (1 + ν)                         # the closed form, ZᵀZ/n = I
      @test count(!iszero, θ) >= 1 && count(iszero, θ) >= 1                # the thresholding acts
      # the gap is zero at the optimum, and D = P there
      ag = augmented_gap(Z, yt, θ, λ, α)
      @test abs(ag.gap) <= 1e-70 && ag.t >= 1 - BigFloat(1e-70)
      # the augmented primal is the objective: ‖ỹ - Zθ‖²/(2n) + λα‖θ‖₁ + (ν/2)‖θ‖²
      r = yt - Z * θ
      @test abs(ag.P - (dot(r, r) / (2n) + a * sum(abs, θ) + ν / 2 * dot(θ, θ))) <= 1e-70
      # the versions with n + p or without √n do not vanish at the optimum: they would fail this control
      for wrong in ((; divisor = :n_plus_p), (; rows = :unscaled), (; divisor = :n_plus_p, rows = :unscaled))
        @test Float64(augmented_gap(Z, yt, θ, λ, α; wrong...).gap) > 1e-4
      end
      # weak duality at other points, and a positive gap away from the optimum
      rng = Xoshiro(round(Int, 100λ + 10α))
      for _ in 1:6
        θr = θ .+ 0.3 .* BigFloat.(randn(rng, p))
        agr = augmented_gap(Z, yt, θr, λ, α)
        @test agr.gap > 0 && agr.D <= agr.P && agr.D <= ag.P
        @test agr.gap > 1e-6
      end
      @test augmented_gap(Z, yt, zeros(BigFloat, p), λ, α).gap > 0
    end
    # above λmax = max|zⱼ'ỹ|/(nα) the zero vector is optimal for the elastic net: gap 0; below it is not
    for α in (0.2, 0.5, 0.8)
      λmax = maximum(abs, c) / BigFloat(α)
      @test abs(augmented_gap(Z, yt, zeros(BigFloat, p), 1.01 * Float64(λmax), α).gap) <= 1e-70
      @test Float64(augmented_gap(Z, yt, zeros(BigFloat, p), 0.9 * Float64(λmax), α).gap) > 1e-4
    end
  end

  # -------------------------------------------------------------------------------
  @testset "alpha = 1 is the lasso, bit for bit, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    X, y = problem(T, 10, 40, 8)
    kw = (; intercept = ic, standardize_x = st)
    for λ in (T(0.3), T(0.05), T(0), 0.1)
      for mk in (() -> cyc(T),
                 () -> CDSolver(tol = tolof(T), rng = Xoshiro(3)),                            # random order, momentum
                 () -> CDSolver(order = :random, momentum = false, tol = tolof(T), rng = Xoshiro(3)))
        base = lasso(X, y, λ, mk(); kw...)
        for one_alpha in (1, one(T), 1.0, BigFloat(1), 1 // 1, true)
          f = lasso(X, y, λ, mk(); alpha = one_alpha, kw...)
          @test bitwise_equal(f.coefficients, base.coefficients) && bitwise_equal(f.residuals, base.residuals)
          @test f.intercept === base.intercept && f.λ === base.λ && f.alpha === one(T) && base.alpha === one(T)
          @test f.hasintercept === base.hasintercept && same_result(f.result, base.result)
        end
      end
    end
  end

  @testset "alpha that rounds to 1 in T is the lasso, Float32" begin
    X, y = problem(Float32, 11, 40, 8)
    for (ic, st) in COMBOS
      kw = (; intercept = ic, standardize_x = st)
      base = lasso(X, y, 0.1f0, cyc(Float32); kw...)
      for alpha in (1 - 1e-9, 1 - 1e-12, prevfloat(1.0), 1 - 2.0^-30)
        @test Float32(alpha) == 1f0 && alpha < 1
        f = lasso(X, y, 0.1f0, cyc(Float32); alpha, kw...)
        @test f.alpha === 1f0 && bitwise_equal(f.coefficients, base.coefficients) && same_result(f.result, base.result)
        @test !occursin("α", sprint(show, f))
      end
      # alpha just above 1 is refused before any conversion, although Float32(1 + 1e-9) == 1f0
      @test Float32(1 + 1e-9) == 1f0
      @test_throws ArgumentError lasso(X, y, 0.1f0, cyc(Float32); alpha = 1 + 1e-9, kw...)
    end
  end

  # -------------------------------------------------------------------------------
  @testset "the effective penalty levels, $T" for T in (Float32, Float64)
    X, y = problem(T, 12, 35, 7)
    n, p = size(X)
    # the stored λ and alpha reproduce the penalty levels bit for bit, and the solver ran on them
    for (λ, alpha) in ((0.1, 0.2), (0.1, 0.5), (0.15, 0.3), (0.05, 0.9), (0.3, 0.45), (1 // 10, 1 // 3), (Float16(0.5), 0.25))
      e = effective(T, λ, alpha)
      for (ic, st) in COMBOS
        fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
        @test fit.λ === e.λ && fit.alpha === e.α
        @test fit.λ * fit.alpha === e.λ₁ && fit.λ * (1 - fit.alpha) === e.λ₂
        # the solver's result is that of ElasticNet(λ₁, λ₂) on the transformed problem
        if !ic && !st
          β = zeros(T, p)
          r = solve!(β, (one(T) / T(n)) * LeastSquaresLoss(y, X), ElasticNet(e.λ₁, e.λ₂), cyc(T))
        else
          d = HDM.preprocess(X, y, ic, st)
          β = zeros(T, p)
          r = solve!(β, (one(T) / T(n)) * LeastSquaresLoss(d.ỹ, d.Z), ElasticNet(e.λ₁, e.λ₂), cyc(T))
        end
        @test same_result(fit.result, r) && bitwise_equal(fit.result.minimizer, β)
      end
    end
    if T === Float32
      # the levels computed in Float64 and rounded are not those computed in T: the test above separates them
      λ, alpha = 0.1, 0.2
      e = effective(T, λ, alpha)
      @test T(λ * alpha) != e.λ₁ && T(λ * (1 - alpha)) != e.λ₂
      fit = lasso(X, y, λ, cyc(T); alpha)
      wrong = solve!(zeros(T, p), (one(T) / T(n)) * LeastSquaresLoss(y, X), ElasticNet(T(λ * alpha), T(λ * (1 - alpha))), cyc(T))
      @test !same_result(fit.result, wrong)
    end
    # λ = 0 with alpha < 1 is least squares: ElasticNet(0, 0)
    fit = lasso(X, y, 0, cyc(T); alpha = 0.5)
    @test fit.λ === zero(T) && fit.alpha === T(0.5)
    r = solve!(zeros(T, p), (one(T) / T(n)) * LeastSquaresLoss(y, X), ElasticNet(zero(T), zero(T)), cyc(T))
    @test same_result(fit.result, r)
    # a level of another type is stored in T
    for λ in (1, 1 // 10, 0.1, Float16(0.5))
      f = lasso(X, y, λ, cyc(T); alpha = 0.5, intercept = true, standardize_x = true)
      @test f.λ isa T && f.λ == T(λ) && f.alpha === T(0.5) && f.result.converged
    end
  end

  @testset "alpha with a rounded or underflowing level, Float32 and Float64" begin
    X, y = problem(Float32, 13, 30, 5)
    # alpha = 1e-50 is valid but is 0 in Float32: a ridge alone, refused
    for kw in ((;), (; intercept = true), (; standardize_x = true), (; intercept = true, standardize_x = true))
      @test_throws "ridge" lasso(X, y, 0.1f0, cyc(Float32); alpha = 1e-50, kw...)
      @test_throws ArgumentError lasso(X, y, 0.1f0, cyc(Float32); alpha = 1e-50, kw...)
      # the ℓ₁ level λ * alpha underflows to 0 for λ > 0 (1e-20 * 1e-30 is below the Float32 range)
      @test_throws "ridge" lasso(X, y, 1f-20, cyc(Float32); alpha = 1e-30, kw...)
      # a level that is 0 in Float32 while λ > 0
      @test_throws ArgumentError lasso(X, y, 1e-60, cyc(Float32); alpha = 0.5, kw...)
      # alpha that is 0 in Float32 is refused also with λ = 0
      @test_throws "ridge" lasso(X, y, 0, cyc(Float32); alpha = 1e-50, kw...)
    end
    # the same alpha is fine in Float64 (1e-50 is a normal Float64) and stored as is
    X64, y64 = problem(Float64, 13, 30, 5)
    f = lasso(X64, y64, 0.1, cyc(Float64); alpha = 1e-50)
    @test f.alpha === 1e-50 && f.λ === 0.1 && f.λ * f.alpha === 1e-51 && f.result.converged
    # alpha = 5e-324 and 1e-323 are subnormal Float64 numbers: the product 0.1 * alpha underflows to 0
    @test_throws "ridge" lasso(X64, y64, 0.1, cyc(Float64); alpha = 5e-324)
    @test_throws "ridge" lasso(X64, y64, 0.1, cyc(Float64); alpha = 1e-323, intercept = true)
    # BigFloat and Rational values of alpha are converted to T
    for alpha in (big"0.5", 1 // 2, Float16(0.5), 0.5f0)
      f = lasso(X64, y64, 0.1, cyc(Float64); alpha)
      @test f.alpha === 0.5
    end
    @test_throws "ridge" lasso(X64, y64, 0.1, cyc(Float64); alpha = big"1e-400")
  end

  @testset "invalid alpha, $T" for T in (Float32, Float64)
    X, y = problem(T, 14, 20, 4)
    for (ic, st) in COMBOS
      kw = (; intercept = ic, standardize_x = st)
      for bad in (0, 0.0, -0.1, -1, 1.5, 2, 1 + 1e-9, nextfloat(1.0), NaN, Inf, -Inf, T(NaN), big"1.5", 3 // 2, -1 // 2, 0 // 1)
        @test_throws ArgumentError lasso(X, y, T(0.1), cyc(T); alpha = bad, kw...)
        @test_throws "alpha" lasso(X, y, T(0.1), cyc(T); alpha = bad, kw...)
        @test_throws "0 < alpha ≤ 1" lasso(X, y, T(0.1); alpha = bad, kw...)
      end
      # alpha of a type that is not Real: an ArgumentError, not a TypeError or a MethodError
      for bad in ("0.5", :half, nothing, missing, 0.5 + 0im, [0.5], (0.5,), 'a')
        @test_throws ArgumentError lasso(X, y, T(0.1), cyc(T); alpha = bad, kw...)
        @test_throws "alpha" lasso(X, y, T(0.1), cyc(T); alpha = bad, kw...)
      end
      # an invalid alpha with λ = 0, and an invalid alpha together with an invalid λ
      @test_throws "alpha" lasso(X, y, 0, cyc(T); alpha = 0, kw...)
      @test_throws ArgumentError lasso(X, y, T(-0.1); alpha = 1.5, kw...)
    end
  end

  @testset "a negative λ is refused with alpha < 1, also when its sign vanishes in rounding, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    # X = (-1, 1)', y = (-1, 1): the least squares coefficient is 1, which a negative λ that became a zero
    # level would return
    X = reshape(T[-1, 1], 2, 1); y = T[-1, 1]
    kw = (; intercept = ic, standardize_x = st)
    # native negative subnormals and small values, and values that convert to -0.0 in T (BigFloat,
    # Rational, a Float64 below the range of Float32), and ordinary negative values
    negatives = Real[-nextfloat(zero(T)), -floatmin(T), T(-0.1), -1, -2, -1 // 10, -1e-100, -big"1e-400", -big"1e-5000", -1 // big(10)^400]
    for λ in negatives
      for alpha in (0.2, 0.5, 0.8)
        @test_throws "nonnegative" lasso(X, y, λ, cyc(T); alpha, kw...)
        @test_throws "λ" lasso(X, y, λ, cyc(T); alpha, kw...)
        @test_throws ArgumentError lasso(X, y, λ, cyc(T); alpha, kw...)
      end
      # with alpha omitted, or 1, or rounding to 1, the refusal is ProximalBase's or ProximalSolvers', unchanged
      direct = try NormL1(λ); nothing catch e; e end
      refusals = [try lasso(X, y, λ, cyc(T); kw...); nothing catch e; e end,
                  try lasso(X, y, λ, cyc(T); alpha = 1, kw...); nothing catch e; e end]
      T === Float32 && push!(refusals, try lasso(X, y, λ, cyc(T); alpha = 1 - 1e-9, kw...); nothing catch e; e end)
      for e in refusals
        @test e isa ArgumentError
        @test !occursin("with alpha < 1", sprint(showerror, e))
        direct === nothing || @test sprint(showerror, e) == sprint(showerror, direct)
      end
    end
    # -0.0 is not negative: it is the zero level, as on the lasso path, and the fit is least squares
    for alpha in (0.2, 0.5, 0.8)
      f = lasso(X, y, -zero(T), cyc(T); alpha, kw...)
      l = lasso(X, y, -zero(T), cyc(T); kw...)
      @test f.λ === -zero(T) && f.alpha === T(alpha) && l.λ === -zero(T)
      @test f.coefficients ≈ l.coefficients atol = 1e-5
      @test f.coefficients ≈ T[1] atol = 1e-5
    end
    # -Inf is refused by the check of λ < 0; NaN and +Inf reach the penalty's levels and are refused downstream
    for λ in (NaN, Inf, -Inf)
      @test_throws ArgumentError lasso(X, y, λ, cyc(T); alpha = 0.5, kw...)
    end
  end

  @testset "errors with alpha < 1 are those of the lasso, $T" for T in (Float32, Float64)
    X, y = problem(T, 15, 20, 4)
    # the fallback method names the keyword alpha
    @test_throws "alpha" lasso(X', y, 0.1; alpha = 0.5)
    @test_throws "alpha" lasso(Int.(round.(X)), Int.(round.(y)), 0.1; alpha = 0.5, intercept = true)
    @test_throws "alpha of type" lasso(X, y, "0.1"; alpha = 0.5)
    for (ic, st) in COMBOS
      kw = (; alpha = 0.5, intercept = ic, standardize_x = st)
      @test_throws ArgumentError lasso(X, y, T(-0.1); kw...)
      @test_throws ArgumentError lasso(X, y, NaN; kw...)
      @test_throws ArgumentError lasso(X, y, Inf; kw...)
      @test_throws ArgumentError lasso(X, y, -Inf; kw...)
      for bad in (NaN, Inf, -Inf)
        Xb = copy(X); Xb[2, 2] = bad
        yb = copy(y); yb[3] = bad
        @test_throws ArgumentError lasso(Xb, y, 0.1; kw...)
        @test_throws ArgumentError lasso(X, yb, 0.1; kw...)
      end
      @test_throws DimensionMismatch lasso(X, y[1:19], 0.1; kw...)
      @test_throws "at least one row" lasso(zeros(T, 0, 3), T[], 0.1; kw...)
      @test_throws DimensionMismatch lasso(zeros(T, 20, 0), y, 0.1; kw...)
      @test_throws ArgumentError lasso(view(X, :, 1:2), y, 0.1; kw...)
      @test_throws ArgumentError lasso(X, y, 0.1, :cyclic; kw...)
      @test_throws ArgumentError lasso(X, y, 0.1, CDSolver(order = :cyclic, momentum = true); kw...)
      if T === Float32
        @test_throws ArgumentError lasso(X, y, 1e60; kw...)
      end
    end
    @test_throws "intercept" lasso(X, y, 0.1, cyc(T); alpha = 0.5, intercept = 1)
    @test_throws "standardize_x" lasso(X, y, 0.1, cyc(T); alpha = 0.5, standardize_x = "yes")
  end

  # -------------------------------------------------------------------------------
  @testset "optimality conditions, duality gap and objective in extended precision, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in ALPHAS, (ic, st) in COMBOS
    problems = [
      ("independent columns", problem(T, 20, 40, 8), 0.15),
      ("correlated columns", problem(T, 21, 50, 10; shared = 2.0), 0.10),
      ("p > n", problem(T, 22, 15, 40), 0.25),
    ]
    # a mean shift of order 1e3 is ill conditioned without an intercept: this problem is run with an intercept
    ic && push!(problems, ("mean shift 1e3", problem(T, 23, 40, 6; shift = 1e3), 0.15))
    for (name, (X, y), frac) in problems
      ref = reference(X, y, ic, st)
      λ = T(frac * Float64(lambda_max(ref, alpha)))
      fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
      @test fit.result.converged
      @test fit.hasintercept == ic && fit.alpha === T(alpha) && fit.λ === λ
      c = check_fit(fit, X, y, ref)
      @test any(!iszero, fit.coefficients) && any(iszero, fit.coefficients)
      # the returned residuals are those of the original data at the returned point
      @test fit.residuals == y .- fit.intercept .- X * fit.coefficients
    end
  end

  @testset "the solve is ProximalSolvers' on the stated problem, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in ALPHAS, (ic, st) in COMBOS
    X, y = problem(T, 24, 35, 7)
    n, p = size(X)
    λ = T(0.12)
    e = effective(T, λ, alpha)
    fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
    β = zeros(T, p)
    if !ic && !st
      r = solve!(β, (one(T) / T(n)) * LeastSquaresLoss(y, X), ElasticNet(e.λ₁, e.λ₂), cyc(T))
      @test bitwise_equal(fit.coefficients, β) && fit.result.minimizer === fit.coefficients
      @test fit.residuals == y - X * β
    else
      d = HDM.preprocess(X, y, ic, st)
      r = solve!(β, (one(T) / T(n)) * LeastSquaresLoss(d.ỹ, d.Z), ElasticNet(e.λ₁, e.λ₂), cyc(T))
      st || @test bitwise_equal(fit.coefficients, β) && fit.result.minimizer === fit.coefficients
      st && @test bitwise_equal(fit.result.minimizer, β) && fit.result.minimizer !== fit.coefficients
      st && @test fit.coefficients == HDM.coefficients_from_scaled(β, d.s)
      β₀ = ic ? HDM.intercept_from_means(d.ȳ, d.x̄, fit.coefficients) : zero(T)
      @test fit.intercept === β₀
      @test fit.residuals == y .- β₀ .- X * fit.coefficients
    end
    @test same_result(fit.result, r)
    @test fit.result.minimum === r.minimum && fit.result.passes == r.passes
  end

  # -------------------------------------------------------------------------------
  @testset "orthogonal designs have the closed form θⱼ = S(zⱼ'ỹ/n, λα)/(1 + λ(1 - α)), $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    # the columns of a Hadamard matrix (n = 8) with column scales and, with an intercept, column means
    # on a grid: the centred columns are orthogonal, ‖x̃ⱼ‖² = n cⱼ² for the column scale cⱼ
    H2 = [1 1; 1 -1]
    H = Float64.(kron(H2, kron(H2, H2)))
    cs = [1.0, 2.0, 0.5, 4.0, 1.5, 3.0, 1.0]
    shifts = ic ? [3.0, -2.0, 0.5, 1.0, -4.0, 2.0, 0.25] : zeros(7)
    X64 = H[:, 2:8] .* transpose(cs) .+ transpose(shifts)
    y64 = [3.0, -1.0, 0.5, 2.0, -2.5, 1.0, 0.25, -0.75] .+ (ic ? 1.0 : 0.0)
    X, y = T.(X64), T.(y64)
    n, p = size(X)
    ref = reference(X, y, ic, st)
    @test maximum(abs, ref.Zb' * ref.Zb - Diagonal(diag(ref.Zb' * ref.Zb))) <= 1e-70
    for (frac, alpha) in ((0.3, 0.5), (0.55, 0.2), (0.2, 0.8), (0.7, 0.5))
      λ = T(frac * Float64(lambda_max(ref, alpha)))
      fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
      a, ν = BigFloat(fit.λ) * BigFloat(fit.alpha), BigFloat(fit.λ) * (1 - BigFloat(fit.alpha))
      d = [dot(ref.Zb[:, j], ref.Zb[:, j]) / n for j in 1:p]                     # 1 with standardization
      θ = [soft(dot(ref.Zb[:, j], ref.yt) / n, a) / (d[j] + ν) for j in 1:p]
      st && @test all(abs.(d .- 1) .<= 1e-70)
      # no coordinate sits on the threshold, where the support is decided by rounding
      corr = [abs(dot(ref.Zb[:, j], ref.yt)) / n for j in 1:p]
      @test all(abs.(corr .- a) .> 1e-3 * a)
      @test count(!iszero, θ) >= 1 && count(iszero, θ) >= 1
      κθ = kappa(fit, X, y, ref)
      A = findall(!iszero, θ)
      bound = coef_bound(ref.Zf, A, maximum(κθ), Float64(ν))
      @test findall(!iszero, fit.result.minimizer) == A
      @test norm(Float64.(fit.result.minimizer) - Float64.(θ), Inf) <= bound
      # the coefficients are θ/s, with the rounding of the division
      β = Float64.(fit.coefficients)
      for j in 1:p
        @test abs(β[j] - Float64(θ[j] / ref.s[j])) <= (bound + 2eps(T) * abs(Float64(θ[j]))) / Float64(ref.s[j]) + eps(T) * abs(β[j])
      end
      check_fit(fit, X, y, ref)
      @test fit.result.passes <= 4
    end
  end

  # -------------------------------------------------------------------------------
  @testset "exact solutions of small problems by enumeration, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in (0.3, 0.7), (ic, st) in COMBOS
    # the exact solution of the problem on the centred and scaled data, with Z rounded to rationals from the
    # 256-bit values; θ = sβ
    cases = [
      ("independent, p = 5", grid_problem(30, 20, 5), 0.30),
      ("correlated, p = 6", grid_problem(31, 25, 6; shared = 2.0), 0.20),
      ("p > n (5 × 7)", grid_problem(32, 5, 7), 0.30),
    ]
    for (name, (X64, y64), frac) in cases
      X, y = T.(X64), T.(y64)
      n, p = size(X)
      ref = reference(X, y, ic, st)
      λ = T(frac * Float64(lambda_max(ref, alpha)))
      e = effective(T, λ, alpha)
      ex = exact_elastic_net(Rational{BigInt}.(ref.Zb), Rational{BigInt}.(ref.yt), Float64(e.λ), Float64(e.α))
      fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
      θ = Float64.(fit.result.minimizer)
      κθ = maximum(kappa(fit, X, y, ref))
      ν = Float64(e.λ) * (1 - Float64(e.α))
      @test findall(!iszero, fit.coefficients) == ex.A
      @test sign.(θ) == sign.(Float64.(ex.θ))
      bound = coef_bound(ref.Zf, ex.A, κθ, ν)
      @test norm(θ - Float64.(ex.θ), Inf) <= bound
      s = Float64.(ref.s)
      β = Float64.(fit.coefficients)
      for j in 1:p
        @test abs(β[j] - Float64(ex.θ[j] / ref.s[j])) <= (bound + 2eps(T) * abs(θ[j])) / s[j] + eps(T) * abs(β[j])
      end
      if ic
        β₀exact = Float64(ref.ȳ - dot(ref.x̄, BigFloat.(ex.θ) ./ ref.s))
        Δβ = [abs(β[j] - Float64(ex.θ[j] / ref.s[j])) for j in 1:p]
        bound₀ = 16 * (p + 4) * eps(T) * (norm(Float64.(y)) / sqrt(n) + sum(j -> abs(β[j]) * norm(Float64.(X[:, j])) / sqrt(n), 1:p)) +
                 sum(j -> abs(Float64(ref.x̄[j])) * Δβ[j], 1:p)
        @test abs(Float64(fit.intercept) - β₀exact) <= bound₀
      else
        @test iszero(fit.intercept) && !fit.hasintercept
      end
    end
  end

  @testset "a fixed problem with hard-coded reference coefficients, $T" for T in (Float32, Float64)
    # Generated by the enumeration of this file (one candidate passes, strictly), with the data below
    # as exact rationals, no intercept and no scaling, λ = 0.375 and α = 0.5 (ν = 0.1875):
    #   exact_elastic_net(X, y, 0.375, 0.5); Float64.(ex.θ)
    # The data and the solution below are literals, independent of the generator and of the method.
    X = [ 1.0625  1.0625  0.5625  0.125
          1.125   1.1875  0.3125  3.8125
         -0.875  -0.5625 -1.3125  0.375
         -0.375   0.3125 -1.125   1.8125
         -0.25    2.3125  1.3125  1.5
         -0.8125 -0.125   0.3125  1.875
          0.125   1.0     2.1875 -0.375
          1.75   -0.3125  2.6875  0.75
         -0.6875  0.125  -0.4375  0.5
          1.25   -0.0625 -0.25    0.0
         -0.6875 -0.3125 -0.625  -1.8125
          0.3125  2.9375  1.0625 -0.125]
    y = [1.0, 0.125, -2.5, -2.875, -2.875, -1.3125, 1.125, 6.75, -1.5625, 2.875, -2.0625, -3.875]
    θ = [1.4965628212682063, -1.3180603923953769, 0.883779916632302, 0.0]            # to 17 digits
    A = [1, 2, 3]
    fit = lasso(T.(X), T.(y), T(0.375), cyc(T); alpha = 0.5)
    ref = reference(T.(X), T.(y), false, false)
    κθ = maximum(kappa(fit, T.(X), T.(y), ref))
    @test findall(!iszero, fit.coefficients) == A
    @test norm(Float64.(fit.coefficients) - θ, Inf) <= coef_bound(X, A, κθ, 0.1875)
    # the literal is the exact solution of the data above
    ex = exact_elastic_net(X, y, 0.375, 0.5)
    @test ex.A == A && norm(Float64.(ex.θ) - θ, Inf) <= 4eps()
  end

  # -------------------------------------------------------------------------------
  @testset "duplicated and correlated predictors: equal coefficients for α < 1, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in ALPHAS, (ic, st) in COMBOS
    X0, y = problem(T, 40, 40, 4; shared = 1.0)
    X = [X0[:, 1:2] X0[:, 1] X0[:, 3:4]]                  # columns 1 and 3 are identical
    ref = reference(X, y, ic, st)
    λ = T(0.15 * Float64(lambda_max(ref, alpha)))
    fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
    check_fit(fit, X, y, ref)
    θ = Float64.(fit.result.minimizer)
    A = findall(!iszero, θ)
    κθ = maximum(kappa(fit, X, y, ref))
    ν = Float64(BigFloat(fit.λ) * (1 - BigFloat(fit.alpha)))
    @test 1 in A && 3 in A
    # the exact solution is unique and has θ₁ = θ₃: the two stored values are within twice the bound
    @test abs(θ[1] - θ[3]) <= 2 * coef_bound(ref.Zf, A, κθ, ν)
    # three highly correlated copies (a column and two small perturbations of it) get close coefficients
    rng = Xoshiro(41)
    base = randn(rng, T, 40)
    Xc = [base base .+ T(1e-3) .* randn(rng, T, 40) base .+ T(1e-3) .* randn(rng, T, 40) randn(rng, T, 40, 2)]
    yc = base .* T(2) .+ T(0.3) .* randn(rng, T, 40)
    fc = lasso(Xc, yc, T(0.1), cyc(T); alpha, intercept = ic, standardize_x = st)
    b = Float64.(fc.result.minimizer)
    @test all(!iszero, b[1:3]) && maximum(b[1:3]) - minimum(b[1:3]) <= 0.3 * maximum(abs, b[1:3])
  end

  @testset "zero and constant columns have the coefficient 0, $T, α = $alpha" for T in (Float32, Float64), alpha in ALPHAS
    X, y = problem(T, 42, 30, 4)
    n = 30
    λ = T(0.15)
    # columns 3, 4 and 5 are constant (0.1 has no exact mean as a sum of copies), zero and constant
    Xz = [X[:, 1:2] fill(T(0.1), n) zeros(T, n) fill(T(-3), n) X[:, 3:4]]
    keep = [1, 2, 6, 7]
    for st in (false, true)
      f = @test_logs lasso(Xz, y, λ, cyc(T); alpha, intercept = true, standardize_x = st)
      @test all(j -> f.coefficients[j] === zero(T), 3:5)
      @test f.result.converged && all(isfinite, f.coefficients) && isfinite(f.intercept)
      base = lasso(X, y, λ, cyc(T); alpha, intercept = true, standardize_x = st)
      refz, refb = reference(Xz, y, true, st), reference(X, y, true, st)
      check_fit(f, Xz, y, refz)
      check_fit(base, X, y, refb)
      # the same fit as without the columns, up to the two optimality bounds
      A = findall(!iszero, base.coefficients)
      ν = Float64(BigFloat(base.λ) * (1 - BigFloat(base.alpha)))
      b1 = coef_bound(refb.Zf, A, maximum(kappa(base, X, y, refb)), ν)
      b2 = coef_bound(refz.Zf[:, keep], A, maximum(kappa(f, Xz, y, refz)), ν)
      @test norm(Float64.(f.result.minimizer[keep]) - Float64.(base.result.minimizer), Inf) <= b1 + b2
    end
    # a zero column also gets 0 without an intercept, with either value of standardize_x
    Xzero = [X[:, 1:2] zeros(T, n) X[:, 3:4]]
    for st in (false, true)
      f = lasso(Xzero, y, λ, cyc(T); alpha, standardize_x = st)
      @test f.coefficients[3] === zero(T) && f.result.converged
      check_fit(f, Xzero, y, reference(Xzero, y, false, st))
    end
    # three columns, one constant at 2 and two zero: the slopes are 0 and the intercept is the mean of y
    f = lasso([fill(T(2), 10) zeros(T, 10, 2)], y[1:10], λ, cyc(T); alpha, intercept = true, standardize_x = true)
    @test iszero(f.coefficients) && f.result.converged && isfinite(f.intercept)
    @test abs(Float64(f.intercept) - sum(Float64.(y[1:10])) / 10) <= 16 * eps(T) * maximum(abs, y[1:10])
  end

  @testset "λ = 0 is least squares, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in (0.3, 0.9), (ic, st) in COMBOS
    X, y = problem(T, 43, 40, 5)
    n, p = size(X)
    ref = reference(X, y, ic, st)
    fit = lasso(X, y, 0, cyc(T); alpha, intercept = ic, standardize_x = st)
    @test iszero(fit.λ) && fit.alpha === T(alpha) && fit.result.converged
    θ = Float64.(fit.result.minimizer)
    θls = Float64.(ref.Zb \ ref.yt)
    c = check_fit(fit, X, y, ref)
    @test norm(θ - θls, Inf) <= coef_bound(ref.Zf, 1:p, maximum(c.κθ), 0.0) + 4eps(T) * maximum(abs, θls)
    # the rank deficient design (a column that is the sum of two others): the fitted values are determined
    Xr = [X X[:, 1] .+ X[:, 2]]
    refr = reference(Xr, y, ic, st)
    f = lasso(Xr, y, 0, cyc(T); alpha, intercept = ic, standardize_x = st)
    @test f.result.converged
    cr = check_fit(f, Xr, y, refr)
    F = svd(refr.Zf)
    r = count(>(max(size(refr.Zf)...) * eps() * F.S[1]), F.S)
    P = F.U[:, 1:r]
    want = P * (P' * Float64.(refr.yt))
    err = norm(refr.Zf * Float64.(f.result.minimizer) - want)
    @test err <= n * sqrt(p + 1) * maximum(cr.κθ) / F.S[r] + 64 * eps() * (F.S[1] / F.S[r]) * norm(Float64.(refr.yt))
  end

  @testset "constant and zero responses, $T, α = $alpha" for T in (Float32, Float64), alpha in ALPHAS
    X, _ = problem(T, 44, 25, 4)
    λ = T(0.2)
    yc = fill(T(3), 25)
    # a constant response with an intercept: the intercept is its value and the coefficients are 0
    for st in (false, true)
      f = lasso(X, yc, λ, cyc(T); alpha, intercept = true, standardize_x = st)
      @test iszero(f.coefficients) && f.intercept === T(3) && iszero(f.residuals) && f.result.converged
    end
    # without an intercept it is an ordinary response
    for st in (false, true)
      f = lasso(X, yc, λ, cyc(T); alpha, standardize_x = st)
      check_fit(f, X, yc, reference(X, yc, false, st))
    end
    for (ic, st) in COMBOS
      f = lasso(X, zeros(T, 25), λ, cyc(T); alpha, intercept = ic, standardize_x = st)
      @test iszero(f.coefficients) && iszero(f.intercept) && iszero(f.residuals) && f.result.converged
    end
  end

  # -------------------------------------------------------------------------------
  @testset "λmax = maxⱼ |zⱼ'ỹ|/(nα): all zero above it, an active coordinate below it, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in ALPHAS, (ic, st) in COMBOS
    X, y = problem(T, 50, 30, 6)
    ref = reference(X, y, ic, st)
    λm = Float64(lambda_max(ref, alpha))
    above = T === Float32 ? 1.001 : 1 + 1e-9
    fit = lasso(X, y, T(above * λm), cyc(T); alpha, intercept = ic, standardize_x = st)
    @test iszero(fit.coefficients) && fit.result.converged
    if ic
      @test abs(Float64(fit.intercept) - Float64(ref.ȳ)) <= 16 * 8 * eps(T) * norm(Float64.(y)) / sqrt(length(y))
    else
      @test iszero(fit.intercept)
    end
    @test fit.residuals ≈ y .- fit.intercept
    # below it the coordinate with the largest |zⱼ'ỹ| enters with the sign of zⱼ'ỹ
    below = lasso(X, y, T((1 - 1e-3) * λm), cyc(T); alpha, intercept = ic, standardize_x = st)
    corr = [iszero(ref.s[k]) ? 0.0 : Float64(dot(ref.Zb[:, k], ref.yt)) for k in 1:ref.p]
    j = argmax(abs.(corr))
    @test below.coefficients[j] != 0 && sign(below.coefficients[j]) == sign(corr[j])
    @test iszero(lasso(X, y, T(1e3 * λm), cyc(T); alpha, intercept = ic, standardize_x = st).coefficients)
    # the boundary moves with alpha: between the lasso's λmax and λmax/α the lasso is zero and the
    # elastic net is not
    lasmax = Float64(lambda_max(ref, 1))
    mid = T((lasmax + Float64(λm)) / 2)
    @test iszero(lasso(X, y, mid, cyc(T); intercept = ic, standardize_x = st).coefficients)
    @test !iszero(lasso(X, y, mid, cyc(T); alpha, intercept = ic, standardize_x = st).coefficients)
  end

  # -------------------------------------------------------------------------------
  @testset "maxpasses = 1 below λmax does not converge, and the gap is measured, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in ALPHAS, (ic, st) in COMBOS
    X, y = problem(T, 60, 40, 6; shared = 2.0)
    n, p = size(X)
    ref = reference(X, y, ic, st)
    λ = T(0.1 * Float64(lambda_max(ref, alpha)))
    capped = CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 1)
    fit = @test_logs (:warn,) lasso(X, y, λ, capped; alpha, intercept = ic, standardize_x = st)
    @test !fit.result.converged && fit.result.passes == 1
    @test all(isfinite, fit.coefficients) && isfinite(fit.intercept) && all(isfinite, fit.residuals)
    @test any(!iszero, fit.coefficients)
    # the result fields are ProximalSolvers' values for the same call on the stated problem
    e = effective(T, λ, alpha)
    d = (ic || st) ? HDM.preprocess(X, y, ic, st) : (; Z = X, ỹ = y)
    r = @test_logs (:warn,) solve!(zeros(T, p), (one(T) / T(n)) * LeastSquaresLoss(d.ỹ, d.Z), ElasticNet(e.λ₁, e.λ₂),
                                   CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 1))
    @test same_result(fit.result, r)
    # the gap of the stopped run is positive and above that of the converged run on the same problem
    done = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
    obj, _ = objective(fit, X, y, ref.s)
    gap = Float64(obj - augmented_gap(ref.Zb, ref.yt, kkt_report(fit, ref).θ, fit.λ, fit.alpha).D)
    objd, _ = objective(done, X, y, ref.s)
    gapd = Float64(objd - augmented_gap(ref.Zb, ref.yt, kkt_report(done, ref).θ, done.λ, done.alpha).D)
    @test isfinite(gap) && gap > 0 && gap > 10 * max(gapd, 0.0)
  end

  # -------------------------------------------------------------------------------
  @testset "the inputs are not modified, and the fit does not alias them, $T, α = $alpha, $(label(ic, st))" for T in (Float32, Float64), alpha in (0.3, 1.0), (ic, st) in COMBOS
    X, y = problem(T, 70, 25, 6)
    X0, y0 = copy(X), copy(y)
    λ = T(0.2)
    fit = lasso(X, y, λ, cyc(T); alpha, intercept = ic, standardize_x = st)
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
    # a failing call leaves the data unchanged, also for an invalid alpha
    Xn = copy(X); Xn[3, 3] = NaN
    yn = copy(y); yn[2] = Inf
    for (Xa, ya, λa, αa) in ((Xn, y, λ, alpha), (X, yn, λ, alpha), (X, y, -λ, alpha), (X, y[1:5], λ, alpha),
                             (X, y, λ, 1.5), (X, y, λ, NaN), (X, y, λ, 0), (X, y, λ, "a"))
      Xa0, ya0 = copy(Xa), copy(ya)
      @test_throws Exception lasso(Xa, ya, λa, cyc(T); alpha = αa, intercept = ic, standardize_x = st)
      @test isequal(Xa, Xa0) && isequal(ya, ya0)
    end
    @test bitwise_equal(X, X0) && bitwise_equal(y, y0)
    if T === Float32
      @test_throws ArgumentError lasso(X, y, λ, cyc(T); alpha = 1e-50, intercept = ic, standardize_x = st)
      @test bitwise_equal(X, X0) && bitwise_equal(y, y0)
    end
  end

  # -------------------------------------------------------------------------------
  @testset "conventions, the stored alpha and the printed form, $T, $(label(ic, st))" for T in (Float32, Float64), (ic, st) in COMBOS
    X, y = problem(T, 80, 30, 6)
    n, p = size(X)
    ref = reference(X, y, ic, st)
    λ = T(0.15 * Float64(lambda_max(ref, 0.5)))
    fit = lasso(X, y, λ, cyc(T); alpha = 0.5, intercept = ic, standardize_x = st)
    @test fit isa LassoSolution{T} && fit.alpha isa T && fit.alpha === T(0.5)
    @test propertynames(fit) == (:coefficients, :residuals, :λ, :alpha, :intercept, :hasintercept, :result)
    @test fit.hasintercept === ic
    @test ic ? coef(fit) == [fit.intercept; fit.coefficients] : coef(fit) === fit.coefficients
    @test fit.residuals == y .- fit.intercept .- X * fit.coefficients
    @test predict(fit, X) == (ic ? X * fit.coefficients .+ fit.intercept : X * fit.coefficients)
    @test dof(fit) isa Int && dof(fit) == count(!iszero, fit.coefficients) + ic
    @test deviance(fit) == sum(abs2, residuals(fit)) && nobs(fit) == n
    # the printed form shows alpha when it is below 1, after λ
    s = sprint(show, fit)
    @test startswith(s, "LassoSolution: $p coefficients, $(count(!iszero, fit.coefficients)) nonzero, λ = $λ, α = $(fit.alpha)")
    @test occursin("intercept = $(fit.intercept)", s) == ic
    @test count(==('\n'), s) == 1 && s == string(fit)
    # and omits it for the lasso, which is stored with alpha = 1
    for lasso_fit in (lasso(X, y, λ, cyc(T); intercept = ic, standardize_x = st), lasso(X, y, λ, cyc(T); alpha = 1, intercept = ic, standardize_x = st))
      @test lasso_fit.alpha === one(T)
      s1 = sprint(show, lasso_fit)
      @test !occursin("α", s1) && startswith(s1, "LassoSolution: $p coefficients, $(count(!iszero, lasso_fit.coefficients)) nonzero, λ = $λ, ")
    end
    # no coefficient: the stored fields are read
    f0 = lasso(X, y, T(1e3 * Float64(lambda_max(ref, 0.5))), cyc(T); alpha = 0.5, intercept = ic, standardize_x = st)
    @test iszero(f0.coefficients) && dof(f0) == ic && f0.alpha === T(0.5) && f0.residuals == y .- f0.intercept
    # the stopped-run message of show
    stopped = @test_logs (:warn,) lasso(X, y, T(0.1 * Float64(lambda_max(ref, 0.5))),
                                        CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 1);
                                        alpha = 0.5, intercept = ic, standardize_x = st)
    @test occursin("not met after 1 passes", sprint(show, stopped)) && occursin("α = 0.5", sprint(show, stopped))
  end

  @testset "element types: Float32 in, Float32 out, alpha converted to T, $(label(ic, st))" for (ic, st) in COMBOS
    X, y = problem(Float32, 81, 30, 5)
    for alpha in (0.5, 0.5f0, 1 // 2, big"0.5")
      fit = lasso(X, y, 0.1, cyc(Float32); alpha, intercept = ic, standardize_x = st)
      @test fit isa LassoSolution{Float32} && fit.alpha === 0.5f0 && fit.λ === 0.1f0
      @test eltype(fit.coefficients) === Float32 && eltype(fit.residuals) === Float32 && fit.intercept isa Float32
      @test eltype(fit.result.minimizer) === Float32 && fit.result.converged
    end
    Xd, yd = problem(Float64, 81, 30, 5)
    fit = lasso(Xd, yd, 0.1, cyc(Float64); alpha = 0.5f0, intercept = ic, standardize_x = st)
    @test fit isa LassoSolution{Float64} && fit.alpha === 0.5
    # alpha = 0.3 (a Float64) is rounded to the nearest Float32
    fit = lasso(X, y, 0.1, cyc(Float32); alpha = 0.3, intercept = ic, standardize_x = st)
    @test fit.alpha === 0.3f0 && fit.alpha != 0.3
  end
end
end

end
