# Controls for the certificates: analytic problems with closed-form values,
# designed problems whose optimum is known by construction (the data are built
# from a primal point and a dual point that satisfy the optimality conditions;
# for the standardized fit with an intercept, from β₀ as well),
# designed nonoptimal points, weak duality and dual feasibility at random
# points, zero solutions at λ ≥ λmax, nonfinite inputs, and the reproducibility
# of the scenarios. Needs only the standard library and Test; the last testset
# runs the quick and control cases of the catalog when HighDimensionalModels can
# be loaded.

using Test, LinearAlgebra, Random

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "certificates.jl"))

const TIGHT = 1e-11       # a gap or residual at a solution known in closed form
const STD_TARGET = 1e-6   # the target of the standardized case (STANDARDIZED_TARGET of cases.jl)

ok(c) = c.status === :ok
mean_of(v) = sum(v) / length(v)
bad(c) = c.status === :uncertifiable && isnan(c.value)

# θ of least norm with X'θ = v (X has full column rank and n ≥ p)
dual_with(X, v) = pinv(transpose(Matrix(X))) * v

# a subgradient of λ‖β‖₁: λ sign(βⱼ) on the support, strictly inside [-λ, λ] off it
l1_subgradient(rng, β, λ) = [β[j] != 0 ? λ * sign(β[j]) : λ * (rand(rng) - 0.5) for j in eachindex(β)]

function sparse_vector(rng, p, k)
  β = zeros(p)
  β[draw_indices(rng, p, k)] .= draw_signs(rng, k) .* (0.5 .+ rand(rng, k))
  β
end

# a lasso whose solution is β by construction: θ = r/n with X'θ ∈ λ∂‖β‖₁, so
# y = Xβ + nθ
function designed_lasso(rng; n = 40, p = 8, λ = 0.1)
  X = randn(rng, n, p) .* transpose(0.5 .+ rand(rng, p)) .+ transpose(randn(rng, p))
  β = sparse_vector(rng, p, 3)
  θ = dual_with(X, l1_subgradient(rng, β, λ))
  y = X * β .+ n .* θ
  (; X, y, β, λ, θ)
end

# a standardized lasso with an intercept whose solution (β₀, β) is known by construction: the
# centred and scaled columns Z and the scales s are those of the certificate; θ is a dual point
# in the range of Z (whose columns sum to zero in exact arithmetic, so θ does too, up to rounding)
# with Z'θ ∈ λ∂‖sβ‖₁, and y = β₀ + Xβ + nθ. The residual of (β₀, β) is then nθ, and the
# optimality conditions of ‖y - β₀ - Xβ‖²/(2n) + λΣ sⱼ|βⱼ| hold in β₀ and β up to rounding.
function designed_standardized(rng; n = 60, p = 8, λ = 0.1, β₀ = 1.5)
  X = randn(rng, n, p) .* transpose(0.5 .+ 2 .* rand(rng, p)) .+ transpose(3 .* randn(rng, p))
  Z, s = centered_scaled(X)
  β = sparse_vector(rng, p, 3)
  θ = dual_with(Z, l1_subgradient(rng, β, λ))
  y = β₀ .+ X * β .+ n .* θ
  (; X, y, β, β₀, λ, θ, Z, s)
end

@testset "Relative gap, residuals and scaling" begin
  @test relative_gap(1.0, 1.0).value == 0 && ok(relative_gap(1.0, 1.0))
  @test relative_gap(2.0, 1.5).value == 0.25
  @test relative_gap(0.0, 0.0).value == 0
  @test relative_gap(1.0, nextfloat(1.0)).value == 0
  @test bad(relative_gap(1.0, 1.001))
  @test occursin("weak duality", relative_gap(1.0, 1.001).note)
  @test bad(relative_gap(Inf, 0.0)) && bad(relative_gap(1.0, NaN))
  @test ok(relative_gap(1.0, 1.0 + 1e-12; scale = 1.0, terms = 10^6))
  @test relative_residual(0.5, 0.25, :kkt_residual).value == 0.5
  @test relative_residual(0.5, 4.0, :kkt_residual).value == 0.125
  @test bad(relative_residual(NaN, 1.0, :kkt_residual))
  @test box_scale([2.0, -0.5, 0.0], 1.0) == 0.5
  @test box_scale([0.5], 1.0) == 1
  @test all(m -> m in CERTIFICATE_MEASURES, (:relative_gap, :kkt_residual))
end

@testset "Lasso certificate" begin
  rng = scenario_rng("certificate-controls-lasso", 1)
  # X = I, n = 2: the loss is ‖y - β‖²/4, whose lasso solution is S(y, 2λ)
  X, y = Matrix(1.0I, 2, 2), [2.0, -1.0]
  @test lasso_gap(X, y, [1.0, 0.0], 0.5).value ≤ 1e-15
  # β = 0: P = 5/4, θ = y/2, t = 1/2, D = 1/2·5/2 - 1/4·5/4 = 15/16
  a = lasso_parts(X, y, zeros(2), 0.5)
  @test a.P == 1.25 && a.D == 0.9375 && a.t == 0.5
  @test lasso_gap(X, y, zeros(2), 0.5).value == 0.25
  # λ = 0 reports the KKT residual of the least squares: |g| = |(y - β)/2| = 1/2 at β = (1, 0)
  c = lasso_gap(X, y, [1.0, 0.0], 0.0)
  @test c.measure === :kkt_residual && c.value == 0.5
  @test lasso_gap(X, y, y, 0.0).value == 0
  # a designed problem: the gap and the dual point at the optimum
  d = designed_lasso(rng)
  c = lasso_gap(d.X, d.y, d.β, d.λ)
  @test ok(c) && c.measure === :relative_gap && c.value ≤ TIGHT
  a = lasso_parts(d.X, d.y, d.β, d.λ)
  @test a.θ ≈ d.θ atol = 1e-12
  # weak duality and dual feasibility at random points: the bounds at these points are below the known optimum
  for _ in 1:5
    β = d.β .+ 0.3 .* randn(rng, length(d.β))
    a = lasso_parts(d.X, d.y, β, d.λ)
    @test ok(lasso_gap(d.X, d.y, β, d.λ))
    @test all(abs.(transpose(d.X) * a.θ) .<= d.λ * (1 + 1e-12))
    @test a.D ≤ a.P && a.D ≤ lasso_parts(d.X, d.y, d.β, d.λ).P + 1e-13
  end
  # a designed least squares problem at λ = 0 has the KKT residual zero at its solution
  z = designed_lasso(rng; λ = 0.0)
  c = lasso_gap(z.X, z.y, z.β, 0.0)
  @test c.measure === :kkt_residual && c.value ≤ TIGHT
  # coefficients that are not optimal miss a gap target of 1e-6, in either direction
  for scale in (0.5, 1.5), shift in (0.0, 1e-3)
    β = scale .* d.β .+ shift
    @test lasso_gap(d.X, d.y, β, d.λ).value > 1e-6
  end
  β = copy(d.β)
  β[findfirst(!iszero, β)] += 1e-3
  @test lasso_gap(d.X, d.y, β, d.λ).value > 1e-6
  # at λ ≥ λmax = ‖X'y‖∞/n the solution is zero
  X, y = d.X, d.y
  λmax = maximum(abs.(transpose(X) * y)) / size(X, 1)
  @test lasso_gap(X, y, zeros(8), λmax).value ≤ TIGHT
  @test lasso_gap(X, y, zeros(8), 2λmax).value ≤ TIGHT
  @test lasso_gap(X, y, zeros(8), 0.9λmax).value > 1e-4
  # element types are converted, not rejected
  X32, y32, β32 = Float32.(d.X), Float32.(d.y), Float32.(d.β)
  @test lasso_gap(X32, y32, β32, Float32(d.λ)).value == lasso_gap(Float64.(X32), Float64.(y32), Float64.(β32), Float64(Float32(d.λ))).value
end

@testset "Standardized lasso certificate with an intercept" begin
  rng = scenario_rng("certificate-controls-standardized", 1)
  d = designed_standardized(rng)
  n = size(d.X, 1)
  @test all(>(0), d.s) && maximum(abs, sum(d.Z; dims = 1)) < 1e-12
  # the designed point is optimal: the gap and the dual point
  c = lasso_gap_standardized(d.X, d.y, d.β₀, d.β, d.λ; target = STD_TARGET)
  @test ok(c) && c.measure === :relative_gap && c.value ≤ TIGHT
  a = lasso_standardized_parts(d.X, d.y, d.β₀, d.β, d.λ)
  @test a.θ ≈ d.θ atol = 1e-12
  # the primal value is the weighted objective in the original coordinates
  r = d.y .- d.β₀ .- d.X * d.β
  @test a.P ≈ dot(r, r) / (2n) + d.λ * dot(d.s, abs.(d.β)) rtol = 1e-12
  # a perturbed intercept raises the primal value by δ²/2 and misses the target of 1e-6, in either direction
  for δ in (1e-2, -1e-2, 0.1)
    p = lasso_standardized_parts(d.X, d.y, d.β₀ + δ, d.β, d.λ)
    @test p.P - a.P ≈ δ^2 / 2 rtol = 1e-6
    @test lasso_gap_standardized(d.X, d.y, d.β₀ + δ, d.β, d.λ; target = STD_TARGET).value > 1e-6
  end
  # perturbed coefficients with the designed intercept miss it too
  for scale in (0.5, 1.5)
    @test lasso_gap_standardized(d.X, d.y, d.β₀, scale .* d.β, d.λ; target = STD_TARGET).value > 1e-6
  end
  # weak duality and dual feasibility at random points
  for _ in 1:5
    β = d.β .+ 0.3 .* randn(rng, length(d.β))
    β₀ = d.β₀ + 0.3 * randn(rng)
    a = lasso_standardized_parts(d.X, d.y, β₀, β, d.λ)
    @test ok(lasso_gap_standardized(d.X, d.y, β₀, β, d.λ; target = STD_TARGET))
    @test all(abs.(transpose(d.Z) * a.θ) .<= d.λ * (1 + 1e-12))
    @test a.feasibility <= 1
    @test a.D ≤ a.P && a.D ≤ lasso_standardized_parts(d.X, d.y, d.β₀, d.β, d.λ).P + 1e-13
  end
  # a column that is constant has scale 0 and is ignored by the penalty
  Xc = [d.X fill(2.0, n)]
  @test lasso_gap_standardized(Xc, d.y, d.β₀, [d.β; 0.0], d.λ; target = STD_TARGET).value ≤ TIGHT
  @test ok(lasso_gap_standardized(Xc, d.y, d.β₀, [d.β; 0.5], d.λ; target = STD_TARGET))
  # at λ ≥ λmax = maxⱼ |zⱼ'ỹ|/n of the centred and scaled problem the solution is β = 0, β₀ = ȳ
  ỹ = d.y .- mean_of(d.y)
  λmax = maximum(abs.(transpose(d.Z) * ỹ)) / n
  @test lasso_gap_standardized(d.X, d.y, mean_of(d.y), zeros(8), λmax; target = STD_TARGET).value ≤ TIGHT
  @test lasso_gap_standardized(d.X, d.y, mean_of(d.y), zeros(8), 2λmax; target = STD_TARGET).value ≤ TIGHT
  @test lasso_gap_standardized(d.X, d.y, mean_of(d.y), zeros(8), 0.9λmax; target = STD_TARGET).value > 1e-4
  # a column whose squares underflow has a representable scale: the gap of β = 0 on
  # X = (1e-200, -1e-200)', y = (1, -1) is not 0. The optimum is β ≈ 9e199 with objective 0.095, and the
  # dual point scaled into |x̃'θ| ≤ λs (t = 0.1) has D = 0.095, while P(0) = 0.5
  Xu, yu = reshape([1e-200, -1e-200], 2, 1), [1.0, -1.0]
  c = lasso_gap_standardized(Xu, yu, 0.0, [0.0], 0.1; target = STD_TARGET)
  @test ok(c) && c.value > 0.8
  a = lasso_standardized_parts(Xu, yu, 0.0, [0.0], 0.1)
  @test a.P == 0.5 && a.feasibility <= 1
  @test a.D ≈ 0.095 rtol = 1e-14
  @test a.t ≈ 0.1 rtol = 1e-14
  @test lasso_gap_standardized(Xu, yu, 0.0, [9e199], 0.1; target = STD_TARGET).value <= TIGHT      # the optimum
  # a small t: with β = -1e150 the residuals are ∓1e150 and the dual point is scaled by t ≈ 1e-250
  # (t² underflows in Float64). The reported D is the dual objective y'θ - n‖θ‖²/2 at the returned
  # dual point, here about 5e-201 (the optimum: λ equals the amplitude of y, so β = 0 is optimal);
  # it is compared with an independent evaluation in 512-bit BigFloat, to a relative 1e-12 (the
  # returned θ is rounded to Float64, which changes D at second order at this point)
  Xs, ys = reshape([-1.0, 1.0], 2, 1), [-1e-100, 1e-100]
  a = lasso_standardized_parts(Xs, ys, 0.0, [-1e150], 1e-100)
  @test a.t > 0 && a.t^2 == 0 && a.feasibility <= 1
  D512 = setprecision(BigFloat, 512) do
    θ512, y512 = BigFloat.(a.θ), BigFloat.(ys)
    dot(y512, θ512) - length(ys) * dot(θ512, θ512) / 2
  end
  @test a.D > 0
  @test abs(BigFloat(a.D) - D512) <= 1e-12 * abs(D512)
  println("small-t control: t = $(a.t), D = $(a.D), 512-bit reference = $(Float64(D512)), relative difference = $(Float64(abs(BigFloat(a.D) - D512) / abs(D512)))")
  c = lasso_gap_standardized(Xs, ys, 0.0, [-1e-100], 1e-100; target = STD_TARGET)
  @test ok(c)
  @test lasso_gap_standardized(Xs, ys, 0.0, [0.0], 1e-100; target = STD_TARGET).value <= TIGHT
  # cancellation in the primal: each column is constant, so every scale is 0, and the exact
  # prediction is 1e120 + 1 - 1e120 = 1 (residuals -1, primal value 0.5, and the optimum is 0 with
  # β = 0). The 256-bit evaluation of Xβ loses the 1, but the error bound G(p) Σ|Xᵢⱼ||βⱼ| ≈ 4e44
  # is carried: the certificate must not be ok
  Xk, yk = repeat([1e120 1.0 -1e120], 2, 1), zeros(2)
  @test lasso_standardized_parts(Xk, yk, 0.0, ones(3), 0.1) isa String
  c = lasso_gap_standardized(Xk, yk, 0.0, ones(3), 0.1; target = STD_TARGET)
  @test bad(c) && occursin("error bound", c.note)
  # the value is an upper bound of the true relative gap, and tight when the bounds are small. For
  # X = (-1, 1)', y = (0.5, 2.5), β₀ = 1.5 (the exact mean offset) and the stored λ the problem is
  # ‖y - β₀ - Xβ‖²/4 + λ|β| = (1 - β)²/2 + λ|β| for β ≥ 0, whose minimum is λ - λ²/2 at β = 1 - λ.
  # The true relative gap of a point β is computed here in rational arithmetic
  Xa, ya = reshape([-1.0, 1.0], 2, 1), [0.5, 2.5]
  λq = Rational{BigInt}(0.1)
  Pq(β) = (1 - β)^2 / 2 + λq * β
  Pmin = λq - λq^2 / 2
  # (tight where the dual point built from the residual of β is good: not above the optimum, where it is not)
  for (β, tight) in ((0.9, true), (0.8, true), (0.5, true), (1.2, false))
    βq = Rational{BigInt}(β)
    exact = (Pq(βq) - Pmin) / Pq(βq)
    c = lasso_gap_standardized(Xa, ya, 1.5, [β], 0.1; target = STD_TARGET)
    @test ok(c)
    @test exact <= BigFloat(c.value)                       # an upper bound
    tight && @test BigFloat(c.value) <= exact + 1e-12      # and nearly tight
  end
  a = lasso_standardized_parts(Xa, ya, 1.5, [0.8], 0.1)
  Pexact = Pq(Rational{BigInt}(0.8))
  # the core inequality |P̂ - P| ≤ eP on the BigFloat values before conversion (P in 512-bit precision)
  @test setprecision(BigFloat, 512) do
    abs(a.Pb - BigFloat(Pexact)) <= a.ePb
  end
  # and the Float64 report, which adds its own rounding of P (this checks the report, not the bound)
  @test abs(BigFloat(a.P) - Pexact) <= a.eP + 8 * eps(Float64(Pexact)) && a.eP < 1e-60 && a.width < 1e-60
  # a certificate is uncertifiable when the interval of the relative gap contains the target in the
  # sense value_lo ≤ target < value_hi; it is ok when the true gap, or the whole interval, lies below
  # the target (a value at most the target) or above it (a value above the target). The data are
  # those of the review: three constant columns, C, 1 and -C, and a column (-1, 1), with
  # β = (1, 1, 1, 0.9) and λ = 0.1, whose gap is set by the intercept β₀ = 0.5 + δ; the error
  # bounds are about 1e-12 relative, since the fitted values cancel C
  C = 1e62
  Xt, yt, βt = [C 1.0 -C -1.0; C 1.0 -C 1.0], [0.5, 2.5], [1.0, 1.0, 1.0, 0.9]
  δof(g) = sqrt(0.19 * (g - 2e-12) / (1 - (g - 2e-12)))
  gapat(g; target = 1e-6) = lasso_gap_standardized(Xt, yt, 0.5 + δof(g), βt, 0.1; target)
  c = gapat(1e-6)
  @test bad(c) && occursin("could change the verdict", c.note) && occursin("1.0e-6", c.note)
  a = lasso_standardized_parts(Xt, yt, 0.5 + δof(1e-6), βt, 0.1)
  @test a.gap_lo <= 1e-6 < a.gap && a.width < 1e-10
  @test ok(gapat(1e-7)) && gapat(1e-7).value <= 1e-6 && gapat(1e-7).value >= 5e-8
  @test ok(gapat(1e-5)) && gapat(1e-5).value > 1e-6 && gapat(1e-5).value <= 2e-5
  # the same value is decided by the target: a target above the bounds is ok, one below is ok with a value above it
  @test ok(gapat(1e-6; target = 1e-5)) && gapat(1e-6; target = 1e-5).value <= 1e-5
  @test ok(gapat(1e-6; target = 1e-7)) && gapat(1e-6; target = 1e-7).value > 1e-7
  @test bad(lasso_gap_standardized(Xa, ya, 1.5, [0.8], 0.1; target = NaN))
  @test bad(lasso_gap_standardized(Xa, ya, 1.5, [0.8], 0.1; target = 0.0))
  # a mean that is lost by cancellation in the column: the error bound of the mean enters the scale
  # and the correlations, and the certificate is valid or refuses; here it is valid
  Xm = reshape([1e120, 1.0, -1e120], 3, 1)
  c = lasso_gap_standardized(Xm, [1.0, 2.0, 3.0], 0.0, [0.0], 0.1; target = STD_TARGET)
  @test ok(c) || bad(c)
  # the exact constant rule: a constant column has scale 0, also when the mean of its copies is not exact in floating point
  @test centered_scaled(fill(0.1, 10, 1))[2] == [0.0]
  @test centered_scaled(fill(1 / 3, 7, 2))[1] == zeros(7, 2)
  c = lasso_gap_standardized([fill(0.1, 10) randn(rng, 10)], randn(rng, 10), 0.0, [0.0, 0.0], 0.1; target = STD_TARGET)
  @test ok(c)
  # a column that is not all zeros whose scale is not a normal Float64 is uncertifiable
  tiny = reshape([5e-324; zeros(99)], :, 1)
  @test bad(lasso_gap_standardized(tiny, randn(rng, 100), 0.0, [0.0], 0.1; target = STD_TARGET))
  @test occursin("scale of column 1", lasso_gap_standardized(tiny, randn(rng, 100), 0.0, [0.0], 0.1; target = STD_TARGET).note)
  sub = reshape(1e-310 .* randn(rng, 20), :, 1)
  @test bad(lasso_gap_standardized(sub, randn(rng, 20), 0.0, [0.0], 0.1; target = STD_TARGET))
  huge = reshape([floatmax(), -floatmax(), floatmax() / 3, 0.0], :, 1)
  @test ok(lasso_gap_standardized(huge, [1.0, 2.0, 3.0, 4.0], 0.0, [0.0], 0.1; target = STD_TARGET))      # centring in BigFloat does not overflow
  # element types are converted, and invalid inputs are uncertifiable
  X32, y32, β32 = Float32.(d.X), Float32.(d.y), Float32.(d.β)
  @test lasso_gap_standardized(X32, y32, 1.5f0, β32, Float32(d.λ); target = STD_TARGET).value ==
        lasso_gap_standardized(Float64.(X32), Float64.(y32), 1.5, Float64.(β32), Float64(Float32(d.λ)); target = STD_TARGET).value
  for v in (NaN, Inf)
    @test bad(lasso_gap_standardized(d.X, d.y, v, d.β, d.λ; target = STD_TARGET))
    @test bad(lasso_gap_standardized(d.X, d.y, d.β₀, [v; d.β[2:end]], d.λ; target = STD_TARGET))
    @test bad(lasso_gap_standardized(d.X, d.y, d.β₀, d.β, v; target = STD_TARGET))
  end
  @test bad(lasso_gap_standardized(d.X, d.y, d.β₀, d.β, 0.0; target = STD_TARGET))
  @test bad(lasso_gap_standardized(d.X, d.y, d.β₀, d.β, -0.1; target = STD_TARGET))
  @test bad(lasso_gap_standardized(d.X, d.y[1:end-1], d.β₀, d.β, d.λ; target = STD_TARGET))
end

@testset "Invalid and nonfinite inputs are uncertifiable" begin
  rng = scenario_rng("certificate-controls-nonfinite", 1)
  X, y = randn(rng, 20, 4), rand(rng, 20)
  @test ok(lasso_gap(X, y, zeros(4), 0.1))
  for v in (NaN, Inf)
    β = [v, 0.0, 0.0, 0.0]
    Xv = copy(X)
    Xv[3, 2] = v
    for (b, Xb, yb) in ((β, X, y), (zeros(4), Xv, y), (zeros(4), X, [v; y[2:end]]))
      @test bad(lasso_gap(Xb, yb, b, 0.1))
      @test bad(lasso_gap(Xb, yb, b, 0.0))
    end
    @test bad(lasso_gap(X, y, zeros(4), v))
  end
  @test bad(lasso_gap(X, y, zeros(4), -0.1))
  @test bad(lasso_gap(X, y, zeros(3), 0.1))
  @test bad(lasso_gap(X, y[1:19], zeros(4), 0.1))
  @test bad(lasso_gap(X[1:0, :], y[1:0], zeros(4), 0.1))
  @test bad(lasso_gap(X, y, "a", 0.1))
end

@testset "Scenarios" begin
  @test rand(scenario_rng("ls-200x50", 1), UInt64) == rand(Xoshiro(0x91bae84adbad28da), UInt64)
  # the hash of repetition 1, as pinned for the earlier harness
  @test data_hash(scenario("ls-200x50", 1)) == "ad8f0a23c3dd3989b99e6c74233b1aaafec8720b647d2099b9bb20de96aa2a38"
  required = ["ls-200x50", "ls-tall-5000x40", "ls-wide-100x400", "ls-corr-300x100", "ls-2000x1000"]
  @test all(in(SCENARIOS), required) && allunique(SCENARIOS)
  for key in SCENARIOS
    a, b, c = scenario(key, 1), scenario(key, 1), scenario(key, 2)
    @test keys(a) == keys(b) && all(getproperty(a, k) == getproperty(b, k) for k in keys(a))
    @test data_hash(a) == data_hash(b) && occursin(r"^[0-9a-f]{64}$", data_hash(a))
    @test data_hash(a) != data_hash(c) && a.y != c.y
    @test all(v -> eltype(v) === Float64, values(a))
    @test all(cert_finite, values(a))
  end
  @test_throws ArgumentError scenario("ls-1x1", 1)
  @test_throws ArgumentError scenario("ls-200x50", 0)
  @test size(scenario("ls-wide-100x400", 1).X) == (100, 400)
  # hashes cover element types, sizes and field names
  @test data_hash((; x = [1.0, 2.0])) != data_hash((; x = Float32[1.0, 2.0]))
  @test data_hash((; x = [1.0, 2.0])) != data_hash((; x = reshape([1.0, 2.0], 1, 2)))
  @test data_hash((; a = 1, b = [1.0])) == data_hash((; b = [1.0], a = 1))
  @test data_hash((; a = 1)) != data_hash((; b = 1))
end

if Base.find_package("HighDimensionalModels") !== nothing && get(ENV, "HDM_CERTIFICATES_PACKAGE", "1") != "0"
  # secondary: tight package fits against the certificate, and the quick and
  # control cases of the catalog against their targets (HDM_CERTIFICATES_PACKAGE=0 skips)
  using HighDimensionalModels
  include(joinpath(@__DIR__, "cases.jl"))
  @testset "Secondary agreement with HighDimensionalModels" begin
    # a cyclic fit to a tight tolerance, on a design that is wide and one that is correlated
    for (key, λ) in (("ls-200x50", 0.05), ("ls-wide-100x400", 0.1), ("ls-corr-300x100", 0.05))
      a = scenario(key, 1)
      X, y = standardized(a.X, a.y)
      fit = lasso(X, y, λ, CDSolver(; order = :cyclic, momentum = false, tol = 1e-12))
      c = lasso_gap(X, y, coef(fit), λ)
      @test ok(c) && c.value ≤ 1e-9
      # the primal value of the certificate is the objective, computed here from the residuals
      r = y .- X * coef(fit)
      @test lasso_parts(X, y, coef(fit), λ).P ≈ dot(r, r) / (2length(y)) + λ * sum(abs, coef(fit)) rtol = 1e-12
    end
    # the standardized fit with an intercept on the raw inputs, against the certificate of its own objective
    for (key, ratio) in (("ls-200x50", 0.05), ("ls-wide-100x400", 0.1))
      a = scenario(key, 1)
      Z, r = standardized(a.X, a.y)
      λ = ratio * lambda_max(Z, r)
      fit = lasso(a.X, a.y, λ, CDSolver(; order = :cyclic, momentum = false, tol = 1e-12); intercept = true, standardize_x = true)
      c = lasso_gap_standardized(a.X, a.y, fit.intercept, fit.coefficients, λ; target = STD_TARGET)
      @test ok(c) && c.value ≤ 1e-9
      # the same objective in the transformed coordinates: the plain certificate of the unweighted lasso on Z
      s = [sqrt(sample_mean(abs2.(a.X[:, j] .- sample_mean(a.X[:, j])))) for j in axes(a.X, 2)]
      @test lasso_gap(Z, r, fit.coefficients .* s, λ).value ≤ 1e-9
    end
    # the case table: the quick lasso cases meet their targets, and the control misses its own
    catalog = build_catalog("extended", ["lasso", "controls"], [r"lasso/lasso/(dense|dense-standardized|float32)$", r"one-pass"])
    @test [c.meta.id for c in catalog] == ["controls/lasso/one-pass", "lasso/lasso/dense", "lasso/lasso/dense-standardized", "lasso/lasso/float32"]
    # the target the certificate of dense-standardized uses is the one of its case
    @test STD_TARGET == STANDARDIZED_TARGET && only(c for c in catalog if endswith(c.meta.id, "dense-standardized")).meta.target == STANDARDIZED_TARGET
    for spec in catalog
      problem = scenario(spec.meta.scenario, 1)
      prepared = spec.build(problem)
      cert = spec.certify(problem, prepared, spec.call(deepcopy(prepared)))
      @test ok(cert) && string(cert.measure) == spec.meta.measure
      spec.meta.group == "controls" ? (@test cert.value > spec.meta.target) : (@test cert.value ≤ spec.meta.target)
    end
    # a reproducible run: equal coefficients (and intercepts) from two fresh copies of the prepared inputs
    @test sort([c.meta.id for c in build_catalog("quick", ["lasso"])]) == ["lasso/lasso/dense", "lasso/lasso/dense-standardized"]
    for spec in build_catalog("quick", ["lasso"])
      prepared = spec.build(scenario(spec.meta.scenario, 1))
      a, b = spec.call(deepcopy(prepared)), spec.call(deepcopy(prepared))
      @test coef(a) == coef(b)
    end
  end
end
