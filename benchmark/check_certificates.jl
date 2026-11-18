# Controls for the certificate: analytic problems with closed-form values,
# designed problems whose optimum is known by construction (the data are built
# from a primal point and a dual point that satisfy the optimality conditions),
# designed nonoptimal points, weak duality and dual feasibility at random
# points, zero solutions at λ ≥ λmax, nonfinite inputs, and the reproducibility
# of the scenarios. Needs only the standard library and Test; the last testset
# runs the quick and control cases of the catalog when HighDimensionalModels can
# be loaded.

using Test, LinearAlgebra, Random

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "certificates.jl"))

const TIGHT = 1e-11       # a gap or residual at a solution known in closed form

ok(c) = c.status === :ok
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
    # the case table: a quick lasso case meets its target, and the control misses its own
    catalog = build_catalog("extended", ["lasso", "controls"], [r"lasso/lasso/(dense|float32)$", r"one-pass"])
    @test [c.meta.id for c in catalog] == ["controls/lasso/one-pass", "lasso/lasso/dense", "lasso/lasso/float32"]
    for spec in catalog
      problem = scenario(spec.meta.scenario, 1)
      prepared = spec.build(problem)
      cert = spec.certify(problem, prepared, spec.call(deepcopy(prepared)))
      @test ok(cert) && string(cert.measure) == spec.meta.measure
      spec.meta.group == "controls" ? (@test cert.value > spec.meta.target) : (@test cert.value ≤ spec.meta.target)
    end
    # a reproducible run: equal coefficients from two fresh copies of the prepared inputs
    spec = only(build_catalog("quick", ["lasso"]))
    prepared = spec.build(scenario(spec.meta.scenario, 1))
    @test coef(spec.call(deepcopy(prepared))) == coef(spec.call(deepcopy(prepared)))
  end
end
