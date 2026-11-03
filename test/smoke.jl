module SmokeTest

using HighDimensionalModels, Test, LinearAlgebra, Random, Statistics, SparseArrays
using HiGHS, StatsModels

logistic(z) = 1 / (1 + exp(-z))

# a tiny sparse regression problem: β = (2, -1.5, 1, 0, …)
function problem(seed, n, p; k = 3)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  X, X * β + 0.5 * randn(rng, n), β
end

@testset "smoke" begin
  @testset "core interface and lasso" begin
    X, y, _ = problem(1, 40, 8)
    λ = 0.15
    # the default coordinate descent tolerance is opt_tol = 1e-8 on the iterate; 1e-6 slack on the gradient is generous
    sol = lasso(X, y, λ, CDOptions(; opt_tol = 1e-10); intercept = true)
    b = coef(sol)
    @test length(b) == 9 && all(isfinite, b)                       # intercept first, then the slopes
    @test nobs(sol) == 40 && length(residuals(sol)) == 40
    @test predict(sol, X) ≈ X * b[2:end] .+ b[1]
    @test residuals(sol) ≈ y - predict(sol, X)
    @test sol.converged
    # KKT of (2n)⁻¹‖y - β₀ - Xβ‖² + λ‖β‖₁: |X'r|/n ≤ λ, with equality and the sign of β on the support
    g = X' * (y - predict(sol, X)) / 40
    β = b[2:end]
    @test abs(sum(y - predict(sol, X))) < 1e-6                     # the intercept is unpenalized
    @test all(abs.(g) .<= λ + 1e-6)
    @test any(!iszero, β) && any(iszero, β)
    @test all(j -> iszero(β[j]) || isapprox(g[j], λ * sign(β[j]); atol = 1e-6), 1:8)

    path = LassoPath(X, y; nλ = 10)
    @test path isa LassoPath && length(path.λpath) == 10
    @test path.λpath[end] < path.λpath[1]
    @test iszero(path.βpath[1]) && count(!iszero, path.βpath[end]) > 0
  end

  @testset "glm_lasso (logistic)" begin
    rng = Xoshiro(2)
    X = randn(rng, 150, 6)
    y = Float64.(rand(rng, 150) .< logistic.(0.2 .+ X * [1.5, -1, 0, 0, 0, 0]))
    sol = glm_lasso(X, y, 0.02)
    b = coef(sol)
    @test all(isfinite, b) && sol.converged
    @test deviance(sol) < nulldeviance(sol)
    @test all(0 .< predict(sol, X) .< 1)                           # fitted probabilities
    @test sign(b[2]) == 1 && sign(b[3]) == -1                      # the two true signals
  end

  @testset "group lasso and sparse group lasso" begin
    X, y, _ = problem(3, 60, 8)                                    # signal in columns 1–3, groups of two
    groups = [1:2, 3:4, 5:6, 7:8]
    big, small = 5.0, 0.01
    s0 = group_lasso(X, y, big, groups)
    @test s0.converged && iszero(coef(s0)[2:end]) && isempty(s0.activegroups)
    s1 = group_lasso(X, y, small, groups)
    @test s1.converged && 1 in s1.activegroups && 2 in s1.activegroups
    @test all(g -> all(iszero, coef(s1)[1 .+ groups[g]]) || all(!iszero, coef(s1)[1 .+ groups[g]]), 1:4)  # whole groups
    t0 = sparse_group_lasso(X, y, big, groups)
    @test t0.converged && iszero(coef(t0)[2:end])
    t1 = sparse_group_lasso(X, y, small, groups)
    @test t1.converged && 1 in t1.activegroups && count(!iszero, coef(t1)[2:end]) > 0
  end

  @testset "quantile_lasso (HiGHS exact)" begin
    X, y, _ = problem(4, 40, 5)
    sol = quantile_lasso(X, y, 0.05; method = :exact, tau = 0.5)
    @test sol.converged && sol.method == :exact
    @test abs(sol.duality_gap) <= 2e-5 * (1 + abs(sol.objective))  # the documented tol = 2e-5 relative gap
    @test sol.kkt_residual <= 2e-5
    @test all(isfinite, coef(sol))
    @test length(coef(sol)) == 6
  end

  @testset "low rank" begin
    rng = Xoshiro(5)
    X = randn(rng, 40, 6)
    Y = X * (randn(rng, 6, 2) * randn(rng, 2, 4)) + 0.1 * randn(rng, 40, 4)   # true rank 2
    rr = reduced_rank(X, Y, 2)
    @test rank(rr.B) <= 2 && rr.rank <= 2 && size(rr.B) == (6, 4)
    @test sum(abs2, residuals(rr)) <= sum(abs2, Y .- mean(Y; dims = 1)) # beats the intercept-only fit
    nr = nuclear_regression(X, Y, 0.2)
    @test nr.converged && eltype(nr.B) == Float64
    @test rank(nr.B; rtol = 1e-8) <= 4
  end

  @testset "fused lasso" begin
    rng = Xoshiro(6)
    truth = [fill(0.0, 15); fill(3.0, 15)]                         # one jump
    y = truth + 0.2 * randn(rng, 30)
    sol = fused_lasso(y, 0.0, 0.1)
    @test sol.converged
    @test count(>(1e-6), abs.(diff(coef(sol)))) <= 3               # recovers a piecewise-constant fit
    @test abs(coef(sol)[end] - 3) < 0.5 && abs(coef(sol)[1]) < 0.5
  end

  @testset "multitask lasso" begin
    rng = Xoshiro(7)
    Xs = [randn(rng, 40, 6) for _ in 1:3]
    ys = [X[:, 1] * 2 - X[:, 2] + 0.3 * randn(rng, 40) for X in Xs]   # rows 1 and 2 shared by all tasks
    s = multitask_lasso(Xs, ys, 0.2)
    @test s.converged && s.activerows == [1, 2]
    @test all(iszero, s.B[3:end, :]) && all(!iszero, s.B[1:2, :])
    @test iszero(multitask_lasso(Xs, ys, 10.0).B)
  end

  @testset "slope and rlasso" begin
    X, y, _ = problem(8, 60, 8)
    s = slope(X, y; calibration = :bh, q = 0.1, sigma = 0.5)
    @test s.converged && length(coef(s)) >= 8 && all(isfinite, coef(s))
    @test issorted(s.lambda; rev = true) && all(isfinite, residuals(s))
    r = rlasso(X, y; rng = Xoshiro(1))
    @test r.converged && all(isfinite, coef(r)) && 1 in r.selected
  end

  @testset "formula (StatsModels)" begin
    X, y, _ = problem(9, 40, 3)
    df = (y = y, x1 = X[:, 1], x2 = X[:, 2], x3 = X[:, 3])
    m = lasso(@formula(y ~ x1 + x2 + x3), df, 0.1)
    s = lasso(X, y, 0.1; intercept = true)
    @test m isa FormulaFit && coefnames(m) == ["(Intercept)", "x1", "x2", "x3"]
    @test coef(m) ≈ coef(s) && responsename(m) == "y"
    @test predict(m, df) ≈ predict(s, X)
  end
end

end
