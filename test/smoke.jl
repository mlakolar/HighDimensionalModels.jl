module SmokeTest
# One cheap fit per element type, checked against the optimality conditions of the lasso.

using HighDimensionalModels, Test, LinearAlgebra, Random

@testset "smoke" begin
  @testset "$T" for T in (Float32, Float64)
    rng = Xoshiro(1)
    n, p = 40, 8
    X = T.(randn(rng, n, p))
    β = T[2, -1.5, 1, 0, 0, 0, 0, 0]
    y = X * β + T(0.5) * randn(rng, T, n)
    λ = T(0.15)
    tol = T === Float32 ? 1e-6 : 1e-10
    # a deterministic run, and the default solver on an explicit generator
    for solver in (CDSolver(order = :cyclic, momentum = false, tol = tol), CDSolver(tol = tol, rng = Xoshiro(2)))
      fit = lasso(X, y, λ, solver)
      b = coef(fit)
      @test fit isa LassoSolution{T}
      @test length(b) == p && all(isfinite, b) && any(!iszero, b) && any(iszero, b)
      @test fit.result.converged
      # KKT of ‖y - Xβ‖²/(2n) + λ‖β‖₁ in Float64: |g_j| ≤ λ, and g_j = λ sign(β_j) on the support
      g = Float64.(X)' * (Float64.(y) - Float64.(X) * Float64.(b)) / n
      slack = 1e-3 * Float64(λ)
      @test all(abs.(g) .<= λ + slack)
      @test all(j -> iszero(b[j]) || abs(g[j] - λ * sign(b[j])) <= slack, 1:p)
      # the StatsAPI methods
      @test nobs(fit) == n && length(residuals(fit)) == n
      @test residuals(fit) == y - X * b
      @test predict(fit, X) ≈ X * b
      @test predict(fit, X) + residuals(fit) ≈ y
    end
  end
end
end
