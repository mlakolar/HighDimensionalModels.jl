module StatsAPITest
# The StatsAPI methods of `LassoSolution`, the exported names and the printed form.

using HighDimensionalModels, Test, LinearAlgebra, Random
using ProximalSolvers: ProximalSolvers

function problem(T, seed, n, p)
  rng = Xoshiro(seed)
  X = randn(rng, T, n, p)
  X, X[:, 1] .- T(0.5) .* X[:, 2] .+ T(0.3) .* randn(rng, T, n)
end

solver(T) = CDSolver(order = :cyclic, momentum = false, tol = T === Float32 ? 1e-6 : 1e-10)

@testset "StatsAPI methods" begin

  @testset "exports" begin
    @test Set(names(HighDimensionalModels)) ==
          Set([:HighDimensionalModels, :lasso, :LassoSolution, :CDSolver, :coef, :predict, :residuals, :nobs, :dof, :deviance, :sqrt_lasso, :SqrtLassoSolution])
    # the six methods are those of StatsAPI, and CDSolver is the one of ProximalSolvers
    for f in (coef, predict, residuals, nobs, dof, deviance)
      @test nameof(parentmodule(f)) === :StatsAPI
      @test any(m -> m.sig <: Tuple{typeof(f), LassoSolution, Vararg}, methods(f))
    end
    @test HighDimensionalModels.coef === coef
    @test CDSolver === ProximalSolvers.CDSolver
  end

  @testset "$T" for T in (Float32, Float64)
    X, y = problem(T, 1, 30, 5)
    fit = lasso(X, y, T(0.1), solver(T))
    b = coef(fit)
    @test b isa Vector{T} && b === fit.coefficients && length(b) == 5

    @test residuals(fit) === fit.residuals
    @test residuals(fit) == y - X * b                            # at the returned point
    @test nobs(fit) == 30 && nobs(fit) isa Int
    @test dof(fit) == count(!iszero, b) && dof(fit) isa Int          # no intercept: the nonzero coefficients
    @test deviance(fit) == sum(abs2, residuals(fit)) && deviance(fit) isa T

    @test predict(fit, X) isa Vector{T} && predict(fit, X) == X * b
    @test predict(fit, X) + residuals(fit) ≈ y rtol = 8 * eps(T)

    # new data: a different number of rows, p columns
    Xnew = randn(Xoshiro(2), T, 7, 5)
    @test predict(fit, Xnew) == Xnew * b && length(predict(fit, Xnew)) == 7
    @test predict(fit, Xnew[1:1, :]) == Xnew[1:1, :] * b
    @test predict(fit, zeros(T, 0, 5)) == T[]
    @test predict(fit, view(Xnew, 2:4, :)) == Xnew[2:4, :] * b       # any AbstractMatrix
    @test predict(fit, Float64.(Xnew)) isa Vector{Float64}           # promoted, as in X * b
    @test predict(fit, Float64.(Xnew)) ≈ Float64.(Xnew) * Float64.(b)

    # a wrong number of columns
    @test_throws DimensionMismatch predict(fit, randn(Xoshiro(3), T, 7, 4))
    @test_throws DimensionMismatch predict(fit, randn(Xoshiro(3), T, 7, 6))
    @test_throws DimensionMismatch predict(fit, zeros(T, 3, 0))
    @test_throws MethodError predict(fit, randn(Xoshiro(3), T, 5))   # a vector is not a design matrix

    # the methods read the stored fields: a fit with no nonzero coefficient
    f0 = lasso(X, y, T(1e3), solver(T))
    @test iszero(coef(f0)) && residuals(f0) == y && predict(f0, Xnew) == zeros(T, 7)
  end

  @testset "show, $T" for T in (Float32, Float64)
    X, y = problem(T, 4, 20, 3)
    fit = lasso(X, y, T(0.1), solver(T))
    s = sprint(show, fit)
    nz = count(!iszero, coef(fit))
    @test startswith(s, "LassoSolution: 3 coefficients, $nz nonzero, λ = $(T(0.1)), 20 observations;")
    @test occursin("not an optimality certificate", s)
    @test occursin("met after $(fit.result.passes) passes", s) && !occursin("not met", s)
    @test s == string(fit) == repr(MIME("text/plain"), fit)
    @test count(==('\n'), s) == 1
    # a run stopped at maxpasses reports that the test was not met
    stopped = @test_logs (:warn,) lasso(X, y, T(0.01), CDSolver(order = :cyclic, momentum = false, tol = T === Float32 ? 1e-6 : 1e-10, maxpasses = 1))
    s = sprint(show, stopped)
    @test occursin("not met after 1 passes", s)
  end
end

end
