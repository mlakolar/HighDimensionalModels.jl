module NamingTest
include("kernels.jl")
using .TestKernels
using HighDimensionalModels, ProximalBase, ProximalSolvers, Test, Random, LinearAlgebra

struct DefaultScaleLoss <: CoordinateDifferentiableFunction end

# Consumer functions keep the property name constant, as estimator code does.
_read_max_iter(o) = o.max_iter
_read_opt_tol(o) = o.opt_tol
_read_options_cd(o) = o.options_cd

@testset "Julia naming and integer inputs" begin
  @test HighDimensionalModels.coordinate_scale(DefaultScaleLoss(), 1) === 1
  @test HighDimensionalModels.response_scale(DefaultScaleLoss()) === 1
  o = CDOptions(; max_iter=Int32(100), num_steps=Int32(5), anderson=Int32(0))
  @test o.max_iter isa Int
  @test (@inferred _read_max_iter(o)) === 100
  @test (@inferred _read_opt_tol(o)) === o.opt_tol
  @test (@inferred _read_options_cd(IterLassoOptions())) isa CDOptions
  @test o.num_steps isa Int
  @test IterLassoOptions(; max_iter=Int32(2), sinit=Int32(3)).sinit isa Int
  @test CDOptions(o; opt_tol=1e-9).max_iter == o.max_iter
  @test IterLassoOptions(IterLassoOptions(); max_iter=3).max_iter == 3
  @test_throws ArgumentError CDOptions(; unknown_option=1)
  @test_throws ArgumentError CDOptions(; max_iter=-1)

  X = randn(Xoshiro(4), 30, 3)
  y = X * [1., -2., 0.]
  x = SparseIterate(3)
  r = coordinate_descent!(x, CDLeastSquaresLoss(y, X), NormL1(0.1); opt_tol=1e-10)
  @test r isa ProximalSolvers.CDResults
  @test r.minimizer === x
  @test r.converged
  @test r.minimum ≈ sum(abs2, y - X*x) / 60 + 0.1 * norm(x, 1)
  @test coef(sqrt_lasso(X, y, 0.1; standardize_x=false)) isa AbstractVector
  @test LassoPath(X, y, [0.2, 0.1]).kkt_checks isa Vector{Int}

  z = collect(range(0., 1.; length=30))
  fit = varying_coefficient(X, z, y, [0.3, 0.7]; kernel=GaussianKernel(), bandwidth=0.3, degree=Int32(1))
  @test size(coef(fit)) == (3, 2)
end
@testset "CI scenario: a failing required test" begin
  @test 1 == 2
end
end
