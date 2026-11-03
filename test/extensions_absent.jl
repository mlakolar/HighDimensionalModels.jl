# Verifies, in a fresh process, the behaviour when the optional backends HiGHS
# and ProximalAlgorithms are not loaded.
using HighDimensionalModels, Random, Test
@testset "quantile API without HiGHS loaded" begin
  @test Base.get_extension(HighDimensionalModels,:HighDimensionalModelsHiGHSExt) === nothing
  X=reshape([1.,2.,3.],:,1);y=[1.,1.5,3.]
  @test isfinite(quantile_lambda(X;nsim=20,rng=Xoshiro(7)))
  err=try
    quantile_lasso(X,y,.1)
  catch e
    e
  end
  @test err isa ArgumentError
  @test occursin("using HiGHS",sprint(showerror,err))
  err=try
    quantile_refit(X,y,Int[])
  catch e
    e
  end
  @test err isa ArgumentError
  @test occursin("requires the optional HiGHS backend",sprint(showerror,err))
end

@testset "low-rank API without ProximalAlgorithms loaded" begin
  @test Base.get_extension(HighDimensionalModels,:HighDimensionalModelsProximalAlgorithmsExt) === nothing
  @test_throws ArgumentError nuclear_regression(ones(3, 2), ones(3, 2), 0.1; solver = :panoc)
end
