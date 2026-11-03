using HighDimensionalModels, Test, LinearAlgebra, Random, Statistics
using ProximalAlgorithms
import ProximalBase

@testset "PANOC loss arithmetic, cancellation and returned diagnostics" begin
  H = HighDimensionalModels
  ext = Base.get_extension(H, :HighDimensionalModelsProximalAlgorithmsExt)
  PA = ProximalAlgorithms
  rng = Xoshiro(1032601)
  for T in (Float32, Float64), (n,p,q) in ((35,7,4), (12,20,5), (25,10,8))
    X, Y, B = T.(randn(rng,n,p)), T.(randn(rng,n,q)), T.(randn(rng,p,q))
    d = H._lr_data(X,Y,false); g = H._lr_gradient(d); f = ext.RegressionLoss(d,g)
    v, G = PA.value_and_gradient(f,B)
    direct = sum(abs2,Y-X*B)/(2n)
    reference_gradient = H._lr_grad(d,g,B)
    @test v ≈ direct rtol=50eps(T)
    @test G ≈ reference_gradient rtol=50eps(T) atol=50eps(T)
    if T === Float32
      @test v == direct
      @test G == reference_gradient
    end
    # A subsequent evaluation cannot overwrite a gradient held by PANOC.
    saved = copy(G)
    PA.value_and_gradient(f,2B)
    @test G == saved
  end
  for T in (Float32, Float64), scale in (T(1), T(1e4))
    X = T.(randn(rng,30,6))
    B = scale .* T.(randn(rng,6,4))
    Y = X*B + T(10eps(T))*scale .* T.(randn(rng,30,4))
    d = H._lr_data(X,Y,false);g = H._lr_gradient(d);f = ext.RegressionLoss(d,g)
    value, _ = PA.value_and_gradient(f,B)
    # This fixture is below the cancellation guard: require the original
    # residual result, including zero and very small positive losses.
    @test value == sum(abs2,Y-X*B)/60
    @test value >= 0
  end
  for T in (Float32, Float64), (n,p,q) in ((30,6,4),(12,20,5))
    X,Y=T.(randn(rng,n,p)),T.(randn(rng,n,q))
    tol = T === Float32 ? 1e-5 : 1e-9
    for λ in (0.,.05,100.)
      s=nuclear_regression(X,Y,λ;solver=:panoc,tol,max_iter=20000)
      @test s.converged
      @test s.rank == rank(s.B)
      @test s.objective ≈ sum(abs2,s.residuals)/(2n)+T(λ)*sum(svdvals(s.B)) rtol=50eps(T)
      @test residuals(s) ≈ Y-predict(s,X) rtol=50eps(T) atol=50eps(T)
      k=isempty(s.activerows) ? 0 : rank((X .- mean(X;dims=1))[:,s.activerows])
      @test dof(s)==s.rank*(k+q-s.rank)+q
    end
    zero=nuclear_regression(zeros(T,n,p),Y,.1;solver=:panoc)
    @test zero.converged && iszero(zero.B) && zero.rank==0
    limit=nuclear_regression(X,Y,.05;solver=:panoc,max_iter=1,tol=1e-14)
    @test !limit.converged
    @test limit.rank==rank(limit.B)
    @test limit.objective ≈ sum(abs2,limit.residuals)/(2n)+T(.05)*sum(svdvals(limit.B)) rtol=50eps(T)
  end
  # The ProximalCore bridge returns the value at the prox point for both
  # aliased and separate input/output arrays.
  for T in (Float32,Float64)
    X=T[3 0;0 1];g=ext.NuclearPenalty(ProximalBase.NuclearNorm(T(.2)))
    Y=similar(X);value=PA.prox!(Y,g,X,T(.5))
    aliased=copy(X);avalue=PA.prox!(aliased,g,aliased,T(.5))
    @test Y≈T[2.9 0;0 .9]
    @test aliased==Y
    @test value≈T(.2)*sum(svdvals(Y))
    @test avalue≈value
  end
end
