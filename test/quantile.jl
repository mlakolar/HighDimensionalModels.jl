using HighDimensionalModels, Test
using HiGHS, Random, LinearAlgebra, SparseArrays
const QMOI = HiGHS.MOI

# Independent primal split formulation, with exactly QR.jl's row orientation.
function primal_qr_oracle(X,y,λ,τ,w;intercept=true)
  n,p=size(X);o=HiGHS.Optimizer();QMOI.set(o,QMOI.Silent(),true)
  QMOI.set(o,QMOI.RawOptimizerAttribute("threads"),1)
  QMOI.set(o,QMOI.RawOptimizerAttribute("parallel"),"off")
  bplus=QMOI.add_variables(o,p);bminus=QMOI.add_variables(o,p)
  rp=QMOI.add_variables(o,n);rn=QMOI.add_variables(o,n)
  a=intercept ? QMOI.add_variable(o) : nothing
  for v in [bplus;bminus;rp;rn];QMOI.add_constraint(o,v,QMOI.GreaterThan(0.0));end
  eq=[]
  for i in 1:n
    terms=[QMOI.ScalarAffineTerm(-X[i,j],bplus[j]) for j in 1:p]
    append!(terms,[QMOI.ScalarAffineTerm(X[i,j],bminus[j]) for j in 1:p])
    push!(terms,QMOI.ScalarAffineTerm(-1.0,rp[i]),QMOI.ScalarAffineTerm(1.0,rn[i]))
    intercept && push!(terms,QMOI.ScalarAffineTerm(-1.0,a))
    push!(eq,QMOI.add_constraint(o,QMOI.ScalarAffineFunction(terms,0.0),QMOI.EqualTo(-y[i])))
  end
  terms=[QMOI.ScalarAffineTerm(λ*w[j],v[j]) for v in [bplus,bminus] for j in 1:p]
  append!(terms,QMOI.ScalarAffineTerm.(fill(τ/n,n),rp))
  append!(terms,QMOI.ScalarAffineTerm.(fill((1-τ)/n,n),rn))
  QMOI.set(o,QMOI.ObjectiveSense(),QMOI.MIN_SENSE)
  QMOI.set(o,QMOI.ObjectiveFunction{QMOI.ScalarAffineFunction{Float64}}(),QMOI.ScalarAffineFunction(terms,0.0))
  QMOI.optimize!(o)
  @test QMOI.get(o,QMOI.TerminationStatus())==QMOI.OPTIMAL
  (;objective=QMOI.get(o,QMOI.ObjectiveValue()),xi=[QMOI.get(o,QMOI.ConstraintDual(),c) for c in eq])
end

@testset "Quantile primal–dual API" begin
  @test !isdefined(HighDimensionalModels, :qr_dual)
  rng=Xoshiro(286);X=randn(rng,61,5);y=.7 .+ X*[1.,.4,0,0,0] .+ randn(rng,61)
  for τ in (.2,.5,.8), int in (true,false), sp in (false,true)
    A=sp ? sparse(X) : X; w=[1.,.7,1.5,1.,.8]
    path=QuantileLassoPath(A,y,[.3,.1,.03];tau=τ,intercept=int,standardize_x=false,penalty_factors=w,tol=1e-6)
    @test path.complete && length(path)==3
    @test size(coef(path))==(5+int,3)
    @test size(dual_scores(path))==(61,3)
    @test predict(path,A) ≈ reduce(hcat,[predict(s,A) for s in path.solutions])
    for (k,s) in enumerate(path.solutions)
      oracle=primal_qr_oracle(X,y,s.λ,τ,w;intercept=int)
      @test s.converged && s.native_converged
      @test s.kkt_residual < 1e-6
      @test abs(s.objective-oracle.objective)<1e-7
      @test -dual_scores(s)/nobs(s) ≈ oracle.xi atol=1e-7
      @test s.duality_gap ≈ 0 atol=1e-7
      @test -1e-7 <= s.certificate_gap <= 1e-6
      @test residuals(s) ≈ y-predict(s,A)
      @test nobs(s)==61 && nobs(path)==61
      @test coef(path,s.λ)==coef(s)
      @test s.dual !== s.certificate_dual
      @test s.dual !== path[mod1(k+1,3)].dual
    end
    direct=quantile_lasso(A,y,.1;tau=τ,intercept=int,standardize_x=false,penalty_factors=w)
    @test direct.objective ≈ path[2].objective atol=1e-7
    @test_throws ArgumentError coef(path,.07)
  end
  # Interior-point representations of inactive slopes must not trigger a
  # discontinuous sign-based KKT failure; preserve their unrounded values.
  nullfit=quantile_lasso(X,y,10.;standardize_x=false)
  perturbed=copy(nullfit.β);perturbed[5]=1e-14
  nearzero=HighDimensionalModels._qsolution(X,y,perturbed,nullfit.intercept,dual_scores(nullfit),10.,ones(5),.5,true,:exact,0.,:OPTIMAL,0,true,1e-6)
  @test nearzero.converged
  @test nearzero.β[5]==1e-14
  @test coef(fit(QuantileLassoSolution,X,y,.1;standardize_x=false)) ≈ coef(quantile_lasso(X,y,.1;standardize_x=false))
  # Zero penalty factors, λ=0, fixed offsets and rescaling are meaningful for refits.
  for w in ([0.,1,1,1,1],ones(5)), λ in (.1,0.)
    s=quantile_lasso(X,y,λ;penalty_factors=w,standardize_x=false,offset=fill(.2,61))
    oracle=primal_qr_oracle(X,y.-.2,λ,.5,w)
    @test s.converged
    @test s.objective ≈ oracle.objective atol=1e-7
    @test residuals(s) ≈ y-predict(s,X;offset=fill(.2,61))
    if λ == 0 || any(iszero,w)
      @test all(iszero,s.certificate_dual)
    else
      @test s.certificate_gap < 1e-6
    end
  end
  scaled=quantile_lasso(X.*[1. 2 3 4 5],y,.1)
  original=quantile_lasso(X,y,.1)
  @test predict(scaled,X.*[1. 2 3 4 5]) ≈ predict(original,X) atol=1e-6
  auto=QuantileLassoPath(X,y;nλ=5,λminratio=.05)
  @test auto.complete && all(s.converged for s in auto.solutions)
  @test !applicable(dual_scores, auto, 1)
  @test norm(auto[1].β,Inf)<1e-6
  @test length(auto)==5
  # Ties and rank deficiency: test optimality, not equality of nonunique duals.
  tied=quantile_lasso(hcat(ones(20),ones(20)),repeat([0.,1.],10),.1;standardize_x=false)
  @test tied.converged && abs(sum(dual_scores(tied)))<1e-7
  tiny=QuantileLassoPath(X,y,[.1,.01];time_limit=1e-12)
  @test !tiny.complete && isempty(tiny.solutions) && tiny.status==:TIME_LIMIT
  @test size(coef(tiny))==(6,0) && size(dual_scores(tiny))==(61,0)
  @test size(predict(tiny,X)) == (61,0)
  @test_throws DimensionMismatch predict(tiny,zeros(4,2))
  @test_throws ErrorException quantile_lasso(X,y,.1;time_limit=1e-12)
  @test dof(nullfit) == 1
  @test dof(quantile_lasso(X,y,1e3;intercept=false)) == 0
  fitted_path=fit(QuantileLassoPath,X,y,[.3,.1];standardize_x=false)
  @test fitted_path.complete && all(s.converged for s in fitted_path.solutions)
  @test predict(fitted_path,X) ≈ reduce(hcat,[predict(s,X) for s in fitted_path.solutions])
  # A bounded native solve preserves an available inaccurate pair explicitly.
  limited_optimizer() = begin
    opt=HiGHS.Optimizer()
    QMOI.set(opt,QMOI.RawOptimizerAttribute("presolve"),"off")
    QMOI.set(opt,QMOI.RawOptimizerAttribute("simplex_iteration_limit"),0)
    opt
  end
  partial,status=HighDimensionalModels._quantile_exact_path(X,y,[.1],ones(5),.5,true;
    tol=1e-6,time_limit=Inf,initial_basis=:default,simplex=:dual,start=time_ns(),optimizer_factory=limited_optimizer)
  @test status==:ITERATION_LIMIT
  @test length(partial)==1 && !only(partial).converged
  # Solvers can retain a pair even at zero time. Use MOI's documented no-result
  # contract to test this separate case without a machine-dependent deadline.
  empty_optimizer() = begin
    opt=QMOI.Utilities.MockOptimizer(QMOI.Utilities.UniversalFallback(QMOI.Utilities.Model{Float64}()))
    QMOI.Utilities.set_mock_optimize!(opt,mock -> QMOI.Utilities.mock_optimize!(mock,QMOI.TIME_LIMIT))
    opt
  end
  unavailable,status=HighDimensionalModels._quantile_exact_path(X,y,[.1],ones(5),.5,true;
    tol=1e-6,time_limit=Inf,initial_basis=:default,simplex=:dual,start=time_ns(),optimizer_factory=empty_optimizer)
  @test isempty(unavailable) && status==:TIME_LIMIT
  # Smoothing exposes its own raw score, and never overwrites it with a certificate.
  for τ in (.2,.5,.8)
    p=QuantileLassoPath(X,y,[.3,.1];method=:smooth,tau=τ,bandwidth=.2,standardize_x=false,tol=3e-5)
    @test p.complete && all(s.converged for s in p.solutions)
    for s in p.solutions
      loss=HighDimensionalModels.ProximalBase.ConvolutionCheck(τ,.2)
      expected=[-last(HighDimensionalModels.ProximalBase.value_and_deriv(loss,predict(s,X)[i],y[i])) for i in eachindex(y)]
      @test dual_scores(s) ≈ expected
      @test s.objective >= s.check_objective-1e-12
      @test abs(s.duality_gap)<1e-4
      @test s.dual !== s.certificate_dual
    end
  end
  @testset "smoothed score accuracy during solving" begin
    loss=HighDimensionalModels.ProximalBase.ConvolutionCheck(.5,.2)
    if HighDimensionalModels.ProximalSolvers._hasconjugate(loss)
      for τ in (.2,.5,.8), screening in (false,true)
        current_loss=HighDimensionalModels.ProximalBase.ConvolutionCheck(τ,.2)
        p=QuantileLassoPath(X,y,[.3,.1,.03];method=:smooth,tau=τ,bandwidth=.2,
          standardize_x=false,tol=1e-6,screening,rng=Xoshiro(19))
        @test p.complete && all(s.converged for s in p.solutions)
        @test all(s.certificate_gap <= 1e-6*(1+abs(s.objective)) for s in p.solutions)
        for s in p.solutions
          expected=[-last(HighDimensionalModels.ProximalBase.value_and_deriv(current_loss,predict(s,X)[i],y[i])) for i in eachindex(y)]
          @test dual_scores(s) ≈ expected
        end
      end
      p=QuantileLassoPath(X,y,[.1,0.];method=:smooth,penalty_factors=[0.,1,1,1,1],
        standardize_x=false,tol=1e-6,rng=Xoshiro(19))
      @test all(s.converged for s in p.solutions)
      @test all(all(iszero,s.certificate_dual) for s in p.solutions)
      stopped=QuantileLassoPath(X,y,[.03];method=:smooth,tol=1e-10,newton_max_iter=1,
        options=CDOptions(max_iter=1),standardize_x=false,rng=Xoshiro(19))
      @test !stopped[1].converged
    else
      # Compatibility with solver releases predating convolution certificates.
      @test !HighDimensionalModels.ProximalSolvers._hasconjugate(loss)
    end
  end
  for kw in ((tau=0.,),(tau=1.,),(tol=NaN,),(time_limit=0.,),(method=:bad,),
             (penalty_factors=[1.,-1.,1,1,1],),(method=:smooth,bandwidth=0.),
             (method=:smooth,time_limit=1.),(initial_basis=:bad,),(simplex=:bad,))
    @test_throws ArgumentError QuantileLassoPath(X,y,[.1];kw...)
  end
  @test_throws ArgumentError QuantileLassoPath(X,y,[.1,.2])
  @test_throws ArgumentError QuantileLassoPath(X,y,Float64[])
  @test_throws ArgumentError QuantileLassoPath(X,y,[NaN])
  @test_throws DimensionMismatch quantile_lasso(X,y[1:20],.1)
  @test_throws DimensionMismatch quantile_lasso(X,y,.1;penalty_factors=ones(2))
  @test_throws ArgumentError QuantileLassoPath(X,y;penalty_factors=[0.,1,1,1,1])
  @test_throws ArgumentError QuantileLassoPath(X,y;nλ=1)
  @test_throws ArgumentError QuantileLassoPath(X,y;λminratio=0.)
end
