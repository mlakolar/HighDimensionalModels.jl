using HighDimensionalModels, Test
using HiGHS, Random, LinearAlgebra, SparseArrays
const QREFITMOI = HiGHS.MOI

# Independent primal LP, with QR.jl's equality orientation.
function qrefit_oracle(X,y,τ;intercept=true)
  n,p=size(X);o=HiGHS.Optimizer();QREFITMOI.set(o,QREFITMOI.Silent(),true)
  QREFITMOI.set(o,QREFITMOI.RawOptimizerAttribute("threads"),1)
  QREFITMOI.set(o,QREFITMOI.RawOptimizerAttribute("parallel"),"off")
  b=QREFITMOI.add_variables(o,p);a=intercept ? QREFITMOI.add_variable(o) : nothing
  rp=QREFITMOI.add_variables(o,n);rn=QREFITMOI.add_variables(o,n)
  for v in [rp;rn];QREFITMOI.add_constraint(o,v,QREFITMOI.GreaterThan(0.0));end
  eq=QREFITMOI.ConstraintIndex{QREFITMOI.ScalarAffineFunction{Float64},QREFITMOI.EqualTo{Float64}}[]
  for i in 1:n
    terms=QREFITMOI.ScalarAffineTerm{Float64}[QREFITMOI.ScalarAffineTerm(-X[i,j],b[j]) for j in 1:p]
    push!(terms,QREFITMOI.ScalarAffineTerm(-1.0,rp[i]),QREFITMOI.ScalarAffineTerm(1.0,rn[i]))
    intercept && push!(terms,QREFITMOI.ScalarAffineTerm(-1.0,a))
    push!(eq,QREFITMOI.add_constraint(o,QREFITMOI.ScalarAffineFunction(terms,0.0),QREFITMOI.EqualTo(-y[i])))
  end
  terms=[QREFITMOI.ScalarAffineTerm.(fill(τ/n,n),rp);QREFITMOI.ScalarAffineTerm.(fill((1-τ)/n,n),rn)]
  QREFITMOI.set(o,QREFITMOI.ObjectiveSense(),QREFITMOI.MIN_SENSE)
  QREFITMOI.set(o,QREFITMOI.ObjectiveFunction{QREFITMOI.ScalarAffineFunction{Float64}}(),QREFITMOI.ScalarAffineFunction(terms,0.0))
  QREFITMOI.optimize!(o)
  @test QREFITMOI.get(o,QREFITMOI.TerminationStatus())==QREFITMOI.OPTIMAL
  (;objective=QREFITMOI.get(o,QREFITMOI.ObjectiveValue()),
    β=Float64[QREFITMOI.get(o,QREFITMOI.VariablePrimal(),v) for v in b],
    intercept=intercept ? QREFITMOI.get(o,QREFITMOI.VariablePrimal(),a) : 0.0,
    xi=[QREFITMOI.get(o,QREFITMOI.ConstraintDual(),c) for c in eq])
end

function qrefit_vertex(X,s)
  Z=s.hasintercept ? hcat(ones(size(X,1)),X) : X
  rows=findall(r -> abs(r)<1e-8,residuals(s));d=size(Z,2)
  @test length(rows)==d
  @test rank(Z[rows,:];atol=1e-8)==d
  @test all(u -> s.tau-1+1e-7 < u < s.tau-1e-7,dual_scores(s)[rows])
end

function qrefit_mock(n,p,terms; scores=zeros(n), pairs=fill(true,length(terms)), after=(k -> nothing))
  opt=QREFITMOI.Utilities.MockOptimizer(QREFITMOI.Utilities.UniversalFallback(QREFITMOI.Utilities.Model{Float64}()))
  k=Ref(0)
  QREFITMOI.Utilities.set_mock_optimize!(opt,mock -> begin
    k[]+=1;j=k[]
    if pairs[j]
      QREFITMOI.Utilities.mock_optimize!(mock,terms[j],copy(scores),QREFITMOI.FEASIBLE_POINT,
        (QREFITMOI.ScalarAffineFunction{Float64},QREFITMOI.Interval{Float64}) => fill(-1/n,p),
        (QREFITMOI.ScalarAffineFunction{Float64},QREFITMOI.EqualTo{Float64}) => [0.0])
      QREFITMOI.set(mock,QREFITMOI.SimplexIterations(),0)
    else
      QREFITMOI.Utilities.mock_optimize!(mock,terms[j])
    end
    after(j)
  end)
  opt,k
end

@testset "quantile refits" begin
  rng=Xoshiro(155);n=53;p=6;X=2rand(rng,n,p).-1
  y=.7 .+ X*[1.4,-.8,0,1.1,0,0] .+ .4 .* (rand(rng,n).-0.5)
  S=[1,2];τ=.43;A=X[:,S]
  @testset "restricted primal and dual agreement" begin
    supplied=reverse(S);r=quantile_refit(X,y,supplied;tau=τ,tol=1e-7)
    @test r isa QuantileRefit && supplied==reverse(S) && r.support==S
    supplied[1]=p
    @test r.support==S
    cold=quantile_lasso(A,y,0.;tau=τ,tol=1e-7)
    oracle=qrefit_oracle(A,y,τ)
    qrefit_vertex(A,r.fit);qrefit_vertex(A,cold)
    @test r.fit.converged && r.fit.native_converged && r.fit.λ==0
    @test r.fit.objective ≈ cold.objective atol=1e-9
    @test r.fit.objective ≈ oracle.objective atol=1e-9
    @test r.fit.check_objective ≈ oracle.objective atol=1e-9
    @test r.fit.β ≈ cold.β atol=1e-8
    @test r.fit.β ≈ oracle.β atol=1e-8
    @test r.intercept ≈ cold.intercept atol=1e-8
    @test r.intercept ≈ oracle.intercept atol=1e-8
    @test -dual_scores(r)/nobs(r) ≈ oracle.xi atol=1e-9
    @test dual_scores(r)==dual_scores(r.fit)
    @test r.β[S]==r.fit.β && all(iszero,r.β[setdiff(1:p,S)])
    @test r.β !== r.fit.β && r.support !== S
    @test coef(r)==[r.intercept;r.β] && nobs(r)==n && dof(r)==length(S)+1
    @test residuals(r) ≈ y-predict(r,X) atol=1e-12
    @test maximum(abs,A'dual_scores(r)/n)<1e-8
    @test maximum(abs,(X'dual_scores(r)/n)[setdiff(1:p,S)])>.05
    @test r.fit.kkt_residual<1e-8 && all(iszero,r.fit.certificate_dual)
    @test r.fit.certificate_gap>.05
    @test_throws DimensionMismatch predict(r,X[:,1:3])
    @test_throws DimensionMismatch predict(r,X;offset=zeros(n-1))
  end

  @testset "quantile grid and offsets" begin
    taus=[.71,.23,.53,.23];off=.2 .* (rand(rng,n).-0.5);z=y.-off
    warm=quantile_refit(X,y,S;tau=taus,offset=off,tol=1e-7)
    @test warm isa Vector{QuantileRefit} && [r.fit.tau for r in warm]==taus
    count=Ref(0);opt=Ref{HiGHS.Optimizer}()
    factory()=(count[]+=1;opt[]=HiGHS.Optimizer())
    fits=HighDimensionalModels._quantile_exact_refit(A,z,taus,true;
      tol=1e-7,time_limit=Inf,initial_basis=:quantile,simplex=:dual,start=time_ns(),optimizer_factory=factory)
    @test count[]==1 && length(fits)==length(taus)
    bounds=QREFITMOI.get(opt[],QREFITMOI.ListOfConstraintIndices{QREFITMOI.VariableIndex,QREFITMOI.Interval{Float64}}())
    @test length(bounds)==n
    for c in bounds
      box=QREFITMOI.get(opt[],QREFITMOI.ConstraintSet(),c)
      @test box.lower==last(taus)-1 && box.upper==last(taus)
    end
    for (j,τ) in enumerate(taus)
      cold=quantile_refit(X,z,S;tau=τ,tol=1e-7)
      oracle=qrefit_oracle(A,z,τ)
      @test warm[j].fit.converged && cold.fit.converged && fits[j].converged
      @test warm[j].fit.objective ≈ cold.fit.objective atol=1e-9
      @test warm[j].fit.objective ≈ oracle.objective atol=1e-9
      @test fits[j].objective ≈ cold.fit.objective atol=1e-9
      @test maximum(abs,A'dual_scores(warm[j])/n)<1e-8
      @test dot(z,-dual_scores(warm[j])/n) ≈ -warm[j].fit.objective atol=1e-9
      qrefit_vertex(A,warm[j].fit);qrefit_vertex(A,cold.fit)
      @test coef(warm[j]) ≈ coef(cold) atol=1e-8
      @test dual_scores(warm[j]) ≈ dual_scores(cold) atol=1e-8
      @test predict(warm[j],X;offset=off) ≈ y-residuals(warm[j]) atol=1e-12
    end
    plain=quantile_refit(X,y,S;tau=τ)
    @test dot(y,-dual_scores(plain)/n) ≈ -plain.fit.objective atol=1e-9
    ts=[.2,.47,.8];vs=[r.fit.objective for r in quantile_refit(X,y,S;tau=ts)]
    @test vs[2] >= ((ts[3]-ts[2])*vs[1]+(ts[2]-ts[1])*vs[3])/(ts[3]-ts[1])-1e-9
    for grid in (range(.2,.8;length=3),sparsevec([.2,.8]))
      rs=quantile_refit(X,y,S;tau=grid)
      @test [r.fit.tau for r in rs]==grid && all(r.fit.converged for r in rs)
    end
    sparsey=quantile_refit(sparse(X),sparsevec(y),S;tau=τ)
    @test sparsey.fit.converged && sparsey.fit.objective≈plain.fit.objective
    saved=deepcopy(warm[2])
    for v in (warm[1].support,warm[1].β,warm[1].fit.β,warm[1].fit.residuals,
              warm[1].fit.dual,warm[1].fit.certificate_dual,warm[1].fit.penalty_factors)
      v[1]+=1
    end
    @test warm[2].support==saved.support && warm[2].β==saved.β
    for f in fieldnames(QuantileLassoSolution);@test isequal(getfield(warm[2].fit,f),getfield(saved.fit,f));end
  end

  @testset "supports and sparse data" begin
    for S in (Int[],collect(1:p)), int in (false,true)
      r=quantile_refit(X,y,S;intercept=int,tau=τ)
      sp=quantile_refit(sparse(X),y,S;intercept=int,tau=τ)
      oracle=qrefit_oracle(X[:,S],y,τ;intercept=int)
      @test r.fit.converged && sp.fit.converged
      @test r.fit.objective ≈ oracle.objective atol=1e-9
      @test sp.fit.objective ≈ r.fit.objective atol=1e-9
      @test dof(r)==length(S)+int && length(coef(r))==p+int
      @test coef(r)==(int ? [r.intercept;r.β] : r.β)
      @test r.fit.hasintercept==int && (int || iszero(r.intercept))
      @test residuals(r) ≈ y-predict(r,X) atol=1e-12
      @test isempty(S) ? all(iszero,r.β) : r.β==r.fit.β
    end
    emptydesign=quantile_refit(zeros(n,0),y,Int[];tau=τ)
    @test isempty(emptydesign.β) && dof(emptydesign)==1 && emptydesign.fit.converged
    @test_throws ArgumentError quantile_lasso(zeros(n,0),y,0.)
    Z=copy(X);Z[:,3].=0
    sol=quantile_lasso(Z,y,.05;penalty_factors=[1.,1,0,1,1,1],standardize_x=false,tau=τ)
    @test sol.converged && iszero(sol.β[3]) && iszero(sol.penalty_factors[3])
    r=quantile_refit(Z,y,sol)
    @test !(3 in r.support) && iszero(r.β[3])
    included=quantile_refit(Z,y,sort([r.support;3]);tau=τ)
    @test 3 in included.support && iszero(included.β[3])
    @test included.fit.converged && dof(included)==length(included.support)+1
    @test included.fit.objective ≈ r.fit.objective atol=1e-9
    # Degenerate scores and coefficients need not agree across optimal bases.
    for B in (hcat(X[:,1],X[:,1]),X[1:4,:])
      yy=size(B,1)==n ? y : y[1:4]
      r=quantile_refit(B,yy,collect(1:size(B,2));tau=τ)
      oracle=qrefit_oracle(B,yy,τ)
      @test r.fit.converged && r.fit.kkt_residual<1e-7
      @test r.fit.objective ≈ oracle.objective atol=1e-8
    end
    tied=quantile_refit(zeros(9,1),repeat([0.,1.,1.],3),[1])
    @test tied.fit.converged && tied.fit.objective≈1/6
    converted=quantile_refit(Int.(round.(2X)),Int.(round.(y)),S;tau=τ)
    @test converted.fit.converged && converted.β isa Vector{Float64}
  end

  @testset "solution and path refits" begin
    off=.2 .* (rand(rng,n).-0.5)
    path=QuantileLassoPath(X,y,[1.,.12,.01];tau=.37,bandwidth=.65,offset=off,standardize_x=false)
    @test path.complete && all(s.converged for s in path.solutions)
    supports=[findall(!iszero,s.β) for s in path.solutions]
    @test length(unique(supports))==3
    sol=path[2];single=quantile_refit(X,y,sol;offset=off)
    explicit=quantile_refit(X,y,supports[2];tau=sol.tau,offset=off)
    @test coef(single) ≈ coef(explicit) atol=1e-9
    @test single.fit.tau==sol.tau && single.fit.method==sol.method && single.fit.hasintercept==sol.hasintercept
    @test [r.fit.tau for r in quantile_refit(X,y,sol;tau=[.3,.7],offset=off)]==[.3,.7]
    sols=path.solutions[[1,3,1,2]]
    repeated=QuantileLassoPath([s.λ for s in sols],sols,[s.λ for s in sols],true,:OPTIMAL,n,p,true,:exact)
    @test findall(!iszero,repeated[1].β)==findall(!iszero,repeated[3].β)!=findall(!iszero,repeated[2].β)
    count=Ref(0)
    factory()=(count[]+=1;HiGHS.Optimizer())
    backend(args...;kw...)=HighDimensionalModels._quantile_exact_refit(args...;kw...,optimizer_factory=factory)
    refits=HighDimensionalModels._qrefit_path(repeated,X,y;offset=off,exact_refit=backend)
    publicrefits=refit_quantile_path(repeated,X,y;offset=off)
    @test refits isa Dict{Vector{Int},QuantileRefit}
    @test Set(keys(refits))==Set(supports) && count[]==length(supports)
    for S in supports
      direct=quantile_refit(X,y,S;tau=.37,offset=off)
      @test refits[S].fit.converged && refits[S].fit.tau==.37
      @test refits[S].fit.objective ≈ direct.fit.objective atol=1e-9
      @test publicrefits[S].fit.objective ≈ direct.fit.objective atol=1e-9
    end
    shifted=refit_quantile_path(path,X,y;tau=.63,offset=off)
    @test all(r.fit.tau==.63 && r.fit.converged for r in values(shifted))
    truncated=QuantileLassoPath(X,y,[1.,.12,.01];tau=.37,offset=off,max_hat_s=0,standardize_x=false)
    @test !truncated.complete && truncated.status==:max_hat_s && length(truncated)==2
    @test Set(keys(refit_quantile_path(truncated,X,y;offset=off)))==Set(supports[1:2])
    empty=QuantileLassoPath(Float64[],QuantileLassoSolution[],[1.],false,:TIME_LIMIT,n,p,true,:exact)
    @test refit_quantile_path(empty,X,y) isa Dict{Vector{Int},QuantileRefit}
    @test isempty(refit_quantile_path(empty,X,y))
    for source in (sol,path)
      refit=source isa QuantileLassoSolution ? ((XX,yy;kw...) -> quantile_refit(XX,yy,source;kw...)) :
        ((XX,yy;kw...) -> refit_quantile_path(source,XX,yy;kw...))
      @test_throws ArgumentError refit(X,y.+.01;offset=off)
      @test_throws ArgumentError refit(X,y;offset=off.+.01)
      @test_throws ArgumentError refit(X,y)
      @test_throws ArgumentError refit(X[:,1:end-1],y;offset=off)
      @test_throws ArgumentError refit(X[1:end-1,:],y[1:end-1];offset=off[1:end-1])
      @test_throws ArgumentError refit(X,y[1:end-1];offset=off)
      @test_throws ArgumentError refit(X,y;offset=off[1:end-1])
      @test_throws ArgumentError refit(2X,y;offset=off)
    end
    @test_throws ArgumentError refit_quantile_path(path,X,y;tau=[.3,.7],offset=off)
    @test_throws ArgumentError refit_quantile_path(path,X,y;tau=0.,offset=off)
    # A design change annihilating β passes the documented residual check.
    nullfit=path[1];@test all(iszero,nullfit.β)
    @test quantile_refit(2X,y,nullfit;offset=off).fit.converged
  end

  @testset "raw scores and interruption" begin
    injected=collect(range(-.8,.9;length=n))
    opt,calls=qrefit_mock(n,2,[QREFITMOI.ITERATION_LIMIT,QREFITMOI.OPTIMAL];scores=injected)
    backend(args...;kw...)=HighDimensionalModels._quantile_exact_refit(args...;kw...,optimizer_factory=() -> opt)
    rs=HighDimensionalModels._qrefit(X,y,S,[.4,.6],true;
      method=:exact,bandwidth=.2,tol=1e-7,time_limit=Inf,initial_basis=:default,simplex=:dual,start=time_ns(),
      options=CDOptions(),newton_tol=1e-10,newton_max_iter=100,screening=true,rng=Xoshiro(9),exact_refit=backend)
    fits=[r.fit for r in rs]
    @test calls[]==2 && length(fits)==2
    r=rs[1]
    @test dual_scores(r)==injected && !r.fit.converged && !r.fit.native_converged
    @test r.fit.status==:ITERATION_LIMIT && fits[2].status==:OPTIMAL && fits[2].native_converged
    @test r.fit.certificate_dual!=injected && all(iszero,r.fit.certificate_dual)
    @test maximum(abs,A'r.fit.certificate_dual)==0 && sum(r.fit.certificate_dual)==0
    @test all(u -> -.6 <= u <= .4,r.fit.certificate_dual)
    opt,calls=qrefit_mock(n,2,[QREFITMOI.OPTIMAL,QREFITMOI.TIME_LIMIT];pairs=[true,false])
    @test_throws r"no primal–dual pair.*TIME_LIMIT" HighDimensionalModels._quantile_exact_refit(A,y,[.4,.6],true;
      tol=1e-7,time_limit=Inf,initial_basis=:default,simplex=:dual,start=time_ns(),optimizer_factory=() -> opt)
    @test calls[]==2
    opt,calls=qrefit_mock(n,2,[QREFITMOI.OPTIMAL])
    @test_throws r"no primal–dual pair.*TIME_LIMIT" HighDimensionalModels._quantile_exact_refit(A,y,[.4],true;
      tol=1e-7,time_limit=1e-9,initial_basis=:default,simplex=:dual,start=time_ns()-UInt(1_000_000_000),optimizer_factory=() -> opt)
    @test calls[]==0
    @test_throws ErrorException quantile_refit(X,y,S;time_limit=1e-12)
    # The public path uses the same start time for every support.
    path=QuantileLassoPath(X,y,[1.,.1];standardize_x=false)
    seen=UInt[]
    backend2(args...;start,kw...)=(push!(seen,start);HighDimensionalModels._quantile_exact_refit(args...;start,kw...))
    rr=HighDimensionalModels._qrefit_path(path,X,y;exact_refit=backend2)
    @test length(rr)==2 && length(seen)==2 && all(==(first(seen)),seen)
    @test_throws ErrorException refit_quantile_path(path,X,y;time_limit=1e-12)
    # A native limit on one support does not skip any later support.
    made=Ref(0)
    function backend3(A,z,taus,int;kw...)
      factory()=(made[]+=1;first(qrefit_mock(length(z),size(A,2),[QREFITMOI.ITERATION_LIMIT])))
      HighDimensionalModels._quantile_exact_refit(A,z,taus,int;kw...,optimizer_factory=factory)
    end
    rr=HighDimensionalModels._qrefit_path(path,X,y;initial_basis=:default,exact_refit=backend3)
    @test made[]==2 && length(rr)==2
    @test all(r.fit.status==:ITERATION_LIMIT && !r.fit.native_converged for r in values(rr))
  end

  @testset "validation" begin
    for S in ([0],[p+1],[1,1],Bool[true,false],Bool[])
      @test_throws ArgumentError quantile_refit(X,y,S)
    end
    for t in (0.,1.,-1.,NaN,Inf,Float64[],[.3,1.],[NaN])
      @test_throws ArgumentError quantile_refit(X,y,S;tau=t)
    end
    @test_throws ArgumentError quantile_refit(X,y,trues(p))
    @test_throws DimensionMismatch quantile_refit(X,y[1:end-1],S)
    @test_throws DimensionMismatch quantile_refit(X,y,S;offset=zeros(n-1))
    @test_throws ArgumentError quantile_refit(zeros(0,p),Float64[],S)
    badX=copy(X);badX[1,p]=NaN
    @test_throws ArgumentError quantile_refit(badX,y,S)
    @test_throws ArgumentError quantile_refit(X,fill(Inf,n),S)
    @test_throws ArgumentError quantile_refit(X,y,S;offset=fill(NaN,n))
    for kw in ((method=:bad,),(tol=NaN,),(tol=0.,),(time_limit=0.,),(time_limit=NaN,),
               (initial_basis=:bad,),(simplex=:bad,),(method=:smooth,bandwidth=0.),
               (method=:smooth,time_limit=1.,))
      @test_throws ArgumentError quantile_refit(X,y,S;kw...)
    end
    for kw in ((penalty_factors=ones(p),),(standardize_x=false,),(max_hat_s=1,),(unknown=1,))
      @test_throws MethodError quantile_refit(X,y,S;kw...)
    end
    @test_throws ArgumentError try quantile_refit(X,y,trues(p));catch e;@test occursin("findall",sprint(showerror,e));rethrow();end
  end
end

qrefitsmooth_scores(r,τ,h) = [τ-.5+.75s-.25s^3 for s in clamp.(residuals(r)./h,-1,1)]

@testset "smoothed quantile refits" begin
  rng=Xoshiro(156);n=47;p=4;X=2rand(rng,n,p).-1
  y=.4 .+ X*[1.2,-.8,.6,0] .+ .3 .* (rand(rng,n).-0.5)
  S=[1,2];h=.6;τ=.37;off=.15 .* (rand(rng,n).-0.5);z=y.-off
  @testset "restricted fits and raw derivatives" begin
    for τ in (.27,.61), int in (true,false)
      r=quantile_refit(X,y,S;method=:smooth,tau=τ,bandwidth=h,intercept=int,
        offset=off,tol=1e-6,rng=Xoshiro(4))
      cold=quantile_lasso(X[:,S],z,0.;method=:smooth,tau=τ,bandwidth=h,
        intercept=int,tol=1e-6,rng=Xoshiro(4))
      @test r.fit.converged && cold.converged
      @test r.support==S && r.fit.method==:smooth && r.fit.bandwidth==h && r.fit.tau==τ
      Z=int ? hcat(ones(n),X[:,S]) : X[:,S]
      @test rank(Z[abs.(residuals(r)).<h,:];atol=1e-8)==size(Z,2)
      @test r.fit.objective ≈ cold.objective atol=1e-10
      @test r.fit.β ≈ cold.β atol=1e-8
      @test r.intercept ≈ cold.intercept atol=1e-8
      @test dual_scores(r) ≈ qrefitsmooth_scores(r,τ,h) atol=1e-12
      @test dual_scores(r) ≈ dual_scores(cold) atol=1e-9
      @test all(iszero,r.β[setdiff(1:p,S)]) && r.β[S]==r.fit.β
      @test maximum(abs,X[:,S]'dual_scores(r)/n)<1e-6
      @test maximum(abs,(X'dual_scores(r)/n)[setdiff(1:p,S)])>.02
      @test predict(r,X;offset=off) ≈ y-residuals(r) atol=1e-12
      @test dof(r)==length(S)+int && length(coef(r))==p+int && nobs(r)==n
      @test all(iszero,r.fit.certificate_dual) && r.fit.dual !== r.fit.certificate_dual
      oracle=qrefit_oracle(X[:,S],z,τ;intercept=int)
      @test r.fit.objective >= r.fit.check_objective-1e-12 >= oracle.objective-1e-9
      rr=z .- oracle.intercept .- X[:,S]*oracle.β
      @test oracle.objective ≈ sum(max.(τ.*rr,(τ-1).*rr))/n atol=1e-9
    end
    grid=[.61,.27,.61]
    rs=quantile_refit(X,y,S;method=:smooth,tau=grid,bandwidth=h,offset=off,tol=1e-6,rng=Xoshiro(8))
    @test [r.fit.tau for r in rs]==grid && all(r.fit.converged for r in rs)
    for (j,τ) in enumerate(grid)
      cold=quantile_refit(X,z,S;method=:smooth,tau=τ,bandwidth=h,tol=1e-6,rng=Xoshiro(8))
      @test rs[j].fit.objective ≈ cold.fit.objective atol=1e-10
      @test dual_scores(rs[j]) ≈ qrefitsmooth_scores(rs[j],τ,h) atol=1e-12
    end
    saved=copy(rs[3].fit.dual);rs[1].fit.dual[1]+=1
    @test rs[3].fit.dual==saved
  end

  @testset "supports, offsets and sparse data" begin
    for S in (Int[],collect(1:p)), int in (true,false)
      r=quantile_refit(X,y,S;method=:smooth,tau=τ,bandwidth=h,intercept=int,offset=off,tol=1e-6,rng=Xoshiro(3))
      sp=quantile_refit(sparse(X),z,S;method=:smooth,tau=τ,bandwidth=h,intercept=int,tol=1e-6,rng=Xoshiro(3))
      @test r.fit.converged && sp.fit.converged
      @test r.fit.objective ≈ sp.fit.objective atol=1e-9
      @test dual_scores(r) ≈ qrefitsmooth_scores(r,τ,h) atol=1e-12
      @test r.fit.hasintercept==int && (int || iszero(r.intercept))
      @test dof(r)==length(S)+int && length(coef(r))==p+int
      @test coef(r)==(int ? [r.intercept;r.β] : r.β)
      @test residuals(r) ≈ y-predict(r,X;offset=off) atol=1e-12
      @test isempty(S) ? all(iszero,r.β) : r.β==r.fit.β
    end
    r=quantile_refit(zeros(n,0),y,Int[];method=:smooth,tau=τ,bandwidth=h)
    @test isempty(r.β) && r.fit.converged && dof(r)==1
    @test_throws ArgumentError quantile_lasso(zeros(n,0),y,0.;method=:smooth)
    Z=copy(X);Z[:,4].=0
    sol=quantile_lasso(Z,y,.05;method=:smooth,tau=τ,bandwidth=h,
      penalty_factors=[1.,1,1,0],standardize_x=false,rng=Xoshiro(9))
    @test sol.converged && iszero(sol.β[4]) && iszero(sol.penalty_factors[4])
    r=quantile_refit(Z,y,sol;rng=Xoshiro(9))
    @test !(4 in r.support) && r.fit.converged
    included=quantile_refit(Z,y,sort([r.support;4]);method=:smooth,tau=τ,bandwidth=h,rng=Xoshiro(9))
    @test 4 in included.support && iszero(included.β[4])
    @test included.fit.converged && dof(included)==length(included.support)+1
    @test included.fit.objective ≈ r.fit.objective atol=1e-8
  end

  @testset "inheritance and solver controls" begin
    path=QuantileLassoPath(X,y,[.8,.09,.01];method=:smooth,tau=τ,bandwidth=h,
      intercept=false,offset=off,standardize_x=false,tol=1e-6,rng=Xoshiro(5))
    @test path.complete && all(s.converged for s in path.solutions)
    sol=path[2];S=findall(!iszero,sol.β)
    r=quantile_refit(X,y,sol;offset=off,tol=1e-6,rng=Xoshiro(7))
    direct=quantile_refit(X,z,S;method=:smooth,tau=τ,bandwidth=h,intercept=false,tol=1e-6,rng=Xoshiro(7))
    @test r.fit.tau==τ && r.fit.bandwidth==h && r.fit.method==:smooth && !r.fit.hasintercept
    @test coef(r) ≈ coef(direct) atol=1e-9
    @test r.fit.converged
    rs=refit_quantile_path(path,X,y;offset=off,tol=1e-6,rng=Xoshiro(7))
    @test Set(keys(rs))==Set(findall(!iszero,s.β) for s in path.solutions)
    for (S,r) in rs
      direct=quantile_refit(X,z,S;method=:smooth,tau=τ,bandwidth=h,intercept=false,tol=1e-6,rng=Xoshiro(7))
      @test r.fit.converged && r.fit.tau==τ && r.fit.bandwidth==h && !r.fit.hasintercept
      @test r.fit.objective ≈ direct.fit.objective atol=1e-9
      @test dual_scores(r) ≈ qrefitsmooth_scores(r,τ,h) atol=1e-12
    end
    shifted=refit_quantile_path(path,X,y;tau=.64,offset=off,tol=1e-6,rng=Xoshiro(7))
    @test all(r.fit.tau==.64 && r.fit.bandwidth==h && r.fit.converged for r in values(shifted))
    truncated=QuantileLassoPath(X,y,[.8,.09,.01];method=:smooth,tau=τ,bandwidth=h,
      intercept=false,offset=off,max_hat_s=0,standardize_x=false,rng=Xoshiro(5))
    @test !truncated.complete && truncated.status==:max_hat_s && length(truncated)==2
    @test length(refit_quantile_path(truncated,X,y;offset=off,rng=Xoshiro(7)))==2
    empty=QuantileLassoPath(Float64[],QuantileLassoSolution[],[1.],false,:max_hat_s,n,p,false,:smooth)
    @test isempty(refit_quantile_path(empty,X,y))
    for kw in ((time_limit=1.,),(bandwidth=0.,),(bandwidth=NaN,))
      @test_throws ArgumentError quantile_refit(X,y,[1];method=:smooth,kw...)
    end
    @test_throws ArgumentError quantile_refit(X,y,sol;offset=off,time_limit=1.)
    @test_throws ArgumentError refit_quantile_path(path,X,y;offset=off,time_limit=1.)
    @test_throws ArgumentError quantile_refit(X,y.+.01,sol;offset=off)
    @test_throws ArgumentError refit_quantile_path(path,X,y;offset=off.+.01)
    for kw in ((method=:exact,),(bandwidth=.4,),(intercept=true,),(penalty_factors=ones(p),),(max_hat_s=1,))
      @test_throws MethodError quantile_refit(X,y,sol;offset=off,kw...)
      @test_throws MethodError refit_quantile_path(path,X,y;offset=off,kw...)
    end
    for gt in (0.,1e-7)
      opts=CDOptions(opt_tol=1e-9,max_iter=10000,gaptol=gt)
      r=quantile_refit(X,y,[1,2];method=:smooth,tau=τ,bandwidth=h,
        tol=1e-6,options=opts,screening=false,newton_tol=1e-11,newton_max_iter=150,rng=Xoshiro(6))
      cold=quantile_lasso(X[:,1:2],y,0.;method=:smooth,tau=τ,bandwidth=h,
        tol=1e-6,options=opts,screening=false,newton_tol=1e-11,newton_max_iter=150,rng=Xoshiro(6))
      @test r.fit.objective ≈ cold.objective atol=1e-10
      @test dual_scores(r) ≈ qrefitsmooth_scores(r,τ,h) atol=1e-12
      @test r.fit.converged==cold.converged && r.fit.native_converged==cold.native_converged
    end
  end
end
