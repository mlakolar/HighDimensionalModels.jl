using HighDimensionalModels, Test
using HiGHS, Random, LinearAlgebra, SparseArrays
const QSMOI = HiGHS.MOI

qstop_support(path) = [count(!iszero,s.β) for s in path.solutions]

function qstop_same(a,b,method)
  for f in fieldnames(QuantileLassoSolution)
    x,y = getfield(a,f),getfield(b,f)
    if method == :smooth && x isa Union{AbstractArray,AbstractFloat}
      @test x ≈ y rtol=1e-10 atol=1e-12
    else
      @test isequal(x,y)
    end
  end
end

function qstop_mock(term,n,p;pair=true)
  opt=QSMOI.Utilities.MockOptimizer(QSMOI.Utilities.UniversalFallback(QSMOI.Utilities.Model{Float64}()))
  QSMOI.Utilities.set_mock_optimize!(opt,mock -> begin
    if pair
      QSMOI.Utilities.mock_optimize!(mock,term,zeros(n),QSMOI.FEASIBLE_POINT,
        (QSMOI.ScalarAffineFunction{Float64},QSMOI.Interval{Float64}) => fill(-1/n,p),
        (QSMOI.ScalarAffineFunction{Float64},QSMOI.EqualTo{Float64}) => [0.0])
      QSMOI.set(mock,QSMOI.SimplexIterations(),0)
    else
      QSMOI.Utilities.mock_optimize!(mock,term)
    end
  end)
  opt
end

@testset "quantile path support stops" begin
  rng=Xoshiro(158);X=2rand(rng,53,6).-1
  y=.8 .+ X*[2.,-1.,.6,0,0,0] .+ .2 .* (rand(rng,53).-0.5)
  ls=[.8,.3,.15,.08,.04,.02,.01,.002]
  for method in (:exact,:smooth)
    @testset "$method" begin
      full=QuantileLassoPath(X,y,ls;method,standardize_x=false,rng=Xoshiro(19))
      ns=qstop_support(full)
      @test full.complete && full.requested_λpath==ls && full.λpath==ls
      @test all(s.converged for s in full.solutions)
      @test all(dof(full[j])-full.hasintercept==ns[j] for j in eachindex(ls))
      unlimited=QuantileLassoPath(X,y,ls;method,max_hat_s=Inf,standardize_x=false,rng=Xoshiro(19))
      @test unlimited.complete && unlimited.status==full.status
      @test unlimited.λpath==ls && unlimited.requested_λpath==ls
      for j in eachindex(ls);qstop_same(unlimited[j],full[j],method);end

      mid=maximum(ns[1:end-1])÷2
      @test 0 < mid < maximum(ns[1:end-1])
      for s in (0,mid,mid+.5)
        k=findfirst(>(s),ns)
        @test k !== nothing
        @test k < length(ls)
        stopped=QuantileLassoPath(X,y,ls;method,max_hat_s=s,standardize_x=false,rng=Xoshiro(19))
        @test length(stopped)==k && stopped.λpath==full.λpath[1:k]
        @test stopped.requested_λpath==ls
        @test !stopped.complete && stopped.status==:max_hat_s
        @test all(<=(s),qstop_support(stopped)[1:end-1])
        @test count(!iszero,stopped[length(stopped)].β)>s
        for j in 1:k;qstop_same(stopped[j],full[j],method);end
      end

      # End at a new support maximum, after a positive equality level.
      records=[j for j in 2:length(ls) if ns[j]>maximum(ns[1:j-1])]
      @test !isempty(records)
      b=last(records);s=maximum(ns[1:b-1])
      @test s>0 && ns[b]>s && all(<=(s),ns[1:b-1])
      eq=findfirst(==(s),ns[1:b-1])
      @test eq !== nothing && eq < b
      boundary=QuantileLassoPath(X,y,ls[1:b];method,max_hat_s=s,standardize_x=false,rng=Xoshiro(19))
      @test boundary.complete && boundary.status==full.status
      @test boundary.λpath==ls[1:b] && boundary.requested_λpath==ls[1:b]
      @test length(boundary)>eq
      for j in 1:b;qstop_same(boundary[j],full[j],method);end

      @test ns[1]==0 && !iszero(full[1].intercept)
      k=findfirst(>(0),ns)
      @test k>1
      single=quantile_lasso(X,y,ls[k];method,max_hat_s=0,standardize_x=false,rng=Xoshiro(19))
      @test count(!iszero,single.β)>0 && single.λ==ls[k]
      @test single.status==full[k].status && single.converged
      free=QuantileLassoPath(X,y,ls;method,max_hat_s=0,penalty_factors=[0.,1,1,1,1,1],
        standardize_x=false,rng=Xoshiro(19))
      @test !iszero(free[1].β[1]) && iszero(free[1].penalty_factors[1])
      @test length(free)==1 && !free.complete && free.status==:max_hat_s
      @test free.requested_λpath==ls

      auto=QuantileLassoPath(X,y;nλ=8,λminratio=.01,method,max_hat_s=0,rng=Xoshiro(19))
      @test !auto.complete && auto.status==:max_hat_s
      @test 1 <= length(auto) < length(auto.requested_λpath)==8
      @test auto.λpath==auto.requested_λpath[1:length(auto)]
      @test all(iszero,qstop_support(auto)[1:end-1]) && count(!iszero,auto[length(auto)].β)>0

      if method == :exact
        sp=QuantileLassoPath(sparse(X),y,ls;max_hat_s=0,standardize_x=false)
        @test sp.λpath==ls[1:k] && sp.requested_λpath==ls
        @test !sp.complete && sp.status==:max_hat_s
        for j in 1:k
          @test coef(sp[j])==coef(full[j]) && dual_scores(sp[j])==dual_scores(full[j])
        end

        ref=Ref{HiGHS.Optimizer}()
        factory() = (ref[]=HiGHS.Optimizer())
        fits,status=HighDimensionalModels._quantile_exact_path(X,y,ls,ones(6),.5,true;
          tol=2e-5,time_limit=Inf,initial_basis=:quantile,simplex=:dual,start=time_ns(),
          max_hat_s=0,optimizer_factory=factory)
        @test length(fits)==k && status==:max_hat_s
        rows=QSMOI.get(ref[],QSMOI.ListOfConstraintIndices{QSMOI.ScalarAffineFunction{Float64},QSMOI.Interval{Float64}}())
        @test length(rows)==size(X,2)
        @test ls[k]!=ls[k+1]
        for row in rows
          bounds=QSMOI.get(ref[],QSMOI.ConstraintSet(),row)
          @test bounds.lower==-size(X,1)*ls[k] && bounds.upper==size(X,1)*ls[k]
        end
      end

      for s in (-1,NaN,-Inf,nothing,1+im)
        @test_throws ArgumentError QuantileLassoPath(X,y,ls;method,max_hat_s=s)
        @test_throws ArgumentError QuantileLassoPath(X,y;nλ=3,method,max_hat_s=s)
      end
    end
  end

  @testset "native termination precedes support" begin
    for term in (QSMOI.TIME_LIMIT,QSMOI.ITERATION_LIMIT)
      fits,status=HighDimensionalModels._quantile_exact_path(X,y,ls,ones(6),.5,true;
        tol=2e-5,time_limit=Inf,initial_basis=:default,simplex=:dual,start=time_ns(),
        max_hat_s=0,optimizer_factory=() -> qstop_mock(term,size(X,1),size(X,2)))
      @test length(fits)==1 && count(!iszero,only(fits).β)>0
      @test status==Symbol(string(term)) && only(fits).status==status
      @test !only(fits).native_converged
    end
    fits,status=HighDimensionalModels._quantile_exact_path(X,y,ls,ones(6),.5,true;
      tol=2e-5,time_limit=Inf,initial_basis=:default,simplex=:dual,start=time_ns(),
      max_hat_s=0,optimizer_factory=() -> qstop_mock(QSMOI.TIME_LIMIT,size(X,1),size(X,2);pair=false))
    @test isempty(fits) && status==:TIME_LIMIT
    timed=QuantileLassoPath(X,y,ls;max_hat_s=0,time_limit=1e-12)
    @test isempty(timed.solutions) && !timed.complete && timed.status==:TIME_LIMIT
  end
end

@testset "quantile CV support stops" begin
  # Opposite slopes cancel on the full data; small first-column values expose
  # the shared slope at lower penalties.
  rng=Xoshiro(159);A=2rand(rng,41,4).-1;A[1:8,1].*=.001
  X=vcat(A,A)
  y=vcat(2 .+ 4A[:,1] .+ .4A[:,2],2 .- 4A[:,1] .+ .4A[:,2])
  folds=repeat(1:2;inner=41);ls=[1.,.2,.1,.06,.04,.02,.005]
  full=QuantileLassoPath(X,y,ls;max_hat_s=0,standardize_x=false)
  @test !full.complete && full.status==:max_hat_s
  @test all(s.converged for s in full.solutions)
  cv=CVQuantileLassoPath(X,y,ls;folds,max_hat_s=0,standardize_x=false,threads=false,rng=Xoshiro(23))
  @test cv.λpath==full.λpath && cv.path.requested_λpath==ls
  @test cv.path.complete==full.complete && cv.path.status==full.status
  @test cv.nnz==qstop_support(full)
  for j in eachindex(full.λpath);qstop_same(cv.path[j],full[j],:exact);end
  stops=Int[]
  for f in 1:2
    tr,te=folds.!=f,folds.==f
    fp=QuantileLassoPath(X[tr,:],y[tr],cv.λpath;max_hat_s=0,standardize_x=false)
    push!(stops,length(fp))
    @test all(s.converged for s in fp.solutions)
    @test fp.status==:max_hat_s && !fp.complete
    @test all(iszero,qstop_support(fp)[1:end-1]) && count(!iszero,fp[length(fp)].β)>0
    @test fp.λpath==cv.λpath[1:length(fp)]
    @test isnan.(cv.fold_errors[f,:])==[j>length(fp) for j in eachindex(cv.λpath)]
    for j in eachindex(fp.λpath)
      r=y[te].-predict(fp[j],X[te,:])
      @test cv.fold_errors[f,j] ≈ sum(abs,r)/(2length(r))
    end
  end
  eligible=vec(all(isfinite,cv.fold_errors;dims=1))
  @test minimum(stops)>=1 && maximum(stops)<length(cv.λpath)
  @test eligible==[j<=minimum(stops) for j in eachindex(cv.λpath)]
  @test any(eligible) && any(j -> all(isnan,cv.fold_errors[:,j]),eachindex(cv.λpath))
  for j in eachindex(cv.λpath)
    if eligible[j]
      @test isfinite(cv.cvm[j]) && isfinite(cv.cvse[j])
    else
      @test cv.cvm[j]==Inf && isnan(cv.cvse[j])
    end
  end
  @test cv.λmin in cv.λpath[eligible] && cv.λ1se in cv.λpath[eligible]
  for s in (-1,NaN)
    @test_throws ArgumentError CVQuantileLassoPath(X,y,ls;folds,max_hat_s=s,threads=false)
  end
end
