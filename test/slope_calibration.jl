using HighDimensionalModels, Test, Random, LinearAlgebra, SparseArrays, StatsAPI, Statistics
import ProximalBase

# Test-local FastProxSL1 (Bogdan et al., Algorithm 4), using block sums and lengths.
function _gaussian_calibration_sorted_l1_prox(u, λ)
  order = sortperm(abs.(u);rev=true)
  z = abs.(u[order]) .- λ
  sums, lengths = similar(z),Vector{Int}(undef,length(z))
  top = 0
  for value in z
    top += 1
    sums[top], lengths[top] = value,1
    while top>1 && sums[top]/lengths[top]>=sums[top-1]/lengths[top-1]
      sums[top-1] += sums[top]
      lengths[top-1] += lengths[top]
      top -= 1
    end
  end
  result = similar(z)
  first = 1
  for block in 1:top
    value = max(sums[block]/lengths[block],zero(eltype(z)))
    for j in first:first+lengths[block]-1
      i = order[j]
      result[i] = sign(u[i])*value
    end
    first += lengths[block]
  end
  result
end

@testset "Gaussian SLOPE test-local proximal map" begin
  rng = Xoshiro(10704)
  cases = [(randn(rng,n),sort!(rand(rng,n);rev=true)) for n in (1,7,16)]
  push!(cases, ([2.,-2.,0.,1.,-1.,2.],[1.8,1.6,1.4,1.2,.6,.2]),
               ([-3.,.25,2.,-1.5,0.,1.],[1.5,1.2,.9,.6,.3,.1]),
               ([.2,-.4,.1,0.],ones(4)))
  for (u,λ) in cases
    expected = similar(u)
    ProximalBase.prox!(expected,ProximalBase.SLOPE(λ),u,1.)
    @test _gaussian_calibration_sorted_l1_prox(u,λ) ≈ expected atol=1e-12 rtol=0
  end
  @test all(iszero,_gaussian_calibration_sorted_l1_prox(cases[end]...))
end

@testset "Gaussian SLOPE calibration" begin
  # Independent normal quantiles and the two-term published recurrence.
  b1, b2 = 1.959963984540054, 1.6448536269514722
  @test slope_gaussian(1;n=100)[1] ≈ 1.6448536269514722 / 10
  @test slope_gaussian(2;n=100) ≈ [b1,b2*sqrt(1+b1^2/98)] ./ 10
  @test slope_gaussian(2;n=100,intercept=true) ≈ [b1,b2*sqrt(1+b1^2/97)] ./ 10
  for n in [2,3,10,100], p in [1,2,10,200], q in [1e-12,0.1,1-1e-12], intercept in [false,true]
    n-Int(intercept)==1 && p>1 && continue
    a = slope_gaussian(p;n,q,intercept)
    @test length(a)==p && all(isfinite,a) && all(>=(0),a) && issorted(a;rev=true)
    @test slope_gaussian(p;n,q,intercept,sigma=3) ≈ 3a
    @test all(iszero,slope_gaussian(p;n,q,intercept,sigma=0))
    @test all(a .>= slope_bh(p;n,q) .- 10eps(Float64))
  end
  @test slope_gaussian(5;n=2) == fill(slope_bh(5;n=2)[1],5)
  for call in [()->slope_gaussian(0;n=100), ()->slope_gaussian(2;n=0),
               ()->slope_gaussian(2;n=1), ()->slope_gaussian(1;n=1,intercept=true),
               ()->slope_gaussian(2;n=100,q=0), ()->slope_gaussian(2;n=100,q=1),
               ()->slope_gaussian(2;n=100,q=NaN), ()->slope_gaussian(2;n=100,sigma=-1),
               ()->slope_gaussian(2;n=100,sigma=Inf)]
    @test_throws ArgumentError call()
  end

  rng = Xoshiro(10701)
  for T in [Float32,Float64], intercept in [false,true], standardize_x in [false,true]
    X = T.(randn(rng,70,12)); X[:,end] .= 2
    intercept && (X[:,1] .+= T(1000))
    y = T.(randn(rng,70)) .+ X[:,2] .- T(.5).*X[:,3]
    original_X, original_y = copy(X),copy(y)
    tol = T==Float32 ? 2e-4 : 1e-9
    weights = slope_gaussian(12;n=70,intercept)
    original_weights = copy(weights)
    a = slope(X,y;calibration=:gaussian,intercept,standardize_x,tol)
    b = slope(X,y,weights;intercept,standardize_x,tol)
    c = slope(sparse(X),y;calibration=:gaussian,intercept,standardize_x,tol)
    @test a.converged && b.converged && c.converged
    @test a.x == b.x && a.intercept == b.intercept
    @test a.lambda == T.(weights)
    @test c.x ≈ a.x rtol=10tol atol=10tol
    @test a.calibration.method==:gaussian && b.calibration.method==:user
    @test a.calibration.sigma==1 && a.calibration.noise==:supplied_sigma
    @test a.calibration.effective_n==70-Int(intercept)
    @test a.calibration.standardize_x==standardize_x
    @test X==original_X && y==original_y
    @test eltype(a.x)==T
    @test coef(a) == (intercept ? [a.intercept;a.x] : a.x)
    @test predict(a,X) ≈ predict(b,X)
    factors = [1.,.8,.6]
    original_factors = copy(factors)
    pa = SLOPEPath(X,y,factors;calibration=:gaussian,intercept,standardize_x,tol)
    pb = SLOPEPath(X,y,factors;lambda=weights,intercept,standardize_x,tol)
    @test all(pa.converged) && all(pb.converged)
    @test coef(pa)==coef(pb) && pa.lambda==pb.lambda
    @test pa.calibration==a.calibration && pb.calibration.method==:user
    @test eltype(pa.λpath)==T && eltype(pa.lambda)==T && eltype(pa.β0path)==T
    @test all(b -> eltype(b)==T, pa.βpath) && eltype(pa.kkt_residuals)==T
    @test eltype(coef(pa))==T && eltype(predict(pa,X))==T
    @test X==original_X && y==original_y
    @test weights==original_weights && factors==original_factors
  end
  X,y = randn(rng,50,5),randn(rng,50)
  bh = slope(X,y)
  @test bh.lambda==slope_bh(5;n=50) && bh.calibration.method==:bh
  supplied = fill(.2,5)
  a = slope(X,y;lambda=supplied,calibration=:gaussian,q=NaN,sigma=NaN)
  @test a.lambda==supplied && a.calibration.method==:user
  @test a.calibration.requested==:gaussian
  @test a.calibration.flat_from===nothing && a.calibration.flatten_reason==:none
  @test a.calibration.q===nothing && a.calibration.sigma===nothing
  p = SLOPEPath(X,y,[1.,.5];lambda=supplied,calibration=:gaussian,q=NaN,sigma=NaN)
  @test p.lambda==supplied && p.calibration.method==:user
  @test p.calibration==a.calibration
  @test_throws ArgumentError slope(X,y;calibration=:user)
  @test_throws ArgumentError SLOPEPath(X,y;calibration=:user)
  @test_throws ArgumentError slope(X,y;lambda=supplied,calibration=:unknown)
  @test_throws ArgumentError SLOPEPath(X,y;lambda=supplied,calibration=:unknown)
  @test_throws ArgumentError SLOPEPath(X,y;calibration=:gaussian,sigma=0)
  z = slope(X,y;calibration=:gaussian,sigma=0)
  @test all(iszero,z.lambda) && z.calibration.sigma==0
  # Legacy positional construction retains accessors and identifies missing metadata.
  for T in (Float32,Float64)
    args = (T.(a.x),T(a.intercept),a.hasintercept,T.(a.residuals),T.(a.lambda),
            a.iterations,a.converged,T(a.kkt_residual))
    for old in (SLOPESolution(args...), SLOPESolution{T}(args...))
      @test old isa SLOPESolution{T}
      @test all(getfield(old,i)==args[i] for i in eachindex(args))
      @test coef(old)==T.(coef(a)) && old.calibration==(method=:unspecified,)
    end
    pathargs = (T.(p.λpath),T.(p.lambda),[T.(b) for b in p.βpath],T.(p.β0path),
                p.intercept,p.nobs,p.converged,p.iterations,T.(p.kkt_residuals),p.working_sizes,p.kkt_repairs)
    for old in (SLOPEPath(pathargs...), SLOPEPath{T}(pathargs...))
      @test old isa SLOPEPath{T}
      @test all(getfield(old,i)==pathargs[i] for i in eachindex(pathargs))
      @test coef(old)==T.(coef(p)) && old.calibration==(method=:unspecified,)
    end
  end
end

@testset "Gaussian SLOPE truncation metadata" begin
  # With effective dimension two the boundary precedes any recurrence update.
  for (n,p,intercept,flat_from,reason) in ((100,2,false,nothing,:none),
      (100,2,true,nothing,:none), (10,10,false,2,:first_nondecrease),
      (10,10,true,2,:first_nondecrease), (2,5,false,2,:dimension_boundary),
      (3,5,true,2,:dimension_boundary))
    X,y = zeros(n,p),zeros(n)
    fit = slope(X,y;calibration=:gaussian,intercept)
    path = SLOPEPath(X,y,[1.,.5];calibration=:gaussian,intercept)
    for result in (fit,path)
      info = result.calibration
      @test info.method==:gaussian && info.requested==:gaussian
      @test info.flat_from===flat_from && info.flatten_reason==reason
      @test info.effective_n==n-Int(intercept)
      @test result.lambda==slope_gaussian(p;n,intercept)
      if flat_from !== nothing
        @test all(==(result.lambda[flat_from-1]), result.lambda[flat_from:end])
      end
    end
  end
end

@testset "Sparse Gaussian SLOPE paths and input preservation" begin
  rng = Xoshiro(10702)
  X = sprandn(rng,60,10,0.15)
  X[1,1] = X[2,2] = 1.
  X[1,1] = X[2,2] = 0.             # retained explicitly stored zeros
  D = Matrix(X)
  y = 1.5 .+ 3X[:,3] .- 2X[:,7] .+ .2randn(rng,60)
  saved_X, saved_D, saved_y = copy(X),copy(D),copy(y)
  @test X isa SparseMatrixCSC && nnz(X)<length(X)÷3
  @test count(iszero,nonzeros(X))>=2
  for intercept in (false,true), standardize_x in (false,true)
    weights = slope_gaussian(10;n=60,q=.2,sigma=.5,intercept)
    saved_weights = copy(weights)
    a = slope(X,y;calibration=:gaussian,q=.2,sigma=.5,intercept,standardize_x,tol=1e-9)
    b = slope(D,y;calibration=:gaussian,q=.2,sigma=.5,intercept,standardize_x,tol=1e-9)
    supplied = slope(X,y,weights;intercept,standardize_x,tol=1e-9)
    @test a.converged && b.converged && supplied.converged
    @test coef(a) ≈ coef(b) atol=2e-7
    @test coef(a)==coef(supplied)
    screened = SLOPEPath(X,y;calibration=:gaussian,q=.2,sigma=.5,intercept,standardize_x,
                         screening=true,nλ=4,λmin_ratio=.2,tol=1e-9)
    full = SLOPEPath(X,y;calibration=:gaussian,q=.2,sigma=.5,intercept,standardize_x,
                    screening=false,nλ=4,λmin_ratio=.2,tol=1e-9)
    factors = copy(screened.λpath)
    saved_factors = copy(factors)
    dense = SLOPEPath(D,y,factors;calibration=:gaussian,q=.2,sigma=.5,intercept,standardize_x,tol=1e-9)
    user = SLOPEPath(X,y,factors;lambda=weights,intercept,standardize_x,tol=1e-9)
    @test all(screened.converged) && all(full.converged) && all(dense.converged) && all(user.converged)
    @test length(screened.λpath)==4 && screened.λpath==full.λpath
    @test screened.λpath[end] ≈ .2screened.λpath[1]
    @test all(>(0),screened.λpath) && issorted(screened.λpath;rev=true)
    @test all(b -> norm(b,Inf)<=1e-7, (screened.βpath[1],full.βpath[1]))
    @test coef(screened) ≈ coef(full) atol=3e-7
    @test coef(screened) ≈ coef(dense) atol=3e-7
    @test coef(screened)==coef(user)
    @test screened.calibration==full.calibration==dense.calibration==a.calibration
    @test X==saved_X && D==saved_D && y==saved_y
    @test X.colptr==saved_X.colptr && X.rowval==saved_X.rowval && X.nzval==saved_X.nzval
    @test weights==saved_weights && factors==saved_factors
  end
  for T in (Float32,Float64)
    A,v = T.(X),T.(y)
    saved_A,saved_v = copy(A),copy(v)
    factors = T[1,.5]
    saved_factors = copy(factors)
    tol = T==Float32 ? 2e-4 : 1e-9
    path = SLOPEPath(A,v,factors;calibration=:gaussian,q=T(.2),sigma=zero(T),
                     intercept=true,standardize_x=true,screening=false,tol)
    @test all(path.converged) && all(iszero,path.lambda)
    @test path.βpath[1] ≈ path.βpath[2] atol=10tol
    @test eltype(path.λpath)==T && eltype(path.lambda)==T && eltype(path.β0path)==T
    @test all(b -> eltype(b)==T,path.βpath) && eltype(path.kkt_residuals)==T
    @test path.calibration.method==:gaussian && path.calibration.requested==:gaussian
    @test path.calibration.q===T(.2) && path.calibration.sigma===zero(T)
    @test path.calibration.effective_n==59 && path.calibration.standardize_x
    @test A==saved_A && v==saved_v && factors==saved_factors
    @test A.colptr==saved_A.colptr && A.rowval==saved_A.rowval && A.nzval==saved_A.nzval
  end
end

@testset "Gaussian SLOPE independent full proximal residual" begin
  rng = Xoshiro(10703)
  for sparse_x in (false,true), intercept in (false,true), standardize_x in (false,true)
    X = sparse_x ? sprandn(rng,40,7,.25) : randn(rng,40,7)
    y = 2 .+ 2X[:,2] .- X[:,5] .+ .2randn(rng,40)
    fit = slope(X,y;calibration=:gaussian,q=.2,sigma=.5,intercept,standardize_x,tol=1e-9)
    @test fit.converged
    # Rebuild the transformation independently from the input and returned point.
    D = Matrix(X)
    xb = intercept ? vec(mean(D;dims=1)) : zeros(7)
    yb = intercept ? mean(y) : 0.
    Z,v = D .- transpose(xb),y .- yb
    scales = standardize_x ? vec(sqrt.(sum(abs2,Z;dims=1)./length(y))) : ones(7)
    replace!(scales,0.0 => 1.0)
    Z ./= transpose(scales)
    point = fit.x .* scales
    grad = Z'*(Z*point-v)/length(v)
    trial = _gaussian_calibration_sorted_l1_prox(point-grad,fit.lambda)
    scale = max(1.,norm(Z'*v/length(v),Inf))
    @test norm(point-trial,Inf)/scale <= 1e-7
    r = y .- fit.intercept .- X*fit.x
    @test fit.residuals ≈ r atol=1e-12
    if intercept
      @test abs(sum(r)/length(r)) <= 1e-7
    else
      @test iszero(fit.intercept)
    end
  end
end

@testset "Gaussian SLOPE constant columns and large offsets" begin
  for T in (Float32,Float64), n in (3,7,10)
    X = fill(T(.1),n,1)
    y = T.(collect(1:n).^2)
    for A in (X,sparse(X))
      fit = slope(A,y;calibration=:gaussian,intercept=true,standardize_x=true,max_iter=100)
      path = SLOPEPath(A,y;calibration=:gaussian,intercept=true,standardize_x=true,nλ=3,max_iter=100)
      @test fit.converged && all(path.converged)
      @test all(iszero,fit.x) && all(b -> all(iszero,b),path.βpath)
      @test fit.intercept ≈ mean(y)
      @test path.β0path ≈ fill(mean(y),3)
    end
  end
  for offset in (1e8,1e12), standardize_x in (false,true)
    X = reshape(offset .+ Float64.(0:9),:,1)
    A,y = sparse(X),Float64.(0:9)
    saved_X,saved_A,saved_y = copy(X),copy(A),copy(y)
    a = slope(A,y;calibration=:gaussian,sigma=.1,intercept=true,standardize_x,max_iter=100,tol=1e-9)
    b = slope(X,y;calibration=:gaussian,sigma=.1,intercept=true,standardize_x,max_iter=100,tol=1e-9)
    @test a.converged && b.converged
    @test a.x ≈ b.x atol=1e-7
    # Evaluate on centered inputs to avoid cancellation in the large intercept.
    D,v = X .- mean(X),y .- mean(y)
    sc = standardize_x ? sqrt(sum(abs2,D)/length(y)) : 1.
    D ./= sc
    point = a.x .* sc
    grad = D'*(D*point-v)/length(v)
    trial = sign.(point-grad) .* max.(abs.(point-grad) .- a.lambda,0.)
    @test norm(point-trial,Inf)/max(1.,norm(D'*v/length(v),Inf)) <= 1e-7
    path = SLOPEPath(A,y,[1.,.5];calibration=:gaussian,sigma=.1,intercept=true,
                     standardize_x,max_iter=100,tol=1e-9)
    dense = SLOPEPath(X,y,[1.,.5];calibration=:gaussian,sigma=.1,intercept=true,
                      standardize_x,max_iter=100,tol=1e-9)
    @test all(path.converged) && all(dense.converged)
    @test reduce(hcat,path.βpath) ≈ reduce(hcat,dense.βpath) atol=1e-7
    @test X==saved_X && A==saved_A && y==saved_y
    @test A.colptr==saved_A.colptr && A.rowval==saved_A.rowval && A.nzval==saved_A.nzval
  end
end
