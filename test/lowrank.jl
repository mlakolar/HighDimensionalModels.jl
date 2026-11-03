using HighDimensionalModels, Test
using LinearAlgebra, Random, Statistics, SparseArrays

@testset "Low-rank multivariate regression" begin
  rng = Xoshiro(8101)
  X = randn(rng, 80, 8)
  B = randn(rng, 8, 2) * randn(rng, 2, 5)
  Y = X*B .+ [1 2 3 4 5] + 0.05randn(rng, 80, 5)
  Xc, Yc = X .- mean(X; dims=1), Y .- mean(Y; dims=1)
  LS = Xc \ Yc
  F = svd(Xc*LS)
  ref = LS * F.V[:, 1:2] * F.V[:, 1:2]'
  s = reduced_rank(X, Y, 2)
  @test s.B ≈ ref atol=1e-11
  @test s.rank == 2
  @test coef(s) ≈ [s.intercept'; s.B]
  @test predict(s, X) + residuals(s) ≈ Y
  @test nobs(s) == 80
  @test dof(s) == 2*(8+5-2)+5
  @test deviance(s) ≈ sum(abs2, Y-predict(s,X))
  @test all(isfinite, (aic(s), bic(s)))
  @test aic(s) ≈ length(Y)*log(deviance(s)/length(Y)) + 2dof(s)
  @test bic(s) ≈ length(Y)*log(deviance(s)/length(Y)) + log(length(Y))*dof(s)
  @test s.df_exact >= dof(s)
  @test lowrank_ic(s;df=:exact) >= lowrank_ic(s)
  @test s.λ == 0
  @test reduced_rank(X,Y,0).B == zeros(8,5)
  @test reduced_rank(X,Y,5).B ≈ LS atol=1e-10
  @test reduced_rank(sparse(X),Y,2).B ≈ s.B
  @test reduced_rank(hcat(X,X[:,1]),Y,2).B[1,:] ≈ s.B[1,:]/2
  wide = randn(rng, 12, 20); yw = randn(rng, 12, 3)
  @test predict(reduced_rank(wide,yw,3;intercept=false),wide) ≈ yw atol=1e-10
  @test_throws ArgumentError select_rank(wide,yw)
  @test select_rank(wide,yw;sigma2=1).rank <= 3
  @test select_rank(X,Y;μ=0).rank == 5
  @test_throws ArgumentError select_rank(X,Y;μ=Inf)
  @test select_rank(X,Y;μ=1e10).rank == 0
  @test reduced_rank(zeros(5,3),ones(5,2),0).rank == 0
  @test select_rank(zeros(5,3),ones(5,2)).rank == 0
  @test select_rank(X,Y;sigma2=0.05^2).rank == 2
  @test select_rank(X,Y).rank == 2

  # Orthogonal-design closed form for the coefficient nuclear penalty.
  Q = Matrix(qr(randn(rng, 30, 6)).Q)[:,1:6] * sqrt(30)
  Z = randn(rng, 30, 4)
  C = Q'Z/30
  S = svd(C)
  for λ in (0.0, 0.05, opnorm(C)*1.1)
    sol = nuclear_regression(Q,Z,λ;intercept=false,tol=1e-10)
    target = S.U*Diagonal(max.(S.S.-λ,0))*S.Vt
    @test sol.converged
    @test sol.B ≈ target atol=1e-9
  end
  λs = [0.6,0.2,0.05]
  path = LowRankPath(X,Y;λpath=λs,tol=1e-9)
  @test all(s->s.converged,path.solutions)
  for (λ, sol) in zip(λs,path.solutions)
    @test sol.B ≈ nuclear_regression(X,Y,λ;tol=1e-9).B atol=2e-7
    @test sol.objective ≈ sum(abs2,residuals(sol))/160 + λ*sum(svdvals(sol.B))
    # Independent nuclear-norm subgradient certificate on a general design.
    SF = svd(sol.B)
    rr = sol.rank
    U,V = SF.U[:,1:rr],SF.V[:,1:rr]
    G = Xc'*(Xc*sol.B-Yc)/80
    W = G + λ*U*V'
    @test norm(U'W) < 2e-7
    @test norm(W*V) < 2e-7
    @test opnorm(W) <= λ + 2e-7
  end
  @test length(predict(path,X)) == 3
  @test length(coef(path)) == 3
  @test LowRankPath(X,Y;nλ=1).solutions[1].rank == 0
  @test nuclear_regression(zeros(4,3),ones(4,2),1).rank == 0
  @test !nuclear_regression(X,Y,0.1;max_iter=1,tol=1e-14).converged

  # Row-sparse full-response rank reduces to the convex multi-response lasso.
  for λ in (0.01,0.3,100.0)
    sol = sparse_reduced_rank(Q,Z,4,λ;intercept=false,tol=1e-10)
    target = copy(C)
    for j in axes(C,1)
      target[j,:] .*= max(0,1-λ/norm(C[j,:]))
    end
    @test sol.converged
    @test sol.B ≈ target atol=1e-8
  end
  sol = sparse_reduced_rank(X,Y,2,0.2)
  @test sol.converged && sol.rank <= 2
  SF = svd(sol.B)
  V = SF.V[:,1:2]
  A = sol.B * V
  G = Xc'*(Xc*A-Yc*V)/80
  for j in axes(A,1)
    @test norm(A[j,:]) == 0 ? norm(G[j,:]) <= 0.2 + 2e-7 : norm(G[j,:] + 0.2A[j,:]/norm(A[j,:])) < 2e-7
  end
  @test sol.objective ≈ sum(abs2,residuals(sol))/160 + 0.2sum(norm(sol.B[j,:]) for j in 1:8)
  @test sparse_reduced_rank(X,Y,2,0).B ≈ s.B atol=1e-6
  @test sparse_reduced_rank(X,Y,0,1).rank == 0
  @test sparse_reduced_rank(X,Y,2,0.1;adaptive=true).converged
  sp = LowRankPath(X,Y;method=:sparse,rank=2,λpath=[100.0,0.2,0.0])
  @test sp.solutions[1].rank == 0
  @test sp.solutions[end].B ≈ s.B atol=1e-5
  @test sp.solutions[2].rank > 0
  rp = LowRankPath(X,Y;method=:rank,ranks=0:3)
  @test rp.ranks == 0:3
  @test all(diff(deviance.(rp.solutions)) .<= 1e-10)
  @test argmin(lowrank_ic(rp)) == 3

  folds = repeat(1:4,20)
  cv = CVLowRankPath(X,Y;method=:rank,ranks=0:3,folds)
  @test cv.index == 3
  @test coef(cv) == coef(cv.path.solutions[cv.index])
  @test predict(cv,X) == predict(cv.path.solutions[cv.index],X)
  errs = zeros(4,4)
  for k in 1:4
    tr,te = findall(!=(k),folds),findall(==(k),folds)
    fp = LowRankPath(X[tr,:],Y[tr,:];method=:rank,ranks=0:3)
    for j in 1:4
      errs[k,j] = mean(abs2,Y[te,:]-predict(fp.solutions[j],X[te,:]))
    end
  end
  @test cv.cvm ≈ vec(mean(errs;dims=1))
  cvn = CVLowRankPath(X,Y;λpath=[0.3,0.1],folds)
  @test all(isfinite,cvn.cvm)
  cvs = CVLowRankPath(X,Y;method=:sparse,rank=2,λpath=[0.2,0.1],folds,adaptive=true)
  @test all(isfinite,cvs.cvm)
  cvss = CVLowRankPath(X,Y;method=:sparse_svd,ranks=0:2,nλ=5,folds)
  @test all(isfinite,cvss.cvm)

  ss = sparse_svd_regression(X,Y,2;nλ=8)
  @test ss.solution.rank <= 2
  @test ss.solution.B ≈ sum(ss.components)
  @test all(x->length(x)==9,ss.scores)
  @test sparse_svd_regression(X,Y,0).solution.rank == 0
  for bad in (-1.0,Inf,NaN)
    @test_throws ArgumentError nuclear_regression(X,Y,bad)
    @test_throws ArgumentError sparse_reduced_rank(X,Y,2,bad)
  end
  @test_throws ArgumentError reduced_rank(X,Y,6)
  @test_throws DimensionMismatch reduced_rank(X,Y[1:3,:],1)
  @test_throws ArgumentError reduced_rank(fill(NaN,4,2),ones(4,2),1)
  @test_throws ArgumentError LowRankPath(X,Y;λpath=[0.1,0.2])
  @test_throws ArgumentError LowRankPath(X,Y;λpath=[])
  @test_throws ArgumentError sparse_reduced_rank(X,Y,1,1;weights=zeros(8))
  @test_throws ArgumentError lowrank_ic(s;criterion=:unknown)
  @test reduced_rank(Float32.(X),Float32.(Y),2).B isa Matrix{Float32}
  sn = nuclear_regression(Float32.(Q),Float32.(Z),0.05;intercept=false,tol=1e-5)
  @test sn.converged && sn.B isa Matrix{Float32}
  @test sn.B ≈ nuclear_regression(Q,Z,0.05;intercept=false).B atol=1e-5
end


@testset "fixed-rank residuals through thin factors" begin
  rng = Xoshiro(101026)
  for T in (Float32,Float64), (n,p,q) in ((60,10,80),(25,40,12),(60,10,8)), intercept in (false,true)
    X,Y = T.(randn(rng,n,p)),T.(randn(rng,n,q))
    for r in (0,1,3,min(n-intercept,p,q))
      fit = reduced_rank(X,Y,r;intercept)
      Xc = intercept ? X .- mean(X;dims=1) : X
      Yc = intercept ? Y .- mean(Y;dims=1) : Y
      E = Yc-Xc*fit.B
      # Both product orders describe the returned coefficients, up to the
      # rounding scale of the original matrix product, including wide fits.
      @test norm(residuals(fit)-E) <= 20eps(T)*max(norm(Yc),norm(Xc)*norm(fit.B))
      @test fit.objective ≈ sum(abs2,E)/(2n) rtol=100eps(T) atol=100eps(T)
      @test predict(fit,X)+residuals(fit) ≈ Y rtol=100eps(T) atol=100eps(T)
      @test fit.rank == rank(fit.B)
      @test fit.activerows == [j for j in 1:p if any(!iszero,fit.B[j,:])]
    end
  end
end
