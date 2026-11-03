using HighDimensionalModels, Test
using LinearAlgebra, Random, Statistics
using ProximalAlgorithms
import ProximalBase

@testset "Low-rank solver choices and diagnostics" begin
  rng = Xoshiro(81082)
  X = randn(rng, 60, 8)
  Y = X * randn(rng, 8, 2) * randn(rng, 2, 5) + 0.1randn(rng, 60, 5)
  @test Base.get_extension(HighDimensionalModels, :HighDimensionalModelsProximalAlgorithmsExt) !== nothing
  for solver in (:apg, :panoc)
    Q = Matrix(qr(randn(rng, 30, 6)).Q)[:, 1:6] * sqrt(30)
    Z = randn(rng, 30, 4)
    F = svd(Q'Z/30)
    for λ in (0.0, 0.05, 2first(F.S))
      s = nuclear_regression(Q, Z, λ; solver, intercept = false, tol = 1e-9)
      target = F.U * Diagonal(max.(F.S .- λ, 0)) * F.Vt
      @test s.converged
      @test s.B ≈ target atol = 1e-8
      @test s.objective ≈ sum(abs2, Z - Q*s.B)/60 + λ*sum(svdvals(s.B))
    end
  end
  for A in (X, hcat(X, X[:, 1:2]), randn(rng, 60, 80))
    apg = nuclear_regression(A, Y, 0.2; tol = 1e-9)
    panoc = nuclear_regression(A, Y, 0.2; solver = :panoc, tol = 1e-9)
    @test apg.converged && panoc.converged
    @test panoc.B ≈ apg.B atol = 2e-6 rtol = 2e-6
    @test predict(panoc, A) + residuals(panoc) ≈ Y
  end
  @test nuclear_regression(zeros(4, 3), ones(4, 2), 1; solver = :panoc).converged
  @test !nuclear_regression(X, Y, 0.1; solver = :panoc, max_iter = 1, tol = 1e-14).converged
  ns = nuclear_regression(Float32.(X), Float32.(Y), 0.2; solver = :panoc, tol = 1e-5)
  @test ns.converged && ns.B isa Matrix{Float32}

  λs = [1000.0, 0.2, 0.05]
  path = LowRankPath(X, Y; solver = :panoc, λpath = λs, tol = 1e-9)
  @test path.solutions[1].rank == 0
  for (λ, s) in zip(λs, path.solutions)
    @test s.converged
    @test s.B ≈ nuclear_regression(X, Y, λ; solver = :panoc, tol = 1e-9).B atol = 2e-6
  end

  Q = Matrix(qr(randn(rng, 30, 6)).Q)[:, 1:6] * sqrt(30)
  Z = randn(rng, 30, 4)
  C = Q'Z/30
  w = [1.0, 2.0, 0.5, 0.7, 3.0, 1.1]
  for λ in (0.0, 0.1, 100.0)
    target = copy(C)
    for j in axes(C, 1)
      target[j, :] .*= max(0, 1 - λ*w[j]/norm(C[j, :]))
    end
    s = sparse_reduced_rank(Q, Z, 4, λ; block_solver = :fista,
      weights = w, intercept = false, tol = 1e-10)
    @test s.converged
    @test s.B ≈ target atol = 1e-8
  end
  sp = LowRankPath(X, Y; method = :sparse, rank = 2, block_solver = :fista,
    λpath = λs, adaptive = true, tol = 1e-8)
  @test all(s -> s.converged, sp.solutions)
  @test sp.solutions[1].rank == 0
  @test sp.solutions[end].rank > 0
  @test sparse_reduced_rank(zeros(5, 3), ones(5, 2), 0, 1; block_solver = :fista).rank == 0
  fs = sparse_reduced_rank(Float32.(Q), Float32.(Z), 4, 0.1;
    block_solver = :fista, tol = 1e-5, inner_tol = 1e-5, intercept = false)
  @test fs.converged && fs.B isa Matrix{Float32}
  folds = repeat(1:3, 20)
  @test all(isfinite, CVLowRankPath(X, Y; solver = :panoc, λpath = [0.3, 0.1], folds).cvm)
  @test all(isfinite, CVLowRankPath(X, Y; method = :sparse, rank = 2,
    block_solver = :fista, λpath = [0.3, 0.1], folds, adaptive = true).cvm)
  @test_logs (:warn, r"did not converge") @test !sparse_reduced_rank(X, Y, 2, 0.5; block_solver = :fista,
    max_iter = 1, inner_max_iter = 1, tol = 1e-14, inner_tol = 1e-14).converged

  for T in (Float32, Float64)
    for (A, Z) in ((X, Y), (hcat(X, X[:, 1:2]), Y), (randn(rng, 12, 20), randn(rng, 12, 4)),
                   (X, zeros(60, 5)), (Matrix{Float64}(I, 5, 5), Diagonal([2.0, 1.0, 0, 0, 0])))
      A, Z = T.(A), Matrix{T}(Z)
      path = LowRankPath(A, Z; method = :rank)
      for s in path.solutions
        @test s.rank == rank(s.B)
        k = isempty(s.activerows) ? 0 : rank((A .- mean(A; dims = 1))[:, s.activerows])
        @test dof(s) == s.rank*(k + size(Z, 2) - s.rank) + size(Z, 2)
        @test residuals(s) ≈ Z - predict(s, A) atol = 50eps(T)*max(one(T), norm(Z))
        @test s.objective ≈ sum(abs2, residuals(s))/(2size(A, 1))
      end
    end
  end
  A = Diagonal([1.0, 1e-8]) * [1.0 1.0; 1.0 -1.0]
  Z = [1.0 2.0 3.0; 0 0 0]
  s = reduced_rank(A, Z, 1; intercept = false, rtol = 0.1)
  @test s.rank == rank(s.B)
  @test dof(s) == s.rank*(rank(A[:, s.activerows]) + 3 - s.rank)
  selected = select_rank(X, Y; μ = 0.1)
  @test selected.solution.B ≈ reduced_rank(X, Y, selected.rank).B
  @test_throws ArgumentError nuclear_regression(X, Y, 1; solver = :unknown)
  @test_throws ArgumentError sparse_reduced_rank(X, Y, 2, 1; block_solver = :unknown)
  @test_throws ArgumentError LowRankPath(X, Y; solver = :unknown)
  @test_throws ArgumentError LowRankPath(X, Y; method = :rank, solver = :panoc)
  @test_throws ArgumentError LowRankPath(X, Y; method = :nuclear, block_solver = :fista)
end

@testset "Low-rank solver boundary behavior and StatsAPI entry points" begin
  # A single sweep on a coupled quadratic is not a converged block solve.
  H = [2.0 1.0; 1.0 2.0]
  C = ones(2, 1)
  A = zeros(2, 1)
  @test !HighDimensionalModels._lr_group!(A, H, C, 0.0, ones(2);
    max_iter = 1, tol = 1e-12)
  @test vec(A) ≈ [0.5, 0.25]
  @test norm(H*A - C) > 1e-12

  fill!(A, 0)
  Cf = reshape([1.0, 0.0], 2, 1)
  @test !HighDimensionalModels._lr_group_fista!(A, H, Cf, 0.0, ones(2);
    max_iter = 1, tol = 1e-12, L = 3.0)
  @test norm(H*A - Cf) > 1e-12

  # A zero quadratic has the zero minimum-norm solution, including with a
  # nonzero warm start. The final KKT check still rejects an inconsistent C.
  fill!(A, 2)
  @test HighDimensionalModels._lr_group_fista!(A, zeros(2, 2), zeros(2, 1),
    0.1, ones(2); max_iter = 1, tol = 1e-12, L = 0.0)
  @test iszero(A)
  @test !HighDimensionalModels._lr_group_fista!(A, zeros(2, 2), C,
    0.1, ones(2); max_iter = 1, tol = 1e-12, L = 0.0)

  u = zeros(2)
  @test !HighDimensionalModels._lr_lasso_gram!(u, H, ones(2), 0.0;
    max_iter = 1, tol = 1e-12)
  @test u ≈ [0.5, 0.25]

  # Both ways to remove a sparse component must return a valid zero fit.
  d = HighDimensionalModels._lr_data(Matrix{Float64}(I, 2, 2), zeros(2, 2), false)
  B, ok, _, df = HighDimensionalModels._lr_unit(d, 0.1, ones(2), ones(2),
    ones(2), [1.0, 0.0]; max_iter = 2, tol = 1e-12,
    inner_max_iter = 1, inner_tol = 1e-12)
  @test ok && iszero(B) && df == 0

  # Correlated predictors can leave nonzero u after one inexact sweep even
  # for Y=0. The exact response update must then remove the whole component.
  d = HighDimensionalModels._lr_data(ones(2, 2), zeros(2, 2), false)
  B, ok, _, df = HighDimensionalModels._lr_unit(d, 0.1, ones(2), ones(2),
    ones(2), [1.0, 0.0]; max_iter = 2, tol = 1e-12,
    inner_max_iter = 1, inner_tol = 1e-12)
  @test ok && iszero(B) && df == 0

  rng = Xoshiro(81083)
  X, Y = randn(rng, 24, 3), randn(rng, 24, 2)
  @test_throws ArgumentError LowRankPath(X, Y; weights = ones(3))
  @test_throws ArgumentError LowRankPath(X, Y; adaptive = true)
  p = fit(LowRankPath, X, Y; method = :rank, ranks = 0:2)
  @test p isa LowRankPath
  @test p.ranks == [0, 1, 2]
  @test p.solutions[end].B ≈ reduced_rank(X, Y, 2).B
  cv = fit(CVLowRankPath, X, Y; method = :rank, ranks = 0:2,
    folds = repeat(1:3, 8))
  @test cv isa CVLowRankPath
  @test predict(cv, X) ≈ predict(cv.path.solutions[cv.index], X)

  # The optional ProximalAlgorithms bridge must expose the penalty value as
  # well as its prox: lambda * sum of the coefficient singular values.
  ext = Base.get_extension(HighDimensionalModels, :HighDimensionalModelsProximalAlgorithmsExt)
  penalty = ext.NuclearPenalty(ProximalBase.NuclearNorm(0.2))
  @test penalty([3.0 0.0; 0.0 1.0]) ≈ 0.8
end


@testset "Exact rank degrees of freedom preserve response scale" begin
  # For X=I and singular values (2,1), rank-one Stein df is
  # 1*(2+2-1) + 2/(4-1) = 11/3, independent of response units.
  for T in (Float32, Float64)
    X = Matrix{T}(I, 2, 2)
    scales = T === Float32 ? T[1e-12, 1, 1e12] : T[1e-100, 1, 1e100]
    for scale in scales
      Y = Matrix(Diagonal(T[2, 1] .* scale))
      s = reduced_rank(X, Y, 1; intercept = false)
      @test isfinite(s.df_exact)
      @test s.df_exact ≈ T(11)/T(3)
      @test isfinite(lowrank_ic(s; df = :exact))
    end
    tied = reduced_rank(X, X, 1; intercept = false)
    @test isinf(tied.df_exact)
    @test isinf(lowrank_ic(tied; df = :exact))
  end
end

@testset "Sparse-path setup preserves the weighted orthogonal solution" begin
  rng = Xoshiro(102261026)
  for T in (Float32,Float64), solver in (:cd,:fista)
    X = T.(Matrix(qr(randn(rng,30,6)).Q)[:,1:6] * sqrt(30))
    Y = T.(randn(rng,30,4)); C = X'Y/30
    w = T[1,2,.5,.7,3,1.1]
    tol = T === Float32 ? 1e-5 : 1e-10
    path = LowRankPath(X,Y;method=:sparse,rank=4,block_solver=solver,
      λpath=T[100,.2,.05],weights=w,intercept=false,tol,inner_tol=tol)
    @test path.solutions[1].rank == 0
    for (λ,s) in zip(path.λpath,path.solutions)
      target=copy(C)
      for j in axes(target,1)
        row=view(target,j,:);nr=norm(row)
        row .*= max(zero(T),1-λ*w[j]/nr)
      end
      @test s.converged
      @test s.B ≈ target atol=100tol rtol=100tol
      @test s.objective ≈ sum(abs2,Y-X*target)/60 + λ*sum(w[j]*norm(view(target,j,:)) for j in axes(target,1)) atol=100tol rtol=100tol
      @test predict(s,X)+residuals(s) ≈ Y atol=100tol rtol=100tol
    end
  end
end

@testset "Factor-based sparse rank keeps dense numerical-rank semantics" begin
  rng=Xoshiro(102261027)
  for T in (Float32,Float64), (p,q) in ((8,6),(80,100),(120,40))
    Q=Matrix(qr(randn(rng,T,p,3)).Q)[:,1:3]
    V=Matrix(qr(randn(rng,T,q,3)).Q)[:,1:3]
    for multiplier in T[.5,2,32], scale in T[1e-8,1e8]
      A=Q*Diagonal(T[1,.3,min(p,q)*eps(T)*multiplier].*scale)
      B=A*V'
      @test HighDimensionalModels._lr_factor_rank(B,A,V')==rank(B)
    end
    A=zeros(T,p,3);B=A*V'
    @test HighDimensionalModels._lr_factor_rank(B,A,V')==rank(B)==0
  end
end

@testset "Sparse thin residuals retain the orthogonal rank-two fit" begin
  rng=Xoshiro(102261028)
  for T in (Float32,Float64), solver in (:cd,:fista)
    X=T.(Matrix(qr(randn(rng,30,10)).Q)[:,1:10]*sqrt(30))
    L=zeros(T,10,2);L[1:4,:].=randn(rng,T,4,2)
    Y=X*(L*randn(rng,T,2,80));C=X'Y/30;target=copy(C)
    for row in eachrow(target)
      nr=norm(row);row .*= nr==0 ? zero(T) : max(zero(T),1-T(.1)/nr)
    end
    tol=T===Float32 ? 1e-4 : 1e-10
    s=sparse_reduced_rank(X,Y,2,T(.1);intercept=false,block_solver=solver,tol,inner_tol=tol)
    @test s.converged
    @test s.B ≈ target atol=100tol rtol=100tol
    @test norm(residuals(s)-(Y-X*s.B)) <= 100eps(T)*max(norm(Y),norm(X)*norm(s.B))
    @test s.rank==rank(s.B)
    @test dof(s)==s.rank*(rank(X[:,s.activerows])+80-s.rank)
  end
end

@testset "Sparse setup handles large scales and subnormal factor ranks" begin
  H = HighDimensionalModels
  rng = Xoshiro(102261029)
  for T in (Float32, Float64)
    a = T === Float32 ? T(5e18) : T(5e153)
    X, Y = fill(a, 4, 4), ones(T, 4, 1)
    d = H._lr_data(X, Y, false)
    f = H._lr_svd(d)
    setup = H._lr_sparse_setup(d, f)
    @test isinf(abs2(first(f.s)))
    @test isfinite(setup.L)
    @test setup.L >= eigmax(Symmetric(setup.XX))
    tol = T === Float32 ? 1e-5 : 1e-10
    s = sparse_reduced_rank(X, Y, 1, a/5; intercept = false,
      block_solver = :fista, tol, inner_tol = tol)
    @test s.converged
    @test all(isfinite, s.B)

    scale = T === Float32 ? T(1e-40) : T(1e-320)
    L = scale .* randn(rng, T, 40, 1)
    V = Matrix(qr(randn(rng, T, 50, 1)).Q)[:, 1:1]
    B = L * V'
    @test rank(B) > 1
    @test H._lr_factor_rank(B, L, V') == rank(B)

    # A large design and small response also reach the subnormal guard
    # through the fixed-rank solution's actual SVD factors.
    design_scale = T === Float32 ? T(1e36) : T(1e300)
    response_scale = T === Float32 ? T(1e-4) : T(1e-20)
    X = design_scale .* Matrix{T}(I, 44, 40)
    Y = response_scale .* vcat(randn(rng, T, 40, 1) * V', zeros(T, 4, 50))
    d = H._lr_data(X, Y, false)
    f = H._lr_svd(d)
    s = H._lr_rrsolution(d, f, 1)
    @test s.rank == rank(s.B)
    @test rank(s.B) > 1
  end
end

@testset "Reused sparse setup agrees with independent block setup" begin
  H = HighDimensionalModels
  rng = Xoshiro(102261030)
  for T in (Float32, Float64), solver in (:cd, :fista)
    X = T.(Matrix(qr(randn(rng, 24, 5)).Q)[:, 1:5] * sqrt(24))
    Y = X * randn(rng, T, 5, 3) + T(0.1) .* randn(rng, T, 24, 3)
    w = T[1, 2, 0.5, 0.7, 3]
    originals = (copy(X), copy(Y), copy(w))
    d = H._lr_data(X, Y, true)
    f = H._lr_svd(d)
    setup = H._lr_sparse_setup(d, f)
    @test isfinite(setup.L) && setup.L > 0
    tol = T === Float32 ? 1e-6 : 1e-11
    opts = (; max_iter = 200, tol, inner_max_iter = 300, inner_tol = tol,
      block_solver = solver)
    for r in (1, 3)
      initial = H._lr_rrcoef(d, f, r)
      cached = H._lr_sparse(d, r, T(0.1), w, initial; opts..., setup)
      fresh = H._lr_sparse(d, r, T(0.1), w, initial; opts..., setup = nothing)
      @test cached.converged == fresh.converged == true
      @test cached.B ≈ fresh.B atol = 300eps(T)*max(one(T), norm(fresh.B)) rtol = 300eps(T)
      @test cached.rank == fresh.rank == rank(cached.B)
      @test cached.activerows == fresh.activerows
      @test dof(cached) == dof(fresh)
      @test cached.objective ≈ fresh.objective rtol = 300eps(T) atol = 300eps(T)
      @test residuals(cached) ≈ residuals(fresh) atol = 300eps(T)*max(one(T), norm(Y)) rtol = 300eps(T)
    end
    @test (X, Y, w) == originals
  end
end

@testset "Automatic sparse grids, zero paths and unchanged inputs" begin
  rng = Xoshiro(102261031)
  for T in (Float32, Float64), solver in (:cd, :fista)
    X = randn(rng, T, 24, 4) .+ T[1 2 3 4]
    Y = X * randn(rng, T, 4, 3) .+ T[2 4 6]
    w = T[0.5, 2, 1, 3]
    originals = (copy(X), copy(Y), copy(w))
    Xc, Yc = X .- mean(X; dims = 1), Y .- mean(Y; dims = 1)
    C = Xc' * Yc / size(X, 1)
    λmax = maximum(norm(view(C, j, :))/w[j] for j in axes(C, 1)) * (1 + sqrt(eps(T)))
    tol = T === Float32 ? 1e-5 : 1e-9
    for r in (0, 1, 3)
      path = LowRankPath(X, Y; method = :sparse, rank = r, nλ = 3,
        λminratio = 0.2, weights = w, block_solver = solver, tol, inner_tol = tol)
      @test first(path.λpath) ≈ λmax rtol = 5eps(T)
      @test path.λpath ≈ λmax .* T[1, sqrt(0.2), 0.2] rtol = 5eps(T)
      @test all(s -> s.converged, path.solutions)
      @test first(path.solutions).rank == 0
      if r == 0
        @test all(s -> iszero(s.B) && s.rank == 0 && dof(s) == 3, path.solutions)
      end
    end
    Z = zeros(T, size(Y))
    zero_path = LowRankPath(X, Z; method = :sparse, rank = 2, nλ = 3,
      weights = w, block_solver = solver, tol, inner_tol = tol)
    @test iszero(zero_path.λpath)
    @test all(s -> s.converged && iszero(s.B) && iszero(residuals(s)), zero_path.solutions)
    @test iszero(Z)
    sparse_reduced_rank(X, Y, 2, λmax/2; weights = w,
      block_solver = solver, tol, inner_tol = tol)
    @test (X, Y, w) == originals
  end
end

@testset "Sparse diagnostic rank and both residual product orders" begin
  rng = Xoshiro(102261032)
  for T in (Float32, Float64)
    U = randn(rng, T, 24, 4)
    U .-= mean(U; dims = 1)
    U = Matrix(qr(U).Q)[:, 1:4]
    V = Matrix(qr(randn(rng, T, 5, 4)).Q)[:, 1:4]
    X = U * Diagonal(T[1, 0.6, 0.3, 9eps(T)]) * V'
    Y = X * ones(T, 5, 1)
    d = HighDimensionalModels._lr_data(X, Y, true)
    f = HighDimensionalModels._lr_svd(d)
    @test f.diagnostic_xrank == rank(d.X) == 4
    @test f.xrank == 3
    for solver in (:cd, :fista)
      tol = T === Float32 ? 1e-5 : 1e-10
      s = sparse_reduced_rank(X, Y, 1, zero(T); block_solver = solver,
        tol, inner_tol = tol)
      @test s.converged
      @test length(s.activerows) == 5
      @test dof(s) == s.rank*(rank(d.X) + 1 - s.rank) + 1
    end

    # Rank one takes the thin product; full response rank takes the dense
    # product. The small noise makes the residual check sensitive to rounding.
    for (p, q, r) in ((6, 20, 1), (4, 3, 3)), solver in (:cd, :fista)
      X = T.(Matrix(qr(randn(rng, 24, p)).Q)[:, 1:p] * sqrt(24))
      X .+= one(T)
      B = randn(rng, T, p, r) * randn(rng, T, r, q)
      Y = X * B .+ T(2) .+ 10eps(T) .* randn(rng, T, 24, q)
      tol = T === Float32 ? 1e-6 : 1e-11
      s = sparse_reduced_rank(X, Y, r, zero(T); block_solver = solver,
        tol, inner_tol = tol)
      @test s.converged
      direct = Y - X*s.B .- s.intercept'
      @test norm(residuals(s) - direct) <= 100eps(T)*max(norm(Y), norm(X)*norm(s.B))
      @test norm(direct) <= 100eps(T)*max(norm(Y), norm(X)*norm(s.B))
    end
  end
end

@testset "Sparse FISTA CV agrees with deterministic training-fold refits" begin
  rng = Xoshiro(102261033)
  for T in (Float32, Float64)
    X = randn(rng, T, 24, 4) .+ T[1 2 3 4]
    Y = X * randn(rng, T, 4, 2) .+ T[2 4]
    w = T[0.5, 2, 1, 3]
    originals = (copy(X), copy(Y), copy(w))
    folds = repeat(1:3, 8)
    tol = T === Float32 ? 1e-5 : 1e-9
    opts = (; method = :sparse, rank = 2, block_solver = :fista,
      λpath = T[0.3, 0.1], weights = w, tol, inner_tol = tol)
    cv = CVLowRankPath(X, Y; opts..., folds)
    @test cv.folds == folds
    errors = zeros(T, 3, 2)
    for k in 1:3
      tr, te = findall(!=(k), folds), findall(==(k), folds)
      path = LowRankPath(X[tr, :], Y[tr, :]; opts...)
      @test all(s -> s.converged, path.solutions)
      for (j, s) in enumerate(path.solutions)
        errors[k, j] = sum(abs2, Y[te, :] - predict(s, X[te, :])) / (length(te)*size(Y, 2))
      end
    end
    # CV exposes the fold-size-weighted mean and its standard error.
    expected = vec(mean(errors; dims = 1))
    @test cv.cvm ≈ expected rtol = 20eps(T) atol = 20eps(T)
    @test cv.cvse ≈ vec(std(errors; dims = 1)) / sqrt(T(3)) rtol = 20eps(T) atol = 20eps(T)
    @test (X, Y, w) == originals

    # Identical halves have equal fold errors, so the reported CV mean is
    # also the error of either fold, independently refitted on its training half.
    A, Z = repeat(X[1:12, :], 2, 1), repeat(Y[1:12, :], 2, 1)
    saved = (copy(A), copy(Z), copy(w))
    paired = CVLowRankPath(A, Z; opts..., folds = repeat(1:2; inner = 12))
    training = LowRankPath(A[13:24, :], Z[13:24, :]; opts...)
    fold_error = [sum(abs2, Z[1:12, :] - predict(s, A[1:12, :])) / 24 for s in training.solutions]
    @test paired.cvm ≈ fold_error rtol = 20eps(T) atol = 20eps(T)
    @test all(s -> s.converged, training.solutions)
    @test (A, Z, w) == saved
  end
end

