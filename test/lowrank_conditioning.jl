using HighDimensionalModels, Test
using LinearAlgebra, Random, Statistics

function lr_conditioning_design(T = Float64)
  rng = Xoshiro(102_005)
  U = Matrix(qr(rand(rng, T, 60, 10) .- T(0.5)).Q)[:, 1:10]
  V = Matrix(qr(rand(rng, T, 10, 10) .- T(0.5)).Q)
  X = U * Diagonal(T.(exp10.(range(0, -4; length = 10)))) * V' * sqrt(T(60))
  B0 = zeros(T, 10, 6)
  B0[1:3, :] .= (rand(rng, T, 3, 2) .- T(0.5)) * (rand(rng, T, 2, 6) .- T(0.5))
  Y = X * B0 .+ T(0.5) .+ T(0.3) .* (rand(rng, T, 60, 6) .- T(0.5))
  C = X' * Y / 60
  λ = T(0.1) * maximum(norm(view(C, j, :)) for j in axes(C, 1))
  X, Y, λ
end

function lr_conditioning_issue(T)
  rng = Xoshiro(102_005)
  U = Matrix(qr(randn(rng, T, 60, 10)).Q)[:, 1:10]; V = Matrix(qr(randn(rng, T, 10, 10)).Q)
  X = U * Diagonal(T.(exp10.(range(0, -4; length = 10)))) * V' * sqrt(T(60))
  B0 = zeros(T, 10, 6); B0[1:3, :] .= randn(rng, T, 3, 2) * randn(rng, T, 2, 6)
  Y = X * B0 .+ T(0.5) .+ T(0.3) .* randn(rng, T, 60, 6)
  C = X' * Y / 60; λ = T(0.1) * maximum(norm(view(C, j, :)) for j in axes(C, 1))
  X, Y, λ
end

function lr_conditioning_weights(X, Y, r, intercept; weights = nothing, adaptive = false)
  d = HighDimensionalModels._lr_data(X, Y, intercept)
  f = HighDimensionalModels._lr_svd(d)
  initial = HighDimensionalModels._lr_rrcoef(d, f, r)
  Float64.(HighDimensionalModels._lr_weights(d, initial, weights, adaptive, 2))
end

function lr_conditioning_stationarity(X, Y, s, r, w)
  X, Y, B = Float64.(X), Float64.(Y), Float64.(s.B)
  if s.hasintercept
    X .-= mean(X; dims = 1)
    Y .-= mean(Y; dims = 1)
  end
  XX, XY = X' * X / size(X, 1), X' * Y / size(X, 1)
  V = svd(B).V[:, 1:r]
  A = B * V
  F = svd(XY' * A; full = false)
  V = F.U * F.Vt
  A = B * V
  C = XY * V
  G = XX * A - C
  err = 0.0
  for j in axes(A, 1)
    a, g = view(A, j, :), view(G, j, :)
    na, penalty = norm(a), Float64(s.λ)*w[j]
    err = max(err, iszero(na) ? max(0, norm(g) - penalty) : norm(g + (penalty/na)*a))
  end
  err / max(1, norm(C))
end

lr_conditioning_objective(A, XX, C, λ, w) = dot(A, XX*A)/2 - dot(A, C) +
  λ*sum(w[j]*norm(view(A, j, :)) for j in axes(A, 1))

@testset "Sparse block Newton direction" begin
  for T in (Float64, Float32), λ in (T(0.3), zero(T))
    rng = Xoshiro(1491)
    M = rand(rng, T, 12, 5) .- T(0.5)
    XX = M' * M + I
    A = rand(rng, T, 5, 2) .- T(0.5)
    A[2, :] .= 0
    C = rand(rng, T, 5, 2) .- T(0.5)
    w = rand(rng, T, 5) .+ T(0.5)
    S = [1, 3, 4, 5]
    a, H = A[S, :], XX[S, S]
    nr = [norm(view(a, j, :)) for j in axes(a, 1)]
    U, c = a ./ nr, λ .* w[S] ./ nr
    F = (XX * A - C)[S, :] + c .* a
    Δ = HighDimensionalModels._lr_group_newton_direction(H, F, c, U)
    @test Δ !== nothing
    # Stack rows so the dense reference matches the block Hessian.
    dense = kron(H, Matrix{T}(I, 2, 2))
    for j in axes(a, 1)
      ix = 2j-1:2j
      u = U[j, :]
      dense[ix, ix] += c[j] * (I - u*u')
    end
    reference = -(dense \ vec(permutedims(F)))
    @test norm(vec(permutedims(Δ)) - reference) <= 50eps(T)*cond(dense)*norm(Δ)
    saved = copy(A[2, :])
    @test HighDimensionalModels._lr_group_newton!(A, XX, C, λ, w; max_iter = 1) == 1
    @test isequal(A[2, :], saved)
  end
end

@testset "Sparse block Newton guards" begin
  for T in (Float64, Float32)
    XX = Matrix(Diagonal(T[eps(T), 1]))
    A, C, w = ones(T, 2, 1), zeros(T, 2, 1), ones(T, 2)
    saved = copy(A)
    @test HighDimensionalModels._lr_group_newton!(A, XX, C, zero(T), w) == 0
    @test isequal(A, saved)

    XX = ones(T, 2, 2)
    A = T[1 0; 0 1]
    C = T[2 1; 1 2]
    e0 = HighDimensionalModels._lr_group_kkt(A, XX*A - C, T(0.5), w)
    @test HighDimensionalModels._lr_group_newton!(A, XX, C, T(0.5), w) > 0
    @test HighDimensionalModels._lr_group_kkt(A, XX*A - C, T(0.5), w) < e0

    A .= 0
    @test HighDimensionalModels._lr_group_newton!(A, XX, C, one(T), w) == 0
    @test iszero(A)
  end
  XX = Matrix(Diagonal(Float32[1e-6, 1]))
  A, C, w = ones(Float32, 2, 1), zeros(Float32, 2, 1), ones(Float32, 2)
  q0 = lr_conditioning_objective(A, XX, C, 1f0, w)
  e0 = HighDimensionalModels._lr_group_kkt(A, XX*A - C, 1f0, w)
  HighDimensionalModels._lr_group_newton!(A, XX, C, 1f0, w)
  @test lr_conditioning_objective(A, XX, C, 1f0, w) <= q0
  @test HighDimensionalModels._lr_group_kkt(A, XX*A - C, 1f0, w) <= e0
end

@testset "Sparse block budgets and Newton mode" begin
  for solver in (:cd, :fista)
    XX = [1.0 1-1e-6; 1-1e-6 1.0]
    target = reshape([1.0, 2.0], 2, 1)
    C, w, L = XX * target, ones(2), opnorm(XX)
    start = target + reshape([0.1, -0.1], 2, 1)
    first = copy(start)
    _, needed = HighDimensionalModels._lr_block!(first, XX, C, 0.0, w, false;
      block_solver = solver, max_iter = 1, tol = 1e-12, L)
    @test needed
    plain, polished = copy(start), copy(start)
    k1 = solver == :cd ? 100 : 1000
    if solver == :cd
      @test HighDimensionalModels._lr_group!(plain, XX, C, 0.0, w; max_iter = k1, tol = 1e-5)
    else
      @test HighDimensionalModels._lr_group_fista!(plain, XX, C, 0.0, w; max_iter = k1, tol = 1e-5, L)
    end
    unchanged = copy(start)
    @test HighDimensionalModels._lr_block!(unchanged, XX, C, 0.0, w, false;
      block_solver = solver, max_iter = k1, tol = 1e-5, L) == (true, false)
    @test isequal(unchanged, plain)
    @test HighDimensionalModels._lr_block!(polished, XX, C, 0.0, w, needed;
      block_solver = solver, max_iter = k1, tol = 1e-5, L) == (true, false)
    @test norm(plain - target) > 0.01
    @test norm(polished - target) < 1e-4*norm(plain - target)

    XX = [1.0 0.5 0.5; 0.5 1.0 0.9; 0.5 0.9 1.0]
    C, w, L = fill(2.0, 3, 1), ones(3), opnorm(XX)
    start = reshape([0.0, 2.0, 2.0], 3, 1)
    for budget in (1, 100, 101, 1000, 1001)
      A = copy(start)
      certified, _ = HighDimensionalModels._lr_block!(A, XX, C, 0.1, w, false;
        block_solver = solver, max_iter = budget, tol = 1e-12, L)
      err = HighDimensionalModels._lr_group_kkt(A, XX*A - C, 0.1, w)
      @test certified == (err <= 1e-12*max(1, norm(C)))
      @test all(isfinite, A)
      if budget == 1
        @test !certified
        @test iszero(A[1, :])
        @test abs((XX*A - C)[1]) > 0.1
      else
        @test certified
      end
    end

    # The active solve can expose a zero row that needs another proximal step.
    XX = [1.0 0.5001 0.4999; 0.5001 1.0 1-1e-6; 0.4999 1-1e-6 1.0]
    start = reshape([0.0, 1.0, 10.0], 3, 1)
    C = vcat(dot(XX[1, :], vec(start)) - 0.1 + 0.0009,
      XX[2:3, 2:3] * [10.0, 1.0] .+ 0.1)
    C = reshape(C, 3, 1)
    L = opnorm(XX)
    for budget in (k1, k1 + 1)
      A = copy(start)
      certified, needed = HighDimensionalModels._lr_block!(A, XX, C, 0.1, w, false;
        block_solver = solver, max_iter = budget, tol = 1e-10, L)
      @test needed
      @test certified == (HighDimensionalModels._lr_group_kkt(A, XX*A - C, 0.1, w) <= 1e-10*max(1, norm(C)))
      if budget == k1
        @test !certified
        @test iszero(A[1, :])
        @test HighDimensionalModels._lr_group_kkt(A[2:3, :], (XX*A - C)[2:3, :], 0.1, w[2:3]) <= 1e-10*max(1, norm(C))
      else
        @test certified
        @test !iszero(A[1, :])
      end
    end
  end
end

@testset "Sparse blocks retain first-order iterates" begin
  XX = [1.0 0.05; 0.05 0.01]
  C, w, λ, tol = reshape([1.2, 0.17], 2, 1), ones(2), 0.1, 1e-12
  L = opnorm(XX)
  for solver in (:cd, :fista)
    k0, k1 = solver == :cd ? (10, 100) : (100, 1000)
    early, plain, A = zeros(2, 1), zeros(2, 1), zeros(2, 1)
    if solver == :cd
      @test !HighDimensionalModels._lr_group!(early, XX, C, λ, w; max_iter = k0, tol)
      @test HighDimensionalModels._lr_group!(plain, XX, C, λ, w; max_iter = k1, tol)
    else
      @test !HighDimensionalModels._lr_group_fista!(early, XX, C, λ, w; max_iter = k0, tol, L)
      @test HighDimensionalModels._lr_group_fista!(plain, XX, C, λ, w; max_iter = k1, tol, L)
    end
    # Keep the old cutoff far from the certificate's rounding scale.
    @test HighDimensionalModels._lr_group_kkt(early, XX*early - C, λ, w) > 100tol*max(1, norm(C))
    certified, needed = HighDimensionalModels._lr_block!(A, XX, C, λ, w, false;
      block_solver = solver, max_iter = k1, tol, L)
    @test certified
    @test !needed
    @test all(a === b for (a, b) in zip(A, plain))
  end
end

@testset "Ill-conditioned sparse reduced-rank fits" begin
  X, Y, λ = lr_conditioning_design()
  for intercept in (false, true)
    fits = [sparse_reduced_rank(X, Y, 2, λ; block_solver, intercept, adaptive = true)
      for block_solver in (:cd, :fista)]
    w = lr_conditioning_weights(X, Y, 2, intercept; adaptive = true)
    for s in fits
      @test s.converged
      @test s.iterations <= 200
      @test lr_conditioning_stationarity(X, Y, s, 2, w) <= 1e-7
    end
    @test fits[1].objective ≈ fits[2].objective rtol = 1e-8
  end
end

@testset "Issue 149 reproduction" begin
  for T in (Float64, Float32)
    X, Y, λ = lr_conditioning_issue(T)
    for solver in (:cd, :fista), intercept in (false, true), adaptive in (false, true)
      max_iter = T === Float32 && adaptive ? 20 : 1000
      logs, s = Test.collect_test_logs() do
        sparse_reduced_rank(X, Y, 2, λ; block_solver = solver, intercept, adaptive, max_iter)
      end
      w = lr_conditioning_weights(X, Y, 2, intercept; adaptive)
      if T === Float32 && adaptive && !s.converged
        @test length(logs) == 1
        @test occursin((:warn, r"did not converge"), only(logs))
        @test occursin("condition number", only(logs).message)
      else
        @test s.converged
        @test isempty(logs)
        @test lr_conditioning_stationarity(X, Y, s, 2, w) <= (T === Float32 ? 1e-3 : 1e-7)
      end
    end
  end
end

@testset "Sparse non-convergence diagnostics" begin
  X, Y, λ = lr_conditioning_design()
  for T in (Float64, Float32)
    A, Z = T.(X), T.(Y)
    logs, s = Test.collect_test_logs() do
      sparse_reduced_rank(A, Z, 2, T(λ); adaptive = true, max_iter = 1)
    end
    @test !s.converged
    @test length(logs) == 1
    log = only(logs)
    @test occursin((:warn, r"did not converge"), log)
    @test occursin("condition number", log.message) == (T === Float32)
    @test occursin("Refitting in Float64", log.message) == (T === Float32)
    @test log.kwargs[:relative_change] > log.kwargs[:tol]
    @test log.kwargs[:block_certified] isa Bool
    w = lr_conditioning_weights(A, Z, 2, true; adaptive = true)
    @test log.kwargs[:stationarity] ≈ lr_conditioning_stationarity(A, Z, s, 2, w) atol = 100eps(T)
    @test log.kwargs[:tol] == max(1e-7, 100eps(T))
    @test log.kwargs[:inner_tol] == max(1e-9, 100eps(T))
  end
  d = HighDimensionalModels._lr_data(X, Y, false)
  f = HighDimensionalModels._lr_svd(d)
  initial = HighDimensionalModels._lr_rrcoef(d, f, 2)
  w = HighDimensionalModels._lr_weights(d, initial, nothing, true, 2)
  logs, s = Test.collect_test_logs() do
    HighDimensionalModels._lr_sparse(d, 2, λ, w, initial;
      max_iter = 1, tol = 1e-7, inner_max_iter = 10, inner_tol = 1e-9, warn = true)
  end
  @test !s.converged
  @test length(logs) == 1
  @test !occursin("condition number", only(logs).message)
end

@testset "Sparse precision defaults and paths" begin
  rng = Xoshiro(1492)
  X = rand(rng, Float32, 40, 5) .- 0.5f0
  B = (rand(rng, Float32, 5, 2) .- 0.5f0) * (rand(rng, Float32, 2, 3) .- 0.5f0)
  Y = X * B .+ 0.5f0 .+ 0.1f0 .* (rand(rng, Float32, 40, 3) .- 0.5f0)
  for solver in (:cd, :fista)
    s = sparse_reduced_rank(X, Y, 2, 0.01f0; block_solver = solver)
    @test s.converged
    w = lr_conditioning_weights(X, Y, 2, true)
    @test lr_conditioning_stationarity(X, Y, s, 2, w) <= 1e-3
    opts = (; method = :sparse, rank = 2, λpath = Float32[0.02, 0.01], block_solver = solver)
    p = @test_logs LowRankPath(X, Y; opts...)
    @test all(s -> s.converged, p.solutions)
    cv = @test_logs CVLowRankPath(X, Y; opts..., folds = repeat(1:2, 20))
    @test all(s -> s.converged, cv.path.solutions)
    @test all(isfinite, cv.cvm)
  end
  logs, s = Test.collect_test_logs() do
    sparse_reduced_rank(X, Y, 2, 0.01f0; tol = 1e-12, inner_tol = 1e-11, max_iter = 1)
  end
  @test !s.converged
  @test length(logs) == 1
  @test occursin((:warn, r"did not converge"), only(logs))
  @test only(logs).kwargs[:tol] == 1e-12
  @test only(logs).kwargs[:inner_tol] == 1e-11
  for T in (Float64, Float32)
    p = LowRankPath(T.(X), T.(Y); method = :nuclear, λpath = T[0.02, 0.01])
    explicit = LowRankPath(T.(X), T.(Y); method = :nuclear, λpath = T[0.02, 0.01], tol = 1e-7)
    @test all(isequal(a.B, b.B) && a.converged == b.converged for (a, b) in zip(p.solutions, explicit.solutions))
  end

  X, Y, λ = lr_conditioning_design()
  w = collect(range(0.5, 1.5; length = size(X, 2)))
  d = HighDimensionalModels._lr_data(X, Y, true)
  C = d.X' * d.Y / size(X, 1)
  λmax = maximum(norm(view(C, j, :))/w[j] for j in axes(C, 1))
  for solver in (:cd, :fista)
    for opts in ((; weights = w, λpath = [λ, λ/2]),
                 (; weights = w, λpath = [2λmax, λ]),
                 (; adaptive = true, λpath = [λ, λ/2]))
      p = @test_logs LowRankPath(X, Y; method = :sparse, rank = 2, block_solver = solver, opts...)
      @test all(s -> s.converged, p.solutions)
      if first(p.λpath) == 2λmax
        @test iszero(p.solutions[1].B)
        @test !iszero(p.solutions[2].B)
      end
    end
    p = @test_logs LowRankPath(X, Y; method = :sparse, rank = 2,
      block_solver = solver, adaptive = true, λpath = [λ, λ/2], max_iter = 1)
    @test any(s -> !s.converged, p.solutions)
  end
end
