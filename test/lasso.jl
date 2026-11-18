module LassoTest
# The contract of `lasso`: result types, boundary penalties, degenerate designs, invalid
# arguments, the inputs left untouched, the reported convergence, the solver argument, and
# agreement with a direct ProximalBase/ProximalSolvers solve. Accuracy of the coefficients
# against exact references is in lasso_reference.jl. Needs ProximalBase and ProximalSolvers
# loadable by name (they are dependencies of the package).

using HighDimensionalModels, Test, Random, LinearAlgebra
using ProximalBase: LeastSquaresLoss, NormL1
using ProximalSolvers: solve!, CDResults

# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------

# deterministic solver settings per element type; the tolerances are those of the relative
# coefficient-change test, above the rounding level of the type
tolof(T) = T === Float32 ? 1e-6 : 1e-12
cyc(T) = CDSolver(order = :cyclic, momentum = false, tol = tolof(T))

function problem(T, seed, n, p; k = min(3, p))
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  T.(X), T.(X * β .+ 0.5 .* randn(rng, n))
end

# λmax = max |X'y| / n, from the stored values in extended precision, as a Float64
λmax(X, y) = setprecision(BigFloat, 256) do
  Float64(maximum(abs, BigFloat.(X)' * BigFloat.(y)) / size(X, 1))
end

# kkt: the largest violation of |g_j| ≤ λ and of g_j = λ sign(β_j) on the support, for
# g = X'(y - Xβ)/n; gap: the duality gap at the residual scaled into the dual feasible set
# (NaN for λ = 0); primal: the objective. Computed in 256-bit BigFloat from the stored values.
function measure(X::Matrix{T}, y::Vector{T}, β::Vector{T}, λ::Real) where T
  setprecision(BigFloat, 256) do
    Xb, yb, βb, λb = BigFloat.(X), BigFloat.(y), BigFloat.(β), BigFloat(λ)
    n = size(X, 1)
    r = yb - Xb * βb
    g = Xb' * r / n
    kkt = zero(BigFloat)
    for j in eachindex(βb)
      kkt = max(kkt, abs(g[j]) - λb)
      iszero(βb[j]) || (kkt = max(kkt, abs(g[j] - λb * sign(βb[j]))))
    end
    l1 = sum(abs, βb)
    fit = dot(r, r) / (2n)
    primal = fit + λb * l1
    gap = BigFloat(NaN)
    if λb > 0
      m = maximum(abs, g)
      s = m <= λb ? one(BigFloat) : λb / m
      gap = primal - (dot(yb, yb) - dot(yb - s * r, yb - s * r)) / (2n)
    end
    (kkt = Float64(kkt), gap = Float64(gap), primal = Float64(primal), l1 = Float64(l1), fit = Float64(fit))
  end
end

colnorm_max(X) = maximum(j -> norm(Float64.(view(X, :, j))), axes(X, 2))

# A bound on the violation of the optimality conditions after a run that ended on the
# coefficient-change test. In the last full pass of plain coordinate descent the test
# requires |Δβ_k| ‖X_k‖ / √n < tolerance (`result.tolerance`) for each coordinate k. In exact
# arithmetic the coordinate j satisfies its condition right after its update, and the later
# updates of the p - 1 other coordinates of the pass move g_j = X_j'r/n by at most
# ‖X_j‖ ‖X_k‖ |Δβ_k| / n each: (p - 1) ‖X_j‖ tolerance / √n in all. p replaces p - 1 for the
# rounding of the last update. The second term is the rounding of g_j in working precision,
# with ‖y‖ as the scale of ‖r‖. The tests hold the runs with momentum to the same bound; this
# argument does not cover them.
function kkt_bound(fit, X::Matrix{T}, y::Vector{T}) where T
  n, p = size(X)
  cmax = colnorm_max(X)
  p * cmax * Float64(fit.result.tolerance) / sqrt(n) + 64 * eps(T) * cmax * norm(Float64.(y)) / n
end

# for λ > 0, if the violations of the optimality conditions are at most κ, the gap is at most
# 2κ‖β‖₁ + (κ/λ)² ‖r‖²/(2n), by the definition of the dual point s r, s = min(1, nλ/‖X'r‖∞)
gap_bound(κ, λ, m) = 2κ * m.l1 + (κ / λ)^2 * m.fit

# ‖β - β′‖₂ for two points with subgradient perturbations at most κ, both with the support
# in A, where the loss is strongly convex with μ = λmin(X_A'X_A / n): at most 2 √|A| κ / μ
function coef_bound(X, A, κ)
  isempty(A) && return 0.0
  XA = Float64.(X[:, A])
  2 * sqrt(length(A)) * κ / eigmin(Symmetric(XA' * XA / size(X, 1)))
end

# the optimality conditions and the duality gap of `fit` for the data (X, y)
function check_optimal(fit, X, y; gap = true)
  m = measure(X, y, coef(fit), fit.λ)
  κ = kkt_bound(fit, X, y)
  @test m.kkt <= κ
  if gap && fit.λ > 0
    @test -1e-12 <= m.gap <= gap_bound(κ, fit.λ, m)
  end
  m
end

# the global random number generator is restored after `f`
function restoring_rng(f)
  state = copy(Random.default_rng())
  try
    f()
  finally
    copy!(Random.default_rng(), state)
  end
end

# a failing call leaves X and y bitwise unchanged
function unchanged(f, X, y)
  X0, y0 = copy(X), copy(y)
  try f() catch end
  isequal(X, X0) && isequal(y, y0)
end

# the tests that call `lasso` with the default solver draw from the global generator, which is
# restored when they end
restoring_rng() do
  @testset "lasso" begin

    @testset "result of a fit, $T" for T in (Float32, Float64)
      X, y = problem(T, 1, 30, 6)
      λ = T(0.2)
      fit = lasso(X, y, λ, cyc(T))
      @test fit isa LassoSolution{T}
      @test coef(fit) isa Vector{T} && length(coef(fit)) == 6
      @test fit.residuals isa Vector{T} && length(fit.residuals) == 30
      @test fit.λ isa T && fit.λ === λ
      @test fit.result isa CDResults
      @test fit.result.minimizer === fit.coefficients                  # the documented field
      @test coef(fit) === fit.coefficients                             # coef returns the stored vector
      @test propertynames(fit) == (:coefficients, :residuals, :λ, :result)
      @test !hasproperty(fit, :converged)                              # no flag of HDM's own
      @test fit.result.converged && fit.result.passes >= 1 && fit.result.maxchange < fit.result.tolerance
      @test fit.residuals == y - X * coef(fit)
      check_optimal(fit, X, y)
      @test any(!iszero, coef(fit)) && any(iszero, coef(fit))
    end

    @testset "λ = 0 is least squares, $T" for T in (Float32, Float64)
      X, y = problem(T, 2, 40, 5)
      fit = lasso(X, y, 0, cyc(T))
      ols = Float64.(X) \ Float64.(y)
      olsbig = setprecision(BigFloat, 256) do
        Float64.(BigFloat.(X) \ BigFloat.(y))
      end
      @test ols ≈ olsbig rtol = 1e-12
      # the solution of the λ = 0 problem differs from the iterate by at most the bound with
      # κ the optimality bound of the run: ‖β - β*‖₂ ≤ √p κ / μ, μ = λmin(X'X/n)
      κ = kkt_bound(fit, X, y)
      @test norm(Float64.(coef(fit)) - ols, Inf) <= coef_bound(X, 1:5, κ)
      check_optimal(fit, X, y)
      @test fit.λ === zero(T)
    end

    @testset "the zero solution from λmax, $T" for T in (Float32, Float64)
      X, y = problem(T, 3, 30, 6)
      lm = λmax(X, y)
      # strictly above λmax (by a margin far above the rounding of λmax) the coefficients are zero
      for λ in (T(1.001 * lm), T(2 * lm), T(1e3 * lm))
        fit = lasso(X, y, λ, cyc(T))
        @test iszero(coef(fit))
        @test fit.residuals == y
        @test fit.result.converged
      end
      # strictly below, the coordinate with the largest |X_j'y| enters with the sign of X_j'y
      fit = lasso(X, y, T(0.999 * lm), cyc(T))
      j = argmax(abs.(Float64.(X)' * Float64.(y)))
      @test coef(fit)[j] != 0 && sign(coef(fit)[j]) == sign(dot(Float64.(X[:, j]), Float64.(y)))
      @test count(!iszero, coef(fit)) >= 1
      check_optimal(fit, X, y)
      @test iszero(coef(lasso(X, y, T(1.001 * lm))))                   # the default solver as well
    end

    @testset "correlated, duplicated and zero columns, $T" for T in (Float32, Float64)
      X, y = problem(T, 4, 40, 5)
      λ = T(0.15)
      base = lasso(X, y, λ, cyc(T))
      # a duplicated column: same objective, optimality conditions hold
      Xd = [X X[:, 2]]
      fd = lasso(Xd, y, λ, cyc(T))
      mb, md = check_optimal(base, X, y), check_optimal(fd, Xd, y)
      @test abs(mb.primal - md.primal) <= gap_bound(kkt_bound(base, X, y), λ, mb) + gap_bound(kkt_bound(fd, Xd, y), λ, md)
      @test coef(fd)[2] * coef(fd)[6] >= 0                             # the pair shares a sign
      # a zero column: its coefficient is zero and the others are those of the fit without it
      Xz = [X[:, 1:2] zeros(T, 40) X[:, 3:5]]
      fz = lasso(Xz, y, λ, cyc(T))
      @test coef(fz)[3] === zero(T)
      @test norm(Float64.(coef(fz)[[1, 2, 4, 5, 6]]) - Float64.(coef(base)[[1, 2, 3, 4, 5]]), Inf) <= coef_bound(X, 1:5, kkt_bound(fz, Xz, y))
      check_optimal(fz, Xz, y)
      # correlated columns (correlation about 0.99)
      rng = Xoshiro(41)
      z = randn(rng, 40)
      Xc = T.([z z .+ 0.15 .* randn(rng, 40) randn(rng, 40, 3)])
      yc = T.(Float64.(Xc[:, 1]) .+ 0.5 .* randn(rng, 40))
      check_optimal(lasso(Xc, yc, λ, cyc(T)), Xc, yc)
      # an all-zero design: nothing to fit
      f0 = lasso(zeros(T, 10, 3), y[1:10], λ, cyc(T))
      @test iszero(coef(f0)) && f0.residuals == y[1:10]
    end

    @testset "rank-deficient and wide designs, $T" for T in (Float32, Float64)
      X, y = problem(T, 5, 30, 4)
      Xr = [X X[:, 1] .+ X[:, 2]]                                      # a column that is a sum of two others
      check_optimal(lasso(Xr, y, T(0.1), cyc(T)), Xr, y)
      Xw, yw = problem(T, 6, 12, 40)                                   # p > n
      λ = T(0.2 * λmax(Xw, yw))
      for solver in (cyc(T), CDSolver(tol = tolof(T), rng = Xoshiro(1)))
        fit = lasso(Xw, yw, λ, solver)
        @test fit.result.converged
        check_optimal(fit, Xw, yw)
        @test 0 < count(!iszero, coef(fit)) <= 12                      # for generic data, at most n nonzero coefficients
      end
      # λ = 0 with p > n: the fit is close to interpolation of the response
      fit = lasso(Xw, yw, 0, cyc(T))
      check_optimal(fit, Xw, yw)
      @test norm(fit.residuals) <= 1e-3 * norm(yw)
    end

    @testset "one observation, one coefficient, constant and zero responses, $T" for T in (Float32, Float64)
      # n = 1: min (y - x'β)²/2 + λ‖β‖₁ puts the weight on the largest |x_j|: β₃ = (|x₃ y| - λ)/x₃²
      X = T[1 2 3]; y = T[2]; λ = T(0.1)
      fit = lasso(X, y, λ, cyc(T))
      β3 = (6 - Float64(λ)) / 9
      @test Float64.(coef(fit)) ≈ [0, 0, β3] atol = 64 * eps(T)
      @test only(fit.residuals) ≈ Float64(λ) / 3 atol = 64 * eps(T)
      @test nobs(fit) == 1
      # p = 1: the soft threshold of x'y/n at λ, divided by x'x/n
      x, yy = problem(T, 7, 20, 1)
      fit = lasso(x, yy, T(0.3), cyc(T))
      c, a = dot(Float64.(x), Float64.(yy)) / 20, dot(Float64.(x), Float64.(x)) / 20
      @test only(coef(fit)) ≈ sign(c) * max(abs(c) - Float64(T(0.3)), 0) / a rtol = 1e-4 atol = 1e-5   # data rounding of Float32 ‖x‖²
      # a constant response, which the lasso has no intercept to absorb
      Xc, _ = problem(T, 8, 25, 4)
      yc = fill(T(3), 25)
      check_optimal(lasso(Xc, yc, T(0.2), cyc(T)), Xc, yc)
      # the zero response: the zero solution
      fit = lasso(Xc, zeros(T, 25), T(0.2), cyc(T))
      @test iszero(coef(fit)) && iszero(fit.residuals) && fit.result.converged
      fit = lasso(Xc, zeros(T, 25), 0, cyc(T))
      @test iszero(coef(fit)) && fit.result.converged
    end

    @testset "invalid arguments, $T" for T in (Float32, Float64)
      X, y = problem(T, 9, 20, 4)
      # the penalty: negative, NaN and infinite levels
      @test_throws ArgumentError lasso(X, y, T(-0.1))
      @test_throws ArgumentError lasso(X, y, -1)
      @test_throws ArgumentError lasso(X, y, NaN)
      @test_throws ArgumentError lasso(X, y, Inf)
      @test_throws ArgumentError lasso(X, y, -Inf)
      @test_throws ArgumentError lasso(X, y, T(NaN))
      # non-finite data
      for bad in (NaN, Inf, -Inf)
        Xb = copy(X); Xb[2, 2] = bad
        yb = copy(y); yb[3] = bad
        @test_throws ArgumentError lasso(Xb, y, 0.1)
        @test_throws ArgumentError lasso(X, yb, 0.1)
      end
      # dimensions
      @test_throws DimensionMismatch lasso(X, y[1:19], 0.1)
      @test_throws DimensionMismatch lasso(X, [y; y[1]], 0.1)
      @test_throws "at least one row" lasso(zeros(T, 0, 3), T[], 0.1)       # n = 0, refused by the package
      @test_throws DimensionMismatch lasso(zeros(T, 20, 0), y, 0.1)          # p = 0
      @test_throws DimensionMismatch lasso(zeros(T, 0, 0), T[], 0.1)
      @test_throws DimensionMismatch lasso(zeros(T, 0, 3), T[], 0.1)
      # unsupported types, each named by an ArgumentError from the package
      @test_throws ArgumentError lasso(Int.(round.(X)), Int.(round.(y)), 0.1)
      @test_throws ArgumentError lasso(Diagonal(ones(T, 20)), y, 0.1)       # a structured matrix
      @test_throws ArgumentError lasso(Symmetric(ones(T, 20, 20)), y, 0.1)
      @test_throws ArgumentError lasso(view(X, :, 1:2), y, 0.1)
      @test_throws ArgumentError lasso(X', y, 0.1)
      @test_throws ArgumentError lasso(X, view(y, 1:20), 0.1)
      @test_throws ArgumentError lasso(X, reshape(y, :, 1), 0.1)             # a matrix response
      @test_throws ArgumentError lasso(BigFloat.(X), BigFloat.(y), 0.1)
      @test_throws ArgumentError lasso(Float16.(X), Float16.(y), 0.1)
      @test_throws ArgumentError lasso(X, y, 0.1, :cyclic)                   # a solver that is not a CDSolver
      @test_throws ArgumentError lasso(X, y, 0.1, nothing)
      @test_throws ArgumentError lasso(X, y, 0.1 + 0im)                      # λ that is not a Real
      @test_throws ArgumentError lasso(X, y, "0.1")
      @test_throws ArgumentError lasso(X, y, missing)
      @test_throws ArgumentError lasso(X, y, nothing)
      other = T === Float32 ? Float64 : Float32
      @test_throws ArgumentError lasso(other.(X), y, 0.1)                    # mixed element types
      @test_throws ArgumentError lasso(X, other.(y), 0.1)
      # λ outside the range of the element type
      if T === Float32
        @test_throws ArgumentError lasso(X, y, 1e-60)
        @test_throws ArgumentError lasso(X, y, 1e60)
      else
        @test iszero(coef(lasso(X, y, 1e300, cyc(T))))
        fit = lasso(X, y, 1e-300, cyc(T))
        @test fit.λ == 1e-300 && fit.result.converged
        @test Float64.(coef(fit)) ≈ Float64.(X) \ Float64.(y) rtol = 1e-7
      end
    end

    @testset "other types of λ, $T" for T in (Float32, Float64)
      X, y = problem(T, 10, 25, 5)
      for λ in (0, 1, true, false, 1 // 10, big"0.1", big(1), π, Float16(0.5), UInt8(1), Float32(0.1), 0.1, T(0.1))
        fit = lasso(X, y, λ, cyc(T))
        @test fit.λ isa T && fit.λ == T(λ)
        ref = lasso(X, y, T(λ), cyc(T))
        @test fit.result.converged
        @test Float64.(coef(fit)) ≈ Float64.(coef(ref)) atol = 1e-4 rtol = 1e-4   # the level reaches the solver in its own type
      end
      # a negative zero: the fit of λ = 0
      fit = lasso(X, y, -0.0, cyc(T))
      @test iszero(fit.λ) && coef(fit) == coef(lasso(X, y, 0.0, cyc(T)))
      # integers reach the solver as integers and give the same fit as the float
      @test coef(lasso(X, y, 1, cyc(T))) == coef(lasso(X, y, T(1), cyc(T)))
    end

    @testset "the inputs are not modified, and the solution does not alias them, $T" for T in (Float32, Float64)
      X, y = problem(T, 11, 25, 6)
      X0, y0 = copy(X), copy(y)
      fit = lasso(X, y, T(0.2), cyc(T))
      @test X == X0 && y == y0
      β0, r0 = copy(coef(fit)), copy(residuals(fit))
      fill!(X, 7); fill!(y, -3)
      @test coef(fit) == β0 && residuals(fit) == r0                    # no reference to X or y
      X, y = copy(X0), copy(y0)
      f1 = lasso(X, y, T(0.2), cyc(T))
      f2 = lasso(X, y, T(0.2), cyc(T))
      @test f1.coefficients !== f2.coefficients && f1.residuals !== f2.residuals
      @test f1.coefficients == f2.coefficients
      f1.coefficients .= 0; f1.residuals .= 0
      @test f2.coefficients == β0 && f2.residuals == r0                # no storage shared between fits
      @test f2.residuals !== y
      # failing calls leave the data unchanged
      Xn = copy(X); Xn[3, 3] = NaN
      yn = copy(y); yn[2] = Inf
      @test unchanged(() -> lasso(X, y, -1), X, y)
      @test unchanged(() -> lasso(X, y, Inf), X, y)
      @test unchanged(() -> lasso(X, y[1:5], 0.1), X, y)
      @test unchanged(() -> lasso(Xn, y, 0.1), Xn, y)
      @test unchanged(() -> lasso(X, yn, 0.1), X, yn)
      @test unchanged(() -> lasso(X, y, 0.1, :solver), X, y)
      big = T === Float32 ? T(1e20) : T(1e200)
      @test_throws ArgumentError lasso(big .* X, y, 0.1)               # the curvature overflows during the solve
      @test unchanged(() -> lasso(big .* X, y, 0.1), big .* X, y)
    end

    @testset "convergence reporting, $T" for T in (Float32, Float64)
      rng = Xoshiro(12)
      z = randn(rng, 40)
      X = T.([z z .+ 0.3 .* randn(rng, 40) randn(rng, 40, 4)])          # correlated pair: several passes
      y = T.(Float64.(X[:, 1]) .+ 0.5 .* randn(rng, 40))
      λ = T(0.05)
      full = lasso(X, y, λ, cyc(T))
      @test full.result.converged && full.result.passes > 1
      @test full.result.maxchange < full.result.tolerance
      one = @test_logs (:warn,) lasso(X, y, λ, CDSolver(order = :cyclic, momentum = false, tol = tolof(T), maxpasses = 1))
      @test !one.result.converged && one.result.passes == 1
      @test one.result.maxchange >= one.result.tolerance
      @test one.result.minimizer === one.coefficients && all(isfinite, coef(one))
      @test any(!iszero, coef(one)) && coef(one) != coef(full)       # the iterate of the first pass is returned as it is
      @test one.residuals == y - X * coef(one)
      @test one.result.minimum > full.result.minimum                  # and it is not the minimum
      zero_passes = @test_logs (:warn,) lasso(X, y, λ, CDSolver(order = :cyclic, momentum = false, maxpasses = 0))
      @test iszero(coef(zero_passes)) && zero_passes.residuals == y
      @test !zero_passes.result.converged && zero_passes.result.passes == 0
      # a threshold below the rounding level of Float32 ends at maxpasses, and the run says so
      if T === Float32
        fit = @test_logs (:warn,) lasso(X, y, λ, CDSolver(order = :cyclic, momentum = false, tol = 1e-12, maxpasses = 50))
        @test !fit.result.converged && fit.result.passes == 50
      end
    end

    @testset "the solver argument, $T" for T in (Float32, Float64)
      X, y = problem(T, 13, 40, 6)
      λ = T(0.2)
      tol = tolof(T)
      # the same seed gives the same run
      a = lasso(X, y, λ, CDSolver(tol = tol, rng = Xoshiro(5)))
      b = lasso(X, y, λ, CDSolver(tol = tol, rng = Xoshiro(5)))
      @test coef(a) == coef(b) && a.result.passes == b.result.passes
      # the caller's generator advances; an explicit generator leaves the global one alone
      rng = Xoshiro(5)
      global_before = copy(Random.default_rng())
      lasso(X, y, λ, CDSolver(tol = tol, rng = rng))
      @test rng != Xoshiro(5)
      @test Random.default_rng() == global_before
      # cyclic order draws nothing
      rng = Xoshiro(5)
      lasso(X, y, λ, CDSolver(order = :cyclic, momentum = false, tol = tol, rng = rng))
      @test rng == Xoshiro(5)
      # orders and the default call agree with each other to the optimality bound
      fits = [lasso(X, y, λ, cyc(T)),
              lasso(X, y, λ, CDSolver(order = :random, momentum = false, tol = tol, rng = Xoshiro(6))),
              lasso(X, y, λ, CDSolver(tol = tol, rng = Xoshiro(7)))]
      restoring_rng() do
        push!(fits, lasso(X, y, λ, CDSolver(tol = tol)))                # default generator
        push!(fits, lasso(X, y, λ))                                     # the default solver, default tolerance
      end
      support = findall(!iszero, coef(fits[1]))
      for f in fits
        check_optimal(f, X, y)
        @test findall(!iszero, coef(f)) == support
      end
      # the bound of the default tolerance 1e-7 is the largest; the runs at `tol` are within their own
      κ = maximum(f -> kkt_bound(f, X, y), fits)
      for f in fits[2:end]
        @test norm(Float64.(coef(f)) - Float64.(coef(fits[1])), Inf) <= coef_bound(X, support, κ)
      end
      @test_throws ArgumentError lasso(X, y, λ, CDSolver(order = :cyclic, momentum = true))   # refused by CDSolver
    end

    @testset "agreement with a direct ProximalBase / ProximalSolvers solve, $T" for T in (Float32, Float64)
      X, y = problem(T, 14, 35, 7)
      n, p = size(X)
      function direct(λ, solver)
        f = (one(T) / T(n)) * LeastSquaresLoss(y, X)
        β = zeros(T, p)
        β, solve!(β, f, NormL1(λ), solver)
      end
      same(r, s) = typeof(r) == typeof(s) && all(i -> isequal(getfield(r, i), getfield(s, i)), 1:fieldcount(typeof(r)))
      for λ in (T(0.3), T(0.05), T(0))
        for mk in (() -> cyc(T),
                   () -> CDSolver(tol = tolof(T), rng = Xoshiro(3)),                        # random order, momentum
                   () -> CDSolver(order = :random, momentum = false, tol = tolof(T), rng = Xoshiro(3)))
          fit = lasso(X, y, λ, mk())
          β, r = direct(λ, mk())
          @test coef(fit) == β                                           # identical, not close
          @test same(fit.result, r) && fit.result.minimizer == β
          @test fit.residuals == y - X * β && fit.λ === λ
        end
      end
    end

    @testset "extreme but representable scales" begin
      X, y = problem(Float64, 15, 30, 6)
      λ = 0.2
      base = lasso(X, y, λ, cyc(Float64))
      b = coef(base)
      # X → cX, y fixed: β → β/c at the level cλ;  y → sy: β → sβ at the level sλ;  both: β fixed at c²λ
      for c in (1e-8, 1e8)
        f = lasso(c .* X, y, c * λ, cyc(Float64))
        @test f.result.converged
        @test coef(f) .* c ≈ b rtol = 1e-8
        f = lasso(X, c .* y, c * λ, cyc(Float64))
        @test f.result.converged
        @test coef(f) ./ c ≈ b rtol = 1e-8
        f = lasso(c .* X, c .* y, c^2 * λ, cyc(Float64))
        @test f.result.converged
        @test coef(f) ≈ b rtol = 1e-8
      end
      # beyond the range of Float64 the curvature or the objective overflows: the solver refuses
      @test_throws ArgumentError lasso(1e200 .* X, y, 1e200 * λ, cyc(Float64))
      @test_throws ArgumentError lasso(X, 1e200 .* y, 1e200 * λ, cyc(Float64))
      # Float32 with a column norm that overflows in Float32
      X32, y32 = problem(Float32, 15, 30, 6)
      @test_throws ArgumentError lasso(1f20 .* X32, y32, 1f20 * 0.2f0, cyc(Float32))
      # and at a large but representable scale
      f = lasso(1f10 .* X32, y32, 1f10 * 0.2f0, cyc(Float32))
      @test f.result.converged
      @test Float64.(coef(f)) .* 1e10 ≈ Float64.(coef(lasso(X32, y32, 0.2f0, cyc(Float32)))) rtol = 1e-3
    end
  end
end

end
