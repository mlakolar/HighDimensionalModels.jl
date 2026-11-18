module LassoReferenceTest
# Accuracy of `lasso` checked without the solver's own stopping rule:
#   * the optimality conditions and the duality gap recomputed in 256-bit BigFloat;
#   * cases with a closed form (orthogonal designs, one column, duplicated columns,
#     λ = 0, block-diagonal designs);
#   * the exact solution of small problems, found by enumerating supports and signs
#     in rational arithmetic, which shares nothing with coordinate descent.
# The objective is ‖y - Xβ‖² / (2n) + λ‖β‖₁.

using HighDimensionalModels, Test, LinearAlgebra, Random

# ---------------------------------------------------------------------------------
# Measurements in extended precision, from the stored Float inputs
# ---------------------------------------------------------------------------------

# kkt: the largest violation of |g_j| ≤ λ and of g_j = λ sign(β_j) on the support, for
# g = X'(y - Xβ)/n. gap: primal value minus the dual value at the residual scaled into the
# dual feasible set (NaN for λ = 0). All values are returned as Float64.
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
      θ = s * r
      gap = primal - (dot(yb, yb) - dot(yb - θ, yb - θ)) / (2n)
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

# From the definition of the dual point s r with s = min(1, nλ/‖X'r‖∞): if the violations
# of the optimality conditions are at most κ, then the gap is at most
#   2κ‖β‖₁ + (κ/λ)² ‖r‖²/(2n)
# (λ > 0). `m` is the measurement of the fit.
gap_bound(κ, λ, m) = 2κ * m.l1 + (κ / λ)^2 * m.fit

# Bound on ‖β - β*‖ for the solver's β and the exact β*, both supported on the set A where
# X_A has full column rank: the strong convexity μ_A = λmin(X_A'X_A / n) of the loss gives
# ‖β - β*‖₂ ≤ √|A| κ / μ_A when the subgradient of β is perturbed by at most κ.
function coef_bound(X, A, κ)
  isempty(A) && return 0.0
  XA = Float64.(X[:, A])
  μ = eigmin(Symmetric(XA' * XA / size(X, 1)))
  sqrt(length(A)) * κ / μ
end

soft(c, t) = sign(c) * max(abs(c) - t, zero(c))

# deterministic solver settings per element type; the tolerances are those of the relative
# coefficient-change test, chosen above the rounding level of the type
cyc(T) = CDSolver(order = :cyclic, momentum = false, tol = T === Float32 ? 1e-6 : 1e-12)

# ---------------------------------------------------------------------------------
# The exact lasso solution of a small problem, by enumeration, in rational arithmetic
# ---------------------------------------------------------------------------------

# For each support A with X_A of full column rank (|A| ≤ n) and each sign vector s on A, the
# stationarity condition X_A'(y - X_A β_A)/n = λ s gives β_A = (X_A'X_A)⁻¹(X_A'y - nλ s). The
# pair is a solution when sign(β_A) = s and |X_j'(y - X_A β_A)|/n ≤ λ off A. When the
# inequalities hold strictly, the solution is unique (X_A has full column rank and the
# dual solution has |g_j| < λ off A). Returns the passing candidates, each (A, β on 1:p, strict).
function exact_candidates(X::Matrix, y::Vector, λ)
  n, p = size(X)
  Q = Rational{BigInt}
  Xr, yr, λr = Q.(X), Q.(y), Q(λ)
  found = NamedTuple{(:A, :β, :strict), Tuple{Vector{Int}, Vector{Q}, Bool}}[]
  for mask in 0:(2^p - 1)
    A = [j for j in 1:p if (mask >> (j - 1)) & 1 == 1]
    k = length(A)
    k <= n || continue
    XA = Xr[:, A]
    G = XA' * XA
    if k > 0
      iszero(det(G)) && continue
      Ginv = inv(G)
      c = XA' * yr
    end
    for s in Iterators.product(ntuple(_ -> (-1, 1), k)...)
      sv = collect(Q, s)
      βA = k == 0 ? Q[] : Ginv * (c - n * λr * sv)
      all(i -> βA[i] * sv[i] > 0, 1:k) || continue
      g = Xr' * (yr - XA * βA) / n
      off = setdiff(1:p, A)
      all(j -> abs(g[j]) <= λr, off) || continue
      β = zeros(Q, p)
      β[A] = βA
      push!(found, (A = A, β = β, strict = all(j -> abs(g[j]) < λr, off)))
    end
  end
  found
end

# the unique solution; an error if the number of passing candidates is not one or it is not strict
function exact_lasso(X, y, λ)
  c = exact_candidates(X, y, λ)
  (length(c) == 1 && only(c).strict) || error("the exact reference is not unique: $(length(c)) candidates")
  only(c)
end

# data on a grid of 1/16 (exact in Float32 and Float64, and small as rationals)
grid(x) = round(x * 16) / 16
function grid_problem(seed, n, p; shared = 0.0, k = min(3, p))
  rng = Xoshiro(seed)
  z = randn(rng, n)
  X = grid.(randn(rng, n, p) .+ shared .* z)
  β = zeros(p); β[1:k] .= [2.0, -1.5, 1.0][1:k]
  X, grid.(X * β .+ 0.5 .* randn(rng, n)), β
end
lammax(X, y) = Float64(maximum(abs, Rational{BigInt}.(X)' * Rational{BigInt}.(y)) / size(X, 1))

# ---------------------------------------------------------------------------------

@testset "lasso against references independent of the solver" begin

  @testset "the measurements detect violations" begin
    X, y, _ = grid_problem(1, 12, 4)
    λ = 0.3 * lammax(X, y)
    ex = exact_lasso(X, y, λ)
    β = Float64.(ex.β)
    m = measure(X, y, β, λ)
    @test m.kkt <= 1e-15 && abs(m.gap) <= 1e-15
    # weak duality: the gap is nonnegative at other points, and positive away from the minimizer
    for δ in (0.01, -0.02)
      b = copy(β); b[ex.A[1]] += δ
      mb = measure(X, y, b, λ)
      @test mb.gap > 1e-4 && mb.kkt > 1e-4
    end
    @test measure(X, y, zeros(4), λ).gap >= 0
    @test measure(X, y, ones(4), λ).gap > 0
    # a zero vector below λmax violates the dual feasibility by (λmax - λ)
    @test measure(X, y, zeros(4), 0.5 * lammax(X, y)).kkt ≈ 0.5 * lammax(X, y) rtol = 1e-12
  end

  @testset "KKT conditions and duality gap, $T" for T in (Float32, Float64)
    problems = Dict(
      "independent columns" => (grid_problem(10, 40, 10)..., 0.15),
      "correlated columns" => (grid_problem(11, 50, 12; shared = 3.0)..., 0.10),
      "p > n" => (grid_problem(12, 15, 40)..., 0.25),
    )
    for (name, (X64, y64, _, frac)) in problems
      X, y = T.(X64), T.(y64)
      λmax = lammax(X, y)
      λ = T(frac * λmax)
      n = size(X, 1)
      for (label, solver) in (("cyclic", cyc(T)),
                              ("random order", CDSolver(order = :random, momentum = false, tol = T === Float32 ? 1e-6 : 1e-12, rng = Xoshiro(2))),
                              ("default momentum", CDSolver(tol = T === Float32 ? 1e-6 : 1e-12, rng = Xoshiro(3))))
        fit = lasso(X, y, λ, solver)
        @test fit.result.converged
        m = measure(X, y, coef(fit), λ)
        κ = kkt_bound(fit, X, y)
        @test m.kkt <= κ
        @test m.gap >= -1e-12
        @test m.gap <= gap_bound(κ, λ, m)
        @test any(!iszero, coef(fit)) && any(iszero, coef(fit))
        # the reported minimum is the objective at the returned point
        @test fit.result.minimum ≈ m.primal rtol = 64 * eps(T)
      end
    end
  end

  @testset "λ = 0: stationarity and the normal equations, $T" for T in (Float32, Float64)
    X64, y64, _ = grid_problem(13, 30, 5; shared = 1.0)
    X, y = T.(X64), T.(y64)
    fit = lasso(X, y, 0, cyc(T))
    m = measure(X, y, coef(fit), 0)
    κ = kkt_bound(fit, X, y)
    @test m.kkt <= κ
    # exact least squares in rational arithmetic
    Xr, yr = Rational{BigInt}.(X), Rational{BigInt}.(y)
    βls = Float64.((Xr' * Xr) \ (Xr' * yr))
    @test norm(Float64.(coef(fit)) - βls, Inf) <= coef_bound(X, 1:5, κ)
  end

  @testset "orthogonal designs have the soft-thresholding solution, $T" for T in (Float32, Float64)
    H2 = [1 1; 1 -1]
    H = Float64.(kron(H2, kron(H2, H2)))                  # 8 × 8, H'H = 8 I
    y = [3.0, -1.0, 0.5, 2.0, -2.5, 1.0, 0.25, -0.75]
    for (d, X64) in ((8, H), (2, H ./ 2))                 # X'X = d I
      X = T.(X64); yT = T.(y)
      n = size(X, 1)
      c = Float64.(X)' * y / n                            # exact: dyadic numbers
      for λ in (0.0, 0.2, 0.6, 1.1, 5.0)
        fit = lasso(X, yT, T(λ), cyc(T))
        βexact = soft.(c, λ) ./ (d / n)                   # minimize d β²/(2n) - c β + λ|β|
        @test norm(Float64.(coef(fit)) - βexact, Inf) <= 32 * eps(T) * maximum(abs, βexact; init = 1.0)
        @test fit.result.passes <= 3
      end
    end
    # orthonormal columns from a QR factorization, run for Float64 (rounded to Float32 the columns are orthonormal to about 1e-7)
    if T === Float64
      Q = Matrix(qr(randn(Xoshiro(5), 20, 5)).Q)          # Q'Q = I, n = 20
      yq = randn(Xoshiro(6), 20)
      c = Q' * yq                                         # minimize β²/(2n) - c β/n + λ|β|
      for λ in (0.02, 0.1, 0.3)
        fit = lasso(Q, yq, λ, cyc(T))
        @test coef(fit) ≈ soft.(c, 20λ) atol = 1e-10
      end
    end
  end

  @testset "one column has the closed form, $T" for T in (Float32, Float64)
    x64 = grid.(randn(Xoshiro(7), 25)); y64 = grid.(1.5 .* x64 .+ 0.5 .* randn(Xoshiro(8), 25))
    X = reshape(T.(x64), :, 1); y = T.(y64)
    n = 25
    for λ in (0.0, 0.1, 0.5, 1.0, 100.0)
      β = setprecision(BigFloat, 256) do
        Float64(soft(BigFloat(dot(Float64.(X[:, 1]), Float64.(y))) / n, BigFloat(λ)) / (BigFloat(dot(Float64.(X[:, 1]), Float64.(X[:, 1]))) / n))
      end
      fit = lasso(X, y, T(λ), cyc(T))
      @test only(coef(fit)) ≈ β rtol = 64 * eps(T) atol = 64 * eps(T) * abs(β)
      @test iszero(only(coef(fit))) == iszero(β)
    end
    # a negative correlation keeps its sign
    fit = lasso(X, -y, T(0.1), cyc(T))
    @test only(coef(fit)) < 0
  end

  @testset "two identical columns: the sum of the coefficients is determined, $T" for T in (Float32, Float64)
    x = T.(grid.(randn(Xoshiro(9), 30))); z = T.(grid.(randn(Xoshiro(10), 30)))
    y = T.(grid.(Float64.(2 .* x .- z) .+ 0.5 .* randn(Xoshiro(11), 30)))
    n = 30
    λ = T(0.2)
    X1 = [x z]
    X2 = [x x z]
    f1 = lasso(X1, y, λ, cyc(T))
    for solver in (cyc(T), CDSolver(order = :random, momentum = false, tol = T === Float32 ? 1e-6 : 1e-12, rng = Xoshiro(1)))
      f2 = lasso(X2, y, λ, solver)
      b = Float64.(coef(f2))
      @test b[1] * b[2] >= 0                                              # the two share a sign
      κ = kkt_bound(f2, X2, y)
      tol = coef_bound(X1, [1, 2], 2κ)
      @test abs(b[1] + b[2] - Float64(coef(f1)[1])) <= tol
      @test abs(b[3] - Float64(coef(f1)[2])) <= tol
      @test measure(X2, y, coef(f2), λ).kkt <= κ
    end
    # λ above λmax: both are zero
    @test iszero(coef(lasso(X2, y, T(1.01 * lammax(X2, y)), cyc(T))))
  end

  @testset "block-diagonal designs split into separate problems, $T" for T in (Float32, Float64)
    XA, yA, _ = grid_problem(20, 10, 3)
    XB, yB, _ = grid_problem(21, 8, 3; shared = 1.0)
    X = T.([XA zeros(10, 3); zeros(8, 3) XB]); y = T.([yA; yB])
    n, nA, nB = 18, 10, 8
    λ = 0.2 * lammax(X, y)
    fit = lasso(X, y, T(λ), cyc(T))
    # (1/2n)‖y - Xβ‖² splits as (nA/n) (1/2nA)‖yA - XAβA‖² + (nB/n) (1/2nB)‖yB - XBβB‖², so each
    # block is the lasso of that block at the level λ n / nᵢ
    fA = lasso(T.(XA), T.(yA), T(λ * n / nA), cyc(T))
    fB = lasso(T.(XB), T.(yB), T(λ * n / nB), cyc(T))
    κ = kkt_bound(fit, X, y)
    tol = max(coef_bound(X, collect(1:6), κ), 0)
    @test norm(Float64.(coef(fit)) - Float64.([coef(fA); coef(fB)]), Inf) <= 4 * tol
    # and both agree with the exact solution of the whole problem
    ex = exact_lasso(Float64.(X), Float64.(y), Float64(T(λ)))
    @test norm(Float64.(coef(fit)) - Float64.(ex.β), Inf) <= coef_bound(X, ex.A, κ)
  end

  @testset "exact solutions of small problems, $T" for T in (Float32, Float64)
    # (name, problem, λ as a fraction of λmax); `exact_lasso` fails unless the solution is unique
    cases = [
      ("independent, p = 5", grid_problem(30, 20, 5), 0.30),
      ("correlated, p = 6", grid_problem(31, 25, 6; shared = 2.0), 0.20),
      ("p > n (6 × 8)", grid_problem(32, 6, 8), 0.30),
      ("nearly collinear pair", begin
         X, y, β = grid_problem(33, 14, 4)
         X[:, 2] = grid.(X[:, 1] .+ 0.25 .* randn(Xoshiro(34), 14))
         X, y, β
       end, 0.15),
      ("small penalty, full support", grid_problem(35, 30, 5), 0.02),
      ("sparse truth, p = 8", grid_problem(36, 30, 8; k = 2), 0.35),
    ]
    for (name, (X64, y64, _), frac) in cases
      X, y = T.(X64), T.(y64)
      λ = T(frac * lammax(X, y))
      ex = exact_lasso(Float64.(X), Float64.(y), Float64(λ))
      fit = lasso(X, y, λ, cyc(T))
      β = Float64.(coef(fit))
      κ = kkt_bound(fit, X, y)
      @test findall(!iszero, β) == ex.A                                  # the same support
      @test norm(β - Float64.(ex.β), Inf) <= coef_bound(X, ex.A, κ)
      @test sign.(β) == sign.(Float64.(ex.β))
    end
    # at and above λmax the exact solution is zero
    X64, y64, _ = grid_problem(30, 20, 5)
    X, y = T.(X64), T.(y64)
    λ = T(1.01 * lammax(X, y))
    ex = exact_lasso(Float64.(X), Float64.(y), Float64(λ))
    @test isempty(ex.A)
    @test iszero(coef(lasso(X, y, λ, cyc(T))))
  end

  @testset "a fixed problem with hard-coded reference coefficients" begin
    # Generated by the exact enumeration of this file (one candidate passes, strictly):
    #   X, y, _ = grid_problem(40, 12, 4; shared = 1.0); λ = 0.25
    #   ex = exact_lasso(X, y, λ); Float64.(ex.β)
    # The data and the solution below are literals, independent of the generator and of the
    # reference method.
    X = [ 1.0625  1.0625  0.5625  0.125
          1.125   1.1875  0.3125  3.8125
         -0.875  -0.5625 -1.3125  0.375
         -0.375   0.3125 -1.125   1.8125
         -0.25    2.3125  1.3125  1.5
         -0.8125 -0.125   0.3125  1.875
          0.125   1.0     2.1875 -0.375
          1.75   -0.3125  2.6875  0.75
         -0.6875  0.125  -0.4375  0.5
          1.25   -0.0625 -0.25    0.0
         -0.6875 -0.3125 -0.625  -1.8125
          0.3125  2.9375  1.0625 -0.125]
    y = [1.0, 0.125, -2.5, -2.875, -2.875, -1.3125, 1.125, 6.75, -1.5625, 2.875, -2.0625, -3.875]
    λ = 0.25
    β = [1.7967069081541913, -1.5039468923980885, 0.9157103812810451, 0.0]   # to 17 digits
    for T in (Float32, Float64)
      fit = lasso(T.(X), T.(y), T(λ), cyc(T))
      κ = kkt_bound(fit, T.(X), T.(y))
      A = findall(!iszero, β)
      @test findall(!iszero, coef(fit)) == A
      @test norm(Float64.(coef(fit)) - β, Inf) <= coef_bound(X, A, κ)
    end
  end
end

end
