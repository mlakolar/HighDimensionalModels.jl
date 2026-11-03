module NumericsExtremesTest

using HighDimensionalModels, HiGHS
using LinearAlgebra, Random, SparseArrays, Statistics, Test

# Numerical extremes: the penalty at its boundary, lambda = 0, degenerate data,
# extreme scales and sparse designs. The expected values are computed here from
# the data, never from a solver's own lambda_max or path.

const N, P, Q = 60, 8, 3
const GROUPS = [1:2, 3:5, 6:8]                    # unequal sizes, so the weights sqrt(|G|) matter

function make_data(seed)
  rng = Xoshiro(seed)
  X = randn(rng, N, P) .* (1 .+ rand(rng, P))' .+ 0.3 .* randn(rng, P)'    # scales and means differ
  y = X[:, 1] .- 0.7 .* X[:, 4] .+ 0.5 .* randn(rng, N)
  Y = X[:, 1:2] * randn(rng, 2, Q) .+ 0.3 .* randn(rng, N, Q)
  yb = Float64.(rand(rng, N) .< 1 ./ (1 .+ exp.(-X[:, 1] .+ mean(X[:, 1]))))
  X, y, Y, yb
end

center(A) = A .- mean(A; dims = 1)
popsd(X) = vec(std(X; dims = 1, corrected = false))
nnz(v) = count(!iszero, v)
fin(v) = all(isfinite, v)
# a solver that cannot deliver must say so; a silent wrong answer is the failure
relerr(a, b) = norm(a .- b) / max(norm(b), eps())

X, y, Y, yb = make_data(1)
Xc, yc, Yc = center(X), y .- mean(y), center(Y)
sd = popsd(X)

@testset "penalty at the boundary: exact zero above lambda_max, nonzero below" begin
  # factors: 1 + 1e-9 sits above lambda_max by far more than rounding in
  # computing it (about 1e-15 relative), and 1 - 1e-3 is far above the solver
  # tolerance (about 1e-7 relative), so the sides are decided by the data
  above, below = 1 + 1e-9, 1 - 1e-3

  # lasso: ‖y - β₀ - Xβ‖²/(2n) + λ ∑ ωⱼ|βⱼ|, with intercept; ω = 1 or the population sds
  for std_x in (false, true)
    ω = std_x ? sd : ones(P)
    λmax = maximum(abs.(Xc' * yc) ./ (N .* ω))
    @test iszero(lasso(X, y, λmax * above; intercept = true, standardize_x = std_x).x)
    @test nnz(lasso(X, y, λmax * below; intercept = true, standardize_x = std_x).x) >= 1
  end

  # logistic: at the null intercept logit(ȳ) the gradient of the mean deviance is X'(ȳ - y)/n
  for std_x in (false, true)
    ω = std_x ? sd : ones(P)
    λmax = maximum(abs.(X' * (yb .- mean(yb))) ./ (N .* ω))
    @test iszero(glm_lasso(X, yb, λmax * above; standardize_x = std_x).x)
    @test nnz(glm_lasso(X, yb, λmax * below; standardize_x = std_x).x) >= 1
  end

  # group lasso (orthonormal): ‖X̃_G β_G‖/√n has weight √|G|, so λmax = maxₖ ‖P_Gₖ ỹ‖ / (√n √|Gₖ|)
  # with P_G the projection on the centered columns of G
  λmax = maximum(g -> norm(Matrix(qr(Xc[:, g]).Q)[:, 1:length(g)]' * yc) / sqrt(N * length(g)), GROUPS)
  @test iszero(group_lasso(X, y, λmax * above, GROUPS).x)
  @test nnz(group_lasso(X, y, λmax * below, GROUPS).x) >= 1

  # multi-response lasso: λ ∑ⱼ ‖sⱼ B[j, :]‖₂, so λmax = maxⱼ ‖X̃ⱼ'Ỹ/n‖ / sⱼ
  for std_x in (false, true)
    ω = std_x ? sd : ones(P)
    λmax = maximum(norm.(eachrow(Xc' * Yc ./ N)) ./ ω)
    @test iszero(multi_lasso(X, Y, λmax * above; standardize_x = std_x).B)
    @test nnz(multi_lasso(X, Y, λmax * below; standardize_x = std_x).B) >= 1
  end

  # nuclear norm: λmax is the spectral norm of X̃'Ỹ/n
  λmax = opnorm(Xc' * Yc ./ N)
  @test rank(nuclear_regression(X, Y, λmax * above).B) == 0
  @test rank(nuclear_regression(X, Y, λmax * below).B) >= 1
end

@testset "lambda = 0 is least squares" begin
  # n > p and a well conditioned random design: the error of an iterate with
  # gradient error 1e-7 (the default tolerances) is below 1e-6
  ols = Xc \ yc
  @test norm(Vector(group_lasso(X, y, 0.0, GROUPS).x) - ols) <= 1e-6
  @test norm(Vector(slope(X, y; lambda = zeros(P), intercept = true).x) - ols) <= 1e-6
  @test norm(Vector(fused_lasso(X, y, 0.0, 0.0; intercept = true).x) - ols) <= 1e-6
  OLS = Xc \ Yc
  @test norm(Matrix(multi_lasso(X, Y, 0.0).B) - OLS) <= 1e-6
  @test norm(Matrix(nuclear_regression(X, Y, 0.0).B) - OLS) <= 1e-6

  # logistic: an independent first-order condition of the unpenalized fit,
  # in the coefficients of X, X'(μ - y)/n = 0 and mean(μ - y) = 0
  sol = glm_lasso(X, yb, 0.0)
  μ = 1 ./ (1 .+ exp.(-(X * Vector(sol.x) .+ sol.intercept)))
  @test sol.converged
  @test norm(X' * (μ .- yb) ./ N, Inf) <= 1e-7
  @test abs(mean(μ .- yb)) <= 1e-7
end

@testset "zero response and zero design give the exact zero fit" begin
  X0, z, Z0 = zeros(N, P), zeros(N), zeros(N, Q)
  zero_fit(s, b) = iszero(b) && s.converged && isfinite(s.intercept isa Number ? s.intercept : sum(s.intercept))
  # zero response: nothing to explain, and the intercept of a zero response is zero
  s = group_lasso(X, z, 0.1, GROUPS); @test zero_fit(s, s.x) && iszero(s.intercept)
  s = sparse_group_lasso(X, z, 0.1, GROUPS); @test zero_fit(s, s.x) && iszero(s.intercept)
  s = multi_lasso(X, Z0, 0.1); @test zero_fit(s, s.B) && iszero(s.intercept)
  s = multitask_lasso([X, X], [z, z], 0.1); @test iszero(s.B) && s.converged && fin(s.intercept)
  s = nuclear_regression(X, Z0, 0.1); @test zero_fit(s, s.B) && isfinite(s.objective) && iszero(s.intercept)
  s = slope(X, z; intercept = true); @test zero_fit(s, s.x) && iszero(s.intercept)
  s = fused_lasso(X, z, 0.1, 0.1; intercept = true); @test zero_fit(s, s.x) && isfinite(s.objective)
  s = trend_filter(Matrix(1.0I, N, N), z, 0.5); @test zero_fit(s, s.x) && isfinite(s.objective)
  s = quantile_lasso(X, z, 0.1); @test zero_fit(s, s.β) && isfinite(s.objective)
  s = dirty_multitask(X, Z0, 0.1, 0.1); @test iszero(s.B) && s.converged && isfinite(s.objective) && fin(s.intercept)

  # zero design: the fit is the intercept alone, the mean of the response
  s = group_lasso(X0, y, 0.1, GROUPS); @test zero_fit(s, s.x) && s.intercept ≈ mean(y)
  s = sparse_group_lasso(X0, y, 0.1, GROUPS); @test zero_fit(s, s.x) && s.intercept ≈ mean(y)
  s = multi_lasso(X0, Y, 0.1); @test zero_fit(s, s.B) && s.intercept ≈ vec(mean(Y; dims = 1))
  s = multitask_lasso([X0, X0], [y, y], 0.1); @test iszero(s.B) && s.converged && fin(s.intercept)
  s = nuclear_regression(X0, Y, 0.1); @test zero_fit(s, s.B) && isfinite(s.objective) && s.intercept ≈ vec(mean(Y; dims = 1))
  s = slope(X0, y; intercept = true); @test zero_fit(s, s.x) && s.intercept ≈ mean(y)
  s = fused_lasso(X0, y, 0.1, 0.1; intercept = true); @test zero_fit(s, s.x) && isfinite(s.objective) && s.intercept ≈ mean(y)
  s = quantile_lasso(X0, y, 0.1); @test zero_fit(s, s.β) && isfinite(s.objective) && isfinite(s.intercept)
  s = dirty_multitask(X0, Y, 0.1, 0.1); @test iszero(s.B) && s.converged && isfinite(s.objective) && fin(s.intercept)

  # a constant response: the exact LP puts it all in the intercept (HiGHS feasibility tolerance 1e-7)
  s = quantile_lasso(X, fill(2.0, N), 0.1)
  @test s.converged && norm(s.β, Inf) <= 1e-8 && abs(s.intercept - 2) <= 1e-6
end

@testset "scale equivariance: y -> s y with λ -> s λ" begin
  # the penalized least squares problems are homogeneous of degree 2 in (y, λ),
  # so the coefficients scale by s; 1e-6 is ten times the default tolerance of
  # the solvers (1e-7) relative to the size of the coefficients
  slopeλ(λ) = fill(λ, P) .* (P:-1:1) ./ P           # nonincreasing
  for s in (1e-150, 1e150)
    f(y, λ) = Vector(group_lasso(X, y, λ, GROUPS).x)
    b0, bs = f(y, 0.05), f(s .* y, s * 0.05)
    @test fin(bs) && nnz(bs) >= 1 && relerr(bs ./ s, b0) <= 1e-6
    b0 = Matrix(multi_lasso(X, Y, 0.05).B)
    bs = Matrix(multi_lasso(X, s .* Y, s * 0.05).B)
    @test fin(bs) && nnz(bs) >= 1 && relerr(bs ./ s, b0) <= 1e-6
  end
  # nuclear_regression, slope and quantile_lasso are not checked at tiny scales:
  # their stopping rules have absolute floors (#214), so only large or
  # moderate scales are asserted
  λ = 0.3 * opnorm(Xc' * Yc / N)
  b0 = nuclear_regression(X, Y, λ).B
  bs = nuclear_regression(X, 1e150 .* Y, 1e150 * λ).B
  @test fin(bs) && rank(bs) >= 1 && relerr(bs ./ 1e150, b0) <= 1e-6
  b0 = Vector(slope(X, y; lambda = slopeλ(0.05), intercept = true).x)
  for s in (1e-3, 1e150)
    bs = Vector(slope(X, s .* y; lambda = slopeλ(s * 0.05), intercept = true).x)
    @test fin(bs) && nnz(bs) >= 1 && relerr(bs ./ s, b0) <= 1e-6
  end
  # the check loss is homogeneous of degree 1 in (y, β), and so λ stays: β -> s β
  # (HiGHS' tolerances are 1e-7 absolute, so the exact LP is compared to 1e-8)
  b0 = quantile_lasso(X, y, 0.05).β
  for s in (1e-3, 1e3)
    bs = quantile_lasso(X, s .* y, 0.05).β
    @test fin(bs) && nnz(bs) >= 1 && relerr(bs ./ s, b0) <= 1e-8
  end
end

@testset "small data scales report convergence at wrong answers (#214)" begin
  # correct: the scaled fit equals the unscaled fit times s (rtol as at the passing
  # scales), or the fit says converged == false, or an ArgumentError is thrown
  function honest(fit, b0, s, rtol)
    try
      sol = fit()
      b = sol isa NamedTuple ? sol.b : sol
      !sol.converged || (fin(b) && relerr(b ./ s, b0) <= rtol)
    catch e
      e isa ArgumentError
    end
  end
  slopeλ(λ) = fill(λ, P) .* (P:-1:1) ./ P
  λn = 0.3 * opnorm(Xc' * Yc / N)
  s = 1e-20
  # #214: slope stops after one iteration at a wrong answer
  b0 = Vector(slope(X, y; lambda = slopeλ(0.05), intercept = true).x)
  @test_broken honest(() -> (r = slope(X, s .* y; lambda = slopeλ(s * 0.05), intercept = true); (b = Vector(r.x), converged = r.converged)), b0, s, 1e-6)
  # #214: nuclear_regression's tolerance has an absolute floor of 1
  b0 = nuclear_regression(X, Y, λn).B
  @test_broken honest(() -> (r = nuclear_regression(X, s .* Y, s * λn); (b = r.B, converged = r.converged)), b0, s, 1e-6)
  # #214: quantile_lasso returns converged with a zero fit
  b0 = quantile_lasso(X, y, 0.05).β
  @test_broken honest(() -> (r = quantile_lasso(X, s .* y, 0.05); (b = r.β, converged = r.converged)), b0, s, 1e-8)
  # #214: at s = 1e150 quantile_lasso throws a raw ErrorException
  @test_broken honest(() -> (r = quantile_lasso(X, 1e150 .* y, 0.05); (b = r.β, converged = r.converged)), b0, 1e150, 1e-8)
end

@testset "scale equivariance of the design: X -> s X" begin
  # lasso without standardization: β -> β/s needs λ -> s λ
  b0 = Vector(lasso(X, y, 0.05; intercept = true).x)
  for s in (1e-150, 1e150)
    bs = Vector(lasso(s .* X, y, s * 0.05; intercept = true).x)
    @test fin(bs) && nnz(bs) >= 1 && relerr(bs .* s, b0) <= 1e-6
  end
  # with standardization the loadings absorb s, so λ is unchanged
  b0 = Vector(lasso(X, y, 0.05; intercept = true, standardize_x = true).x)
  for s in (1e-150, 1e150)
    bs = Vector(lasso(s .* X, y, 0.05; intercept = true, standardize_x = true).x)
    @test fin(bs) && nnz(bs) >= 1 && relerr(bs .* s, b0) <= 1e-6
  end

  # #200: lasso returns a converged all-zero fit when the design's squares overflow
  # correct: the fit of the unscaled problem / s, or a refusal (converged == false or ArgumentError)
  s = 1e200
  b0 = Vector(lasso(X, y, 0.05; intercept = true).x)
  ok = try
    sol = lasso(s .* X, y, s * 0.05; intercept = true)
    bs = Vector(sol.x)
    !sol.converged || (fin(bs) && relerr(bs .* s, b0) <= 1e-6)
  catch e
    e isa ArgumentError
  end
  @test_broken ok
end

@testset "sparse designs" begin
  rng = Xoshiro(1)
  Xs = sprandn(rng, 80, 10, 0.4)
  Xd = Matrix(Xs)
  ys = Xd[:, 1] .+ 0.5 .* randn(rng, 80)
  gs = [2k-1:2k for k in 1:5]
  # either the sparse fit is the dense fit, or a clean ArgumentError; never an ErrorException from LinearAlgebra
  agree(f) = try
    norm(Vector(f(Xs)) - Vector(f(Xd))) <= 1e-7
  catch e
    e isa ArgumentError
  end
  # #201: group_lasso (default standardize = :orthonormal) throws "Pivoting Strategies are not supported by SparseMatrixCSC"
  @test_broken agree(X -> group_lasso(X, ys, 0.02, gs).x)
  # #201: rlasso throws the same ErrorException from the pivoted QR
  @test_broken agree(X -> rlasso(X, ys).x)
end

end # module
