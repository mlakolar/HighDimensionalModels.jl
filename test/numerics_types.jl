module NumericsTypesTest

# Element types and integer inputs (T9, T10 of the #199 audit); test/numerics_contracts.jl has the rest.
using HighDimensionalModels, ProximalBase, HiGHS
using LinearAlgebra, Random, SparseArrays, Statistics, Test

# One small design shared by all the tables; every fit is on at most 60 rows.
const n, p, q = 60, 8, 3
const rng = Xoshiro(2024)
const X = randn(rng, n, p)
const y = X[:, 1] .- 0.5 .* X[:, 2] .+ 0.4 .* randn(rng, n)
const Y = X[:, 1:2] * randn(rng, 2, q) .+ 0.3 .* randn(rng, n, q)
const yb = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-X[:, 1])))
const groups = [1:2, 3:4, 5:6, 7:8]

# the coefficients of any solution as a Float vector, whatever the field is called
function coefs(s)
  b = hasproperty(s, :x) ? s.x : hasproperty(s, :B) ? s.B : hasproperty(s, :W) ? s.W : s.β
  b isa AbstractVector ? Vector(b) : vec(Array(b))
end
dist(a, b) = norm(coefs(a) .- coefs(b))
# the element type of an array, or of the arrays in a list of arrays
leaftype(a::AbstractArray{<:AbstractArray}) = leaftype(first(a))
leaftype(a::AbstractArray) = eltype(a)

# ---------------------------------------------------------------------------------------------
# T9: Float32 data gives Float32 output, for a Float32 and for a Float64 λ
# ---------------------------------------------------------------------------------------------
# quantile_lasso and fixed_knockoffs are not pinned: they return Float64 for Float32 input today
# and #202 asks for a decision on the contract.
# Tolerances are explicit where the default 1e-7 is below Float32 rounding (class #185).
const F32_CASES = (
  ("glm_lasso", (d, λ) -> glm_lasso(d.X, d.yb, λ, ProximalBase.Logistic(), CDOptions(opt_tol = 1e-5)), d -> d.X),
  ("group_lasso", (d, λ) -> group_lasso(d.X, d.y, λ, groups, CDOptions(opt_tol = 1e-5)), d -> d.X),
  ("sparse_group_lasso", (d, λ) -> sparse_group_lasso(d.X, d.y, λ, groups, CDOptions(opt_tol = 1e-5)), d -> d.X),
  ("multi_lasso", (d, λ) -> multi_lasso(d.X, d.Y, λ, CDOptions(opt_tol = 1e-5)), d -> d.X),
  ("multitask_lasso", (d, λ) -> multitask_lasso([d.X, d.X], [d.y, d.y], λ, CDOptions(opt_tol = 1e-5)), d -> [d.X, d.X]),
  ("dirty_multitask", (d, λ) -> dirty_multitask(d.X, d.Y, λ, λ / 2, CDOptions(opt_tol = 1e-5)), d -> d.X),
  ("slope", (d, λ) -> slope(d.X, d.y; lambda = fill(λ, p), tol = 1e-4), d -> d.X),
  ("fused_lasso", (d, λ) -> fused_lasso(d.X, d.y, λ, λ), d -> d.X),
  ("trend_filter", (d, λ) -> trend_filter(Matrix{eltype(d.X)}(I, n, n), d.y, 10λ), d -> Matrix{eltype(d.X)}(I, n, n)),
  ("rlasso", (d, λ) -> rlasso(d.X, d.y), d -> d.X),
  ("nuclear_regression", (d, λ) -> nuclear_regression(d.X, d.Y, λ; tol = 1e-4), d -> d.X),
  ("sparse_nuclear_regression", (d, λ) -> sparse_nuclear_regression(d.X, d.Y, λ, λ; tol = 1e-4), d -> d.X),
  ("lowrank_plus_sparse", (d, λ) -> lowrank_plus_sparse(d.X, d.Y, λ, λ; tol = 1e-4), d -> d.X),
  ("reduced_rank", (d, λ) -> reduced_rank(d.X, d.Y, 1), d -> d.X),
)

@testset "Float32 element types" begin
  d = (X = Float32.(X), y = Float32.(y), Y = Float32.(Y), yb = Float32.(yb))
  for (name, fit, newX) in F32_CASES, λ in (0.05f0, 0.05)
    @testset "$name λ::$(typeof(λ))" begin
      s = fit(d, λ)
      @test eltype(coef(s)) == Float32
      @test leaftype(predict(s, newX(d))) == Float32
      for f in (:intercept, :objective)
        hasproperty(s, f) && @test eltype(getproperty(s, f)) == Float32
      end
      hasproperty(s, :converged) && @test all(s.converged)
      @test all(isfinite, coefs(s))
    end
  end
end

@testset "Float32 X with Float64 y promotes to Float64" begin
  X32 = Float32.(X)
  for (name, s) in (("glm_lasso", glm_lasso(X32, yb, 0.05)), ("group_lasso", group_lasso(X32, y, 0.05, groups)),
                    ("multi_lasso", multi_lasso(X32, Y, 0.05)), ("fused_lasso", fused_lasso(X32, y, 0.05, 0.05)))
    @testset "$name" begin
      @test eltype(coef(s)) == Float64
      hasproperty(s, :intercept) && @test eltype(s.intercept) == Float64
    end
  end
end

# ---------------------------------------------------------------------------------------------
# T10: Int, BitVector inputs equal their Float64 conversion
# ---------------------------------------------------------------------------------------------
@testset "integer inputs equal their Float64 conversion" begin
  r = Xoshiro(10)
  Xi = rand(r, -3:3, n, p); Yi = rand(r, -5:5, n, q); yi = Yi[:, 1]
  Xf, Yf, yf = Float64.(Xi), Float64.(Yi), Float64.(yi)
  # the conversion is exact, so with the same rng (the CD order is random) the fits are identical;
  # 1e-10 only guards against reordered sums
  cases = (
    ("group_lasso", (X, y) -> group_lasso(X, y, 0.5, groups), yi, yf),
    ("multi_lasso", (X, y) -> multi_lasso(X, y, 0.3; rng = Xoshiro(1)), Yi, Yf),
    ("nuclear_regression", (X, y) -> nuclear_regression(X, y, 0.3), Yi, Yf),
    ("slope", (X, y) -> slope(X, y), yi, yf),
    ("quantile_lasso", (X, y) -> quantile_lasso(X, y, 0.2), yi, yf),
    ("fused_lasso", (X, y) -> fused_lasso(X, y, 0.5, 0.5), yi, yf),
    ("rlasso", (X, y) -> rlasso(X, y), yi, yf),
    ("dirty_multitask", (X, y) -> dirty_multitask(X, y, 0.5, 0.3; rng = Xoshiro(1)), Yi, Yf),
  )
  for (name, fit, yint, yflt) in cases
    @testset "$name" begin
      a, b = fit(Xi, yint), fit(Xf, yflt)
      @test dist(a, b) <= 1e-10
      @test eltype(coef(a)) == Float64
    end
  end
  @testset "glm_lasso Int and BitVector y" begin
    ref = glm_lasso(X, yb, 0.05; rng = Xoshiro(1))
    for yy in (Int.(yb), BitVector(yb .> 0))
      a = glm_lasso(X, yy, 0.05; rng = Xoshiro(1))
      @test dist(a, ref) <= 1e-10
      @test a.intercept ≈ ref.intercept atol = 1e-10
      @test a.converged
    end
  end
end

end # module
