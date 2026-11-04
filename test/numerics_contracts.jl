module NumericsContractsTest

# Iteration limits, non-mutation, sparse designs and rejected data (T11-T13, T5 of the #199 audit);
# test/numerics_types.jl has the element types.
using HighDimensionalModels, ProximalBase, HiGHS
using LinearAlgebra, Logging, Random, SparseArrays, Statistics, Test

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
# T11: max_iter exhaustion
# ---------------------------------------------------------------------------------------------
@testset "max_iter exhaustion gives converged == false and finite output, and no poisoned state" begin
  o1 = CDOptions(; max_iter = 1, opt_tol = 1e-14, gaptol = 0)
  I60 = Matrix(1.0I, n, n)
  D = difference_matrix(p)
  # (name, capped fit, uncapped fit on the same data, warns, iterations of the capped fit)
  # warns: whether the solver reports the exhaustion with @warn; the low-rank and structured solvers
  # report it only through `converged`. glm_lasso sums the passes over its Newton steps.
  cases = (
    ("glm_lasso", () -> glm_lasso(X, yb, 0.01, ProximalBase.Logistic(), o1), () -> glm_lasso(X, yb, 0.01), true, nothing),
    ("group_lasso", () -> group_lasso(X, y, 0.01, groups, o1), () -> group_lasso(X, y, 0.01, groups), true, 1),
    ("sparse_group_lasso", () -> sparse_group_lasso(X, y, 0.01, groups, o1), () -> sparse_group_lasso(X, y, 0.01, groups), true, 1),
    ("multi_lasso", () -> multi_lasso(X, Y, 0.01, o1), () -> multi_lasso(X, Y, 0.01), true, 1),
    ("trend_filter", () -> trend_filter(I60, y, 0.1; max_iter = 1, abstol = 1e-14, reltol = 1e-14), () -> trend_filter(I60, y, 0.1), false, 1),
    ("generalized_lasso", () -> generalized_lasso(X, y, D, 0.1; max_iter = 1, abstol = 1e-14, reltol = 1e-14), () -> generalized_lasso(X, y, D, 0.1), false, 1),
    ("nuclear_regression", () -> nuclear_regression(X, Y, 0.05; max_iter = 1, tol = 1e-14), () -> nuclear_regression(X, Y, 0.05), false, 1),
    ("sparse_reduced_rank", () -> sparse_reduced_rank(X, Y, 1, 0.05; max_iter = 1, tol = 1e-14), () -> sparse_reduced_rank(X, Y, 1, 0.05), true, 1),
    ("lowrank_plus_sparse", () -> lowrank_plus_sparse(X, Y, 0.05, 0.05; max_iter = 1, tol = 1e-14), () -> lowrank_plus_sparse(X, Y, 0.05, 0.05), false, 1),
    ("rlasso", () -> rlasso(X, y, o1; max_iter = 1, tol = 1e-14), () -> rlasso(X, y), nothing, 1),
  )
  for (name, capped, free, warns, its) in cases
    @testset "$name" begin
      s = warns === true ? (@test_logs (:warn,) match_mode = :any capped()) : with_logger(capped, NullLogger())
      @test !s.converged
      its === nothing ? @test(s.iterations >= 1) : @test(s.iterations == its)
      @test all(isfinite, coefs(s))
      t = free()
      @test t.converged
      @test all(isfinite, coefs(t))
    end
  end
end

# ---------------------------------------------------------------------------------------------
# T12: the fits do not write to their arguments, and do not return them
# ---------------------------------------------------------------------------------------------
@testset "inputs are not mutated or aliased" begin
  w = 0.5 .+ rand(Xoshiro(1), n)
  off = 0.1 .* randn(Xoshiro(2), n)
  pf = [0.5; ones(p - 1)]
  gw = [1.0, 2.0, 1.0, 1.0]
  folds = repeat(1:3, 20)
  Xs, ys = [X, X], [y, y]
  Xd, yd = X[:, 2:end], X[:, 1]
  Xid = Matrix(1.0I, p, p); Yid = Y[1:p, :]
  # (name, input arrays, fit, result fields that may hold an input); the identity design is the one a
  # solver could return by reference. lasso keeps `weights` and group_lasso `groupweights` as given.
  cases = (
    ("lasso", (X, y, w, off, pf), (X, y, w, off, pf) -> lasso(X, y, 0.05, pf; intercept = true, weights = w, offset = off), (:weights,)),
    ("sqrt_lasso", (X, y), (X, y) -> sqrt_lasso(X, y, 0.3; intercept = true)),
    ("glm_lasso", (X, yb, w, off), (X, yb, w, off) -> glm_lasso(X, yb, 0.02; weights = w, offset = off)),
    ("group_lasso", (X, y, groups, gw), (X, y, g, gw) -> group_lasso(X, y, 0.05, g; groupweights = gw), (:groupweights,)),
    ("sparse_group_lasso", (X, y, groups), (X, y, g) -> sparse_group_lasso(X, y, 0.05, g)),
    ("multi_lasso", (X, Y), (X, Y) -> multi_lasso(X, Y, 0.05)),
    ("multitask_lasso", (Xs, ys, pf), (Xs, ys, pf) -> multitask_lasso(Xs, ys, 0.05; penalty_factors = pf)),
    ("dirty_multitask", (X, Y), (X, Y) -> dirty_multitask(X, Y, 0.1, 0.05)),
    ("transfer_multitask", (X, Y), (X, Y) -> transfer_multitask((X, Y), [(X, Y)], 0.1)),
    ("slope", (X, y), (X, y) -> slope(X, y; intercept = true)),
    ("fused_lasso", (X, y), (X, y) -> fused_lasso(X, y, 0.05, 0.05)),
    ("trend_filter", (X, y), (X, y) -> trend_filter(Matrix(1.0I, n, n), y, 0.5)),
    ("nuclear_regression", (X, Y), (X, Y) -> nuclear_regression(X, Y, 0.05)),
    ("nuclear_regression identity design", (Xid, Yid), (X, Y) -> nuclear_regression(X, Y, 0.01)),
    ("reduced_rank", (X, Y), (X, Y) -> reduced_rank(X, Y, 1)),
    ("sparse_reduced_rank", (X, Y), (X, Y) -> sparse_reduced_rank(X, Y, 1, 0.05)),
    ("quantile_lasso", (X, y, pf), (X, y, pf) -> quantile_lasso(X, y, 0.05; penalty_factors = pf)),
    ("rlasso", (X, y), (X, y) -> rlasso(X, y)),
    ("double_selection", (y, Xd, yd), (y, Xd, yd) -> double_selection(y, yd, Xd)),
    ("CVLassoPath user folds", (X, y, folds), (X, y, folds) -> CVLassoPath(X, y; folds = folds, threads = false)),
  )
  for (name, args, fit, exempt...) in cases
    @testset "$name" begin
      skip = isempty(exempt) ? () : only(exempt)
      before = deepcopy(args)
      s = fit(args...)
      @test args == before
      # no array held by the result is one of the inputs (or an element of an input list)
      inputs = Any[]
      for a in args
        push!(inputs, a)
        a isa AbstractVector{<:AbstractMatrix} && append!(inputs, a)
      end
      for f in propertynames(s)
        f in skip && continue
        v = getproperty(s, f)
        v isa AbstractArray && @test !any(a -> a === v, inputs)
      end
    end
  end
end

# ---------------------------------------------------------------------------------------------
# T13: a SparseMatrixCSC design equals the dense design
# ---------------------------------------------------------------------------------------------
@testset "sparse design equals dense design" begin
  r = Xoshiro(13)
  Xsp = sprandn(r, n, p, 0.4)
  Xdn = Matrix(Xsp)
  ysp = Xdn[:, 1] .+ 0.5 .* randn(r, n)
  Ysp = Xdn[:, 1:2] * randn(r, 2, q) .+ 0.3 .* randn(r, n, q)
  # the two paths differ only in the order of floating point sums; the solver tolerance is 1e-7,
  # so agreement to 1e-7 relative (the fits have norm O(1)) is the contract, not 1e-12
  cases = (
    ("multi_lasso", X -> multi_lasso(X, Ysp, 0.02)),
    ("nuclear_regression", X -> nuclear_regression(X, Ysp, 0.02)),
    ("dirty_multitask", X -> dirty_multitask(X, Ysp, 0.1, 0.05)),
    ("fused_lasso", X -> fused_lasso(X, ysp, 0.05, 0.05)),
    ("lowrank_plus_sparse", X -> lowrank_plus_sparse(X, Ysp, 0.05, 0.05)),
    ("transfer_multitask", X -> transfer_multitask((X, Ysp), [(X, Ysp)], 0.1)),
    ("sparse_nuclear_regression", X -> sparse_nuclear_regression(X, Ysp, 0.05, 0.05)),
  )
  for (name, fit) in cases
    @testset "$name" begin
      a, b = fit(Xsp), fit(Xdn)
      @test norm(coefs(a) .- coefs(b)) <= 1e-7 * max(1, norm(coefs(b)))
      hasproperty(a, :converged) && @test all(a.converged) == all(b.converged)
    end
  end
end

# ---------------------------------------------------------------------------------------------
# T5: NaN, Inf, and out-of-domain data are rejected with an ArgumentError
# ---------------------------------------------------------------------------------------------
@testset "non-finite and out-of-domain data are rejected" begin
  bad(A, idx...; v = NaN) = (B = float.(A); B[idx...] = v; B)
  Xn, Xi = bad(X, 2, 3), bad(X, 5, 2; v = Inf)
  yn, Yi = bad(y, 4), bad(Y, 3, 2; v = Inf)
  ybn = bad(yb, 4)
  w1 = ones(n)
  cases = (
    ("glm y NaN", () -> glm_lasso(X, ybn, 0.05)),
    ("glm weights NaN", () -> glm_lasso(X, yb, 0.05; weights = bad(w1, 1))),
    ("glm weights negative", () -> glm_lasso(X, yb, 0.05; weights = bad(w1, 1; v = -1.0))),
    ("glm offset NaN", () -> glm_lasso(X, yb, 0.05; offset = fill(NaN, n))),
    ("glm Poisson y < 0", () -> glm_lasso(X, bad(round.(abs.(y)), 3; v = -1.0), 0.05, ProximalBase.Poisson())),
    ("glm logistic y > 1", () -> glm_lasso(X, 2 .* yb, 0.05)),
    ("glm logistic y < 0", () -> glm_lasso(X, yb .- 1, 0.05)),
    ("group_lasso X", () -> group_lasso(Xn, y, 0.05, groups)),
    ("group_lasso y", () -> group_lasso(X, yn, 0.05, groups)),
    ("sparse_group_lasso y", () -> sparse_group_lasso(X, yn, 0.05, groups)),
    ("multi_lasso X", () -> multi_lasso(Xn, Y, 0.05)),
    ("multi_lasso Y", () -> multi_lasso(X, Yi, 0.05)),
    ("dirty_multitask X", () -> dirty_multitask(Xn, Y, 0.1, 0.05)),
    ("dirty_multitask Y", () -> dirty_multitask(X, Yi, 0.1, 0.05)),
    ("nuclear_regression Y", () -> nuclear_regression(X, Yi, 0.05)),
    ("reduced_rank Y", () -> reduced_rank(X, Yi, 1)),
    ("trend_filter y", () -> trend_filter(Matrix(1.0I, n, n), bad(y, 4), 0.5)),
    ("quantile_lasso X", () -> quantile_lasso(Xn, y, 0.05)),
    ("quantile_lasso y", () -> quantile_lasso(X, yn, 0.05)),
    ("CVLassoPath y", () -> CVLassoPath(X, yn)),
  )
  for (name, f) in cases
    @testset "$name" begin
      @test_throws ArgumentError f()
    end
  end
end

end # module
