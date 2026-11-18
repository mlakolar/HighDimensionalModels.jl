module GroupLassoTest
# `group_lasso`: the objective, for disjoint nonempty groups Gₖ that cover the columns, weights wₖ
# and X̃ the columns of X centred with an intercept (X otherwise),
#   ‖y - β₀1 - Xβ‖² / (2n) + λ Σₖ wₖ ‖X̃_Gₖ β_Gₖ‖₂ / √n      (standardize = :orthonormal)
#   ‖y - β₀1 - Xβ‖² / (2n) + λ Σₖ wₖ ‖β_Gₖ‖₂                  (standardize = :none).
# The references share nothing with the orthonormalization or with block coordinate descent:
#   * the group optimality conditions in 256-bit BigFloat, in the coordinates β, from the returned
#     β₀ and β, with the projectors onto the groups built from the columns known to be independent
#     (normal equations), so that they do not use the QR of the package;
#   * least squares at λ = 0 and the closed form of a single group, `(1 - √n λ w/‖P_G ỹ‖)₊ P_G ỹ`,
#     and its decoupling over groups that are orthogonal after centring;
#   * λmax from the data, with the groups of weight zero profiled out;
#   * invariance of the fit to rescaling or recombining the columns of a group, asserted only
#     where the computed rank and the condition number of the transformation permit it;
#   * the rows of the numerical regression suite of PR #215 that apply to group_lasso.
# The tolerances of the optimality conditions are formulas stated at `kkt` below. The rank decision
# (`HighDimensionalModels.group_basis`, with its provisional tolerance) is exercised only on cases
# whose diagonal entries of R are far from any plausible threshold: a dependent column
# (|Rᵢᵢ|/|R₁₁| of the order of eps(T)) and genuine directions (at least 1e-2 relative).

using HighDimensionalModels, Test, LinearAlgebra, Random
using ProximalBase: BlockIterate
using ProximalSolvers: ProximalSolvers

const HDM = HighDimensionalModels

# ---------------------------------------------------------------------------------
# settings and data
# ---------------------------------------------------------------------------------

tolof(T) = T === Float32 ? 1e-6 : 1e-10
cyc(T; tol = tolof(T), maxpasses = 20_000) = CDSolver(order = :cyclic, tol = tol, maxpasses = maxpasses)

# Gaussian columns with different scales and means; the response uses the columns `active`
function problem(T, seed, n, p; active = (1, 3), noise = 0.5)
  rng = Xoshiro(seed)
  scales = 0.5 .+ 1.5 .* rand(rng, p)
  X = randn(rng, n, p) .* transpose(scales) .+ transpose(randn(rng, p))
  β = zeros(p)
  for (i, j) in enumerate(active)
    β[j] = (-1)^i * (1 + 0.5i)
  end
  T.(X), T.(1.0 .+ X * β .+ noise .* randn(rng, n))
end

# ---------------------------------------------------------------------------------
# references in extended precision (call inside setprecision(BigFloat, 256))
# ---------------------------------------------------------------------------------

bigcentred(A, intercept) = intercept ? A .- sum(A; dims = 1) ./ size(A, 1) : A
# the projection of v on the span of the columns of A, which have full column rank: normal equations
bigproj(A, v) = A * ((A' * A) \ (A' * v))

# The group optimality conditions of the objective at the returned point, from the stored inputs and
# the returned β₀, β, λ and weights. With `r = y - β₀ - Xβ` (so `Σr = 0` with an intercept, up to
# the rounding of β₀ in T) and `g = X̃_Gᵀr/n`, for the group G of weight w:
#   :orthonormal, β_G ≠ 0:  g = λ w X̃_GᵀX̃_Gβ_G / (√n ‖X̃_Gβ_G‖);
#   :orthonormal, β_G = 0:  ‖P_G r‖ ≤ √n λ w, with P_G the projector on the span of the columns
#                            `indep[k]` of X̃ (known to be independent);
#   :none, β_G ≠ 0:         g = λ w β_G/‖β_G‖;
#   :none, β_G = 0:         ‖g‖ ≤ λ w.
# Returns, for each group, the size `res` of the violation (the norm of the difference of the two sides
# of the equation, or `max(‖P_G r‖/√n - λ w, 0)` and `max(‖g‖ - λ w, 0)` for a zero group) and the
# tolerance `tol`, with ε = eps(T), `stol` the tolerance of the solver, K the number of groups,
# ys = ‖ỹ‖/√n, smax = ‖X̃_G‖₂/√n and κ = cond(X̃_G):
#   active, :orthonormal:  8 smax (K stol ys + κ ε (ys + λw + smax ‖β_G‖));
#   active, :none:         8 smax (K stol ys + ε (ys + λw + smax ‖β_G‖));
#   zero, :orthonormal:    8 (K stol ys + ε (ys + λw));
#   zero, :none:           8 smax (K stol ys + ε (ys + λw)).
# The first term is the stopping rule of the solver for K blocks coupled by correlations (the change of
# a block, in the units of ‖ỹ‖/√n, is below stol), the others are rounding in T, with the factor κ
# for the solve `β = R̃ \ θ` that maps the solver's coordinates back (cond(R̃) = κ).
# Also returns `sumres = |Σr|/n` against `sumtol = 4 (p + 2) ε (|ȳ| + Σⱼ|x̄ⱼ βⱼ|)` with an intercept.
function kkt(X::Matrix{T}, y::Vector{T}, fit, stol; indep = fit.groups) where {T}
  setprecision(BigFloat, 256) do
    n, p = size(X)
    Xb, yb = BigFloat.(X), BigFloat.(y)
    ic = fit.hasintercept
    Xc = bigcentred(Xb, ic)
    ỹ = ic ? yb .- sum(yb) / n : yb
    β, β₀, λ = BigFloat.(fit.coefficients), BigFloat(fit.intercept), BigFloat(fit.λ)
    r = yb .- β₀ .- Xb * β
    ε = BigFloat(eps(T))
    ys = norm(ỹ) / sqrt(BigFloat(n))
    K = length(fit.groups)
    rows = map(1:K) do k
      G, w = fit.groups[k], BigFloat(fit.groupweights[k])
      A, bG = Xc[:, G], β[G]
      A64 = Float64.(A)
      smax, κ = opnorm(A64) / sqrt(n), cond(A64)
      g = A' * r / n
      active = !iszero(bG)
      orth = fit.standardize === :orthonormal
      if orth && active
        c = A * bG
        res = norm(g .- λ * w .* (A' * c) ./ (sqrt(BigFloat(n)) * norm(c)))
        tol = 8 * smax * (K * stol * ys + κ * ε * (ys + λ * w + smax * norm(bG)))
      elseif orth
        res = max(norm(bigproj(Xc[:, indep[k]], r)) / sqrt(BigFloat(n)) - λ * w, 0)
        tol = 8 * (K * stol * ys + ε * (ys + λ * w))
      elseif active
        res = norm(g .- λ * w .* bG ./ norm(bG))
        tol = 8 * smax * (K * stol * ys + ε * (ys + λ * w + smax * norm(bG)))
      else
        res = max(norm(g) - λ * w, 0)
        tol = 8 * smax * (K * stol * ys + ε * (ys + λ * w))
      end
      (; group = k, active, res = Float64(res), tol = Float64(tol))
    end
    x̄ = ic ? vec(sum(Xb; dims = 1) ./ n) : zeros(BigFloat, p)
    ȳ = ic ? sum(yb) / n : zero(BigFloat)
    sumtol = ic ? 4 * (p + 2) * ε * (abs(ȳ) + sum(abs.(x̄ .* β))) : zero(BigFloat)
    (; rows, sumres = Float64(abs(sum(r)) / n), sumtol = Float64(sumtol))
  end
end

# the objective of group_lasso at (β₀, β) in BigFloat, with the weights of the fit
function objective(X, y, fit)
  setprecision(BigFloat, 256) do
    n = size(X, 1)
    Xb, yb = BigFloat.(X), BigFloat.(y)
    Xc = bigcentred(Xb, fit.hasintercept)
    β = BigFloat.(fit.coefficients)
    r = yb .- BigFloat(fit.intercept) .- Xb * β
    pen = sum(BigFloat(fit.groupweights[k]) *
              (fit.standardize === :orthonormal ? norm(Xc[:, G] * β[G]) / sqrt(BigFloat(n)) : norm(β[G]))
              for (k, G) in enumerate(fit.groups))
    sum(abs2, r) / (2n) + BigFloat(fit.λ) * pen
  end
end

# λmax with the groups of weight zero profiled out: the smallest λ at which every group of positive
# weight is zero. With U the span of the groups of weight zero and r₀ = ỹ - P_U ỹ (ỹ itself if there is
# none): max over the groups of positive weight of ‖P_G r₀‖/(√n w) (:orthonormal) or ‖X̃_Gᵀr₀‖/(n w) (:none).
function lambda_max(X, y, groups, w, intercept, standardize)
  setprecision(BigFloat, 256) do
    n = size(X, 1)
    Xc = bigcentred(BigFloat.(X), intercept)
    yb = BigFloat.(y)
    ỹ = intercept ? yb .- sum(yb) / n : yb
    free = reduce(vcat, [G for (k, G) in enumerate(groups) if iszero(w[k])]; init = Int[])
    r₀ = isempty(free) ? ỹ : ỹ .- bigproj(Xc[:, free], ỹ)
    top = zero(BigFloat)
    for (k, G) in enumerate(groups)
      iszero(w[k]) && continue
      v = standardize === :orthonormal ? norm(bigproj(Xc[:, G], r₀)) / (sqrt(BigFloat(n)) * w[k]) :
                                         norm(Xc[:, G]' * r₀) / (n * w[k])
      top = max(top, v)
    end
    Float64(top)
  end
end

# the fitted values y - residuals of a fit, in Float64
fitted(y, fit) = Float64.(y) .- Float64.(fit.residuals)

# The rank (the number of columns of Z) that `group_lasso` computes for the columns G of X
computed_rank(X, G; intercept = true) =
  size(HDM.group_basis(HDM.preprocess(X, zeros(eltype(X), size(X, 1)), intercept, false).Z[:, G]).Z, 2)

# The centred columns G of X as a Float64 matrix (centred in BigFloat from the stored entries)
centred64(X, G; intercept = true) =
  setprecision(BigFloat, 256) do
    Float64.(bigcentred(BigFloat.(X[:, G]), intercept))
  end

# The condition number of the centred columns G (a Float64 computation, for a group of full rank)
cond_centred(X, G; intercept = true) = cond(centred64(X, G; intercept))

# The smallest singular value of the centred columns G
smin_centred(X, G; intercept = true) = minimum(svdvals(centred64(X, G; intercept)))

# The tolerance for the fitted values of a fit of a design with condition number κ (of the centred
# design) and K groups, against the exact minimizer or another fit. The fitted values are determined
# up to n‖κ_res‖/σmin(X̃) for a violation κ_res of the optimality conditions, which is of the order of
# smax K stol ys (see `kkt`), so that the difference in norm is of the order of κ K stol ‖ỹ‖, plus the
# rounding κ ε ‖ỹ‖ of the data and of the back-map. With a factor of 16 for the comparison of two fits and a margin:
#   tolfit = 16 κ (K stol + ε) ‖ỹ‖.
# Where the block steps are exact (one group, or groups orthogonal to one another), a pass solves the
# problem up to rounding and the stopping rule does not enter:
#   tolexact = 16 κ ε ‖ỹ‖.
tolfit(T, κ, K, ynorm) = 16 * κ * (K * tolof(T) + Float64(eps(T))) * ynorm
tolexact(T, κ, ynorm) = 16 * κ * Float64(eps(T)) * ynorm

centred_norm(y) = (v = Float64.(y); norm(v .- sum(v) / length(v)))

# ---------------------------------------------------------------------------------

@testset "group_lasso" begin

  # 1. the group optimality conditions of the objective in BigFloat, in the coordinates β
  @testset "group optimality conditions in BigFloat, $T, $std" for T in (Float32, Float64), std in (:orthonormal, :none)
    X, y = problem(T, 1, 50, 8)
    stol = tolof(T)
    seen_active, seen_zero = false, false
    for groups in ([[1, 2], [3, 4, 5], [6, 7, 8]], [[1, 4], [2, 3, 6], [5, 7, 8]]), λ in (T(0.15), T(0.4))
      fit = group_lasso(X, y, λ, groups, cyc(T); standardize = std)
      @test fit.result.converged
      k = kkt(X, y, fit, stol)
      for row in k.rows
        @test row.res ≤ row.tol
        seen_active |= row.active
        seen_zero |= !row.active
      end
      @test k.sumres ≤ k.sumtol
      @test fit.activegroups == [row.group for row in k.rows if row.active]
    end
    @test seen_active && seen_zero                # the equations and the inequalities were checked
  end

  # the minimum reported by the solver is the objective of the fit, up to rounding
  @testset "result.minimum is the objective, $T, $std" for T in (Float32, Float64), std in (:orthonormal, :none)
    X, y = problem(T, 2, 40, 6)
    groups = [[1, 2], [3, 4], [5, 6]]
    fit = group_lasso(X, y, T(0.1), groups, cyc(T); standardize = std)
    κ = cond_centred(X, 1:6)
    @test fit.result isa ProximalSolvers.CDResults && fit.result.minimizer isa BlockIterate
    @test Float64(fit.result.minimum) ≈ Float64(objective(X, y, fit)) rtol = 64 * κ * eps(T)
  end

  # 2. exact references
  @testset "exact references, $T" for T in (Float32, Float64)
    stol, ε = tolof(T), Float64(eps(T))
    @testset "λ = 0 is least squares on the span of the centred columns, $std" for std in (:orthonormal, :none)
      X, y = problem(T, 3, 40, 6)
      groups = [[1, 2], [3, 4, 5], [6]]
      fit = group_lasso(X, y, 0, groups, cyc(T); standardize = std)
      ref = setprecision(BigFloat, 256) do
        Xb, yb = BigFloat.(X), BigFloat.(y)
        ȳ = sum(yb) / length(yb)
        Float64.(ȳ .+ bigproj(bigcentred(Xb, true), yb .- ȳ))
      end
      @test fit.result.converged
      @test norm(fitted(y, fit) - ref) ≤ tolfit(T, cond_centred(X, 1:6), 3, centred_norm(y))
    end

    @testset "one group: (1 - √n λ w/‖P_G ỹ‖)₊ P_G ỹ" begin
      X, y = problem(T, 4, 40, 5)
      groups = [collect(1:5)]
      n = size(X, 1)
      for factor in (0.3, 0.8, 1.5)              # the last is above the threshold, where the fit is zero
        setprecision(BigFloat, 256) do
          Xb, yb = BigFloat.(X), BigFloat.(y)
          ȳ = sum(yb) / n
          Xc, ỹ = bigcentred(Xb, true), yb .- ȳ
          Pỹ = bigproj(Xc, ỹ)
          w = sqrt(BigFloat(5))
          λ = T(factor * Float64(norm(Pỹ) / (sqrt(BigFloat(n)) * w)))
          fit = group_lasso(X, y, λ, groups, cyc(T))
          shrink = max(1 - sqrt(BigFloat(n)) * BigFloat(λ) * w / norm(Pỹ), 0)
          ref = Float64.(ȳ .+ shrink .* Pỹ)
          @test norm(fitted(y, fit) - ref) ≤ tolexact(T, cond_centred(X, 1:5), centred_norm(y))
          @test (factor > 1) == iszero(fit.coefficients)
        end
      end
    end

    @testset "a group with more columns than rows" begin
      # n = 10 rows and 15 columns in one group: the centred columns span the n - 1 = 9 dimensional space of the
      # vectors that sum to 0, so the rank is 9 and P_G ỹ = ỹ; the fit is (1 - √n λ w/‖ỹ‖)₊ ỹ plus the mean
      rng = Xoshiro(21)
      X, y = T.(randn(rng, 10, 15) .+ 0.5), T.(randn(rng, 10))
      fit = group_lasso(X, y, T(0.02), [collect(1:15)], cyc(T))
      @test computed_rank(X, collect(1:15)) == 9 && count(!iszero, fit.coefficients) ≤ 9
      s = svdvals(centred64(X, 1:15))
      ref = setprecision(BigFloat, 256) do
        yb = BigFloat.(y)
        ȳ = sum(yb) / 10
        Float64.(ȳ .+ max(1 - sqrt(BigFloat(10)) * BigFloat(T(0.02)) * sqrt(BigFloat(15)) / norm(yb .- ȳ), 0) .* (yb .- ȳ))
      end
      @test norm(fitted(y, fit) - ref) ≤ tolexact(T, s[1] / s[9], centred_norm(y))
    end

    @testset "groups orthogonal after centring decouple" begin
      n, sizes = 48, (2, 3, 2)
      rng = Xoshiro(5)
      A = randn(rng, n, sum(sizes))
      A .-= sum(A; dims = 1) ./ n
      Q = Matrix(qr(A).Q)[:, 1:sum(sizes)]            # orthonormal and orthogonal to the ones vector
      X = T.(sqrt(n) .* Q .* transpose([1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.7]) .+ transpose([0.3, -1.0, 2.0, 0.0, 1.0, -0.5, 0.2]))
      y = T.(1.0 .+ X[:, 1] .- 0.7 .* X[:, 3] .+ 0.4 .* randn(rng, n))
      groups = [[1, 2], [3, 4, 5], [6, 7]]
      for λ in (T(0.1), T(0.35))
        fit = group_lasso(X, y, λ, groups, cyc(T))
        ref = setprecision(BigFloat, 256) do
          Xb, yb = BigFloat.(X), BigFloat.(y)
          ȳ = sum(yb) / n
          Xc, ỹ = bigcentred(Xb, true), yb .- ȳ
          total = fill(ȳ, n)
          for G in groups
            Pỹ = bigproj(Xc[:, G], ỹ)
            total .+= max(1 - sqrt(BigFloat(n)) * BigFloat(λ) * sqrt(BigFloat(length(G))) / norm(Pỹ), 0) .* Pỹ
          end
          Float64.(total)
        end
        @test norm(fitted(y, fit) - ref) ≤ tolexact(T, 1.0, centred_norm(y)) + 16 * ε * sqrt(3) * centred_norm(y)
        @test 0 < length(fit.activegroups) < 3
      end
    end
  end

  # 3. λmax, with the groups of weight zero profiled out
  @testset "λmax, $T, $std, $kind" for T in (Float32, Float64), std in (:orthonormal, :none), kind in (:default, :zero_weight)
    X, y = problem(T, 6, 40, 8; active = (1, 4, 7))
    groups = [[1, 2], [3, 4, 5], [6, 7, 8]]
    w = kind === :default ? sqrt.(Float64.(length.(groups))) : [0.0, sqrt(3), sqrt(3)]
    top = lambda_max(X, y, groups, w, true, std)
    margin = T === Float32 ? 1e-3 : 1e-9
    kw = kind === :default ? (;) : (; groupweights = w)
    above = group_lasso(X, y, T(top * (1 + margin)), groups, cyc(T); standardize = std, kw...)
    penalized = [k for k in eachindex(groups) if w[k] > 0]
    @test all(k -> all(iszero, above.coefficients[groups[k]]), penalized)
    kind === :zero_weight && @test above.activegroups == [1]       # the unpenalized group stays active
    below = group_lasso(X, y, T(top * (1 - 1e-3)), groups, cyc(T); standardize = std, kw...)
    @test any(k -> k in below.activegroups, penalized)
    # the zero fit is exact above λmax: the residuals are those of the profiled-out least squares
    kind === :default && @test above.intercept == HDM.column_mean(y) && above.residuals == y .- above.intercept
  end

  # 4. invariance of the orthonormal fit, where the computed rank is preserved and the transformation
  #    is well conditioned
  @testset "invariance under :orthonormal, $T" for T in (Float32, Float64)
    X, y = problem(T, 7, 50, 8)
    groups = [[1, 2, 3], [4, 5, 6], [7, 8]]
    λ = T(0.12)
    base = group_lasso(X, y, λ, groups, cyc(T))
    @test 1 in base.activegroups && length(base.activegroups) < 3
    κ = cond_centred(X, 1:8)
    ynorm = centred_norm(y)
    @testset "rescaling a column by $s" for s in (T(4), T(0.25))
      Xs = copy(X)
      Xs[:, 2] .*= s
      @test computed_rank(Xs, [1, 2, 3]) == computed_rank(X, [1, 2, 3]) == 3
      @test cond(Float64.(Diagonal([1, s, 1]))) ≤ 10
      fit = group_lasso(Xs, y, λ, groups, cyc(T))
      @test norm(fitted(y, fit) - fitted(y, base)) ≤ tolfit(T, κ * 4, 3, ynorm)
      # only the coefficient of the column changes, by the factor 1/s: the coefficients of the
      # scaled design, multiplied by the column scales, are those of the original design, to the
      # fitted-value tolerance divided by the smallest singular value of the centred design
      back = copy(fit.coefficients)
      back[2] *= s
      @test norm(back - base.coefficients) ≤ tolfit(T, κ * 4, 3, ynorm) / (smin_centred(X, 1:8) / 4)
    end
    @testset "recombining the columns of a group" begin
      rng = Xoshiro(11)
      M = Matrix{Float64}(I, 3, 3) .+ 0.4 .* randn(rng, 3, 3)
      @test cond(M) ≤ 10
      Xr = copy(X)
      Xr[:, 1:3] = X[:, 1:3] * T.(M)
      @test computed_rank(Xr, [1, 2, 3]) == computed_rank(X, [1, 2, 3]) == 3
      fit = group_lasso(Xr, y, λ, groups, cyc(T))
      @test norm(fitted(y, fit) - fitted(y, base)) ≤ tolfit(T, κ * cond(M), 3, ynorm)
      @test fit.activegroups == base.activegroups
      # the coefficients of the group transform by M: M β_new = β_old on the group, and are equal elsewhere
      back = copy(fit.coefficients)
      back[1:3] = M * back[1:3]
      @test norm(back - base.coefficients) ≤ tolfit(T, κ * cond(M), 3, ynorm) / smin_centred(X, 1:8)
    end
  end

  # 5. a redundant column: pivoted QR does not decide which coefficient of the group is zero, so the
  #    assertions are on what does not depend on the pivots; the weights are given, since the default
  #    √|G| changes when a column is added
  @testset "a redundant column, $T" for T in (Float32, Float64)
    X0, y = problem(T, 8, 50, 6; active = (1, 4))
    a = T[0.5, -1.0, 2.0]
    d = X0[:, 1:3] * a                     # a combination of the columns 1:3, formed in T
    w = [sqrt(3.0), sqrt(3.0)]
    λ = T(0.12)
    base = group_lasso(X0, y, λ, [[1, 2, 3], [4, 5, 6]], cyc(T); groupweights = w)
    @test 1 in base.activegroups
    κ, ynorm = cond_centred(X0, 1:6), centred_norm(y)
    contribution(X, fit, k) = centred64(X, fit.groups[k]) * Float64.(fit.coefficients[fit.groups[k]])
    groups = [[1, 2, 3, 4], [5, 6, 7]]
    placements = (hcat(X0[:, 1:3], d, X0[:, 4:6]), hcat(d, X0[:, 1:3], X0[:, 4:6]))   # d last and first
    @testset "$where, :orthonormal" for (where, X1) in zip(("d last in its group", "d first in its group"), placements)
      @test computed_rank(X1, groups[1]) == 3                      # the constructed rank
      @test computed_rank(X0, [1, 2, 3]) == 3
      fit = group_lasso(X1, y, λ, groups, cyc(T); groupweights = w)
      @test fit.result.converged
      @test fit.groupweights == T.(w)
      @test norm(fitted(y, fit) - fitted(y, base)) ≤ tolfit(T, κ, 2, ynorm)
      @test norm(contribution(X1, fit, 1) - contribution(X0, base, 1)) ≤ tolfit(T, κ, 2, ynorm)
      @test norm(contribution(X1, fit, 2) - contribution(X0, base, 2)) ≤ tolfit(T, κ, 2, ynorm)
      @test count(!iszero, fit.coefficients[groups[1]]) ≤ 3         # at most the rank
      @test fit.activegroups == base.activegroups
    end
    # :none has no such equivalence with the group without the column: the penalty ‖β_G‖ of the
    # group with the column is √(b'Mb) for the contribution b = X₀(β₀ + a β_d), with M = I - aa'/(1 + ‖a‖²)
    # (the least norm of β over the splits of b between the columns and the added one). So the fitted values
    # are those of the problem with the penalty ‖β_G‖ on the design X₀ M^(-1/2) = X₀ (I + (s - 1) ââ'), where
    # â = a/‖a‖ and s = √(1 + ‖a‖²)
    @testset ":none, d last in its group" begin
      a64 = Float64.(a)
      â, s = a64 ./ norm(a64), sqrt(1 + sum(abs2, a64))
      Mred = Matrix{Float64}(I, 3, 3) .+ (s - 1) .* (â * â')
      Xr = copy(X0)
      Xr[:, 1:3] = X0[:, 1:3] * T.(Mred)
      ref = group_lasso(Xr, y, λ, [[1, 2, 3], [4, 5, 6]], cyc(T); standardize = :none, groupweights = w)
      fit = group_lasso(placements[1], y, λ, groups, cyc(T); standardize = :none, groupweights = w)
      @test 1 in ref.activegroups
      @test fit.result.converged && ref.result.converged
      @test norm(fitted(y, fit) - fitted(y, ref)) ≤ tolfit(T, κ * s, 2, ynorm)
    end
  end

  # 6. constant and zero data
  @testset "constant and zero data, $T" for T in (Float32, Float64)
    n = 40
    X, y = problem(T, 9, n, 4)
    λ = T(0.1)
    c = fill(T(0.1), n)
    @testset "a lone constant column, $std" for std in (:orthonormal, :none)
      X1 = hcat(X[:, 1:2], c, X[:, 3:4])
      fit = group_lasso(X1, y, λ, [[1, 2], [3], [4, 5]], cyc(T); standardize = std)
      ref = group_lasso(X, y, λ, [[1, 2], [3, 4]], cyc(T); standardize = std)
      std === :orthonormal && @test computed_rank(X1, [3]) == 0
      @test fit.coefficients[3] == 0 && !(2 in fit.activegroups)
      # the zero column changes the schedule of the passes, not the problem: the fit is that of the design
      # without the column, to the tolerance of the fitted values
      κ, ynorm = cond_centred(X, 1:4), centred_norm(y)
      @test norm(fitted(y, fit) - fitted(y, ref)) ≤ tolfit(T, κ, 3, ynorm)
      @test norm(fit.coefficients[[1, 2, 4, 5]] - ref.coefficients) ≤ tolfit(T, κ, 3, ynorm) / smin_centred(X, 1:4)
      @test abs(fit.intercept - ref.intercept) ≤ 16 * 6 * eps(T) * norm(y) + tolfit(T, κ, 3, ynorm) / sqrt(n) * 8
      @test fit.result.converged
      # the group stays in the fit as one column of zeros
      std === :orthonormal && @test length(fit.result.minimizer) == 5
    end
    @testset "a group of constant and zero columns" begin
      X2 = hcat(X[:, 1:2], c, fill(T(3), n), zeros(T, n))
      fit = group_lasso(X2, y, λ, [[1, 2], [3, 4, 5]], cyc(T))
      @test computed_rank(X2, [3, 4, 5]) == 0
      @test all(iszero, fit.coefficients[3:5]) && fit.activegroups == [1]
      # zero columns are rank zero without an intercept too
      fit0 = group_lasso(hcat(X[:, 1:2], zeros(T, n, 2)), y, λ, [[1, 2], [3, 4]], cyc(T); intercept = false)
      @test all(iszero, fit0.coefficients[3:4]) && fit0.activegroups == [1]
    end
    @testset "no intercept: a constant column is an ordinary column" begin
      yc = T(2) .* c .+ T(0.05) .* X[:, 1]
      fit = group_lasso(hcat(c, X[:, 2:3]), yc, T(0.02), [[1], [2, 3]], cyc(T); intercept = false)
      @test computed_rank(hcat(c, X[:, 2:3]), [1]; intercept = false) == 1
      @test fit.coefficients[1] != 0 && 1 in fit.activegroups
      @test fit.intercept == 0 && !fit.hasintercept
    end
    @testset "zero design, zero response, constant response" begin
      for std in (:orthonormal, :none), ic in (true, false)
        zero_design = group_lasso(zeros(T, n, 4), y, λ, [[1, 2], [3, 4]], cyc(T); standardize = std, intercept = ic)
        @test all(iszero, zero_design.coefficients) && isempty(zero_design.activegroups) && zero_design.result.converged
        @test zero_design.intercept == (ic ? HDM.column_mean(y) : T(0))
        @test zero_design.residuals == y .- zero_design.intercept
        zero_response = group_lasso(X, zeros(T, n), λ, [[1, 2], [3, 4]], cyc(T); standardize = std, intercept = ic)
        @test all(iszero, zero_response.coefficients) && zero_response.result.converged
        @test iszero(zero_response.intercept) && all(iszero, zero_response.residuals)
      end
      constant_response = group_lasso(X, fill(T(0.1), n), λ, [[1, 2], [3, 4]], cyc(T))
      @test all(iszero, constant_response.coefficients) && constant_response.intercept == T(0.1)
      @test all(iszero, constant_response.residuals)
    end
  end

  # 7. groups
  @testset "noncontiguous groups and every validation failure, $T" for T in (Float32, Float64)
    X, y = problem(T, 10, 40, 5)
    λ = T(0.12)
    @testset "permuting the columns permutes the coefficients, $std" for std in (:orthonormal, :none)
      perm = [1, 4, 2, 3, 5]
      fit = group_lasso(X, y, λ, [[1, 4], [2, 3], [5]], cyc(T); standardize = std)
      fitp = group_lasso(X[:, perm], y, λ, [[1, 2], [3, 4], [5]], cyc(T); standardize = std)
      @test fit.coefficients[perm] == fitp.coefficients
      @test fit.activegroups == fitp.activegroups
      # the residuals and the intercept sum the same terms in another order
      @test norm(fit.residuals - fitp.residuals) ≤ 16 * 7 * eps(T) * norm(y)
      @test abs(fit.intercept - fitp.intercept) ≤ 16 * 7 * eps(T) * norm(y)
      @test fit.groups == [[1, 4], [2, 3], [5]]
      # ranges and vectors of integers of other types are groups too
      @test group_lasso(X, y, λ, [1:2, 3:5], cyc(T); standardize = std).coefficients ==
            group_lasso(X, y, λ, [[1, 2], [3, 4, 5]], cyc(T); standardize = std).coefficients
      @test group_lasso(X, y, λ, [Int8[1, 2], UInt16[3, 4, 5]], cyc(T); standardize = std).groups == [[1, 2], [3, 4, 5]]
    end
    groups = [[1, 2], [3], [4, 5]]
    messages = String[]
    function fails(f, E)
      err = try f(); nothing catch e; e end
      @test err isa E
      push!(messages, err isa E ? err.msg : "")
      err isa E ? err.msg : ""
    end
    @test occursin("column 2 is in the groups 1 and 2", fails(() -> group_lasso(X, y, λ, [[1, 2], [2, 3], [4, 5]], cyc(T)), ArgumentError))
    @test occursin("group 1 repeats the column 1", fails(() -> group_lasso(X, y, λ, [[1, 1, 2], [3], [4, 5]], cyc(T)), ArgumentError))
    @test occursin("[3, 4, 5] are in no group", fails(() -> group_lasso(X, y, λ, [[1, 2]], cyc(T)), ArgumentError))
    @test occursin("the index 0", fails(() -> group_lasso(X, y, λ, [[0, 1], [2, 3, 4, 5]], cyc(T)), ArgumentError))
    @test occursin("the index 6", fails(() -> group_lasso(X, y, λ, [[1, 2, 3], [4, 5, 6]], cyc(T)), ArgumentError))
    @test occursin("group 1 is empty", fails(() -> group_lasso(X, y, λ, [Int[], [1, 2, 3, 4, 5]], cyc(T)), ArgumentError))
    @test occursin("not an integer", fails(() -> group_lasso(X, y, λ, [[1.0, 2.0], [3, 4, 5]], cyc(T)), ArgumentError))
    @test occursin("not an integer", fails(() -> group_lasso(X, y, λ, [Any[true, 2], [3, 4, 5]], cyc(T)), ArgumentError))
    @test occursin("must be a vector of column indices", fails(() -> group_lasso(X, y, λ, Any[[1, 2], 3, [4, 5]], cyc(T)), ArgumentError))
    @test occursin("2 group weights", fails(() -> group_lasso(X, y, λ, groups, cyc(T); groupweights = [1.0, 1.0]), DimensionMismatch))
    @test occursin("group 2 is -1.0", replace(fails(() -> group_lasso(X, y, λ, groups, cyc(T); groupweights = [1.0, -1.0, 1.0]), ArgumentError), "the weight of " => ""))
    @test occursin("is NaN", fails(() -> group_lasso(X, y, λ, groups, cyc(T); groupweights = [NaN, 1.0, 1.0]), ArgumentError))
    @test occursin("is Inf", fails(() -> group_lasso(X, y, λ, groups, cyc(T); groupweights = [1.0, Inf, 1.0]), ArgumentError))
    T === Float32 && @test occursin("not representable", fails(() -> group_lasso(X, y, λ, groups, cyc(T); groupweights = [1e300, 1.0, 1.0]), ArgumentError))
    @test occursin("vector of real numbers", fails(() -> group_lasso(X, y, λ, groups, cyc(T); groupweights = (1.0, 1.0, 1.0)), ArgumentError))
    @test occursin("standardize must be", fails(() -> group_lasso(X, y, λ, groups, cyc(T); standardize = :foo), ArgumentError))
    @test occursin("standardize must be", fails(() -> group_lasso(X, y, λ, groups, cyc(T); standardize = "orthonormal"), ArgumentError))
    @test occursin("intercept must be true or false", fails(() -> group_lasso(X, y, λ, groups, cyc(T); intercept = 1), ArgumentError))
    @test occursin("y has 39 entries", fails(() -> group_lasso(X, y[1:39], λ, groups, cyc(T)), DimensionMismatch))
    @test occursin("at least one row", fails(() -> group_lasso(zeros(T, 0, 5), T[], λ, groups, cyc(T)), DimensionMismatch))
    @test occursin("at least one column", fails(() -> group_lasso(zeros(T, 5, 0), zeros(T, 5), λ, Vector{Int}[], cyc(T)), DimensionMismatch))
    @test occursin("group_lasso needs", fails(() -> group_lasso(X, y, λ, (1:2, 3:5), cyc(T)), ArgumentError))
    # each kind of failure of the groups has its own message: overlap, repeated index, column in no group,
    # index outside 1:p, empty group, entry that is not an integer, and a group that is not a vector
    @test allunique(messages[[1, 2, 3, 4, 6, 7, 9]])
  end

  # 8. a group of weight zero is unpenalized
  @testset "weight zero, $T, $std" for T in (Float32, Float64), std in (:orthonormal, :none)
    X, y = problem(T, 12, 50, 6; active = (1, 2))
    groups = [[1, 2], [3, 4], [5, 6]]
    w = [0.0, sqrt(2.0), sqrt(2.0)]
    top = lambda_max(X, y, groups, w, true, std)
    ynorm = centred_norm(y)
    # above λmax only the unpenalized group is in the fit: the least squares on its columns
    fit = group_lasso(X, y, T(2top), groups, cyc(T); standardize = std, groupweights = w)
    ref = setprecision(BigFloat, 256) do
      yb = BigFloat.(y)
      ȳ = sum(yb) / length(yb)
      Float64.(ȳ .+ bigproj(bigcentred(BigFloat.(X[:, 1:2]), true), yb .- ȳ))
    end
    @test fit.activegroups == [1] && all(iszero, fit.coefficients[3:6])
    @test norm(fitted(y, fit) - ref) ≤ tolfit(T, cond_centred(X, 1:2), 3, ynorm)
    # below λmax the group stays active, and the optimality conditions hold with the weight 0
    fit = group_lasso(X, y, T(0.5top), groups, cyc(T); standardize = std, groupweights = w)
    @test 1 in fit.activegroups && length(fit.activegroups) ≥ 2
    k = kkt(X, y, fit, tolof(T))
    @test all(row -> row.res ≤ row.tol, k.rows) && k.sumres ≤ k.sumtol
  end


  # 10. element types, λ and the solver
  @testset "element types, λ and the solver, $T" for T in (Float32, Float64)
    X, y = problem(T, 13, 40, 6)
    groups = [[1, 2], [3, 4], [5, 6]]
    @testset "the type of the fit, $std, λ::$(typeof(λ))" for std in (:orthonormal, :none), λ in (T(0.1), 0.1)
      fit = group_lasso(X, y, λ, groups, cyc(T); standardize = std)
      @test fit isa GroupLassoSolution{T}
      @test eltype(fit.coefficients) == T && eltype(fit.residuals) == T && eltype(fit.groupweights) == T
      @test typeof(fit.intercept) == T && typeof(fit.λ) == T && fit.λ === T(λ)
      @test eltype(coef(fit)) == T && eltype(predict(fit, X)) == T
      @test fit.result.converged && all(isfinite, coef(fit)) && all(isfinite, fit.residuals)
    end
    @testset "λ of another real type is converted to $T" begin
      ref = group_lasso(X, y, T(0.1), groups, cyc(T))
      for λ in (big"0.1", 1 // 10, 0.1)
        fit = group_lasso(X, y, λ, groups, cyc(T))
        @test fit.λ === T(0.1) && fit.coefficients == ref.coefficients && fit.intercept == ref.intercept
      end
      @test group_lasso(X, y, 1, groups, cyc(T)).λ === T(1)
      @test group_lasso(X, y, big(1), groups, cyc(T)).λ === T(1)
      # λ = 0 and the weights too
      @test group_lasso(X, y, 0, groups, cyc(T)).λ === T(0)
      @test group_lasso(X, y, T(0.1), groups, cyc(T); groupweights = [1, big"2.0", 3 // 2]).groupweights == T[1, 2, 1.5]
    end
    @testset "invalid λ" begin
      for λ in (-0.1, NaN, Inf, -Inf, big"-1.0", big"NaN")
        @test_throws ArgumentError group_lasso(X, y, λ, groups, cyc(T))
      end
      if T === Float32
        @test_throws ArgumentError group_lasso(X, y, 1e300, groups, cyc(T))      # converts to Inf
        @test_throws ArgumentError group_lasso(X, y, big(10)^400, groups, cyc(T))
        @test_throws ArgumentError group_lasso(X, y, 1e-50, groups, cyc(T))      # converts to 0
        err = try group_lasso(X, y, 1e300, groups, cyc(T)); nothing catch e; e end
        @test occursin("not representable in Float32", err.msg)
      end
    end
    @testset "arguments of other types are refused, nothing is converted" begin
      other = T === Float32 ? Float64 : Float32
      @test_throws ArgumentError group_lasso(other.(X), y, T(0.1), groups, cyc(T))        # mixed element types
      @test_throws ArgumentError group_lasso(X, other.(y), T(0.1), groups, cyc(T))
      @test_throws ArgumentError group_lasso(round.(Int, 10 .* X), round.(Int, 10 .* y), 0.1, groups, cyc(T))
      @test_throws ArgumentError group_lasso(view(X, :, :), y, T(0.1), groups, cyc(T))    # a design that is not a Matrix
      @test_throws ArgumentError group_lasso(X, view(y, :), T(0.1), groups, cyc(T))
      @test_throws ArgumentError group_lasso(X, reshape(y, :, 1), T(0.1), groups, cyc(T))
      @test_throws ArgumentError group_lasso(X, y, T(0.1), groups, :cyclic)               # a solver that is not a CDSolver
      @test_throws ArgumentError group_lasso(X, y, "0.1", groups, cyc(T))
    end
    @testset "the solver" begin
      # momentum = true is refused by ProximalSolvers for a group lasso, and its error is passed on
      err = try group_lasso(X, y, T(0.1), groups, CDSolver(momentum = true)); nothing catch e; e end
      @test err isa ArgumentError && occursin("momentum", err.msg) && !startswith(err.msg, "group_lasso")
      # the default CDSolver() and a seeded random order converge to the optimality conditions
      for solver in (CDSolver(), CDSolver(rng = Xoshiro(2), tol = tolof(T), maxpasses = 20_000))
        for std in (:orthonormal, :none)
          fit = group_lasso(X, y, T(0.1), groups, solver; standardize = std)
          @test fit.result.converged
          k = kkt(X, y, fit, solver.tol)
          @test all(row -> row.res ≤ row.tol, k.rows)
        end
      end
      # one pass at 0.1 λmax stops at maxpasses, with finite output and the solver's warning
      for std in (:orthonormal, :none)
        w = sqrt.(Float64.(length.(groups)))
        λ = T(0.1 * lambda_max(X, y, groups, w, true, std))
        fit = @test_logs (:warn,) match_mode = :any group_lasso(X, y, λ, groups, cyc(T; maxpasses = 1); standardize = std)
        @test !fit.result.converged && fit.result.passes == 1
        @test all(isfinite, fit.coefficients) && all(isfinite, fit.residuals) && isfinite(fit.intercept)
        @test group_lasso(X, y, λ, groups, cyc(T); standardize = std).result.converged
      end
    end
  end

  # 11. conventions
  @testset "conventions, $T" for T in (Float32, Float64)
    n = 40
    X, y = problem(T, 14, n, 6)
    groups = [[1, 4], [2, 3], [5, 6]]
    gw = [1.0, 2.0, 1.0]
    Xbefore, ybefore, gbefore, wbefore = copy(X), copy(y), deepcopy(groups), copy(gw)
    @testset "intercept = $ic, $std" for ic in (true, false), std in (:orthonormal, :none)
      fit = group_lasso(X, y, T(0.08), groups, cyc(T); intercept = ic, standardize = std, groupweights = gw)
      @test fit.result.converged
      # the arguments are not modified, and the fit holds none of them
      @test X == Xbefore && y == ybefore && groups == gbefore && gw == wbefore
      @test !any(a -> a === fit.coefficients || a === fit.residuals || a === fit.groupweights || a === fit.groups, (X, y, gw, groups))
      @test all(G -> !any(H -> G === H, groups), fit.groups)
      # the fields
      @test fit.groups == groups && fit.groupweights == T.(gw) && fit.standardize === std
      @test fit.hasintercept == ic && fit.λ === T(0.08)
      @test fit.activegroups == [k for (k, G) in enumerate(groups) if any(!iszero, fit.coefficients[G])]
      @test 0 < length(fit.activegroups)
      # the StatsAPI methods
      b = fit.coefficients
      ic ? (@test coef(fit) == [fit.intercept; b]) : (@test coef(fit) === b && fit.intercept == 0)
      @test residuals(fit) === fit.residuals && fit.residuals == y .- fit.intercept .- X * b
      @test predict(fit, X) == X * b .+ (ic ? fit.intercept : T(0))
      @test nobs(fit) == n && nobs(fit) isa Int
      @test dof(fit) == count(!iszero, b) + ic && dof(fit) isa Int
      @test deviance(fit) == sum(abs2, residuals(fit)) && deviance(fit) isa T
      Xnew = problem(T, 15, 7, 6)[1]
      @test predict(fit, Xnew) == Xnew * b .+ (ic ? fit.intercept : T(0)) && length(predict(fit, Xnew)) == 7
      @test predict(fit, view(Xnew, 2:4, :)) ≈ Xnew[2:4, :] * b .+ (ic ? fit.intercept : T(0)) rtol = 64 * eps(T)     # a view takes another product
      @test_throws DimensionMismatch predict(fit, zeros(T, 7, 5))
      # with an intercept, β₀ = ȳ - x̄ᵀβ, from the data
      if ic
        x̄ = vec(sum(Float64.(X); dims = 1)) ./ n
        @test abs(Float64(fit.intercept) - (sum(Float64.(y)) / n - dot(x̄, Float64.(b)))) ≤ 16 * 8 * eps(T) * (norm(Float64.(y)) / sqrt(n) + norm(x̄) * norm(b))
      end
      # a later change of the arguments does not change the fit
      X2, y2, groups2 = copy(X), copy(y), deepcopy(groups)
      fit2 = group_lasso(X2, y2, T(0.08), groups2, cyc(T); intercept = ic, standardize = std, groupweights = copy(gw))
      X2 .= 0; y2 .= 0; foreach(G -> (G .= 1), groups2)
      @test fit2.coefficients == b && fit2.groups == groups && fit2.residuals == fit.residuals
    end
    @testset "the default weights are √|G| of the groups given" begin
      fit = group_lasso(X, y, T(0.08), groups, cyc(T))
      @test fit.groupweights == sqrt.(T.(length.(groups)))
      # a group of three columns of which one is redundant has the weight √3
      X3 = hcat(X[:, 1:2], X[:, 1] .- T(0.5) .* X[:, 2], X[:, 3:6])
      fit3 = group_lasso(X3, y, T(0.08), [[1, 2, 3], [4, 5, 6, 7]], cyc(T))
      @test fit3.groupweights == sqrt.(T[3, 4])
    end
    @testset "show" begin
      fit = group_lasso(X, y, T(0.08), groups, cyc(T))
      str = sprint(show, fit)
      @test startswith(str, "GroupLassoSolution: 3 groups, $(length(fit.activegroups)) active, 6 coefficients, ")
      @test occursin("not an optimality certificate", str) && occursin("standardize = :orthonormal", str)
      @test occursin("met after $(fit.result.passes) passes", str) && !occursin("not met", str)
      @test str == string(fit) == repr(MIME("text/plain"), fit) && count(==('\n'), str) == 1
    end
  end

  # The rows of the numerical regression suite of PR #215 (test/numerics_extremes.jl, numerics_contracts.jl,
  # numerics_types.jl on the archive tag of the PR) that apply to group_lasso, on its data and groups
  @testset "numerical regression rows of PR #215" begin
    N, P = 60, 8
    GROUPS = [1:2, 3:5, 6:8]                       # unequal sizes, so the weights √|G| matter
    rng = Xoshiro(1)
    X = randn(rng, N, P) .* (1 .+ rand(rng, P))' .+ 0.3 .* randn(rng, P)'
    y = X[:, 1] .- 0.7 .* X[:, 4] .+ 0.5 .* randn(rng, N)
    Xc, yc = X .- sum(X; dims = 1) ./ N, y .- sum(y) / N
    nnz(v) = count(!iszero, v)
    relerr(a, b) = norm(a .- b) / max(norm(b), eps())
    solver() = CDSolver(rng = Xoshiro(3))           # the default tolerance and a random order

    @testset "penalty at the boundary: exact zero above λmax, nonzero below" begin
      # λmax = maxₖ ‖P_Gₖ ỹ‖ / (√n √|Gₖ|); 1 + 1e-9 is above it by more than the rounding in computing it
      # and 1 - 1e-3 is far above the solver tolerance
      above, below = 1 + 1e-9, 1 - 1e-3
      λmax = maximum(g -> norm(Matrix(qr(Xc[:, g]).Q)[:, 1:length(g)]' * yc) / sqrt(N * length(g)), GROUPS)
      @test λmax ≈ lambda_max(X, y, collect.(GROUPS), sqrt.(Float64.(length.(GROUPS))), true, :orthonormal) rtol = 1e-12
      @test iszero(group_lasso(X, y, λmax * above, GROUPS, solver()).coefficients)
      @test nnz(group_lasso(X, y, λmax * below, GROUPS, solver()).coefficients) >= 1
    end

    @testset "λ = 0 is least squares" begin
      # n > p and a well conditioned random design: the error of an iterate with gradient error 1e-7 is below 1e-6
      @test norm(group_lasso(X, y, 0.0, GROUPS, solver()).coefficients - Xc \ yc) <= 1e-6
      @test norm(group_lasso(X, y, 0.0, GROUPS, solver(); standardize = :none).coefficients - Xc \ yc) <= 1e-6
    end

    @testset "zero response and zero design give the exact zero fit" begin
      for std in (:orthonormal, :none)
        s = group_lasso(X, zeros(N), 0.1, GROUPS, solver(); standardize = std)
        @test iszero(s.coefficients) && s.result.converged && iszero(s.intercept)
        s = group_lasso(zeros(N, P), y, 0.1, GROUPS, solver(); standardize = std)
        @test iszero(s.coefficients) && s.result.converged && isfinite(s.intercept) && s.intercept ≈ sum(y) / N
      end
    end

    @testset "scale equivariance: y -> s y with λ -> s λ" begin
      # the penalized least squares problem is homogeneous of degree 2 in (y, λ), so the coefficients scale by s;
      # 1e-6 is ten times the default tolerance of the solver relative to the size of the coefficients
      for s in (1e-150, 1e150)
        f(y, λ) = group_lasso(X, y, λ, GROUPS, CDSolver(order = :cyclic)).coefficients
        b0, bs = f(y, 0.05), f(s .* y, s * 0.05)
        @test all(isfinite, bs) && nnz(bs) >= 1 && relerr(bs ./ s, b0) <= 1e-6
      end
    end

    @testset "max-iterations exhaustion: converged == false, finite output, the uncapped solve converges" begin
      for T in (Float32, Float64)
        XT, yT = T.(X), T.(y)
        capped = @test_logs (:warn,) match_mode = :any group_lasso(XT, yT, T(0.01), GROUPS, CDSolver(order = :cyclic, maxpasses = 1, tol = 1e-14))
        @test !capped.result.converged && capped.result.passes == 1 && all(isfinite, coef(capped))
        @test group_lasso(XT, yT, T(0.01), GROUPS, cyc(T)).result.converged
      end
    end

    @testset "the arguments are not mutated and not returned" begin
      gw = [1.0, 2.0, 1.0]
      args = (X, y, [collect(g) for g in GROUPS], gw)
      before = deepcopy(args)
      s = group_lasso(args[1], args[2], 0.05, args[3], solver(); groupweights = args[4])
      @test args == before
      for f in propertynames(s)
        v = getproperty(s, f)
        v isa AbstractArray && @test !any(a -> a === v, args)         # `groupweights` is a copy too
      end
    end

    @testset "non-finite data are rejected with an ArgumentError" begin
      bad(A, idx...; v = NaN) = (B = copy(A); B[idx...] = v; B)
      @test_throws ArgumentError group_lasso(bad(X, 2, 3), y, 0.05, GROUPS, solver())
      @test_throws ArgumentError group_lasso(bad(X, 5, 2; v = Inf), y, 0.05, GROUPS, solver())
      @test_throws ArgumentError group_lasso(X, bad(y, 4), 0.05, GROUPS, solver())
      @test_throws ArgumentError group_lasso(X, bad(y, 4; v = -Inf), 0.05, GROUPS, solver())
      for std in (:orthonormal, :none)             # without an intercept the entries are checked on another path
        @test_throws ArgumentError group_lasso(bad(X, 2, 3), y, 0.05, GROUPS, solver(); intercept = false, standardize = std)
        @test_throws ArgumentError group_lasso(X, bad(y, 4), 0.05, GROUPS, solver(); intercept = false, standardize = std)
      end
      @test occursin("column 3 of X has a non-finite entry", try group_lasso(bad(X, 2, 3), y, 0.05, GROUPS, solver()) catch e; e.msg end)
    end

    @testset "Float32 data give Float32 output, for a Float32 and a Float64 λ" begin
      X32, y32 = Float32.(X), Float32.(y)
      for λ in (0.05f0, 0.05), std in (:orthonormal, :none)
        s = group_lasso(X32, y32, λ, GROUPS, cyc(Float32; tol = 1e-5); standardize = std)
        @test eltype(coef(s)) == Float32 && eltype(predict(s, X32)) == Float32
        @test typeof(s.intercept) == Float32 && s.result.converged && all(isfinite, coef(s))
      end
    end

    @testset "Float32 X with Float64 y is refused, and so are integer data (adapted: nothing is promoted or converted)" begin
      @test_throws ArgumentError group_lasso(Float32.(X), y, 0.05, GROUPS, solver())
      Xi, yi = rand(Xoshiro(10), -3:3, N, P), rand(Xoshiro(11), -5:5, N)
      @test_throws ArgumentError group_lasso(Xi, yi, 0.5, GROUPS, solver())
      @test_throws ArgumentError group_lasso(Float64.(Xi), yi, 0.5, GROUPS, solver())
      # the conversion by the caller is exact and gives the fit of the Float64 data
      fit = group_lasso(Float64.(Xi), Float64.(yi), 0.5, GROUPS, cyc(Float64))
      @test eltype(coef(fit)) == Float64 && fit.result.converged
    end
  end

end

end
