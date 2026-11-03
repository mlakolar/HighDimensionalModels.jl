module HDMGoldenTest

using HighDimensionalModels, DelimitedFiles, LinearAlgebra, Random, Statistics, Test

# Frozen baselines were generated with one BLAS thread; pin it so atol 1e-12 holds in any order.
BLAS.set_num_threads(1)

const ROOT = joinpath(@__DIR__, "data", "hdm")
const DATASETS = ("hetero", "highdim", "forced", "nearthreshold", "empty")

# Julia baselines use 1e-12 * max(1, abs(expected)) elementwise, allowing one-ulp
# cross-platform differences at the nearthreshold design's 1e6 loading scale.
# Cross-package comparisons use absolute tolerances. Post=true results
# only use OLS after identical selection, hence a 1e-11 roundoff allowance.
# For post=false, hdm stops shooting at L1 step <1e-5, versus Julia's relative
# coordinate step 1e-7, with different starts; both outer loops stop at 1e-5.
# Use 1e-5 for coefficients. Propagate this through e=y-b0-Xb using the
# observed design, rather than silently increasing the residual tolerance.
const BASELINE_ATOL = 1e-12
const OLS_ATOL = 1e-11
const SHOOTING_ATOL = 1e-5

readmat(dir, file) = readdlm(joinpath(dir, file * ".csv"), ',', Float64)
readvec(dir, file) = vec(readmat(dir, file))

function readfixture(path)
  rows, header = readdlm(path, ',', Any; header = true)
  @assert vec(header) == ["fit", "field", "index", "value"]
  result = Dict{Tuple{String, String}, Vector{Float64}}()
  for row in eachrow(rows)
    key = (String(row[1]), String(row[2]))
    index, value = Int(row[3]), Float64(row[4])
    @assert isfinite(value)
    if index == 0
      @assert key[2] in ("selected", "index") && value == 0 && !haskey(result, key)
      result[key] = Float64[]
    else
      values = get!(result, key, Float64[])
      @assert index == length(values) + 1
      push!(values, value)
    end
  end
  result
end

maxdiff(x, y) = isempty(x) && isempty(y) ? 0.0 : maximum(abs, x .- y)

function baselinecheck(baseline, name, field, value; integer = false)
  actual = value isa Number ? [value] : collect(value)
  expected = baseline[name, field]
  @test length(actual) == length(expected)
  if integer
    @test all(isinteger, expected)
    @test actual == expected
  else
    @test all(i -> abs(actual[i] - expected[i]) <= BASELINE_ATOL * max(1, abs(expected[i])),
              eachindex(expected))
  end
end

@testset "hdm U1 golden references and unchanged Julia baseline" begin
  for (k, dataset) in enumerate(DATASETS)
    @testset "$dataset" begin
      dir = joinpath(ROOT, dataset)
      X, y, d = readmat(dir, "X"), readvec(dir, "y"), readvec(dir, "d")
      n, p = size(X)
      forced = dataset == "forced" ? Int.(readvec(dir, "forced")) : Int[]
      baseline = readfixture(joinpath(dir, "baseline-julia.csv"))
      reference = readfixture(joinpath(dir, "reference-r.csv"))
      metadata, header = readdlm(joinpath(dir, "metadata.csv"), ',', String; header = true)
      meta = Dict(row[1] => row[2] for row in eachrow(metadata))
      @test vec(header) == ["key", "value"]
      @test startswith(meta["R"], "R version 4.5.1")
      @test meta["hdm"] == "0.3.2" && meta["sandwich"] == "3.1.1"
      @test meta["hdm_archive_sha256"] == "3354b22752f0aa0c0d9c8c12b38810202b6380f2587e539be11d9a358412fe6a"
      @test meta["data_script"] == "test/data/hdm/golden_data.jl"
      @test meta["reference_script"] == "test/data/hdm/golden.R"
      @test meta["baseline_script"] == "test/data/hdm/baseline.jl"

      # Load both layouts' references; their final stages are checked below
      # on hdm's own residuals or selection, independently of Julia selection.
      for method in (:doubleselection, :partialling)
        name, final = "effect_$method", "final_$method"
        for field in ("alpha", "se", "t", "pval")
          @test length(reference[name, field]) == 1
        end
        @test all(isinteger, reference[name, "index"])
        @test length(reference[name, "epsilon"]) == n
        @test length(reference[name, "v"]) == n
        for field in ("coefficient", "vcov", "HC0", "HC1", "HC2", "HC3")
          @test length(reference[final, field]) == 1
        end
        @test length(reference[final, "hatvalues"]) == n
        if method == :partialling
          @test length(reference[final, "yr"]) == n
          @test length(reference[final, "dr"]) == n
        end
      end

      if dataset != "nearthreshold"
        @testset "hdm double-selection final stage" begin
          S = Int.(reference["effect_doubleselection", "index"])
          # This internal layout only reads selected. Feed it hdm's full
          # union, including I3, rather than fitting Julia nuisances again.
          layout = HighDimensionalModels._effect_layout(y, d, X, (; selected = S),
            (; selected = Int[]), :doubleselection, Int[], nothing)
          @test layout.selected == S && layout.s == length(S)
          @test issubset(forced, S)
          varerr, haterr, alphaerr, epsilonerr, verr = 0.0, 0.0, 0.0, 0.0, 0.0
          for type in (:homoskedastic, :hdm, :HC0, :HC1, :HC2, :HC3)
            core = HighDimensionalModels._effect_final(layout.v, layout.w, layout.h_W,
              layout.k, layout.s, type, :doubleselection;
              target_norm = norm(d .- mean(d)), outcome_norm = norm(y .- mean(y)))
            V = if type === :hdm
              only(reference["effect_doubleselection", "se"])^2
            else
              only(reference["final_doubleselection", type === :homoskedastic ? "vcov" : string(type)])
            end
            epsilon = core.residuals .* sqrt(n / (n - layout.s - 1))
            @test core.variance ≈ V rtol = 1e-10 atol = 0
            @test core.leverage ≈ reference["final_doubleselection", "hatvalues"] atol = 1e-10 rtol = 0
            @test core.estimate ≈ only(reference["final_doubleselection", "coefficient"]) atol = 1e-10 rtol = 0
            @test epsilon ≈ reference["effect_doubleselection", "epsilon"] atol = 1e-10 rtol = 0
            @test layout.v ≈ reference["effect_doubleselection", "v"] atol = 1e-10 rtol = 0
            varerr = max(varerr, abs(core.variance / V - 1))
            haterr = max(haterr, maxdiff(core.leverage, reference["final_doubleselection", "hatvalues"]))
            alphaerr = max(alphaerr, abs(core.estimate - only(reference["final_doubleselection", "coefficient"])))
            epsilonerr = max(epsilonerr, maxdiff(epsilon, reference["effect_doubleselection", "epsilon"]))
            verr = max(verr, maxdiff(layout.v, reference["effect_doubleselection", "v"]))
          end
          println("hdm $dataset double-selection final max_relative_variance=$varerr max_abs_leverage=$haterr max_abs_alpha=$alphaerr max_abs_epsilon=$epsilonerr max_abs_v=$verr")
        end

        @testset "hdm partialling final stage" begin
          u = reference["final_partialling", "yr"]
          v = reference["final_partialling", "dr"]
          u, v = u .- mean(u), v .- mean(v)
          varerr, haterr, alphaerr = 0.0, 0.0, 0.0
          homovariance = NaN
          for type in (:homoskedastic, :hdm, :HC0, :HC1, :HC2, :HC3)
            core = HighDimensionalModels._effect_final(v, u, fill(1/n, n), 2, 0, type, :partialling;
              target_norm = norm(d .- mean(d)), outcome_norm = norm(y .- mean(y)))
            field = type in (:homoskedastic, :hdm) ? "vcov" : string(type)
            V = only(reference["final_partialling", field])
            @test core.variance ≈ V rtol = 1e-10 atol = 0
            @test core.leverage ≈ reference["final_partialling", "hatvalues"] atol = 1e-10 rtol = 0
            @test core.estimate ≈ only(reference["final_partialling", "coefficient"]) atol = 1e-10 rtol = 0
            if type === :homoskedastic
              homovariance = core.variance
            elseif type === :hdm
              @test core.variance == homovariance
            end
            varerr = max(varerr, abs(core.variance / V - 1))
            haterr = max(haterr, maxdiff(core.leverage, reference["final_partialling", "hatvalues"]))
            alphaerr = max(alphaerr, abs(core.estimate - only(reference["final_partialling", "coefficient"])))
          end
          println("hdm $dataset partialling final max_relative_variance=$varerr max_abs_leverage=$haterr max_abs_alpha=$alphaerr")
        end
      end

      coeferr, residerr, loadingerr, lambdaerr = 0.0, 0.0, 0.0, 0.0
      loadingerr_by_post = Dict(true => 0.0, false => 0.0)
      for (j, (post, response)) in enumerate(((true, :y), (true, :d), (false, :y), (false, :d)))
        target = response == :y ? y : d
        Random.seed!(11900 + 10k + j)
        r = rlasso(X, target; post)
        name = "rlasso_$(response)_$post"
        baselinecheck(baseline, name, "selected", r.selected; integer = true)
        baselinecheck(baseline, name, "iterations", r.iterations; integer = true)
        baselinecheck(baseline, name, "intercept", r.intercept)
        baselinecheck(baseline, name, "coefficients", Vector(r.x))
        baselinecheck(baseline, name, "residuals", r.residuals)
        baselinecheck(baseline, name, "loadings", r.loadings)
        baselinecheck(baseline, name, "lambda0", r.λ0)
        baselinecheck(baseline, name, "sigma", r.σ)

        @test length(reference[name, "coefficients"]) == p + 1
        @test length(reference[name, "intercept"]) == 1
        @test length(reference[name, "residuals"]) == n
        @test length(reference[name, "loadings"]) == p
        @test length(reference[name, "lambda0"]) == 1
        @test length(reference[name, "iter"]) == 1
        @test all(isinteger, reference[name, "index"])
        @test isinteger(only(reference[name, "iter"]))
        Rsel = Int.(reference[name, "index"])
        dc = maxdiff([r.intercept; Vector(r.x)], reference[name, "coefficients"])
        de = maxdiff(r.residuals, reference[name, "residuals"])
        dl = maxdiff(r.loadings, reference[name, "loadings"])
        dλ = abs(r.λ0 - only(reference[name, "lambda0"]) / (2n))
        coeferr, residerr = max(coeferr, dc), max(residerr, de)
        loadingerr, lambdaerr = max(loadingerr, dl), max(lambdaerr, dλ)
        loadingerr_by_post[post] = max(loadingerr_by_post[post], dl)
        @test dλ <= 1e-15
        if dataset == "nearthreshold"
          # Orthogonal, centered design: only coordinate 2 differs, in every
          # nuisance fit, for both post values. hdm discards it at 1e-6.
          @test Rsel == [1]
          @test r.selected == [1, 2]
          @test setdiff(r.selected, Rsel) == [2]
          @test 0 < abs(r.x[2]) < 1e-6
          @test reference[name, "coefficients"][3] == 0
          if post
            @test abs(r.x[2] - (response == :y ? 8e-7 : 6e-7)) <= 1e-15
          end
        else
          @test r.selected == Rsel
          coefficient_tol = post ? OLS_ATOL : SHOOTING_ATOL
          residual_tol = post ? OLS_ATOL : SHOOTING_ATOL * (1 + maximum(sum(abs, X[:, Rsel]; dims = 2)))
          @test dc <= coefficient_tol
          @test de <= residual_tol
          if post || isempty(Rsel)
            # OLS on common supports (or common initialization for the empty
            # first fit): only floating-point roundoff enters the loadings.
            @test dl <= OLS_ATOL
          else
            # Reverse triangle inequality for Γ_j=||Xc_j .* e||₂/sqrt(n):
            # |ΔΓ_j| <= ||Xc_j .* Δe||₂/sqrt(n). Propagate U1's frozen
            # coefficient/intercept budget pointwise through the raw design.
            Xc = X .- mean(X; dims = 1)
            residual_budget = SHOOTING_ATOL .* (1 .+ vec(sum(abs, X[:, Rsel]; dims = 2)))
            loading_budget = vec(sqrt.(mean(abs2.(Xc .* residual_budget); dims = 1)))
            @test all(abs.(r.loadings - reference[name, "loadings"]) .<= loading_budget)
          end
          if dataset == "forced"
            @test isempty(intersect(forced, r.selected))
          elseif dataset == "empty"
            @test isempty(r.selected)
            @test r.iterations == 1
          end
        end
      end

      alphaerr, seerr, terr, perr = 0.0, 0.0, 0.0, 0.0
      for (j, method) in enumerate((:doubleselection, :partialling))
        force = method == :doubleselection ? forced : Int[]
        Random.seed!(11900 + 10k + 4 + j)
        r = double_selection(y, d, X; method, forced = force)
        name = "effect_$method"
        for field in (:estimate, :se, :t, :pvalue, :ci, :selected)
          baselinecheck(baseline, name, string(field), getproperty(r, field); integer = field == :selected)
        end
        da = abs(r.estimate - only(reference[name, "alpha"]))
        ds = abs(r.se - only(reference[name, "se"]))
        dt = abs(r.t - only(reference[name, "t"]))
        dp = abs(r.pvalue - only(reference[name, "pval"]))
        alphaerr, seerr = max(alphaerr, da), max(seerr, ds)
        terr, perr = max(terr, dt), max(perr, dp)
        if dataset == "nearthreshold"
          @test r.selected == [1, 2]
          @test reference[name, "index"] == [1]
          @test abs(r.estimate - 0.5) <= OLS_ATOL
          @test abs(only(reference[name, "alpha"]) - 0.8) <= OLS_ATOL
        else
          @test r.selected == Int.(reference[name, "index"])
          @test da <= OLS_ATOL
          @test ds <= OLS_ATOL
          @test dt <= OLS_ATOL
          @test dp <= OLS_ATOL
          if dataset == "forced"
            @test method == :doubleselection ? issubset(forced, r.selected) : isempty(intersect(forced, r.selected))
          end
        end
      end
      println("hdm $dataset max_abs: coefficients=$coeferr residuals=$residerr loadings=$loadingerr lambda0=$lambdaerr alpha=$alphaerr se=$seerr t=$terr pval=$perr")
      println("hdm $dataset loadings max_abs by post: true=$(loadingerr_by_post[true]) false=$(loadingerr_by_post[false])")
      if dataset == "forced"
        for file in ("X", "y", "d")
          @test read(joinpath(dir, file * ".csv")) == read(joinpath(ROOT, "hetero", file * ".csv"))
        end
      end
    end
  end
end

end
