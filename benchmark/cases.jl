# The case table: a scenario of problems.jl, one call of the public `lasso` with
# a fixed solver configuration, the certificate of certificates.jl for its
# objective, and a target. Each case states its target and settings in its
# add_case! call; a target is a threshold on the certificate's value.
#
# build(problem) prepares the inputs once, untimed: the design and response in
# the element type of the case, the penalty level and the solver, with its
# seeded random number generator. Each sample times call on a fresh copy of
# them, so the generator restarts, and certify(problem, prepared, output) checks
# that output from the prepared inputs and the returned coefficients; the
# certifiers below do not read the solver's convergence flag or minimum.

struct CaseSpec
  meta::NamedTuple
  build::Function
  call::Function
  certify::Function
end

const CASE_GROUPS = ("lasso", "group", "controls")
# The negative controls run only when HDM_GROUPS names them.
const DEFAULT_GROUPS = filter(!=("controls"), CASE_GROUPS)

# The measures of the certificates, as the catalog records them
const MEASURE = (gap = "relative_gap", kkt = "kkt_residual")

# The random coordinate order starts from the same state in every sample; the
# default generator is reseeded with SOLVER_SEED too.
const SOLVER_SEED = 20261028
const HARNESS_SEED_SCHEME = "inputs: problems.jl scenario seeds by key and repetition; " *
  "solvers: Xoshiro from SHA-256 of solver, reset per sample; " *
  "order: Xoshiro from SHA-256 of measurement_order and repetition"

# A Xoshiro generator whose state is four words of the SHA-256 digest of parts
function harness_rng(parts...)
  digest = sha256(join(string.(parts), '/'))
  word(k) = foldl((h, b) -> (h << 8) | UInt64(b), digest[8k-7:8k]; init = UInt64(0))
  Xoshiro(word(1), word(2), word(3), word(4))
end
solver_rng() = harness_rng("solver")

# The data hash of a case: its scenario and the penalty levels that the case
# derives from it
case_data_hash(problem, prepared) =
  data_hash(merge(problem, (; case_levels = Float64.(prepared.levels))))

function add_case!(catalog, group, estimator, variant; scenario, measure, target,
                   build, call, certify, suite = "quick")
  parts = (group, estimator, variant)
  any(part -> occursin(r"[/\t\n\r]", part), parts) && error("invalid case identifier")
  group in CASE_GROUPS || error("unknown group $group")
  suite in ("quick", "extended") || error("unknown suite $suite")
  target == "record" || target isa Float64 && isfinite(target) && target > 0 ||
    error("the target must be record or a positive threshold")
  meta = (; id = join(parts, '/'), group, scenario, estimator, variant, phase = "call", suite,
            inputs = "draw", measure, target)
  push!(catalog, CaseSpec(meta, build, call, certify))
end

######################################################################
#
#   Inputs and penalty levels
#
######################################################################

case64(A::AbstractMatrix) = Matrix{Float64}(A)
case64(v::AbstractVector) = Vector{Float64}(collect(v))
case64(x::Real) = Float64(x)

# Columns centered and scaled to mean square one (a column whose entries are all equal
# has that value as its mean and becomes zero) and the response centered. The scales
# are computed from squares, which is adequate for the moderate magnitudes of the
# scenarios of problems.jl. These are the coordinates in which the earlier
# standardized lasso with an intercept solved the same scenarios, and in which
# the cases other than `dense-standardized` call `lasso` with its defaults: the
# objective with an intercept and scales in the original coordinates is this one
# in the new, with the same ratio λ/λmax. The sums add in a fixed
# order, so that they do not depend on the BLAS or Statistics of the Julia version.
function standardized(X, y)
  Z = similar(X, Float64)
  for j in axes(X, 2)
    x = view(X, :, j)
    c = all(==(first(x)), x) ? first(x) : sample_mean(x)
    s = sqrt(sample_mean(abs2.(x .- c)))
    Z[:, j] .= s > 0 ? (x .- c) ./ s : 0.0
  end
  Z, y .- sample_mean(y)
end

# The smallest λ with a zero lasso solution, maxⱼ |xⱼ'y|/n, with the sums in a fixed order
function lambda_max(X, y)
  top = 0.0
  for j in axes(X, 2)
    s = 0.0
    for i in axes(X, 1)
      s += Float64(X[i, j]) * Float64(y[i])
    end
    top = max(top, abs(s))
  end
  top / size(X, 1)
end

######################################################################
#
#   The catalog
#
######################################################################

# The target of `dense-standardized`, which its certificate also needs to decide whether the rounding
# uncertainty could change the verdict
const STANDARDIZED_TARGET = 1e-6

function lasso_cases!(catalog)
  # λ = ratio λmax on the standardized scenario, in element type T, with the
  # solver of fixed configuration. The stopping tolerance tol is that of the
  # case.
  function lasso_inputs(p; T = Float64, ratio = 0.1, tol = 1e-7)
    Z, r = standardized(p.X, p.y)
    X, y = Matrix{T}(Z), Vector{T}(r)
    λ = T(ratio * lambda_max(X, y))
    solver = CDSolver(; tol, rng = solver_rng())
    (; X, y, λ, solver, levels = [case64(λ)])
  end
  lasso_call(s) = lasso(s.X, s.y, s.λ, s.solver)
  lasso_certificate(s, out) = lasso_gap(s.X, s.y, coef(out), s.λ)

  # The raw inputs of the scenario, fit with an intercept and standardization at the level
  # ratio λmax of the centered and scaled problem (the level of `dense`). The certificate
  # reads the stored intercept and coefficients: coef(out) has p + 1 entries here.
  function standardized_inputs(p; ratio = 0.1, tol = 1e-7)
    Z, r = standardized(p.X, p.y)
    λ = ratio * lambda_max(Z, r)
    solver = CDSolver(; tol, rng = solver_rng())
    (; X = p.X, y = p.y, λ, solver, levels = [λ])
  end
  standardized_call(s) = lasso(s.X, s.y, s.λ, s.solver; intercept = true, standardize_x = true)
  standardized_certificate(s, out) = lasso_gap_standardized(s.X, s.y, out.intercept, out.coefficients, s.λ; target = STANDARDIZED_TARGET)

  add_case!(catalog, "lasso", "lasso", "dense"; scenario = "ls-200x50",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> lasso_inputs(p), call = lasso_call, certify = (p, s, out) -> lasso_certificate(s, out))
  add_case!(catalog, "lasso", "lasso", "dense-standardized"; scenario = "ls-200x50",
    measure = MEASURE.gap, target = STANDARDIZED_TARGET,
    build = p -> standardized_inputs(p), call = standardized_call,
    certify = (p, s, out) -> standardized_certificate(s, out))
  for (variant, scenario) in (("tall", "ls-tall-5000x40"), ("wide", "ls-wide-100x400"),
                              ("correlated", "ls-corr-300x100"), ("large", "ls-2000x1000"))
    add_case!(catalog, "lasso", "lasso", variant; scenario, suite = "extended",
      measure = MEASURE.gap, target = 1e-6,
      build = p -> lasso_inputs(p), call = lasso_call, certify = (p, s, out) -> lasso_certificate(s, out))
  end
  # Float32 arithmetic limits the attainable relative gap; the target is 1e-4,
  # and the solver's stopping tolerance is 1e-6 instead of its Float64 default.
  add_case!(catalog, "lasso", "lasso", "float32"; scenario = "ls-200x50", suite = "extended",
    measure = MEASURE.gap, target = 1e-4,
    build = p -> lasso_inputs(p; T = Float32, tol = 1e-6), call = lasso_call,
    certify = (p, s, out) -> lasso_certificate(s, out))
end

function control_cases!(catalog)
  # One pass of coordinate descent is expected to leave a relative gap above 1e-8
  # on this scenario: this case must end target_missed, with no timing-eligible
  # sample. The solver warns that it stopped at its pass limit; the warning is
  # silenced.
  add_case!(catalog, "controls", "lasso", "one-pass"; scenario = "ls-200x50",
    measure = MEASURE.gap, target = 1e-8,
    build = p -> begin
      Z, r = standardized(p.X, p.y)
      λ = 0.1 * lambda_max(Z, r)
      (; X = Z, y = r, λ, solver = CDSolver(; maxpasses = 1, rng = solver_rng()), levels = [λ])
    end,
    call = s -> Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
      lasso(s.X, s.y, s.λ, s.solver)
    end,
    certify = (p, s, out) -> lasso_gap(s.X, s.y, coef(out), s.λ))
end

# ---- group lasso (#224) ----
# The group lasso on the raw inputs of the scenario "groups-200x60" (twelve groups of five columns), with an
# intercept and the default weights √|G|, at the level λ = 0.1 λmax. Certified at the stored intercept and
# coefficients by `group_gap`, a duality gap on the full column span of each group.

# The target of the group cases, which their certificate also needs to decide whether the rounding of its
# bounds could change the verdict
const GROUP_TARGET = 1e-6

# The dot product of two vectors, added in order
function ordered_dot(a, b)
  s = 0.0
  for i in eachindex(a, b)
    s += a[i] * b[i]
  end
  s
end

# gᵀG⁻¹g for a symmetric positive definite G, by elimination in order (no BLAS)
function ordered_quadratic(G, g)
  A, b, value = copy(G), copy(g), 0.0
  for i in eachindex(b)
    value += b[i]^2 / A[i, i]
    for j in i+1:length(b)
      f = A[j, i] / A[i, i]
      for l in i:length(b)
        A[j, l] -= f * A[i, l]
      end
      b[j] -= f * b[i]
    end
  end
  value
end

# The smallest λ with a zero group lasso solution, for weights above zero: maxₖ ‖P_Gₖ ỹ‖/(√n wₖ) with
# `orthonormal`, P_G the projector on the span of the centred columns of the group, and maxₖ ‖X̃_Gₖ'ỹ‖/(n wₖ)
# without. The sums and the elimination add in a fixed order, so that λ does not depend on the BLAS or the
# Julia version.
function group_lambda_max(X, y, groups, weights; orthonormal)
  n = size(X, 1)
  ỹ = y .- sample_mean(y)
  top = 0.0
  for (k, G) in enumerate(groups)
    cols = [X[:, j] .- sample_mean(view(X, :, j)) for j in G]
    g = [ordered_dot(c, ỹ) for c in cols]
    v = orthonormal ? sqrt(ordered_quadratic([ordered_dot(a, b) for a in cols, b in cols], g)) / (sqrt(n) * weights[k]) :
                      sqrt(ordered_dot(g, g)) / (n * weights[k])
    top = max(top, v)
  end
  top
end

function group_cases!(catalog)
  # λ = ratio λmax of the problem of the case, with the solver of fixed configuration; `orthonormal` selects
  # standardize = :orthonormal or :none
  function group_inputs(p; orthonormal, tol, ratio = 0.1)
    X, y = case64(p.X), case64(p.y)
    groups = grouped_groups(GROUPED_SHAPE.ngroups, GROUPED_SHAPE.size)
    weights = sqrt.(Float64.(length.(groups)))
    λ = ratio * group_lambda_max(X, y, groups, weights; orthonormal)
    solver = CDSolver(; tol, rng = solver_rng())
    (; X, y, groups, weights, λ, solver, orthonormal, levels = [λ])
  end
  group_call(s) = group_lasso(s.X, s.y, s.λ, s.groups, s.solver; intercept = true,
                              standardize = s.orthonormal ? :orthonormal : :none)
  group_certificate(s, out) = group_gap(s.X, s.y, out.intercept, out.coefficients, s.λ, s.groups, s.weights;
                                        orthonormal = s.orthonormal, target = GROUP_TARGET)
  # The tolerance of the solver is explicit. In a pilot on repetitions 1 to 10 (Julia 1.13.4) the stopping
  # tolerance 1e-7 left relative gaps of at most 6e-8 with orthonormalization; without it 1e-7 left up to 7e-7
  # (about 3 times the tolerance) and 1e-8 left at most 7e-8.
  add_case!(catalog, "group", "group_lasso", "orthonormal"; scenario = "groups-200x60",
    measure = MEASURE.gap, target = GROUP_TARGET,
    build = p -> group_inputs(p; orthonormal = true, tol = 1e-7), call = group_call,
    certify = (p, s, out) -> group_certificate(s, out))
  add_case!(catalog, "group", "group_lasso", "none"; scenario = "groups-200x60", suite = "extended",
    measure = MEASURE.gap, target = GROUP_TARGET,
    build = p -> group_inputs(p; orthonormal = false, tol = 1e-8), call = group_call,
    certify = (p, s, out) -> group_certificate(s, out))
end

function all_cases()
  catalog = CaseSpec[]
  lasso_cases!(catalog)
  control_cases!(catalog)
  group_cases!(catalog)
  ids = [c.meta.id for c in catalog]
  allunique(ids) || error("duplicate case ids")
  unknown = setdiff(unique(c.meta.scenario for c in catalog), SCENARIOS)
  isempty(unknown) || error("scenarios missing from problems.jl: $(join(unknown, ", "))")
  catalog
end

# The cases of the suite (quick cases also run in extended), the groups and the
# id patterns (none for all), sorted by id
function build_catalog(suite, groups, patterns = Regex[])
  suite in ("quick", "extended") || throw(ArgumentError("HDM_SUITE must be quick or extended"))
  all(in(CASE_GROUPS), groups) || throw(ArgumentError("unknown HDM_GROUPS entry"))
  selected = filter(all_cases()) do c
    c.meta.group in groups && (suite == "extended" || c.meta.suite == "quick") &&
      (isempty(patterns) || any(p -> occursin(p, c.meta.id), patterns))
  end
  isempty(selected) && error("no case matches HDM_SUITE, HDM_GROUPS and HDM_CASES")
  sort!(selected; by = c -> c.meta.id)
end
