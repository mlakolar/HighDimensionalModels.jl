using HighDimensionalModels
import ProximalBase, ProximalSolvers
using LinearAlgebra, Random, Statistics, TOML, SHA, Profile

include("problems.jl")
include("certificates.jl")
include("cases.jl")

const CATALOG_HEADER = ("id", "group", "scenario", "estimator", "variant", "phase", "suite",
                        "inputs", "measure", "target")
const METRICS_HEADER = ("id", "group", "phase", "rep", "status", "samples", "ok_samples",
  "median_ns", "mean_ns", "sd_ns", "min_ns", "max_ns", "p95_ns", "gc_share",
  "median_bytes", "mean_bytes", "min_bytes", "max_bytes", "measure", "target",
  "worst_value", "median_value", "data_hash")
const SAMPLES_HEADER = ("id", "rep", "sample", "time_ns", "gc_ns", "bytes", "value", "status")
const ERRORS_HEADER = ("id", "rep", "status", "exception", "stack")
const CASE_STATUS_ORDER = (:error, :timeout, :uncertifiable, :target_missed)
const MIN_SAMPLES = 3

tsv_field(::Nothing) = ""
tsv_field(x::AbstractFloat) = repr(x)
tsv_field(x) = replace(string(x), '\t' => ' ', '\n' => ' ', '\r' => ' ')
tsv_line(io, values...) = println(io, join(tsv_field.(values), '\t'))
finite_field(x) = isfinite(x) ? x : nothing

function fingerprint(root, files)
  io = IOBuffer()
  for file in sort(files)
    write(io, relpath(file, root), '\0', read(file), '\0')
  end
  bytes2hex(sha1(take!(io)))
end

function package_fingerprint(root)
  files = [joinpath(root, "Project.toml")]
  for (folder, _, names) in walkdir(joinpath(root, "src")), name in names
    endswith(name, ".jl") && push!(files, joinpath(folder, name))
  end
  fingerprint(root, files)
end

# The Julia files of this folder, not those of generated environments below it
harness_files() = [joinpath(@__DIR__, name) for name in readdir(@__DIR__) if endswith(name, ".jl")]

# The commit and whether the tree differs from it, for a checkout at its
# repository root (untracked files count with `untracked = true`);
# "unavailable" and dirty for a directory that is not one
function git_identity(root; untracked::Bool = false)
  mode = untracked ? "--untracked-files=all" : "--untracked-files=no"
  revision = try
    top = readchomp(pipeline(`git -C $root rev-parse --show-toplevel`; stderr = devnull))
    realpath(top) == realpath(root) || error("not the root of its repository")
    readchomp(pipeline(`git -C $root rev-parse HEAD`; stderr = devnull))
  catch
    "unavailable"
  end
  dirty = try
    !isempty(readchomp(pipeline(`git -C $root status --porcelain $mode`; stderr = devnull)))
  catch
    true
  end
  revision, dirty
end

# The checkout that a loaded package comes from: its root, commit, dirty flag and version
function source_identity(mod::Module; untracked::Bool = false)
  root = realpath(dirname(dirname(pathof(mod))))
  revision, dirty = git_identity(root; untracked)
  version = pkgversion(mod)
  (; root, revision, dirty, version = version === nothing ? "unknown" : string(version))
end

# The manifest of the active environment, by the names Pkg looks for
function active_manifest()
  folder, v = dirname(Base.active_project()), "v$(VERSION.major).$(VERSION.minor)"
  for name in ("JuliaManifest-$v.toml", "Manifest-$v.toml", "JuliaManifest.toml", "Manifest.toml")
    path = joinpath(folder, name)
    isfile(path) && return path
  end
  error("no manifest in $folder; prepare the environment with benchmark/setup.jl")
end

# The version and tree hash of every package of the active manifest that is not a
# standard library (registered, tracked by repository or by path), except the
# measured package, which the package keys identify. A package tracked by path
# has no tree hash.
function dependency_identity()
  identity = Dict{String,Any}()
  for (name, entries) in TOML.parsefile(active_manifest())["deps"], entry in entries
    name == "HighDimensionalModels" && continue
    any(key -> haskey(entry, key), ("git-tree-sha1", "path", "repo-url")) || continue
    identity[name] = Dict{String,Any}("version" => get(entry, "version", "unknown"),
                                      "tree_hash" => get(entry, "git-tree-sha1", ""))
  end
  identity
end

function selected_patterns(text)
  patterns = filter!(!isempty, strip.(split(text, ',')))
  # Globs on case ids: * matches any characters and ? one character.
  [Regex("^" * replace(escape_glob(p), "\\*" => ".*", "\\?" => ".") * "\$") for p in patterns]
end
escape_glob(pattern) = replace(pattern, r"[\\^$.|+()\[\]{}*?]" => s -> "\\" * s)

# The loaded ProximalBase and ProximalSolvers; their trees count untracked files
dependency_sources() = (source_identity(ProximalBase; untracked = true),
                        source_identity(ProximalSolvers; untracked = true))

# Records identify ProximalBase and ProximalSolvers by commit, so a measurement
# needs both loaded from clean checkouts with a known commit
function require_clean_dependencies()
  for (name, source) in zip(("ProximalBase", "ProximalSolvers"), dependency_sources())
    source.revision == "unavailable" &&
      error("the loaded $name at $(source.root) is not the root of a git checkout; its commit is unknown")
    source.dirty &&
      error("the loaded $name at $(source.root) has tracked or untracked changes; use a clean checkout")
  end
end

function runner_settings()
  suite = get(ENV, "HDM_SUITE", "quick")
  suite in ("quick", "extended") || error("HDM_SUITE must be quick or extended")
  groups = sort!(unique(strip.(split(get(ENV, "HDM_GROUPS", join(DEFAULT_GROUPS, ',')), ','))))
  !isempty(groups) && all(in(CASE_GROUPS), groups) ||
    error("HDM_GROUPS entries must be among $(join(CASE_GROUPS, ", "))")
  cases = selected_patterns(get(ENV, "HDM_CASES", ""))
  rep = parse(Int, get(ENV, "HDM_REPETITION", "1"))
  1 <= rep <= 10 || error("HDM_REPETITION must be between 1 and 10")
  samples = parse(Int, get(ENV, "HDM_SAMPLES", suite == "quick" ? "50" : "100"))
  seconds = parse(Float64, get(ENV, "HDM_SECONDS", suite == "quick" ? "1" : "3"))
  case_seconds = parse(Float64, get(ENV, "HDM_CASE_SECONDS", "60"))
  profile_seconds = parse(Float64, get(ENV, "HDM_PROFILE_SECONDS", "0"))
  samples >= MIN_SAMPLES && isfinite(seconds) && seconds > 0 && isfinite(case_seconds) &&
    case_seconds > 0 && isfinite(profile_seconds) && profile_seconds >= 0 ||
    error("HDM_SAMPLES must be at least $MIN_SAMPLES and time limits positive; " *
          "profile seconds must be nonnegative")
  list = get(ENV, "HDM_LIST", "false")
  list in ("true", "false") || error("HDM_LIST must be true or false")
  output = abspath(get(ENV, "HDM_OUTPUT", joinpath(@__DIR__, "results")))
  expected = get(ENV, "HDM_EXPECTED_PACKAGE_ROOT", "")
  root = source_identity(HighDimensionalModels).root
  if !isempty(expected)
    expected = realpath(expected)
    root == expected || error("loaded HighDimensionalModels from $root; expected $expected")
  end
  (; suite, groups, cases, rep, samples, seconds, case_seconds, profile_seconds,
     list = list == "true", output, expected, root)
end

function start_metadata(settings)
  s = settings
  package = source_identity(HighDimensionalModels)
  base, solvers = dependency_sources()
  Dict{String,Any}("julia" => string(VERSION), "rep" => s.rep, "suite" => s.suite,
    "groups" => join(s.groups, ','),
    "harness_hash" => fingerprint(@__DIR__, harness_files()),
    "package_path" => package.root, "package_sha" => package.revision, "package_dirty" => package.dirty,
    "package_version" => package.version, "package_hash" => package_fingerprint(s.root),
    "proximalbase_path" => base.root, "proximalbase_sha" => base.revision,
    "proximalbase_dirty" => base.dirty, "proximalbase_version" => base.version,
    "proximalsolvers_path" => solvers.root, "proximalsolvers_sha" => solvers.revision,
    "proximalsolvers_dirty" => solvers.dirty, "proximalsolvers_version" => solvers.version,
    "expected_package_root" => s.expected, "julia_threads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(), "blas" => string(BLAS.get_config()),
    "cpu" => Sys.CPU_NAME, "cpu_model" => Sys.cpu_info()[1].model,
    "samples_target" => s.samples, "seconds_limit" => s.seconds,
    "case_seconds" => s.case_seconds, "profile_seconds" => s.profile_seconds,
    "seed_scheme" => HARNESS_SEED_SCHEME, "job_id" => get(ENV, "SLURM_JOB_ID", "none"),
    "dependencies" => dependency_identity())
end

function write_metadata(path, metadata)
  open(path, "w") do io
    TOML.print(io, metadata; sorted = true)
  end
end

function certificate_status(cert, meta)
  cert.status == :uncertifiable && return cert.status
  cert.status == :ok || error("invalid certificate status $(cert.status)")
  string(cert.measure) == meta.measure ||
    error("certificate measure $(cert.measure) differs from the catalog's $(meta.measure)")
  isfinite(cert.value) && cert.value >= 0 || return :uncertifiable
  meta.target isa Real && cert.value > meta.target ? :target_missed : :ok
end

# The input draw of a case, and the state of each sample: fresh copies of the
# prepared inputs, whose random number generators restart with them, and the
# default generator reseeded for the calls that draw from it. The call gets the
# copy and the certificate the prepared inputs, which no call sees, so that a
# call that changes its inputs cannot certify itself.
struct CaseState{S}
  problem::NamedTuple
  prepared::S
  hash::String
end

function case_state(spec, rep)
  problem = scenario(spec.meta.scenario, rep)
  prepared = spec.build(problem)
  CaseState(problem, prepared, case_data_hash(problem, prepared))
end

function reset_state(state)
  sample = deepcopy(state.prepared)
  Random.seed!(SOLVER_SEED)
  sample
end

# A function barrier, so that the timed call dispatches statically
timed_call(f::F, sample::S) where {F, S} = @timed f(sample)

function observe!(state, spec, case_seconds)
  meta = spec.meta
  try
    sample = reset_state(state)
    timed = timed_call(spec.call, sample)
    ns = round(Int, timed.time * 1e9)
    gc = round(Int, timed.gctime * 1e9)
    cert, status, detail, stack = nothing, :ok, "", ""
    try
      cert = spec.certify(state.problem, state.prepared, timed.value)
      status = certificate_status(cert, meta)
      detail = status == :target_missed ?
        "value $(cert.value) exceeds the target $(meta.target); $(cert.note)" : cert.note
    catch err
      cert = nothing
      status = :error
      detail = sprint(showerror, err)
      stack = sprint(Base.show_backtrace, catch_backtrace())
    end
    if timed.time > case_seconds && status != :error
      status = :timeout
      detail = "single call exceeds HDM_CASE_SECONDS"
    end
    value = cert === nothing ? NaN : Float64(cert.value)
    (; time_ns = ns, gc_ns = gc, bytes = Int(timed.bytes), value, status, detail, stack)
  catch err
    (; time_ns = nothing, gc_ns = nothing, bytes = nothing, value = NaN,
       status = :error, detail = sprint(showerror, err),
       stack = sprint(Base.show_backtrace, catch_backtrace()))
  end
end

function aggregate_status(statuses)
  for status in CASE_STATUS_ORDER
    status in statuses && return status
  end
  :ok
end

function profile_case!(io, state, spec, seconds)
  seconds > 0 || return
  Profile.clear()
  deadline = time_ns() + round(UInt64, seconds * 1e9)
  while time_ns() < deadline
    sample = reset_state(state)
    output = Profile.@profile spec.call(sample)
    certificate_status(spec.certify(state.problem, state.prepared, output), spec.meta) == :ok ||
      error("profile output fails certification")
  end
  println(io, "CASE\t", spec.meta.id)
  Profile.print(io; format = :flat, C = true, sortedby = :count, mincount = 1)
  Profile.print(io; format = :tree, C = true, maxdepth = 22, mincount = 1)
  flush(io)
end

function measure_case(spec, settings, profiles)
  rows = NamedTuple[]
  observations = NamedTuple[]
  hash = ""
  try
    state = case_state(spec, settings.rep)
    hash = state.hash
    # The first call compiles; it is certified but contributes no sample.
    validation = observe!(state, spec, Inf)
    push!(observations, validation)
    if validation.status != :error && validation.time_ns !== nothing &&
        validation.time_ns / 1e9 > settings.case_seconds
      validation = observe!(state, spec, settings.case_seconds)
      push!(observations, validation)
    end
    if !(validation.status in (:error, :timeout))
      timed_seconds = 0.0
      while length(rows) < MIN_SAMPLES ||
          (length(rows) < settings.samples && timed_seconds < settings.seconds)
        row = observe!(state, spec, settings.case_seconds)
        push!(rows, row)
        push!(observations, row)
        row.time_ns === nothing || (timed_seconds += row.time_ns / 1e9)
        row.status in (:error, :timeout) && break
      end
    end
    if aggregate_status([r.status for r in observations]) == :ok && profiles !== nothing
      profile_case!(profiles, state, spec, settings.profile_seconds)
    end
  catch err
    push!(observations, (; status = :error, detail = sprint(showerror, err),
                        stack = sprint(Base.show_backtrace, catch_backtrace())))
  end
  status = aggregate_status([r.status for r in observations])
  failure = status == :ok ? nothing : first(r for r in observations if r.status == status)
  (; rows, status, failure, hash)
end

function timing_statistics(rows)
  ok = filter(r -> r.status == :ok, rows)
  isempty(ok) && return ntuple(_ -> nothing, 11)
  times = [Int(r.time_ns) for r in ok]
  bytes = [Int(r.bytes) for r in ok]
  integer(x) = round(Int, x)
  (integer(median(times)), integer(mean(times)),
   integer(std(times; corrected = false)),
   minimum(times), maximum(times), integer(quantile(times, 0.95)),
   sum(r.gc_ns for r in ok) / max(sum(times), 1),
   integer(median(bytes)), integer(mean(bytes)), minimum(bytes), maximum(bytes))
end

function record_case!(metrics, raw, errors, spec, settings, result)
  meta, rep, rows = spec.meta, settings.rep, result.rows
  for (i, row) in enumerate(rows)
    tsv_line(raw, meta.id, rep, i, row.time_ns, row.gc_ns, row.bytes,
             finite_field(row.value), row.status)
  end
  values = [r.value for r in rows if isfinite(r.value)]
  worst = isempty(values) ? nothing : maximum(values)
  typical = isempty(values) ? nothing : median(values)
  tsv_line(metrics, meta.id, meta.group, meta.phase, rep, result.status, length(rows),
           count(r -> r.status == :ok, rows), timing_statistics(rows)...,
           meta.measure, meta.target, worst, typical, result.hash)
  if result.status != :ok
    failure = result.failure
    tsv_line(errors, meta.id, rep, result.status, failure.detail, failure.stack)
  end
  flush(raw)
  flush(metrics)
  flush(errors)
end

function run_records(settings, catalog)
  started = time()
  mkpath(settings.output)
  stem = "julia-$(VERSION)-rep$(lpad(settings.rep, 2, '0'))"
  any(name -> startswith(name, stem), readdir(settings.output)) &&
    error("Refusing to overwrite files with prefix $stem; use a fresh HDM_OUTPUT")
  prefix = joinpath(settings.output, stem)
  metadata = start_metadata(settings)
  write_metadata(prefix * "-metadata-start.toml", metadata)
  open(prefix * "-catalog.tsv", "w") do io
    tsv_line(io, CATALOG_HEADER...)
    for spec in catalog
      tsv_line(io, (getproperty(spec.meta, Symbol(field)) for field in CATALOG_HEADER)...)
    end
  end
  order = shuffle!(harness_rng("measurement_order", settings.rep), copy(catalog))
  failures = 0
  profiles = settings.profile_seconds > 0 ? open(prefix * "-profiles.txt", "w") : nothing
  try
    profiles === nothing || Profile.init(n = 2_000_000, delay = 0.001)
    open(prefix * "-metrics.tsv", "w") do metrics
      open(prefix * "-samples.tsv", "w") do raw
        open(prefix * "-errors.tsv", "w") do errors
          tsv_line(metrics, METRICS_HEADER...)
          tsv_line(raw, SAMPLES_HEADER...)
          tsv_line(errors, ERRORS_HEADER...)
          for spec in order
            result = measure_case(spec, settings, profiles)
            record_case!(metrics, raw, errors, spec, settings, result)
            failures += result.status != :ok
            println("CASE ", spec.meta.id, " status=", result.status)
          end
        end
      end
    end
  finally
    profiles === nothing || close(profiles)
  end
  merge!(metadata, Dict("cases" => length(catalog), "ok_cases" => length(catalog) - failures,
    "nonok_cases" => failures, "elapsed_s" => time() - started,
    "maxrss_bytes" => Int(Sys.maxrss()), "finished" => true))
  # Publish completion only after every other record has closed successfully.
  temporary = prefix * "-metadata-finalizing.toml"
  write_metadata(temporary, metadata)
  mv(temporary, prefix * "-metadata.toml")
  println("REPETITION_FINISHED cases=", length(catalog), " nonok=", failures,
          " elapsed_s=", metadata["elapsed_s"])
  failures == 0 ? 0 : 2
end

function main()
  try
    settings = runner_settings()
    settings.list || require_clean_dependencies()
    catalog = build_catalog(settings.suite, settings.groups, settings.cases)
    if settings.list
      for spec in catalog
        tsv_line(stdout, spec.meta.id, spec.meta.suite, spec.meta.target)
      end
      return 0
    end
    Threads.nthreads() == 1 || error("run with JULIA_NUM_THREADS=1")
    BLAS.set_num_threads(1)
    run_records(settings, catalog)
  catch err
    println(stderr, "RECORD_FAILURE ", sprint(showerror, err))
    return 3
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  exit(main())
end
