# Inputs of the benchmark cases: for a scenario key and a repetition,
# scenario(key, rep) returns a NamedTuple of raw Float64 arrays, drawn with the
# standard library only. Keys are family-shape, as ls-200x50; SCENARIOS lists
# them. Fields: X and y, and truth, the coefficients that generated the data.
# Float32 copies are made by the caller.

using LinearAlgebra, Random, Statistics, SHA

# BLAS, LAPACK and Statistics may round differently between Julia versions. So
# the inputs are computed from the draws by elementwise operations and by the
# products and moments below, which add in a fixed order.
function product(X::AbstractMatrix, b::AbstractVector)
  y = zeros(size(X, 1))
  for j in axes(X, 2), i in axes(X, 1)
    y[i] += X[i, j] * b[j]
  end
  y
end

function sample_mean(v)
  s = 0.0
  for t in v
    s += t
  end
  s / length(v)
end

function sample_sd(v)
  m, s = sample_mean(v), 0.0
  for t in v
    s += abs2(t - m)
  end
  sqrt(s / (length(v) - 1))
end

# The stream of a key and repetition: Xoshiro seeded with the first 8 bytes of
# sha256("$key|$rep") read as a big-endian integer (not Base.hash, which can
# change between Julia versions). Its scalar uniform draws are the same on
# Julia 1.10 and 1.13, but randn and randexp are not (their tail algorithms
# differ), nor are the bulk fills of long arrays. So every draw is a scalar
# rand: normals by Marsaglia's polar method and exponentials by inversion, with
# log and sqrt, which also agree; randperm and sampling from ranges are not used.
function scenario_rng(key::AbstractString, rep::Integer)
  digest = sha256("$key|$rep")
  Xoshiro(foldl((a, b) -> (a << 8) | UInt64(b), digest[1:8]; init = UInt64(0)))
end

function normal_draw(rng)
  while true
    u, v = 2rand(rng) - 1, 2rand(rng) - 1
    s = u * u + v * v
    0 < s < 1 && return u * sqrt(-2log(s) / s)
  end
end

exponential_draw(rng) = -log1p(-rand(rng))
gaussian(rng, dims::Integer...) = reshape([normal_draw(rng) for _ in 1:prod(dims)], dims...)
uniform(rng, dims::Integer...) = reshape([rand(rng) for _ in 1:prod(dims)], dims...)

# k distinct indices of 1:p, in random order
draw_indices(rng, p, k) = sortperm(uniform(rng, p))[1:k]
draw_signs(rng, k) = [rand(rng) < 0.5 ? -1.0 : 1.0 for _ in 1:k]

# a coefficient vector with k nonzeros of magnitudes in [0.5, 1.5]
function sparse_truth(rng, p, k)
  β = zeros(p)
  β[draw_indices(rng, p, k)] .= draw_signs(rng, k) .* (0.5 .+ uniform(rng, k))
  β
end

# the noise level that gives a signal-to-noise variance ratio snr
noise_sd(signal, snr) = (σ = sample_sd(signal); σ > 0 ? σ / sqrt(snr) : 1.0)

# Gaussian columns with shifted means and scales in [0.5, 2], so that centering
# and loadings matter; ρ > 0 makes the columns equicorrelated
function gaussian_design(rng, n, p; ρ = 0.0)
  Z = gaussian(rng, n, p)
  ρ > 0 && (Z .= sqrt(1 - ρ) .* Z .+ sqrt(ρ) .* gaussian(rng, n))
  m, s = 0.3 .* gaussian(rng, p), 0.5 .+ 1.5 .* uniform(rng, p)
  (Z .+ transpose(m)) .* transpose(s)
end

function least_squares(rng, n, p; k = 5, ρ = 0.0)
  X = gaussian_design(rng, n, p; ρ)
  β = sparse_truth(rng, p, k)
  signal = product(X, β)
  (; X, y = 1.0 .+ signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n), truth = β)
end

const SCENARIO_BUILDERS = [
  # quick shape
  "ls-200x50" => rng -> least_squares(rng, 200, 50),
  # extended shapes
  "ls-tall-5000x40" => rng -> least_squares(rng, 5000, 40),
  "ls-wide-100x400" => rng -> least_squares(rng, 100, 400),
  "ls-corr-300x100" => rng -> least_squares(rng, 300, 100; ρ = 0.5),
  "ls-2000x1000" => rng -> least_squares(rng, 2000, 1000; k = 10),
]

const SCENARIOS = String[first(b) for b in SCENARIO_BUILDERS]

function scenario(key::AbstractString, rep::Integer)
  k = findfirst(b -> first(b) == key, SCENARIO_BUILDERS)
  k === nothing && throw(ArgumentError("unknown scenario $key; the scenarios are $(join(SCENARIOS, ", "))"))
  rep >= 1 || throw(ArgumentError("the repetition must be positive, but it is $rep"))
  last(SCENARIO_BUILDERS[k])(scenario_rng(key, rep))
end

# A lowercase hex SHA-256 (64 characters) of the inputs over their sizes,
# element types and bytes, stable across processes and Julia versions: fields
# in the order of their names, arrays in column-major order, numbers little-endian
function hash_data!(io, a::AbstractArray)
  write(io, "Array{$(eltype(a))}")
  hash_data!(io, size(a))
  if isbitstype(eltype(a))
    for v in a
      write(io, htol(v))
    end
  else
    foreach(v -> hash_data!(io, v), a)
  end
end

function hash_data!(io, a::Tuple)
  write(io, htol(Int64(length(a))))
  foreach(v -> hash_data!(io, Int64(v)), a)
end

hash_data!(io, a::Union{Integer, AbstractFloat}) = (write(io, string(typeof(a))); write(io, htol(a)))
hash_data!(io, a::Bool) = write(io, a ? "true" : "false")
hash_data!(io, a::Union{AbstractString, Symbol}) = (s = string(a); write(io, htol(Int64(ncodeunits(s))), s))
hash_data!(io, ::Nothing) = write(io, "nothing")

function data_hash(nt::NamedTuple)
  io = IOBuffer()
  for name in sort!(collect(keys(nt)); by = string)
    write(io, string(name), '\0')
    hash_data!(io, getproperty(nt, name))
  end
  bytes2hex(sha256(take!(io)))
end
