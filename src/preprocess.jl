# The preprocessing of `lasso` when `intercept` or `standardize_x` is true. The functions are
# internal. They use no package beyond Base.

# The largest power of two that is at most `a`, for a finite `a > 0`. A division by it is exact
# when the quotient is a normal number, so that the sums below add the entries themselves, scaled
# by a power of two.
pow2floor(a::T) where {T<:AbstractFloat} = ldexp(one(T), exponent(a))

# The mean of the entries of `v` (finite). A vector whose entries are all equal to the first one
# has that value as its mean. Otherwise, with `m` the power of two below `max |vᵢ|`, the mean is
# `m * (Σᵢ vᵢ/m / n)`: each quotient is below 2 in absolute value, so that the sum does not
# overflow, and the product is at most about `max |vᵢ|`.
function column_mean(v::AbstractVector{T}) where {T<:AbstractFloat}
  v1 = first(v)
  all(==(v1), v) && return v1
  m = pow2floor(maximum(abs, v))
  m * (sum(x -> x / m, v) / length(v))
end

# The root mean square of the entries of `v`, which has a nonzero entry and finite entries:
# `m * sqrt(Σᵢ (vᵢ/m)² / n)` with `m` the power of two below `max |vᵢ|`. The quotients are below
# 2 in absolute value, so the sum is below `4n`, and the result is at most about `max |vᵢ|`, which
# is finite, whereas `norm(v)` can exceed `floatmax` when the root mean square does not. A square
# that underflows is rounded to a subnormal number or to 0.
function root_mean_square(v::AbstractVector{T}) where {T<:AbstractFloat}
  m = pow2floor(maximum(abs, v))
  m * sqrt(sum(x -> abs2(x / m), v) / length(v))
end

nonfinite_error(what, step) = throw(ArgumentError("lasso: $what $step"))

# The data of the problem `min ‖ỹ - Zθ‖²/(2n) + λ‖θ‖₁`, which is the problem of `lasso` with the
# intercept and the column scales of the keywords (`θⱼ = sⱼβⱼ`). Returns `(; Z, ỹ, x̄, ȳ, s)`:
#
# - `x̄` and `ȳ`: the column means and the mean of `y` with `intercept`, zeros without it. A column
#   (or `y`) whose entries are all equal has that value as its mean (see `column_mean`), so that
#   it centres to exact zeros.
# - `Z`: the columns `(xⱼ - x̄ⱼ)/sⱼ`, or `xⱼ - x̄ⱼ` when `standardize` is false, in fresh arrays.
# - `ỹ`: `y - ȳ` (a fresh array) with `intercept`, `y` itself without it.
# - `s`: `nothing` when `standardize` is false. Otherwise the root mean square of the centred
#   column, which is 0 for a column whose centred entries are all zero (a zero column,
#   or a constant column with an intercept); such a column of `Z` is exact zeros.
#
# An `ArgumentError` that names `X` or `y` is thrown when an entry of `X` (or of `y`, with an
# intercept), a mean, a scale, an entry of `Z` or of `ỹ` is not finite, or when the scale of a
# column that is not all zeros underflows to 0.
function preprocess(X::Matrix{T}, y::Vector{T}, intercept::Bool, standardize::Bool) where {T<:AbstractFloat}
  n, p = size(X)
  x̄ = zeros(T, p)
  ȳ = zero(T)
  ỹ = y
  if intercept
    all(isfinite, y) || nonfinite_error("y", "has a non-finite entry")
    ȳ = column_mean(y)
    isfinite(ȳ) || nonfinite_error("the mean of y", "is not finite")
    ỹ = y .- ȳ
    all(isfinite, ỹ) || nonfinite_error("centring y", "overflowed")
  end
  Z = similar(X)
  s = standardize ? zeros(T, p) : nothing
  for j in 1:p
    xj, zj = view(X, :, j), view(Z, :, j)
    all(isfinite, xj) || nonfinite_error("column $j of X", "has a non-finite entry")
    if intercept
      x̄[j] = column_mean(xj)
      isfinite(x̄[j]) || nonfinite_error("the mean of column $j of X", "is not finite")
      zj .= xj .- x̄[j]
      all(isfinite, zj) || nonfinite_error("centring column $j of X", "overflowed")
    else
      copyto!(zj, xj)
    end
    standardize || continue
    iszero(maximum(abs, zj)) && continue                     # zero scale: s[j] = 0, zj stays zero
    sj = root_mean_square(zj)
    isfinite(sj) || nonfinite_error("the scale of column $j of X", "is not finite")
    iszero(sj) && nonfinite_error("the scale of column $j of X", "underflows to 0")
    s[j] = sj
    zj .= zj ./ sj                                           # a division: 1/sj can overflow for a small subnormal sj
    all(isfinite, zj) || nonfinite_error("dividing column $j of X by its scale", "overflowed")
  end
  (; Z, ỹ, x̄, ȳ, s)
end
