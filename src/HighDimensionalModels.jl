"""
    HighDimensionalModels

The Gaussian lasso at one penalty level, [`lasso`](@ref), with an optional intercept and column
scales, as a thin layer on
ProximalBase (the loss and the penalty) and ProximalSolvers (coordinate descent).

The square-root lasso at one penalty level, [`sqrt_lasso`](@ref), minimizes the normalized
`‖y - β₀1 - Xβ‖/√n + λ‖θ‖₁` with ProximalSolvers' square-root coordinate descent and its duality
certificate.
"""
module HighDimensionalModels

using ProximalBase: LeastSquaresLoss, NormL1
using ProximalBase: SqrtLeastSquaresLoss
using ProximalSolvers: CDSolver, solve!
import StatsAPI: coef, predict, residuals, nobs, dof, deviance

export lasso, LassoSolution, CDSolver, coef, predict, residuals, nobs, dof, deviance
export sqrt_lasso, SqrtLassoSolution

include("preprocess.jl")
include("lasso.jl")
include("sqrt_lasso.jl")

end
