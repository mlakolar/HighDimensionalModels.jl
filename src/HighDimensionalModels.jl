"""
    HighDimensionalModels

The Gaussian lasso at one penalty level, [`lasso`](@ref), with an optional intercept and column
scales, as a thin layer on
ProximalBase (the loss and the penalty) and ProximalSolvers (coordinate descent).
"""
module HighDimensionalModels

using ProximalBase: LeastSquaresLoss, NormL1
using ProximalSolvers: CDSolver, solve!
import StatsAPI: coef, predict, residuals, nobs, dof, deviance

export lasso, LassoSolution, CDSolver, coef, predict, residuals, nobs, dof, deviance

include("preprocess.jl")
include("lasso.jl")

end
