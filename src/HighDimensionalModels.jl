"""
    HighDimensionalModels

The Gaussian lasso at one penalty level, [`lasso`](@ref), with an optional intercept and column
scales, and the group lasso at one penalty level, [`group_lasso`](@ref), with an optional
intercept and orthonormalized groups, as a thin layer on
ProximalBase (the loss and the penalty) and ProximalSolvers (coordinate descent).
"""
module HighDimensionalModels

using LinearAlgebra: ColumnNorm, I, UpperTriangular, qr!
using ProximalBase: BlockIterate, GroupNormL2, LeastSquaresLoss, NormL1
using ProximalSolvers: CDSolver, solve!
import StatsAPI: coef, predict, residuals, nobs, dof, deviance

export lasso, LassoSolution, group_lasso, GroupLassoSolution, CDSolver, coef, predict, residuals, nobs, dof, deviance

include("preprocess.jl")
include("lasso.jl")
include("group_lasso.jl")

end
