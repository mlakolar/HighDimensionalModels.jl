module AquaTest
# The sequential runner ran Aqua last, with the optional backends of the test environment
# loaded by earlier files; load them here so the checks see their extensions.
using HighDimensionalModels, Aqua, Test
using HiGHS, ProximalAlgorithms, StatsModels

@testset "Aqua" begin
  Aqua.test_all(HighDimensionalModels)
end
end
