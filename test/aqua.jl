module AquaTest
using HighDimensionalModels, Aqua, Test

@testset "Aqua" begin
  Aqua.test_all(HighDimensionalModels)
end
end
