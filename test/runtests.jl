using Test
using ARFF
using Random
using Statistics

@testset "Data Sets" begin
    @test include("datasets/data1.jl")
    @test include("datasets/scalings1.jl")
    @test include("datasets/scalings2.jl")
    @test include("datasets/scalings3.jl")
end

