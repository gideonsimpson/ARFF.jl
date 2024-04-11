using Test
using ARFF
using Random
using Statistics
using SpecialFunctions

@testset "Data Sets" begin
    @test include("datasets/data1.jl")
    @test include("datasets/scalings1.jl")
    @test include("datasets/scalings2.jl")
    @test include("datasets/scalings3.jl")
end

@testset "Linear Algebra" begin
    @test include("linalg/matrix1.jl")
    @test include("linalg/normal1.jl")
    @test include("linalg/normal2.jl")
end

@testset "Options" begin
    @test include("opts/opts1.jl")
end

@testset "Utilities" begin
    @test include("utils/utils1.jl")
end


@testset "Training" begin
    @test include("train/train1.jl")
end