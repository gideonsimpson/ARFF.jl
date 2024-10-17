using Test
using ARFF
using Random
using Statistics
using LinearAlgebra
using SpecialFunctions

@testset "Scalar Data Sets" begin
    @test include("datasets/data1.jl")
    @test include("datasets/scalings1.jl")
    @test include("datasets/scalings2.jl")
    @test include("datasets/scalings3.jl")
end


@testset "Vector Data Sets" begin
    @test include("datasets/data2.jl")
    @test include("datasets/scalings4.jl")
    @test include("datasets/scalings5.jl")
    @test include("datasets/scalings6.jl")
end


@testset "Linear Algebra" begin
    @test include("linalg/matrix1.jl")
    @test include("linalg/matrix2.jl")
    @test include("linalg/normal1.jl")
    @test include("linalg/normal2.jl")
    @test include("linalg/normal3.jl")
end

# @testset "Options" begin
#     @test include("opts/opts1.jl")
# end

@testset "Utilities" begin
    @test include("utils/utils1.jl")
    @test include("utils/eval1.jl")
    @test include("utils/eval2.jl")
    @test include("utils/eval3.jl")
    @test include("utils/eval4.jl")
    @test include("utils/eval5.jl")
    @test include("utils/bias1.jl")
    @test include("utils/bias2.jl")
end


@testset "Training" begin
    @test include("train/train1.jl")
    @test include("train/train2.jl")
    @test include("train/train3.jl")
    @test include("train/train4.jl")
    # @test include("train/train5.jl")
    # @test include("train/train6.jl")
end