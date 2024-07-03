module ARFF

using LinearAlgebra
using Distributions
using Statistics
using Accessors
using ProgressMeter
using Flux
include("types.jl")
include("fourier.jl")
include("data.jl")
include("scalings.jl")
include("opts.jl")
include("utils.jl")
include("linear.jl")
include("train_scalar.jl")
include("train_vector.jl")
include("loss.jl")
include("features.jl")

export FourierModel, DataSet, ARFFOptions
export train_rwm, train_rwm!
export solve_normal!, solve_normal_svd!
export optimal_γ, get_scalings, scale_data!,rescale_data!, convert_problem, convert_dataset


end # module ARFF
