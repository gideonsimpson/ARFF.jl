module ARFF

using LinearAlgebra
using Distributions
using Statistics
using Accessors
using ProgressMeter
using Flux

include("structs.jl")
include("utils.jl")
include("linear.jl")
include("train.jl")
include("loss.jl")

export FourierModel, DataSet, ARFFOptions, DataScalings
export train_rwm, train_rwm!
export optimal_γ, get_scalings, scale_data!,rescale_data!, convert_problem, convert_dataset


end # module ARFF
