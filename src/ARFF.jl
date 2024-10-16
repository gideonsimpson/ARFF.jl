module ARFF

using LinearAlgebra
using Distributions
using Statistics
using Accessors
using ProgressMeter
#using Flux
include("types.jl")
include("activations.jl")
include("fourier.jl")
include("rwm.jl")
include("data.jl")
include("scalings.jl")
include("bias.jl")
include("opts.jl")
include("fourier_eval.jl")
include("utils.jl")
include("linear.jl")
include("loss.jl")
include("solver.jl")
include("train.jl")
include("train2.jl")
include("train_scalar.jl")
include("train_scalar2.jl")
include("train_vector.jl")
include("train_vector2.jl")

export FourierModel, DataSet, ActivationFunction, ARFFOptions, ARFFSolver
export train_rwm!, train_rwm
export train_rwm2!
export solve_normal!, solve_normal_svd!
export optimal_γ, get_scalings, scale_data!,rescale_data!
export append_bias
#, convert_problem, convert_dataset
export FourierActivation, SigmoidActivation, ArcTanActivation

end # module ARFF
