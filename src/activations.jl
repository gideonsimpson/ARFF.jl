"""
    ActivationFunction{TB,TF} <: AbstractActivationFunction where {TB<:Number,TF<:Function}

Activation function data structure.  `TB` should match the desired type of `y`
from the data.
### Fields
* `σ` - A scalar valued function that returns the `TB` data type.
"""
struct ActivationFunction{TB,TF} <: AbstractActivationFunction where {TB<:Number,TF<:Function}
    σ::TF # activation function that is TB valued
end
ActivationFunction{TB}(σ::TF) where {TB,TF} = ActivationFunction{TB,TF}(σ)

"""
    (ϕ::ActivationFunction{TB})(x::Vector{TR}, ω::Vector{TR}) where {TB<:Number,TR<:AbstractFloat}

Overload evaluation operator so that `ϕ(x,ω)` can be evaluated, where `ϕ`
represents an activation function, and what is evaluated is `ϕ.σ(x⋅ω)`
### Fields
* `x` - Point for evaluation
* `ω` - Vector for the feature
"""
function (ϕ::ActivationFunction{TB})(x::Vector{TR}, ω::Vector{TR}) where {TB<:Number,TR<:AbstractFloat}
    return ϕ.σ(x⋅ω)
end


"""
    fourier(z)

TBW
"""
function fourier(z) 
    return exp(im * z) 
end

"""
    sigmoid(z)

TBW
"""
function sigmoid(z) 
    return 1/(1 + exp(-z))
end

"""
    arctan(z)

TBW
"""
function arctan(z) 
    return atan(z)
end

# convenience constructions
FourierActivation = ActivationFunction{ComplexF64}(fourier);
SigmoidActivation = ActivationFunction{Float64}(sigmoid);
ArcTanActivation = ActivationFunction{Float64}(arctan);