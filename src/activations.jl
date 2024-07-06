struct ActivationFunction{TB,TF} <: AbstractActivationFunction where {TB<:Number,TF<:Function}
    σ::TF # activation function that is TB valued
end
ActivationFunction{TB}(σ::TF) where {TB,TF} = ActivationFunction{TB,TF}(σ)

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