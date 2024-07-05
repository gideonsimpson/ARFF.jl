struct ActivationFunction{TB<:Number} <: AbstractActivationFunction
    σ::Function # activation function that is TB valued
end

function (ϕ::ActivationFunction{TB})(x::TW, ω::TW) where {TB<:Number,TR<:AbstractFloat,TW<:AbstractArray{TR}}
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

# convenience structures
FourierActivation = ActivationFunction{ComplexF64}(fourier);
SigmoidActivation = ActivationFunction{Float64}(sigmoid);
ArcTanActivation = ActivationFunction{Float64}(arctan);