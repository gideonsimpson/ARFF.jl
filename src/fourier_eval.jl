"""
    (F::ScalarFourierModel{TR,TY,TI,TA})(x::Vector{TR}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.
"""
function (F::ScalarFourierModel{TR,TY,TI,TA})(x::Vector{TR}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    y = zero(TY)
    for (β, ω) in F
        y += β * F.ϕ(x, ω)
    end
    return y

end


"""
    (F::VectorFourierModel{TR,TY,TI,TA})(x::Vector{TR}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.
"""
function (F::VectorFourierModel{TR,TY,TI,TA})(x::Vector{TR}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    y = zeros(TY, F.dy)
    for (β, ω) in F
        y += β * F.ϕ(x, ω)
    end
    return y

end


"""
    (F::ScalarFourierModel{TR,TY,TI,TA})(x::Vector{TR}, scalings::ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.  This takes in the scalings argument
so that x can be in the original units.
"""
function (F::ScalarFourierModel{TR,TY,TI,TA})(x::Vector{TR}, scalings::ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    y = zero(TY)
    x_scaled = (x - scalings.μx) ./ sqrt.(scalings.σ2x)
    for (β, ω) in F
        y += β * F.ϕ(x_scaled, ω)
    end

    y *= sqrt(scalings.σ2y)
    y += scalings.μy

    return y

end


"""
    (F::VectorFourierModel{TR,TY,TI,TA})(x::Vector{TR}, scalings::VectorDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.  This takes in the scalings argument
so that x can be in the original units.
"""
function (F::VectorFourierModel{TR,TY,TI,TA})(x::Vector{TR}, scalings::VectorDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    y = zero(TY, F.dy)
    x_scaled = (x - scalings.μx) ./ sqrt.(scalings.σ2x)
    for (β, ω) in F
        y += β * exp(im * (ω ⋅ x_scaled))
    end

    @. y *= sqrt(scalings.σ2y)
    @. y += scalings.μy

    return y

end

