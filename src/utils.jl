
"""
    optimal_γ(d::Integer)

Compute the optimal γ parameter as a function of dimension `d`
### Fields
* `d` - the dimension of the x coordinate
"""
function optimal_γ(d::Integer)
    return 3 * d - 2
end

"""
    (F::ScalarFourierModel{TB,TR,TW})(x::TW) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.
"""
function (F::ScalarFourierModel{TR,TB,TI,TA})(x::Vector{TR}) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    y = zero(TB)
    # for (β, ω) in F
    #     y += β * F.ϕ(x,ω)
    # end
    for i in 1:F.K
        y += F.β[i] * F.ϕ(x, F.ω[i]);
    end
    return y

end


"""
    (F::VectorFourierModel{TC,TR,Vector{TR},Vector{TC}})(x::Vector{TR}) where {TC<:Complex,TR<:AbstractFloat}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.
"""
function (F::VectorFourierModel{TR,TB,TI,TA})(x::Vector{TR}) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    y = zeros(TB, F.dy)
    for (β, ω) in F
        y += β * F.ϕ(x, ω)
    end
    return y

end


"""
    (F::ScalarFourierModel{TB,TR,TW})(x::TW, scalings::ScalarDataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.  This takes in the scalings argument
so that x can be in the original units.
"""
function (F::ScalarFourierModel{TR,TB,TI,TA})(x::Vector{TR}, scalings::ScalarDataScalings{TR,TB}) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    y = zero(TB)
    x_scaled = (x - scalings.μx) ./ sqrt.(scalings.σ2x)
    for (β, ω) in F
        y += β * F.ϕ(x_scaled,  ω)
    end

    y *= sqrt(scalings.σ2y)
    y += scalings.μy

    return y

end


"""
    (F::FourierModel{TB,TR,TW})(x::TW, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.  This takes in the scalings argument
so that x can be in the original units.
"""
function (F::VectorFourierModel{TR,TB,TI,TA})(x::Vector{TR}, scalings::VectorDataScalings{TR,TB}) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    y = zero(TB, F.dy)
    x_scaled = (x - scalings.μx) ./ sqrt.(scalings.σ2x)
    for (β, ω) in F
        y += β * exp(im * (ω ⋅ x_scaled))
    end

    @. y *= sqrt(scalings.σ2y)
    @. y += scalings.μy

    return y

end


"""
    convert_problem(model::FourierModel)

Convert a Fourier Problem (non-Flux compatible) to a Flux Model
"""
function convert_problem(model::ScalarFourierModel)
    W = transpose(hcat(model.ω...))
    b = reshape(copy(model.β), 1, length(model.β))
    return Chain(
        Dense(W, false, fourier_features),
        Dense(b, false, identity)
    )
end

"""
    convert_dataset(data::DataSet)

Convert a dataset designed to work with FourierModel to work with RFF
"""
function convert_dataset(data::ScalarDataSet)
    return collect(zip(data.x, data.y))
end

