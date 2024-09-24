
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
    append_bias(data::ScalarDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}

Create a new `DataSet` with a bias term appended to the end of the `x` coordinate.
### Fields
* `data` - data set to be augmented
"""
function append_bias(data::ScalarDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    return ScalarDataSet([[x_; one(TR)] for x_ in data.x], data.y, data.N, data.dx + 1)
end

"""
    append_bias(data::VectorDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}

Create a new `DataSet` with a bias term appended to the end of the `x` coordinate.
### Fields
* `data` - data set to be augmented
"""
function append_bias(data::VectorDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    return VectorDataSet([[x_; one(TR)] for x_ in data.x], data.y, data.N, data.dx + 1)
end

"""
    append_bias(scalings::ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}

Create a new `DataScalings` accounting for a bias term appended to the end of the `x` coordinate.
### Fields
* `scalings` - scalings to be augmented
"""
function append_bias(scalings::ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
    return ScalarDataScalings([scalings.μx; zero(TR)], [scalings.μx; one(TR)], scalings.μy, scalings.σ2y)
end


"""
    append_bias(scalings::VectorDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}

Create a new `DataScalings` accounting for a bias term appended to the end of the `x` coordinate.
### Fields
* `scalings` - scalings to be augmented
"""
function append_bias(scalings::VectorDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
    return VectorDataScalings([scalings.μx; zero(TR)], [scalings.μx; one(TR)], scalings.μy, scalings.σ2y)
end

