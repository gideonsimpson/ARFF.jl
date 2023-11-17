
"""
    optimal_γ(d::Integer)

Compute the optimal γ parameter as a function of dimension
"""
function optimal_γ(d::Integer)
    return 3 * d - 2
end

"""
    (F::FourierModel{TB,TR,TW})(x::TW) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.
"""
function (F::FourierModel{TB,TR,TW})(x::TW) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    y = zero(TB)
    for (β, ω) in F
        y += β * exp(im * (ω ⋅ x))
    end
    return y

end

"""
    (F::FourierModel{TB,TR,TW})(x::TW, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Overload evaluation operator so that `F(x)` can be evaluated, where `F`
corresponds to the Fourier feature model.  This takes in the scalings argument
so that x can be in the original units.
"""
function (F::FourierModel{TB,TR,TW})(x::TW, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    y = zero(TB)
    x_scaled = (x - scalings.μx) ./ sqrt.(scalings.σ2x)
    for (β, ω) in F
        y += β * exp(im * (ω ⋅ x_scaled))
    end

    y *= sqrt(scalings.σ2y)
    y += scalings.μy

    return y

end

"""
    get_scalings(data::DataSet{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Find the means and variances of the data for scaling.
"""
function get_scalings(data::DataSet{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    μx = mean(data.x)
    σ2x = var(data.x)
    μy = mean(data.y)
    σ2y = var(data.y)

    return DataScalings(μx, σ2x, μy, σ2y)
end

"""
    scale_data!(data::DataSet{TB,TR,TW}, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Scale the data set (in-place) according to the specified scalings
"""
function scale_data!(data::DataSet{TB,TR,TW}, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

    for i in 1:length(data)
        @. data.x[i] = (data.x[i] - scalings.μx) / sqrt(scalings.σ2x)
        data.y[i] = (data.y[i] - scalings.μy) / sqrt(scalings.σ2y)
    end

    data
end

"""
    rescale_data!(data::DataSet{TB,TR,TW}, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Rescale the data set (in-place) according back to the original units
"""
function rescale_data!(data::DataSet{TB,TR,TW}, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

    for i in 1:length(data)
        @. data.x[i] = scalings.μx + sqrt(scalings.σ2x) * data.x[i]
        data.y[i] = scalings.μy + sqrt(scalings.σ2y) * data.y[i]
    end

    data
end

"""
    fourier_features(x, W)

Helper function to evaluate RFF models (activation function)
"""
function fourier_features(x)
    return exp(im * x)
end

"""
    convert_problem(model::FourierModel)

Convert a Fourier Problem (non-Flux compatible) to a Flux Model
"""
function convert_problem(model::FourierModel)
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
function convert_dataset(data::DataSet)
    return collect(zip(data.x, data.y))
end