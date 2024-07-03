"""
    mse_loss(F::ScalarFourierModel{TC,TR,TW}, data_x::Vector{TW}, data_y::Vector{TC}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::ScalarFourierModel{TC,TR,TW,TI}, data_x::Vector{TW}, data_y::Vector{TC}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer}
    return mean(abs2, F.(data_x) - data_y)
end

"""
    mse_loss(F::ScalarFeatureModel{TC,TR,TW}, data_x::Vector{TW}, data_y::Vector{TC}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::ScalarFeatureModel{TC,TR,TW,TI}, data_x::Vector{TW}, data_y::Vector{TC}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer}
    return mean(abs2, F.(data_x) - data_y)
end

"""
    mse_loss(F::VectorFourierModel{TC,TR,TW,TB}, data_x::Vector{TW}, data_y::Vector{TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC}}

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::VectorFourierModel{TC,TR,TW,TB,TI}, data_x::Vector{TW}, data_y::Vector{TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC},TI<:Integer}
    return mean(norm.(F.(data_x) - data_y).^2)
end

"""
    mse_loss(F::VectorFeatureModel{TC,TR,TW,TB,TI,TF}, data_x::Vector{TW}, data_y::Vector{TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC}, TI<:Integer, TF<:Function}

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::VectorFeatureModel{TC,TR,TW,TB,TI,TF}, data_x::Vector{TW}, data_y::Vector{TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC},TI<:Integer,TF<:Function}
    return mean(norm.(F.(data_x) - data_y).^2)
end