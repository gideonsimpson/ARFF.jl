"""
    mse_loss(F, data_x, data_y)

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::ScalarFourierModel{TC,TR,TW}, data_x::Vector{TW}, data_y::Vector{TC}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    return (norm(data_y - F.(data_x))^2) / length(data_x)
end