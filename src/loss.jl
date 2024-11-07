"""
    mse_loss(F, data_x, data_y)

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::ScalarFourierModel{TR,TY,TI,TA}, data_x::AbstractVector{Vector{TR}}, data_y::AbstractVector{TY}) where {TR<:AbstractFloat,TY<:Number,TI<:Integer,TA<:ActivationFunction{TY}}
    return mean(abs2, F.(data_x) - data_y)
end


"""
    mse_loss(F, data_x, data_y)

Mean squared error loss function
### Fields
* `F` - A `FourierModel` structure
* `data_x` - the x coordinates in training data
* `data_y` - the y coordinates in training data
"""
function mse_loss(F::VectorFourierModel{TR,TY,TI,TA}, data_x::AbstractVector{Vector{TR}}, data_y::AbstractVector{Vector{TY}}) where {TR<:AbstractFloat,TY<:Number,TI<:Integer,TA<:ActivationFunction{TY}}
    return mean(norm.(F.(data_x) - data_y).^2)
end
