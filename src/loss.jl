"""
    mse_loss(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Mean squared error loss function
"""
function mse_loss(F::FourierModel{TB,TR,TW}, data_x::Vector{TW}, data_y::Vector{TB}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    return (norm(data_y - F.(data_x))^2) / length(data_x)
end