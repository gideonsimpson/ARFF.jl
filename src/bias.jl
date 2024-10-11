"""
    append_bias(data) 

Create a new `DataSet` with a bias term appended to the end of the `x`
coordinate.
### Fields
* `data` - data set to be augmented
"""
function append_bias(data::ScalarDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    return ScalarDataSet([[deepcopy(x_); one(TR)] for x_ in data.x], deepcopy(data.y), data.N, data.dx + 1)
end

"""
    append_bias(data)

Create a new `DataSet` with a bias term appended to the end of the `x`
coordinate.
### Fields
* `data` - data set to be augmented
"""
function append_bias(data::VectorDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    return VectorDataSet([[deepcopy(x_); one(TR)] for x_ in data.x], deepcopy(data.y), data.N, data.dx + 1, data.dy)
end

"""
    append_bias(scalings)

Create a new `DataScalings` accounting for a bias term appended to the end of
the `x` coordinate.
### Fields
* `scalings` - scalings to be augmented
"""
function append_bias(scalings::ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
    return ScalarDataScalings([deepcopy(scalings.μx); zero(TR)], [deepcopy(scalings.σ2x); one(TR)], scalings.μy, scalings.σ2y)
end


"""
    append_bias(scalings)

Create a new `DataScalings` accounting for a bias term appended to the end of
the `x` coordinate.
### Fields
* `scalings` - scalings to be augmented
"""
function append_bias(scalings::VectorDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
    return VectorDataScalings([deepcopy(scalings.μx); zero(TR)], [deepcopy(scalings.σ2x); one(TR)], deepcopy(scalings.μy), deepcopy(scalings.σ2y))
end

