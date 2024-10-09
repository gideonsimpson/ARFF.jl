"""
    append_bias(data) 

Create a new `DataSet` with a bias term appended to the end of the `x`
coordinate.
### Fields
* `data` - data set to be augmented
"""
function append_bias(data::ScalarDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    return ScalarDataSet([[x_; one(TR)] for x_ in data.x], data.y, data.N, data.dx + 1)
end

"""
    append_bias(data)

Create a new `DataSet` with a bias term appended to the end of the `x`
coordinate.
### Fields
* `data` - data set to be augmented
"""
function append_bias(data::VectorDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    return VectorDataSet([[x_; one(TR)] for x_ in data.x], data.y, data.N, data.dx + 1)
end

"""
    append_bias(scalings)

Create a new `DataScalings` accounting for a bias term appended to the end of
the `x` coordinate.
### Fields
* `scalings` - scalings to be augmented
"""
function append_bias(scalings::ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
    return ScalarDataScalings([scalings.μx; zero(TR)], [scalings.σ2x; one(TR)], scalings.μy, scalings.σ2y)
end


"""
    append_bias(scalings)

Create a new `DataScalings` accounting for a bias term appended to the end of
the `x` coordinate.
### Fields
* `scalings` - scalings to be augmented
"""
function append_bias(scalings::VectorDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
    return VectorDataScalings([scalings.μx; zero(TR)], [scalings.σ2x; one(TR)], scalings.μy, scalings.σ2y)
end

