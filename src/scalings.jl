"""
    ScalarDataScalings{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Data structure holding the scalings of a `DataSet` type.  Used for centering and
scaling the data to improve training.
### Fields
* `μx` - Mean in `x`
* `σ2x` - Variance in `x`
* `μy` - Mean in `y`
* `σ2y` - Variance in `y`
"""
struct ScalarDataScalings{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}} <: AbstractDataScalings
    μx::TW
    σ2x::TW
    μy::TC
    σ2y::TR
end

"""
    VectorDataScalings{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}}

Data structure holding the scalings of a `DataSet` type.  Used for centering and
scaling the data to improve training.
### Fields
* `μx` - Mean in `x`
* `σ2x` - Variance in `x`
* `μy` - Mean in `y`
* `σ2y` - Variance in `y`
"""
struct VectorDataScalings{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}} <: AbstractDataScalings
    μx::TW
    σ2x::TW
    μy::TB
    σ2y::TR
end


"""
    get_scalings(data::ScalarDataSet{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Find the means and variances of the data for scaling
### Fields
* `data` - The training data set
"""
function get_scalings(data::ScalarDataSet{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    μx = mean(data.x)
    σ2x = var(data.x)
    μy = mean(data.y)
    σ2y = var(data.y)

    return ScalarDataScalings(μx, σ2x, μy, σ2y)
end



"""
    get_scalings(data::VectorDataSet{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Find the means and variances of the data for scaling
### Fields
* `data` - The training data set
"""
function get_scalings(data::VectorDataSet{TC,TR,TW,TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}}
    μx = mean(data.x)
    σ2x = var(data.x)
    μy = mean(data.y)
    σ2y = var(data.y)

    return VectorDataScalings(μx, σ2x, μy, σ2y)
end

"""
    scale_data!(data::ScalarDataSet{TC,TR,TW}, scalings::ScalarDataScalings{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Scale the data set (in-place) according to the specified scalings
### Fields
* `data` - Data set to be scale
* `scalings` - Scalings to apply to `data`
"""
function scale_data!(data::ScalarDataSet{TC,TR,TW}, scalings::ScalarDataScalings{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

    for i in 1:data.N
        @. data.x[i] = (data.x[i] - scalings.μx) / sqrt(scalings.σ2x)
        data.y[i] = (data.y[i] - scalings.μy) / sqrt(scalings.σ2y)
    end

    data
end

"""
    scale_data!(data::ScalarDataSet{TC,TR,TW}, scalings::ScalarDataScalings{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Scale the data set (in-place) according to the specified scalings
### Fields
* `data` - Data set to be scale
* `scalings` - Scalings to apply to `data`
"""
function scale_data!(data::VectorDataSet{TC,TR,TW,TB}, scalings::VectorDataScalings{TC,TR,TW,TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC}}

    for i in 1:data.N
        @. data.x[i] = (data.x[i] - scalings.μx) / sqrt(scalings.σ2x)
        @. data.y[i] = (data.y[i] - scalings.μy) / sqrt(scalings.σ2y)
    end

    data
end

"""
    rescale_data!(data::ScalarDataSet{TC,TR,TW}, scalings::ScalarDataScalings{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Rescale the data set (in-place) according back to the original units
### Fields
* `data` - Data set to be scale
* `scalings` - Scalings to apply to `data`
"""
function rescale_data!(data::ScalarDataSet{TC,TR,TW}, scalings::ScalarDataScalings{TC,TR,TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

    for i in 1:length(data)
        @. data.x[i] = scalings.μx + sqrt(scalings.σ2x) * data.x[i]
        data.y[i] = scalings.μy + sqrt(scalings.σ2y) * data.y[i]
    end

    data
end


"""
    rescale_data!(data::VectorDataSet{TC,TR,TW,TB}, scalings::VectorDataScalings{TC,TR,TW,TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC}}

Rescale the data set (in-place) according back to the original units
### Fields
* `data` - Data set to be scale
* `scalings` - Scalings to apply to `data`
"""
function rescale_data!(data::VectorDataSet{TC,TR,TW,TB}, scalings::VectorDataScalings{TC,TR,TW,TB}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}, TB<:AbstractArray{TC}}

    for i in 1:length(data)
        @. data.x[i] = scalings.μx + sqrt(scalings.σ2x) * data.x[i]
        @. data.y[i] = scalings.μy + sqrt(scalings.σ2y) * data.y[i]
    end

    data
end

