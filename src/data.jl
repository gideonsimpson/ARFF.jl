
"""
    ScalarDataSet{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer}

Training data containing (x,y) data pairs stored in arrays of x values and
arrays of y values.
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of complex scalars
* `N` - Number of data points
* `dx` - Dimension of `x` coordinates
"""
struct ScalarDataSet{TR,TB,TI} <: AbstractDataSet where {TR<:AbstractFloat, TB<:Number,  TI<:Integer}
    x::Vector{Vector{TR}}
    y::Vector{TB}
    N::TI
    dx::TI
end


"""
    VectorDataSet{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC},TI<:Integer}

Training data containing (x,y) data pairs stored in arrays of x values and
arrays of y values.
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of complex valued vectors
* `yt` - Tranposed array of complex valued vectors
* `N` - Number of data points
* `dx` - Dimension of `x` coordinates
* `dy` - Dimension of `y` coordinates
"""
struct VectorDataSet{TR,TY,TI} <: AbstractDataSet where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
    x::Vector{Vector{TR}}
    y::Vector{Vector{TY}}
    yt::Vector{Vector{TY}}
    N::TI
    dx::TI
    dy::TI
end

"""
    Base.length(D::TD) where {TD<:AbstractDataSet}

Return the number of data points in the data set.
"""
function Base.length(D::TD) where {TD<:AbstractDataSet}
    return D.N
end

"""
    Base.isempty(D::TD) where {TD<:AbstractDataSet}

Determine if the data set is trivial (zero entries) or not. 
"""
function Base.isempty(D::TD) where {TD<:AbstractDataSet}
    return isempty(D.x)
end


"""
    Base.size(D::TD) where {TD<:ScalarDataSet}

Returns the tuple `(N, dx)` of the number of samples, `N`, and the dimension of
the domain, `dx`, of a `ScalarDataSet`.
"""
function Base.size(D::TD) where {TD<:ScalarDataSet}
    return (D.N, D.dx)
end

"""
    Base.size(D::TD) where {TD<:VectorDataSet}

Returns the tuple `(N, dx, dy)` of the number of samples, `N`, the dimension of
the domain, `dx`, and hte dimension of the range, `dy`, of a `VectorDataSet`.
"""
function Base.size(D::TD) where {TD<:VectorDataSet}
    return (D.N, D.dx, D.dy)
end


"""
    Base.iterate(D::TD, state=1) where {TD<:ScalarDataSet}

TBW
"""
function Base.iterate(D::TD, state=1) where {TD<:ScalarDataSet}
    if state > D.N
        return nothing
    end
    return (D.x[state], D.y[state]), state + 1
end


"""
    Base.iterate(D::TD, state=1) where {TD<:VectorDataSet}

TBW
"""
function Base.iterate(D::TD, state=1) where {TD<:VectorDataSet}
    if state > D.N
        return nothing
    end
    return (D.x[state], [D.yt[state]]), state + 1
end


"""
    DataSet(x, y)

Constructor for a `ScalarDataSet`
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of scalars
"""
function DataSet(x::Vector{Vector{TR}}, y::Vector{TY}) where {TR<:AbstractFloat,TY<:Number}
    N = length(x);
    dx = length(x[1]);
    return ScalarDataSet(deepcopy(x), deepcopy(y), N, dx)
end


"""
    DataSet(x, y)

Constructor for a `VectorDataSet`
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of vectors
"""
function DataSet(x::Vector{Vector{TR}}, y::Vector{Vector{TY}}) where {TY<:Number, TR<:AbstractFloat}
    N = length(x)
    dx = length(x[1])
    dy = length(y[1])
    return VectorDataSet(deepcopy(x), deepcopy(y), deepcopy([[y_[i] for y_ in y] for i = 1:dy]), N, dx, dy)
end

