
"""
    ScalarDataSet{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer}

Training data containing (x,y) data pairs stored in arrays of x values and
arrays of y values.
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of complex scalars
* `N` - Number of data points
"""
struct ScalarDataSet{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer} <: AbstractDataSet
    x::Vector{TW}
    y::Vector{TC}
    N::TI
    dx::TI
end


"""
    VectorDataSet{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC},TI<:Integer}

Training data containing (x,y) data pairs stored in arrays of x values and
arrays of y values.
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of complex scalars
* `N` - Number of data points
"""
struct VectorDataSet{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC},TI<:Integer} <: AbstractDataSet
    x::Vector{TW}
    y::Vector{TC}
    N::TI
    dx::TI
    dy::TI
end

"""
    Base.length(D::TD) where {TD<:AbstractDataSet}

TBW
"""
function Base.length(D::TD) where {TD<:AbstractDataSet}
    return D.N
end

"""
    Base.isempty(D::TD) where {TD<:AbstractDataSet}

TBW
"""
function Base.isempty(D::TD) where {TD<:AbstractDataSet}
    return isempty(D.x)
end


"""
    Base.size(D::TD) where {TD<:ScalarDataSet}

TBW
"""
function Base.size(D::TD) where {TD<:ScalarDataSet}
    return (D.N, D.dx)
end

"""
    Base.size(D::TD) where {TD<:VectorDataSet}

TBW
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
    return (D.x[state], [D.y[i][state] for i in 1:D.d]), state + 1
end



"""
    DataSet(x::Vector{TW}, y::Vector{TR}) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}

Convenience constructor for real valued y data
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of real scalars
"""
function DataSet(x::Vector{TW}, y::Vector{TR}) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}
    N = length(x);
    dx = length(x[1]);
    return ScalarDataSet(x, Complex.(y), N, dx)
end


"""
    DataSet(x::Vector{TW}, y::Vector{TR}) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}

Convenience constructor for real valued y data
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of real scalars
"""
function DataSet(x::Vector{TW}, y::Vector{TB}) where {TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TR}}
    N = length(x)
    dx = length(x[1])
    dy = length(y[1])
    return VectorDataSet(x, [Complex.([y_[i] for y_ in y]) for i = 1:dy], N, dx, dy)
end

