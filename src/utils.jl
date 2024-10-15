
"""
    optimal_γ(d)

Compute the optimal `γ` parameter as a function of dimension `d`
### Fields
* `d` - the dimension of the x coordinate
"""
function optimal_γ(d::Integer)
    return 3 * d - 2
end

function subsample(a::AbstractVector, rows::TI) where {TI<:Integer}
    return a[rows];

end

function subsample(a::AbstractVector, rows::AbstractVector{TI}) where {TI<:Integer}
    return @view a[rows];

end

function subsample(a::AbstractMatrix, rows::TI) where {TI<:Integer}
    return @view a[rows, :];

end

function subsample(a::AbstractMatrix, rows::AbstractVector{TI}) where {TI<:Integer}
    return @view a[rows,:];

end

function copy_entries!(a::AbstractVector, b::AbstractVector, rows::TI) where {TI<:Integer}
    a[rows] = b[rows];

end

function copy_entries!(a::AbstractMatrix, b::AbstractMatrix, rows::TI) where {TI<:Integer}
    @. a[rows,:] = b[rows,:];

end
