
"""
    optimal_γ(d)

Compute the optimal `γ` parameter as a function of dimension `d`
### Fields
* `d` - the dimension of the x coordinate
"""
function optimal_γ(d::Integer)
    return 3 * d - 2
end

"""
    subsample(a::AbstractVector, rows::TI) where {TI<:Integer}

TBW
"""
function subsample(a::AbstractVector, rows::TI) where {TI<:Integer}
    return a[rows];

end

"""
    subsample(a::AbstractVector, rows::AbstractVector{TI}) where {TI<:Integer}

TBW
"""
function subsample(a::AbstractVector, rows::AbstractVector{TI}) where {TI<:Integer}
    return @view a[rows];

end

"""
    subsample(a::AbstractMatrix, rows::TI) where {TI<:Integer}

TBW
"""
function subsample(a::AbstractMatrix, rows::TI) where {TI<:Integer}
    return @view a[rows, :];

end

"""
    subsample(a::AbstractMatrix, rows::AbstractVector{TI}) where {TI<:Integer}

TBW
"""
function subsample(a::AbstractMatrix, rows::AbstractVector{TI}) where {TI<:Integer}
    return @view a[rows,:];

end

"""
    copy_entries!(a::AbstractVector, b::AbstractVector, rows::TI) where {TI<:Integer}

TBW
"""
function copy_entries!(a::AbstractVector, b::AbstractVector, rows::TI) where {TI<:Integer}
    a[rows] = b[rows];

end

"""
    copy_entries!(a::AbstractMatrix, b::AbstractMatrix, rows::TI) where {TI<:Integer}

TBW
"""
function copy_entries!(a::AbstractMatrix, b::AbstractMatrix, rows::TI) where {TI<:Integer}
    @. a[rows,:] = b[rows,:];

end

"""
    trivial_mutate!(F, x, y, S, epoch)

TBW
"""
function trivial_mutate!(F, x, y, S, epoch)
    F

end

"""
    trivial_resample!(F, x, y, S, epoch)

TBW
"""
function trivial_resample!(F, x, y, S, epoch)
    F
end