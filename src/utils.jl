
"""
    optimal_γ(d)

Compute the optimal `γ` parameter as a function of dimension `d`
### Fields
* `d` - the dimension of the x coordinate
"""
function optimal_γ(d::Integer)
    return 3 * d - 2
end

function subsample(a::TA, rows) where {TA<:AbstractVector}
    return @view a[rows];

end

function subsample(a::TA, rows) where {TA<:AbstractMatrix}
    return @view a[rows,:]

end
