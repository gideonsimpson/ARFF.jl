"""
    sigmoid(ω, x)

TBW
"""
function sigmoid(ω, x)
    return 1/(1 + exp(-ω ⋅ x))
end

"""
    arctan(ω, x)

TBW
"""
function arctan(ω, x)
    return atan(ω ⋅ x)
end