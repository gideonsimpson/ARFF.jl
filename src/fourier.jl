"""
    ScalarFourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
"""
struct ScalarFourierModel{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer} <: AbstractFourierModel
    β::Vector{TC}
    ω::Vector{TW}
    K::TI
    dx::TI
end


"""
    VectorFourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
* `dy` - Dimension of `y` coordinate
"""
struct VectorFourierModel{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC},TI<:Integer} <: AbstractFourierModel
    β::Vector{TB}
    ω::Vector{TW}
    K::TI
    dx::TI
    dy::TI
end

"""
    Base.length(F::TF) where {TF<:AbstractFourierModel}

TBW
"""
function Base.length(F::TF) where {TF<:AbstractFourierModel}
    return F.K
end

"""
    Base.size(F::TF) where {TF<:ScalarFourierModel}

TBW
"""
function Base.size(F::TF) where {TF<:ScalarFourierModel}
    return (F.K, F.dx)
end


"""
    Base.size(F::TF) where {TF<:ScalarFourierModel}

TBW
"""
function Base.size(F::TF) where {TF<:VectorFourierModel}
    return (F.K, F.dx, F.dy)
end


"""
    Base.isempty(F::TF) where {TF<:FourierModel}

TBW
"""
function Base.isempty(F::TF) where {TF<:AbstractFourierModel}
    return isempty(F.β)
end

"""
    Base.iterate(F::TF, state=1) where {TF<:ScalarFourierModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:ScalarFourierModel}
    if state > F.K
        return nothing
    end
    return (F.β[state], F.ω[state]), state + 1
end


"""
    Base.iterate(F::TF, state=1) where {TF<:ScalarFourierModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}
    if state > F.K
        return nothing
    end
    return ([F.β[i][state] for i in 1:F.dy], F.ω[state]), state + 1
end

"""
    FourierModel(β::Vector{TC}, ω::Vector{TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

TBW
"""
function FourierModel(β::Vector{TC}, ω::Vector{TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    K = length(ω)
    dx = length(ω[1])
    return ScalarFourierModel(β, ω, K, dx)
end

"""
    FourierModel(β::Vector{TB}, ω::Vector{TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}}

TBW
"""
function FourierModel(β::Vector{TB}, ω::Vector{TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])
    return VectorFourierModel(β, ω, K, dx, dy)
end