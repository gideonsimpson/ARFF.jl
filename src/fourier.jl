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
    ScalarFourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
"""
struct ScalarFeatureModel{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TI<:Integer,FF<:Function} <: AbstractFourierModel
    β::Vector{TC}
    ω::Vector{TW}
    K::TI
    dx::TI
    ϕ::FF
end


"""
    VectorFourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `βt` - Transposed array of complex coefficients
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
* `dy` - Dimension of `y` coordinate
"""
struct VectorFourierModel{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC},TI<:Integer} <: AbstractFourierModel
    β::Vector{TB}
    βt::Vector{TB}
    ω::Vector{TW}
    K::TI
    dx::TI
    dy::TI
end

"""
    VectorFeatureModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `βt` - Transposed array of complex coefficients
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
* `dy` - Dimension of `y` coordinate
"""
struct VectorFeatureModel{TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC},TI<:Integer,FF<:Function} <: AbstractFourierModel
    β::Vector{TB}
    βt::Vector{TB}
    ω::Vector{TW}
    K::TI
    dx::TI
    dy::TI
    ϕ::FF
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
    Base.size(F::TF) where {TF<:ScalarFeatureModel}

TBW
"""
function Base.size(F::TF) where {TF<:ScalarFeatureModel}
    return (F.K, F.dx)
end


"""
    Base.size(F::TF) where {TF<:VectorFourierModel}

TBW
"""
function Base.size(F::TF) where {TF<:VectorFourierModel}
    return (F.K, F.dx, F.dy)
end

"""
    Base.size(F::TF) where {TF<:VectorFeatureModel}

TBW
"""
function Base.size(F::TF) where {TF<:VectorFeatureModel}
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
    Base.iterate(F::TF, state=1) where {TF<:ScalarFeatureModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:ScalarFeatureModel}
    if state > F.K
        return nothing
    end
    return (F.β[state], F.ω[state]), state + 1
end


"""
    Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}
    if state > F.K
        return nothing
    end
    return (F.β[state], F.ω[state]), state + 1
end

"""
    Base.iterate(F::TF, state=1) where {TF<:VectorFeatureModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:VectorFeatureModel}
    if state > F.K
        return nothing
    end
    return (F.β[state], F.ω[state]), state + 1
end

"""
    FourierModel(β::Vector{TC}, ω::Vector{TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

TBW
"""
function FourierModel(β::Vector{TR}, ω::Vector{TW}) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}
    K = length(ω)
    dx = length(ω[1])
    return ScalarFourierModel(complex.(β), ω, K, dx)
end

"""
    FourierModel(β::Vector{TC}, ω::Vector{TW}, ϕ::Function) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

TBW
"""
function FourierModel(β::Vector{TR}, ω::Vector{TW}, ϕ::Function) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}
    K = length(ω)
    dx = length(ω[1])
    return ScalarFeatureModel(complex.(β), ω, K, dx, ϕ)
end

"""
    FourierModel(β::Vector{TB}, ω::Vector{TW}) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}}

TBW
"""
function FourierModel(β::Vector{TB}, ω::Vector{TW}) where {TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TR}}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])
    return VectorFourierModel(complex.(β), [complex.([β_[d_] for β_ in β]) for d_ = 1:dy], ω, K, dx, dy)
end

"""
    FourierModel(β::Vector{TB}, ω::Vector{TW}, ϕ::TF) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}, TF<:Function}

TBW
"""
function FourierModel(β::Vector{TB}, ω::Vector{TW}, ϕ::TF) where {TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TR}, TF<:Function}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])
    return VectorFeatureModel(complex.(β), [complex.([β_[d_] for β_ in β]) for d_ = 1:dy], ω, K, dx, dy, ϕ)
end

"""
    copy_from_transpose!(F::VectorFourierModel)

TBW
"""
function copy_from_transpose!(F::VectorFourierModel)
    for d_ in 1:F.dy
        for k in 1:F.K
            F.β[k][d_] = F.βt[d_][k]
        end
    end
    F
end

"""
    copy_from_transpose!(F::VectorFeatureModel)

TBW
"""
function copy_from_transpose!(F::VectorFeatureModel)
    for d_ in 1:F.dy
        for k in 1:F.K
            F.β[k][d_] = F.βt[d_][k]
        end
    end
    F
end

"""
    copy_to_transpose!(F::VectorFourierModel)

TBW
"""
function copy_to_transpose!(F::VectorFourierModel)
    for d_ in 1:F.dy
        for k in 1:F.K
            F.βt[d_][k] = F.β[k][d_]
        end
    end
    F
end

"""
    copy_to_transpose!(F::VectorFeatureModel)

TBW
"""
function copy_to_transpose!(F::VectorFeatureModel)
    for d_ in 1:F.dy
        for k in 1:F.K
            F.βt[d_][k] = F.β[k][d_]
        end
    end
    F
end