"""
    ScalarFourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
* `ϕ` - Activation function
"""
struct ScalarFourierModel{TR,TB,TI,TA} <: AbstractFourierModel where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    β::Vector{TB}
    ω::Vector{Vector{TR}}
    K::TI
    dx::TI
    ϕ::TA
end

"""
    VectorFourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - 2D array of coefficients; K by dy in size
* `ω` - Array of wave vectors
* `K` - Number of Fourier features
* `dx` - Dimension of `x` coordinate
* `dy` - Dimension of `y` coordinate
"""
struct VectorFourierModel{TR,TY,TI,TA} <: AbstractFourierModel where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    β::Matrix{TY}
    ω::Vector{Vector{TR}}
    K::TI
    dx::TI
    dy::TI
    ϕ::TA
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
    Base.size(F::TF) where {TF<:VectorFourierModel}

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
    Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}
    if state > F.K
        return nothing
    end
    return (F.β[state, :], F.ω[state]), state + 1
end

"""
    FourierModel(β::Vector{TR}, ω::Vector{Vector{TR}}) where {TR<:AbstractFloat}

TBW
"""
function FourierModel(β::Vector{TB}, ω::Vector{Vector{TR}}) where {TB <: Number,TR <: AbstractFloat}
    K = length(ω)
    dx = length(ω[1])
    TC = typeof(complex(β[1]))

    return ScalarFourierModel(complex.(β), ω, K, dx, ActivationFunction{TC}(fourier))
end

"""
    FourierModel(β::Vector{TB}, ω::Vector{Vector{TR}}, ϕ::TA) where {TB<:Number,TR<:AbstractFloat,TA<:ActivationFunction{TB}}

TBW
"""
function FourierModel(β::Vector{TB}, ω::Vector{Vector{TR}}, ϕ::TA) where {TB<:Number,TR<:AbstractFloat,TA<:ActivationFunction{TB}}
    K = length(ω)
    dx = length(ω[1])
    return ScalarFourierModel(β, ω, K, dx, ϕ)
end

"""
    FourierModel(β::Vector{Vector{TR}}, ω::Vector{Vector{TR}}) where {TR<:AbstractFloat}

TBW
"""
function FourierModel(β::Matrix{TB}, ω::Vector{Vector{TR}}) where {TB<:Number,TR<:AbstractFloat}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])
    TC = typeof(complex(β[1,1]))

    return VectorFourierModel(complex.(β), ω, K, dx, dy, ActivationFunction{TC}(fourier))
end


"""
    FourierModel(β::Vector{Vector{TB}}, ω::Vector{Vector{TR}}, ϕ::TA) where {TR<:AbstractFloat,TB<:Number,TA<:ActivationFunction{TB}}

TBW
"""
function FourierModel(β::Matrix{TB}, ω::Vector{Vector{TR}}, ϕ::TA) where {TR<:AbstractFloat,TB<:Number,TA<:ActivationFunction{TB}}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])

    return VectorFourierModel(β, ω, K, dx, dy, ϕ)
end


"""
    FourierModel(β::Vector{Vector{TR}}, ω::Vector{Vector{TR}}) where {TR<:AbstractFloat}

TBW
"""
function FourierModel(β::Vector{Vector{TB}}, ω::Vector{Vector{TR}}) where {TB <: Number, TR <: AbstractFloat}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])
    TC = typeof(complex(β[1][1]));

    β_ = zeros(TC, K, dx);
    for k in 1:K
        @. β_[k,:] = β[k];
    end

    return VectorFourierModel(β_, ω, K, dx, dy, ActivationFunction{TC}(fourier))
end

"""
    FourierModel(β::Vector{Vector{TB}}, ω::Vector{Vector{TR}}, ϕ::TA) where {TR<:AbstractFloat,TB<:Number,TA<:ActivationFunction{TB}}

TBW
"""
function FourierModel(β::Vector{Vector{TB}}, ω::Vector{Vector{TR}}, ϕ::TA) where {TR<:AbstractFloat,TB<:Number,TA<:ActivationFunction{TB}}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])

    β_ = zeros(TB, K, dx)
    for k in 1:K
        @. β_[k, :] = β[k]
    end


    return VectorFourierModel(β_, ω, K, dx, dy, ϕ)
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
