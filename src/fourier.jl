"""
    ScalarFourierModel

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
    VectorFourierModel

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - 2D array of coefficients; `K` by `dy` in size
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

Return the number of terms, `K`, of the Fourier model.
"""
function Base.length(F::TF) where {TF<:AbstractFourierModel}
    return F.K
end

"""
    Base.size(F::TF) where {TF<:ScalarFourierModel}

Returns the tuple `(K, dx)` of the number of terms, `K`, and the dimension of
the domain, `dx`, of a scalar valued Fourier model.
"""
function Base.size(F::TF) where {TF<:ScalarFourierModel}
    return (F.K, F.dx)
end

"""
    Base.size(F::TF) where {TF<:VectorFourierModel}

Returns the tuple `(K, dx, dy)` of the number of terms, `K`, the dimension of
the domain, `dx`, and the dimension of the range, `dy`, of a vector valued
Fourier model.
"""
function Base.size(F::TF) where {TF<:VectorFourierModel}
    return (F.K, F.dx, F.dy)
end


"""
    Base.isempty(F::TF) where {TF<:FourierModel}

Determine if the Fourier model is trivial (zero terms) or not. 
"""
function Base.isempty(F::TF) where {TF<:AbstractFourierModel}
    return isempty(F.β)
end

"""
    Base.iterate(F::TF, state=1) where {TF<:ScalarFourierModel}

Iterate through the `(β, ω)` pairs characterizng the Fourier model
"""
function Base.iterate(F::TF, state=1) where {TF<:ScalarFourierModel}
    if state > F.K
        return nothing
    end
    return (F.β[state], F.ω[state]), state + 1
end

"""
    Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}

Iterate through the `(β, ω)` pairs characterizng the Fourier model
"""
function Base.iterate(F::TF, state=1) where {TF<:VectorFourierModel}
    if state > F.K
        return nothing
    end
    return (F.β[state, :], F.ω[state]), state + 1
end

"""
    FourierModel(β, ω)

Constructor for a Fourier features model. Defaults to complex exponentials for activation functions.
### Fields
* `β` - Array of coefficients
* `ω` - Array of wave numbers
"""
function FourierModel(β::Vector{TB}, ω::Vector{Vector{TR}}) where {TB <: Number,TR <: AbstractFloat}
    K = length(ω)
    dx = length(ω[1])
    TC = typeof(complex(β[1]))

    return ScalarFourierModel(complex.(β), ω, K, dx, ActivationFunction{TC}(fourier))
end

"""
    FourierModel(β, ω, ϕ) 

Constructor for a Fourier features model.
### Fields
* `β` - Array of coefficients
* `ω` - Array of wave numbers
* `ϕ` - Activation function of `ActivationFunction` type.  The data type of the
  `β` must agree with the data type of the range of `ϕ`.
"""
function FourierModel(β::Vector{TB}, ω::Vector{Vector{TR}}, ϕ::TA) where {TB<:Number,TR<:AbstractFloat,TA<:ActivationFunction{TB}}
    K = length(ω)
    dx = length(ω[1])
    return ScalarFourierModel(β, ω, K, dx, ϕ)
end

"""
    FourierModel(β, ω)

Constructor for a Fourier features model. Defaults to complex exponentials for activation functions.
### Fields
* `β` - Array of coefficients
* `ω` - Array of wave numbers
"""
function FourierModel(β::Matrix{TB}, ω::Vector{Vector{TR}}) where {TB<:Number,TR<:AbstractFloat}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])
    TC = typeof(complex(β[1,1]))

    return VectorFourierModel(complex.(β), ω, K, dx, dy, ActivationFunction{TC}(fourier))
end


"""
    FourierModel(β, ω, ϕ) 

Constructor for a Fourier features model.
### Fields
* `β` - Array of coefficients
* `ω` - Array of wave numbers
* `ϕ` - Activation function of `ActivationFunction` type.  The data type of the
  `β` must agree with the data type of the range of `ϕ`.
"""
function FourierModel(β::Matrix{TB}, ω::Vector{Vector{TR}}, ϕ::TA) where {TR<:AbstractFloat,TB<:Number,TA<:ActivationFunction{TB}}
    K = length(ω)
    dx = length(ω[1])
    dy = length(β[1])

    return VectorFourierModel(β, ω, K, dx, dy, ϕ)
end


"""
    FourierModel(β, ω) 

Constructor for a `VectorFourierModel`. Defaults to complex exponentials for
activation functions.
### Fields
* `β` - Array of coefficients
* `ω` - Array of wave numbers
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
    FourierModel(β, ω, ϕ) 

Constructor for a `VectorFourierModel`. Defaults to complex exponentials for
activation functions.
### Fields
* `β` - Array of coefficients 
* `ω` - Array of wave numbers 
* `ϕ` - Activation function of `ActivationFunction` type.  The data type of the
  `β` must agree with the data type of the range of `ϕ`.
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
