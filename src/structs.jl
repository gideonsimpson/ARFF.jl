
"""
    FourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}} 

Structure containing a scalar valued fourier model which will be learned
### Fields
* `β` - Array of complex coefficients
* `ω` - Array of wave vectors
"""
struct FourierModel{TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    β::Vector{TB}
    ω::Vector{TW}
end

"""
    Base.length(F::TF) where {TF<:FourierModel}

TBW
"""
function Base.length(F::TF) where {TF<:FourierModel}
    return length(F.β)
end

"""
    Base.isempty(F::TF) where {TF<:FourierModel}

TBW
"""
function Base.isempty(F::TF) where {TF<:FourierModel}
    return isempty(F.β)
end

"""
    Base.iterate(F::TF, state=1) where {TF<:FourierModel}

TBW
"""
function Base.iterate(F::TF, state=1) where {TF<:FourierModel}
    if state > length(F)
        return nothing
    end
    return (F.β[state], F.ω[state]), state + 1
end

"""
    DataSet{TB<:Complex,TW<:AbstractArray{AbstractFloat}}

Training data containing (x,y) data pairs stored in arrays of x values and
arrays of y values.
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of complex scalars
"""
struct DataSet{TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    x::Vector{TW}
    y::Vector{TB}
end

"""
    DataSet(x::Vector{TW}, y::Vector{TR}) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}

Convenience constructor for real valued y data
### Fields
* `x` - Array of real valued vectors 
* `y` - Array of real scalars
"""
function DataSet(x::Vector{TW}, y::Vector{TR}) where {TR<:AbstractFloat,TW<:AbstractArray{TR}}
    return DataSet(x, Complex.(y))
end

"""
    Base.length(D::TD) where {TD<:DataSet}

TBW
"""
function Base.length(D::TD) where {TD<:DataSet}
    return length(D.x)
end

"""
    Base.isempty(D::TD) where {TD<:DataSet}

TBW
"""
function Base.isempty(D::TD) where {TD<:DataSet}
    return isempty(D.x)
end

"""
    Base.iterate(D::TD, state=1) where {TD<:DataSet}

TBW
"""
function Base.iterate(D::TD, state=1) where {TD<:DataSet}
    if state > length(D)
        return nothing
    end
    return (D.x[state], D.y[state]), state + 1
end

"""
    DataScalings{TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}

Data structure holding the scalings of a `DataSet` type.  Used for centering and
scaling the data to improve training.
### Fields
* `μx` - Mean in `x`
* `σ2x` - Variance in `x`
* `μy` - Mean in `y`
* `σ2y` - Variance in `y`
"""
struct DataScalings{TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}
    μx::TW
    σ2x::TW
    μy::TB
    σ2y::TR
end

"""
    ARFFOptions{TI<:Integer,TF<:AbstractFloat,TB<:Bool,TL,TS}

Data structure containing the training options and parameters
### Fields
* `n_epochs` - Number of training epochs
* `n_ω_steps` - Number of internal RWM steps
* `δ` - RWM proposal step size
* `n_burn` - Number of epochs before the covariance adaptation begins
* `γ` - Metropolis-Hastings exponent
* `ω_max` - Maximum wave number norm cutoff
* `adapt_covariance` - Boolean for adaptivity
* `linear_solve!` - User specified solver for the normal equations
* `loss` - User specified loss function
"""
struct ARFFOptions{TI<:Integer,TF<:AbstractFloat,TB<:Bool,TL,TS}
    n_epochs::TI # M in text, total number of iterations
    n_ω_steps::TI # m in text, number of steps between full β updates
    δ::TF # RWM step size scaling
    n_burn::TI # burnin time, in epochs
    γ::TI # Metropolis-Hastings exponent parameter
    ω_max::TF # maximum frequency (in norm)
    adapt_covariance::TB
    linear_solve!::TS
    loss::TL
end

"""
    ARFFOptions(n_epochs::TI, n_ω_steps::TI, n_burn::TI, γ::TI,
    ω_max::TF, adapt_covariance::TB, linear_solve!::TS, loss::TL) where {TI<:Integer,TF<:AbstractFloat,TB<:Bool,TL,TS}

Convenience constructor for older version of options
"""
function ARFFOptions(n_epochs::TI, n_ω_steps::TI, n_burn::TI, γ::TI,
    ω_max::TF, adapt_covariance::TB, linear_solve!::TS, loss::TL) where {TI<:Integer,TF<:AbstractFloat,TB<:Bool,TL,TS}
    return ARFFOptions(n_epochs, n_ω_steps, 1.0, n_burn, γ, ω_max, adapt_covariance, linear_solve!, loss)
end