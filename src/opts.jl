


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