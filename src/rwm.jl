"""
    RWMSampler{TS,TI,TM<:AbstractVecOrMat,TR,TMN,TX<:AbstractVecOrMat,TY<:AbstractVecOrMat} <: AbstractRWMSampler

Data structure containing random walk Metrpolis sampler parameters and structures
### Fields
* `linear_solve!` - User specified solver for the normal equations
* `n_rwm_steps` - Number of internal RWM steps
* `n_burn` - Number of epochs before the covariance adaptation begins
* `β_proposal` - workspace for β vector
* `ω_proposal` - workspace for ω vector
* `Σ` - Covariance matrix
* `γ` - Metropolis-Hastings exponent
* `δ` - RWM proposal step size
* `acceptance_rate` - Array recording acceeptance rate
* `ω_max` - Maximum wave number norm cutoff
* `mv_normal` - Multivariate normal for RWM
"""
struct RWMSampler{TS,TI,TM<:AbstractVecOrMat,TR,TMN,TX<:AbstractVecOrMat,TY<:AbstractVecOrMat} <: AbstractRWMSampler
    linear_solve!::TS
    n_rwm_steps::TI
    β_proposal::TY
    ω_proposal::TX
    Σ::TM
    γ::TI
    δ::TR
    acceptance_rate::AbstractVector
    ω_max::TR
    mv_normal::TMN
end

"""
    RWMSampler(F, linear_solve!, n_rwm_steps, Σ, γ, δ, ω_max) 

Constructor for the `RWMSampler` data structure
### Fields
* `F` - Fourier feature model; used for setting types and dimensions
* `linear_solve!` - User specified solver for the normal equations
* `n_rwm_steps` - Number of internal RWM steps
* `Σ` - Covariance matrix
* `γ` - Metropolis-Hastings exponent
* `δ` - RWM proposal step size
* `ω_max` - Maximum wave number norm cutoff
"""
function RWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, Σ::TM, γ::TI, δ::TR, ω_max::TR) where {TF<:AbstractFourierModel,TS,TI<:Integer,TM<:AbstractMatrix,TR<:AbstractFloat}
    return RWMSampler(linear_solve!, n_rwm_steps, deepcopy(F.β), deepcopy(F.ω), deepcopy(Σ), γ, δ, TR[], ω_max, MvNormal(Σ))
end

"""
    RWMSampler(F, linear_solve!, n_rwm_steps, δ)

Constructor for the `RWMSampler` data structure.  Defaults to `ω_max = Inf`, `γ
= optimal_γ(dx)`, and `Σ= I`
### Fields
* `F` - Fourier feature model; used for setting types and dimensions
* `linear_solve!` - User specified solver for the normal equations
* `n_rwm_steps` - Number of internal RWM steps
* `δ` - RWM proposal step size
"""
function RWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, δ::TR) where {TF<:AbstractFourierModel,TS,TI<:Integer,TR<:AbstractFloat}
    ω_max = Inf
    γ = optimal_γ(F.dx)
    Σ = Matrix{TR}(I(F.dx))
    return RWMSampler(linear_solve!, n_rwm_steps, deepcopy(F.β), deepcopy(F.ω), deepcopy(Σ), γ, δ, TR[], ω_max, MvNormal(Σ))
end


"""
    AdaptiveRWMSampler{TS,TI,TM<:AbstractMatrix,TR,TMN,TX<:AbstractVector,TY<:AbstractVecOrMat} <: AbstractRWMSampler

Data structure containing adaptive random walk Metrpolis sampler parameters and structures
### Fields
* `linear_solve!` - User specified solver for the normal equations
* `n_rwm_steps` - Number of internal RWM steps
* `n_burn` - Number of epochs before the covariance adaptation begins
* `β_proposal` - workspace for β vector
* `ω_proposal` - workspace for ω vector
* `γ` - Metropolis-Hastings exponent
* `δ` - RWM proposal step size
* `acceptance_rate` - Array recording acceeptance rate
* `ω_max` - Maximum wave number norm cutoff
* `ω_mean_` - Instantaneous mean of wave numbers
* `ω_mean` - Time averaged mean of wave numbers
* `Σ_mean_` - Instantaneous covariance matrix
* `Σ_mean` - Time averaged covariance matrix
* `mv_normal` - Multivariate normal for RWM
"""
struct AdaptiveRWMSampler{TS,TI,TM<:AbstractMatrix,TR,TMN,TX<:AbstractVector,TY<:AbstractVecOrMat} <: AbstractRWMSampler
    linear_solve!::TS
    n_rwm_steps::TI
    n_burn::TI
    β_proposal::TY
    ω_proposal::AbstractVector{TX}
    γ::TI
    δ::TR
    acceptance_rate::AbstractVector
    ω_max::TR
    ω_mean_::TX
    ω_mean::TX
    Σ_mean_::TM
    Σ_mean::TM
    mv_normal::TMN
end

"""
    AdaptiveRWMSampler(F, linear_solve!, n_rwm_steps, n_burn, Σ0, γ, δ, ω_max)

Constructor for the `AdaptiveRWMSampler` data structure
### Fields
* `F` - Fourier feature model; used for setting types and dimensions
* `linear_solve!` - User specified solver for the normal equations
* `n_rwm_steps` - Number of internal RWM steps
* `n_burn` - Number of epochs before the covariance adaptation begins
* `Σ0` - Initial covariance matrix
* `γ` - Metropolis-Hastings exponent
* `δ` - RWM proposal step size
* `ω_max` - Maximum wave number norm cutoff
"""
function AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, Σ0::TM, γ::TI, δ::TR, ω_max::TR) where {TF<:AbstractFourierModel,TS,TI<:Integer,TM<:AbstractMatrix,TR<:AbstractFloat}
    
    return AdaptiveRWMSampler(linear_solve!, n_rwm_steps, n_burn, deepcopy(F.β), deepcopy(F.ω), γ, δ, TR[], ω_max, deepcopy(F.ω[1]), deepcopy(F.ω[1]), deepcopy(Σ0), deepcopy(Σ0), MvNormal(Σ0))
end

"""
    AdaptiveRWMSampler(F, linear_solve!, n_rwm_steps, n_burn, δ)

Constructor for the `AdaptiveRWMSampler` data structure. Defaults to `ω_max =
Inf`, `γ = optimal_γ(dx)`, and `Σ0= I`
### Fields
* `F` - Fourier feature model; used for setting types and dimensions
* `linear_solve!` - User specified solver for the normal equations
* `n_rwm_steps` - Number of internal RWM steps
* `n_burn` - Number of epochs before the covariance adaptation begins
* `δ` - RWM proposal step size
"""
function AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, δ::TR) where {TF<:AbstractFourierModel,TS,TI<:Integer,TR<:AbstractFloat}
    ω_max = Inf
    γ = optimal_γ(F.dx);
    Σ0 = Matrix{TR}(I(F.dx))
    return AdaptiveRWMSampler(linear_solve!, n_rwm_steps, n_burn, deepcopy(F.β), deepcopy(F.ω), γ, δ, TR[], ω_max, deepcopy(F.ω[1]), deepcopy(F.ω[1]), deepcopy(Σ0), deepcopy(Σ0), MvNormal(Σ0))
end


"""
    likelihood(β_new, β_old, γ)

Likelihood ratio for Metropois-Hastings with exponent `γ`
### Fields
* `β_new` - New (proposed) `β`
* `β_old` - Old `β`
* `γ` - Metropolis-Hastings exponent
"""
function likelihood(β_new::TY, β_old::TY, γ::TI) where {TY<:Number,TI<:Integer}

    return (abs(β_new) / abs(β_old))^γ

end

"""
    likelihood(β_new, β_old, γ)

Likelihood ratio for Metropois-Hastings with exponent `γ`
### Fields
* `β_new` - New (proposed) `β`
* `β_old` - Old `β`
* `γ` - Metropolis-Hastings exponent    
"""
function likelihood(β_new::AbstractVector{TY}, β_old::AbstractVector{TY}, γ::TI) where {TY<:Number,TI<:Integer}

    return (norm(β_new) / norm(β_old))^γ

end

"""
    rwm!(F, sampler, x, y, S, epoch) 

Perform random walk Metrpolis with the specified parameters
### Fields
* `F` - Fourier features model, which is modified in place
* `sampler` - The RWM sampler which contains parameters and workspace
* `x` - x coordinates from data used during RWM
* `y` - y coordinates from data used during RWM
* `S` - S matrix used during RWM
* `epoch` - Current training epoch
"""
function rwm!(F::TF, sampler::RWMSampler, x, y, S, epoch) where {TF<:AbstractFourierModel}
    K = length(F);
    accept_ = 0.0

    for j in 1:sampler.n_rwm_steps
        # generate proposal

        @. sampler.ω_proposal = F.ω + sampler.δ * rand((sampler.mv_normal,))
        assemble_matrix!(S, F.ϕ, x, sampler.ω_proposal)
        sampler.linear_solve!(sampler.β_proposal, sampler.ω_proposal, x, y, S, epoch)

        # apply Metroplis step
        for k in 1:K
            ζ = rand()
            if (likelihood(subsample(sampler.β_proposal, k), subsample(F.β, k), sampler.γ) > ζ) && (norm(sampler.ω_proposal[k]) < sampler.ω_max)
                copy_entries!(F.ω, sampler.ω_proposal, k)
                copy_entries!(F.β, sampler.β_proposal, k)
                accept_ += 1.0 / (K * sampler.n_rwm_steps)
            end
        end
    end

    if (epoch > 1)
        push!(sampler.acceptance_rate, sampler.acceptance_rate[end] + (accept_ - sampler.acceptance_rate[end]) / epoch)
    else
        push!(sampler.acceptance_rate, accept_)
    end

    F, sampler
end

"""
    rwm!(F, sampler, x, y, S, epoch) 

* `F` - Fourier features model, which is modified in place
* `sampler` - The RWM sampler which contains parameters and workspace
* `x` - x coordinates from data used during RWM
* `y` - y coordinates from data used during RWM
* `S` - S matrix used during RWM
* `epoch` - Current training epoch
"""
function rwm!(F::TF, sampler::AdaptiveRWMSampler, x, y, S, epoch) where {TF<:AbstractFourierModel}
    K = length(F)
    accept_ = 0.0
    for j in 1:sampler.n_rwm_steps
        # generate proposal

        @. sampler.ω_proposal = F.ω + sampler.δ * rand((sampler.mv_normal,))
        assemble_matrix!(S, F.ϕ, x, sampler.ω_proposal)
        sampler.linear_solve!(sampler.β_proposal, sampler.ω_proposal, x, y, S, epoch)

        # apply Metroplis step
        for k in 1:K
            ζ = rand()
            if (likelihood(subsample(sampler.β_proposal, k), subsample(F.β, k), sampler.γ) > ζ) && (norm(sampler.ω_proposal[k]) < sampler.ω_max)
                copy_entries!(F.ω, sampler.ω_proposal, k)
                copy_entries!(F.β, sampler.β_proposal, k)
                accept_ += 1.0 / (K * sampler.n_rwm_steps)
            end
        end

        # update running mean and covariance

        # compute instantaneous ensemble averages
        sampler.ω_mean_ .= mean(F.ω)
        sampler.Σ_mean_ .= cov(F.ω, corrected=false)
        # update cumulative averages 
        l = (epoch - 1) * sampler.n_rwm_steps + j
        @. sampler.Σ_mean *= (l - 1) / l
        @. sampler.Σ_mean += 1 / l * sampler.Σ_mean_ + (l - 1) / l^2 * (sampler.ω_mean_ - sampler.ω_mean) * (sampler.ω_mean_ - sampler.ω_mean)'
        # ensure symmetry
        @. sampler.Σ_mean = 0.5 * (sampler.Σ_mean + sampler.Σ_mean')

        @. sampler.ω_mean += (sampler.ω_mean_ - sampler.ω_mean) / l
        # switch to dynamic covariance matrix ater i n_burn epochs
        if (epoch > sampler.n_burn)
            @set sampler.mv_normal = MvNormal(sampler.Σ_mean)
        end
    end
    if (epoch > 1)
        push!(sampler.acceptance_rate, sampler.acceptance_rate[end] + (accept_ - sampler.acceptance_rate[end]) / epoch)
    else
        push!(sampler.acceptance_rate, accept_)
    end

    F, sampler

end