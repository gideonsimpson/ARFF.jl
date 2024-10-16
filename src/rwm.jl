struct RWMSampler{TS,TI,TM<:AbstractVecOrMat,TR,TMN,TX<:AbstractVecOrMat,TY<:AbstractVecOrMat}
    linear_solve!::TS
    n_rwm_steps::TI
    n_burn::TI
    n_epochs::TI
    β_proposal::TY
    ω_proposal::TX
    Σ::TM
    γ::TI
    δ::TR
    acceptance_rate::AbstractVector
    ω_max::TR
    mv_normal::TMN
end

function RWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, n_epochs::TI, Σ::TM, γ::TI, δ::TR, ω_max::TR) where{TF<:AbstractFourierModel, TS, TI, TM, TR}
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    return RWMSampler(linear_solve!, n_rwm_steps,n_burn, n_epochs, deepcopy(β_proposal), deepcopy(ω_proposal), deepcopy(Σ), γ, δ, zeros(n_epochs), ω_max, MvNormal(Σ))
end



struct AdaptiveRWMSampler{TS,TI,TM<:AbstractMatrix,TR,TMN,TX<:AbstractVector,TY<:AbstractVecOrMat}
    linear_solve!::TS
    n_rwm_steps::TI
    n_burn::TI
    n_epochs::TI
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

function AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, n_epochs::TI, Σ0::TM, γ::TI, δ::TR, ω_max::TR) where {TF<:AbstractFourierModel,TS,TI,TM,TR}
    return AdaptiveRWMSampler(linear_solve!, n_rwm_steps, n_burn, n_epochs, deepcopy(F.β), deepcopy(F.ω),
        γ, δ, zeros(n_epochs), ω_max, deepcopy(F.ω[1]), deepcopy(F.ω[1]), deepcopy(Σ0), deepcopy(Σ0), MvNormal(Σ0))
end

function likelihood(β_new::TY, β_old::TY, γ::TI) where {TY<:Number,TI<:Integer}

    return (abs(β_new) / abs(β_old))^γ

end

function likelihood(β_new::AbstractVector{TY}, β_old::AbstractVector{TY}, γ::TI) where {TY<:Number,TI<:Integer}

    return (norm(β_new) / norm(β_old))^γ

end

function rwm!(F::TF, sampler::RWMSampler, x, y, S, epoch) where {TF<:AbstractFourierModel}
    K = length(F);
    accept_ = 0.0

    for j in sampler.n_rwm_steps
        # generate proposal

        @. sampler.ω_proposal = F.ω + sampler.δ * rand((sampler.mv_normal,))
        assemble_matrix!(S, F.ϕ, x, sampler.ω_proposal)
        sampler.linear_solve!(sampler.β_proposal, S, y, sampler.ω_proposal)

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
        sampler.acceptance_rate[epoch] = sample.accept[epoch-1] + (accept_ - acceptance_rate[epoch-1]) / epoch
    else
        sampler.acceptance_rate[epoch] = accept_;
    end

    F, sampler
end

function rwm!(F::TF, sampler::AdaptiveRWMSampler, x, y, S, epoch) where {TF<:AbstractFourierModel}
    K = length(F)
    accept_ = 0.0
    for j in 1:sampler.n_rwm_steps
        # generate proposal

        @. sampler.ω_proposal = F.ω + sampler.δ * rand((sampler.mv_normal,))
        assemble_matrix!(S, F.ϕ, x, sampler.ω_proposal)
        sampler.linear_solve!(sampler.β_proposal, S, y, sampler.ω_proposal)

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
        sampler.acceptance_rate[epoch] = sampler.acceptance_rate[epoch-1] + (accept_ - sampler.acceptance_rate[epoch-1]) / epoch
    else
        sampler.acceptance_rate[epoch] = accept_
    end

    F, sampler

end