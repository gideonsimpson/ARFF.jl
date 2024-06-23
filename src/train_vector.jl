
"""
    train_rwm!(F::ScalarFourierModel{TC,TR,TW}, data::ScalarDataSet{TC,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F` - The `FourierModel` to be trained
* `data`- The `DataSet` training data
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TC,TR,TW,TB,TI}, data::VectorDataSet{TC,TR,TW,TB,TI}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TC<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TB<:AbstractArray{TC}, TI<:Integer, TM<:AbstractMatrix}

    # extract values
    K = length(F)
    (N,dx,dy) = size(data);

    # initialize data structures
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    S = zeros(TC, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    # fit initial coefficients
    assemble_matrix!(S, data.x, F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.β[d_], S, data.y[d_], F.ω)
    end

    loss = Float64[]
    p = Progress(options.n_epochs; enabled=show_progress)

    # @showprogress "Training..." 
    for i in 1:options.n_epochs

        accept_ = 0.0
        loss_ = 0.0
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            assemble_matrix!(S, data.x, ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, data.y[d_], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.β[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    F.β[k] = β_proposal[k]
                    accept_ += 1.0 / (K * options.n_ω_steps)
                end
            end


            # update running mean and covariance
            if (options.adapt_covariance)
                # compute instantaneous ensemble averages
                ω_mean_ .= mean(F.ω)
                Σ_mean_ .= cov(F.ω, corrected=false)
                # update cumulative averages 
                l = (i - 1) * options.n_ω_steps + j
                @. Σ_mean *= (l - 1) / l
                @. Σ_mean += 1 / l * Σ_mean_ + (l - 1) / l^2 * (ω_mean_ - ω_mean) * (ω_mean_ - ω_mean)'
                # ensure symmetry
                @. Σ_mean = 0.5 * (Σ_mean + Σ_mean')

                @. ω_mean += (ω_mean_ - ω_mean) / l
                # switch to dynamic covariance matrix ater i n_burn epochs
                if (i > options.n_burn)
                    @set mv_normal = MvNormal(Σ_mean)
                end
            end
        end

        # perform full β update
        assemble_matrix!(S, data.x, F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.β[d_], S, data.y[d_], F.ω)
        end


        # record loss
        if record_loss
            loss_ = options.loss(F, data.x, data.y)
            push!(loss, loss_)
        end

        # record acceptance rate
        if (i > 1)
            push!(acceptance_rate, acceptance_rate[end] + (accept_ - acceptance_rate[end]) / i)
        else
            push!(acceptance_rate, accept_)
        end
        next!(p; showvalues=[(:loss, loss_), (:accept, accept_)])
    end

    return Σ_mean, acceptance_rate, loss
end
