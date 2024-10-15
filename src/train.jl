"""
    train_arff!(F, data_sets, batch_size::Integer, Σ, solver::ARFFSolver, options::ARFFOptions; show_progress=true, record_loss=true)

TBW
"""
function train_arff!(F::AbstractFourierModel, data_sets::TD, batch_size::TI, Σ, solver::ARFFSolver, options::ARFFOptions; show_progress=true, record_loss=true) where {TD,TI<:Integer}

    # extract values
    K = length(F)
    _, dx, dy = size(F);
    N = length(Iterators.first(data_sets));

    # initialize data structures
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    S = zeros(typeof(F.β[1,1]), batch_size, K)

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
    loss = Float64[]

    # fit initial coefficients
    rows = 1:N
    if (batch_size < N)
        rows = sample(1:N, batch_size, replace=false)
    end
    # assemble_matrix!(S, F.ϕ, Iterators.first(data_sets).x[rows], F.ω)
    assemble_matrix!(S, F.ϕ, subsample(Iterators.first(data_sets).x,rows), F.ω)
    solver.linear_solve!(F.β, S, subsample(Iterators.first(data_sets).y, rows), F.ω)

    p = Progress(options.n_epochs; enabled=show_progress)

    for (i, data) in enumerate(data_sets)
        if i> options.n_epochs
            break
        end

        accept_ = 0.0
        loss_ = 0
        if (batch_size < N)
            rows = sample(1:N, batch_size, replace=false)
        end

        for j in 1:options.n_ω_steps
            # generate proposal

            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            # assemble_matrix!(S, F.ϕ, data.x[rows], ω_proposal)
            assemble_matrix!(S, F.ϕ, subsample(data.x, rows), ω_proposal)
            solver.linear_solve!(β_proposal, S, subsample(data.y, rows), ω_proposal)

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                # if ((norm(β_proposal[k, :]) / norm(F.β[k, :]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                if (likelihood(subsample(β_proposal,k), subsample(F.β,k), options.γ) > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    # @. F.β[k, :] = β_proposal[k, :]
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
        # assemble_matrix!(S, F.ϕ, data.x[rows], ω_proposal)
        assemble_matrix!(S, F.ϕ, subsample(data.x,rows), F.ω)
        solver.linear_solve!(F.β, S, subsample(data.y, rows), F.ω)

        # record loss
        if record_loss
            loss_ = options.loss(F, subsample(data.x,rows), subsample(data.y,rows))
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
