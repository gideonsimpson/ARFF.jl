

"""
    train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}

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
function train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}

    # extract values
    K = length(F)
    N = length(data)
    d = length(data.x[1])

    # initialize data structures
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(d)
    Σ_mean_ = zeros(d, d)

    # cumulative averages 
    ω_mean = zeros(d)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    # fit initial coefficients
    assemble_matrix!(S, data.x, F.ω)
    options.linear_solve!(F.β, S, data.y, F.ω)

    loss = Float64[];
    p = Progress(options.n_epochs; enabled=show_progress)

    # @showprogress "Training..." 
    for i in 1:options.n_epochs

        accept_ = 0.0;
        loss_ = 0.0;
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            assemble_matrix!(S, data.x, ω_proposal)
            options.linear_solve!(β_proposal, S, data.y, ω_proposal)

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((abs(β_proposal[k]) / abs(F.β[k]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
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
        options.linear_solve!(F.β, S, data.y, F.ω)

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

"""
    train_rwm!(F::FourierModel{TB,TR,TW}, batched_data::Vector{DataSet{TB,TR,TW}}, Σ::TM, options::ARFFOptions) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}

Train the Fourier feature model using a random walk Metropolis exploration
strategy with batched data, which is cycled through from epoch to epoch.

### Fields
* `F` - The `FourierModel` to be trained
* `batched_data`- A vector of `DataSet` training data sets, for the purpose of minibatching.  These are presumed to all be the same size.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::FourierModel{TB,TR,TW}, batched_data::Vector{DataSet{TB,TR,TW}}, Σ::TM, options::ARFFOptions; show_progress=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}

    # extract values
    K = length(F)
    N = length(batched_data[1])
    d = length(batched_data[1].x[1])
    n_batch = length(batched_data);

    # initialize data structures
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(d)
    Σ_mean_ = zeros(d, d)

    # cumulative averages 
    ω_mean = zeros(d)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]

    p = Progress(options.n_epochs; enabled=show_progress)

    for i in 1:options.n_epochs

        i_batch = mod1(i, n_batch); # current data set index
        loss_ = 0;
        accept_ = 0.0
        # fit initial coefficients at each epoch
        assemble_matrix!(S, batched_data[i_batch].x, F.ω)
        options.linear_solve!(F.β, S, batched_data[i_batch].y, F.ω)

        for j in 1:options.n_ω_steps
            # generate proposal

            @. ω_proposal = F.ω + rand((mv_normal,))
            assemble_matrix!(S, batched_data[i_batch].x, ω_proposal)
            options.linear_solve!(β_proposal, S, batched_data[i_batch].y, ω_proposal)

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((abs(β_proposal[k]) / abs(F.β[k]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
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
        assemble_matrix!(S, batched_data[i_batch].x, F.ω)
        options.linear_solve!(F.β, S, batched_data[i_batch].y, F.ω)

        # record loss
        if record_loss
            loss_ = options.loss(F, batched_data[i_batch].x, batched_data[i_batch].y)
            push!(loss, loss_)
        end

        # record acceptance rate
        if (i > 1)
            push!(acceptance_rate, acceptance_rate[end] + (accept_ - acceptance_rate[end]) / i)
        else
            push!(acceptance_rate, accept_)
        end
        # record loss
        next!(p; showvalues=[(:loss, loss_), (:accept, accept_)])
    end

    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, batch_size::TI, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix,TI<:Integer}

Train the Fourier feature model using a random walk Metropolis exploration
strategy with minibatching, randomly subsampling at each epoch.

* `F` - The `FourierModel` to be trained
* `batch_size`- Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, batch_size::TI, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix,TI<:Integer}

    # extract values
    K = length(F)
    N = length(data)
    d = length(data.x[1])

    # initialize data structures
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    S = zeros(TB, batch_size, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(d)
    Σ_mean_ = zeros(d, d)

    # cumulative averages 
    ω_mean = zeros(d)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    # fit initial coefficients
    rows = sample(1:N, batch_size, replace=false);
    assemble_matrix!(S, data.x[rows], F.ω)
    options.linear_solve!(F.β, S, data.y[rows], F.ω)

    loss = Float64[];
    p = Progress(options.n_epochs; enabled=show_progress)

    # @showprogress "Training..." 
    for i in 1:options.n_epochs

        accept_ = 0.0;
        loss_ = 0.0;
        rows = sample(1:N, batch_size, replace=false)
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))

            assemble_matrix!(S, data.x[rows], ω_proposal)
            options.linear_solve!(β_proposal, S, data.y[rows], ω_proposal)

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((abs(β_proposal[k]) / abs(F.β[k]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
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
        # rows = sample(1:N, batch_size, replace=false)

        assemble_matrix!(S, data.x[rows], F.ω)
        options.linear_solve!(F.β, S, data.y[rows], F.ω)
        
        # record loss
        if record_loss
            loss_ = options.loss(F, data.x[rows], data.y[rows])
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

"""
    train_rwm(F₀::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}

Train the Fourier feature model using a random walk Metropolis exploration
strategy.  Returns the entire trajectory of models during training.
"""
function train_rwm(F₀::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}


    F = deepcopy(F₀)
    F_trajectory = FourierModel{TB,TR,TW}[]

    # extract values
    K = length(F)
    N = length(data)
    d = length(data.x[1])

    # initialize data structures
    β_proposal = similar(F.β)
    ω_proposal = similar(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(d)
    Σ_mean_ = zeros(d, d)

    # cumulative averages 
    ω_mean = zeros(d)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]
    p = Progress(options.n_epochs; enabled=show_progress)

    for i in 1:options.n_epochs

        accept_ = 0.0;
        loss_ = 0.;
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + rand((mv_normal,))
            assemble_matrix!(S, data.x, ω_proposal)
            options.linear_solve!(β_proposal, S, data.y, ω_proposal)

            # apply Metroplis step
            ζ = rand()
            for k in 1:K
                if ((abs(β_proposal[k]) / abs(F.β[k]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
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
        options.linear_solve!(F.β, S, data.y, F.ω)

        # record F_trajectory
        push!(F_trajectory, deepcopy(F))

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

    return F_trajectory, Σ_mean, acceptance_rate, loss
end


