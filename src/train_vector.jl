
"""
    train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, 
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F` - The `VectorFourierModel` to be trained
* `data`- The `DataSet` training data
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, 
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    # extract values
    K = length(F)
    (N,dx,dy) = size(data);

    # initialize data structures
    β_proposal = deepcopy(F.βt)
    ω_proposal = deepcopy(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # fit initial coefficients
    assemble_matrix!(S, F.ϕ, data.x, F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.βt[d_], S, data.yt[d_], F.ω)
    end

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]
    p = Progress(options.n_epochs; enabled=show_progress)
    
    for i in 1:options.n_epochs

        accept_ = 0.0
        loss_ = 0.0
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            assemble_matrix!(S, F.ϕ, data.x, ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, data.yt[d_], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.βt[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    for d_ in 1:dy
                        F.βt[d_][k] = β_proposal[d_][k];
                    end
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
        assemble_matrix!(S, F.ϕ, data.x, F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.βt[d_], S, data.yt[d_], F.ω)
        end

        copy_from_transpose!(F)
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
    train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, batched_data::Vector{VectorDataSet{TR,TB,TI}},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F` - The `VectorFourierModel` to be trained
* `data`- The `DataSet` training data
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, batched_data::Vector{VectorDataSet{TR,TB,TI}},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    # extract values
    K = length(F)
    (N, dx, dy) = size(batched_data[1])
    n_batch = length(batched_data)

    # initialize data structures
    β_proposal = deepcopy(F.βt)
    ω_proposal = deepcopy(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    i_batch = 1
    # fit initial coefficients at each epoch
    assemble_matrix!(S, F.ϕ, batched_data[i_batch].x, F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.βt[d_], S, batched_data[i_batch]data.yt[d_], F.ω)
    end

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]
    
    p = Progress(options.n_epochs; enabled=show_progress)


    # rearrange data for computing loss function
    for i in 1:options.n_epochs

        i_batch = mod1(i, n_batch); # current data set index
        accept_ = 0.0
        loss_ = 0.0


        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            assemble_matrix!(S, F.ϕ, batched_data[i_batch].x, ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, batched_data[i_batch]data.yt[d_], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.βt[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    for d_ in 1:dy
                        F.βt[d_][k] = β_proposal[d_][k]
                    end
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
        assemble_matrix!(S, F.ϕ, batched_data[i_batch].x, F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.βt[d_], S, batched_data[i_batch]data.yt[d_], F.ω)
        end

        copy_from_transpose!(F);

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
        next!(p; showvalues=[(:loss, loss_), (:accept, accept_)])
    end

    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, 
    batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

Train the Fourier feature model using a random walk Metropolis exploration
strategy with minibatching, randomly subsampling at each epoch.

* `F` - The `VectorFourierModel` to be trained
* `batch_size`- Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, 
    batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    # extract values
    K = length(F)
    (N, dx, dy) = size(data);

    # initialize data structures
    β_proposal = deepcopy(F.βt)
    ω_proposal = deepcopy(F.ω)
    S = zeros(TB, batch_size, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # fit initial coefficients
    rows = sample(1:N, batch_size, replace=false)
    assemble_matrix!(S, F.ϕ, data.x[rows], F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.βt[d_], S, data.yt[d_][rows], F.ω)
    end

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]
    p = Progress(options.n_epochs; enabled=show_progress)

    # @showprogress "Training..." 
    for i in 1:options.n_epochs

        accept_ = 0.0
        loss_ = 0.0
        rows = sample(1:N, batch_size, replace=false)
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))

            assemble_matrix!(S, F.ϕ, data.x[rows], ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, data.yt[d_][rows], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.βt[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    for d_ in 1:dy
                        F.βt[d_][k] = β_proposal[d_][k]
                    end
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
        assemble_matrix!(S, F.ϕ, data.x[rows], F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.βt[d_], S, data.yt[d_][rows], F.ω)
        end
        copy_from_transpose!(F)
        # record loss
        if record_loss
            loss_ = options.loss(F, data.x[rows], data.y[rows]);
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
    train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

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
function train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    F = deepcopy(F₀)
    F_trajectory = typeof(F)[];

    # extract values
    K = length(F)
    (N, dx, dy) = size(data)

    # initialize data structures
    β_proposal = deepcopy(F.βt)
    ω_proposal = deepcopy(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # fit initial coefficients
    assemble_matrix!(S, F.ϕ, data.x, F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.βt[d_], S, data.yt[d_], F.ω)
    end

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]
    p = Progress(options.n_epochs; enabled=show_progress)

    for i in 1:options.n_epochs

        accept_ = 0.0
        loss_ = 0.0
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            assemble_matrix!(S, F.ϕ, data.x, ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, data.yt[d_], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.βt[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    for d_ in 1:dy
                        F.βt[d_][k] = β_proposal[d_][k]
                    end
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
        assemble_matrix!(S, F.ϕ, data.x, F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.βt[d_], S, data.yt[d_], F.ω)
        end
        copy_from_transpose!(F)
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


"""
    train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, batched_data::Vector{VectorDataSet{TR,TB,TI}},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

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
function train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, batched_data::Vector{VectorDataSet{TR,TB,TI}},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    F = deepcopy(F₀)
    F_trajectory = typeof(F)[];

    # extract values
    K = length(F)
    (N, dx, dy) = size(batched_data[1])
    n_batch = length(batched_data)

    # initialize data structures
    β_proposal = deepcopy(F.βt)
    ω_proposal = deepcopy(F.ω)
    S = zeros(TB, N, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    i_batch = 1
    # fit initial coefficients at each epoch
    assemble_matrix!(S, F.ϕ, batched_data[i_batch].x, F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.βt[d_], S, batched_data[i_batch]data.yt[d_], F.ω)
    end

    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]

    p = Progress(options.n_epochs; enabled=show_progress)

    # rearrange data for computing loss function
    for i in 1:options.n_epochs

        i_batch = mod1(i, n_batch) # current data set index
        accept_ = 0.0
        loss_ = 0.0
        # fit initial coefficients at each epoch

        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))
            assemble_matrix!(S, F.ϕ, batched_data[i_batch].x, ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, batched_data[i_batch]data.yt[d_], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.βt[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    for d_ in 1:dy
                        F.βt[d_][k] = β_proposal[d_][k]
                    end
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
        assemble_matrix!(S, F.ϕ, batched_data[i_batch].x, F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.βt[d_], S, batched_data[i_batch]data.yt[d_], F.ω)
        end
        copy_from_transpose!(F)
        # record F_trajectory
        push!(F_trajectory, deepcopy(F))


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
        next!(p; showvalues=[(:loss, loss_), (:accept, accept_)])
    end

    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

Train the Fourier feature model using a random walk Metropolis exploration
strategy with minibatching, randomly subsampling at each epoch.

* `F` - The `FourierModel` to be trained
* `batch_size`- Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    F = deepcopy(F₀)
    F_trajectory = typeof(F)[]
    
    # extract values
    K = length(F)
    (N, dx, dy) = size(data)

    # initialize data structures
    β_proposal = deepcopy(F.βt)
    ω_proposal = deepcopy(F.ω)
    S = zeros(TC, batch_size, K)

    # instantaneous ensemble averages
    ω_mean_ = zeros(dx)
    Σ_mean_ = zeros(dx, dx)

    # cumulative averages 
    ω_mean = zeros(dx)
    Σ_mean = deepcopy(Σ)

    # initialize RWM distribution
    mv_normal = MvNormal(Σ_mean)

    # fit initial coefficients
    rows = sample(1:N, batch_size, replace=false)
    assemble_matrix!(S, F.ϕ, data.x[rows], F.ω)
    for d_ in 1:dy
        options.linear_solve!(F.βt[d_], S, data.yt[d_][rows], F.ω)
    end


    # track acceptance rate and loss over
    acceptance_rate = Float64[]
    loss = Float64[]
    p = Progress(options.n_epochs; enabled=show_progress)

    # @showprogress "Training..." 
    for i in 1:options.n_epochs

        accept_ = 0.0
        loss_ = 0.0
        rows = sample(1:N, batch_size, replace=false)
        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + options.δ * rand((mv_normal,))

            assemble_matrix!(S, F.ϕ, data.x[rows], ω_proposal)
            for d_ in 1:dy
                options.linear_solve!(β_proposal[d_], S, data.yt[d_][rows], ω_proposal)
            end

            # apply Metroplis step
            for k in 1:K
                ζ = rand()
                if ((norm([β_proposal[d_][k] for d_ in 1:dy]) / norm([F.βt[d_][k] for d_ in 1:dy]))^options.γ > ζ) && (norm(ω_proposal[k]) < options.ω_max)
                    @. F.ω[k] = ω_proposal[k]
                    for d_ in 1:dy
                        F.βt[d_][k] = β_proposal[d_][k]
                    end
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
        assemble_matrix!(S, F.ϕ, data.x[rows], F.ω)
        for d_ in 1:dy
            options.linear_solve!(F.βt[d_], S, data.yt[d_][rows], F.ω)
        end
        copy_from_transpose!(F)
        # record F_trajectory
        push!(F_trajectory, deepcopy(F))

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



