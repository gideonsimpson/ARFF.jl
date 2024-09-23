
"""
    train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ScalarDataSet{TR,TB,TI}}, N::TI, batch_size::TI,  Σ::Matrix{TR}, options::ARFFOptions; show_progress=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

Train the Fourier feature model using a random walk Metropolis exploration
strategy with batched data, which is cycled through from epoch to epoch.

### Fields
* `F` - The `FourierModel` to be trained
* `data_sets`- An iterable of `DataSet` training data sets.  These are presumed
  to all be the same size.  They are cycled through each epoch.
* `N` - Length of each data set in `data_sets`.
* `batch_size` - Minibatching parameter, for subsampling at each epoch.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
# function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data_sets::Abstract{ScalarDataSet{TR,TB,TI}}, N::TI, batch_size::TI,  Σ::Matrix{TR}, options::ARFFOptions; show_progress=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data_sets, N::TI, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    # extract values
    K = length(F)
    d = F.dx;

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
    loss = Float64[]

    # fit initial coefficients
    rows = 1:N
    if (batch_size < N)
        rows = sample(1:N, batch_size, replace=false)
    end
    assemble_matrix!(S, F.ϕ, Iterators.first(data_sets).x[rows], F.ω)
    options.linear_solve!(F.β, S, Iterators.first(data_sets).y[rows], F.ω)

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
            assemble_matrix!(S, F.ϕ, data.x[rows], ω_proposal)
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
        assemble_matrix!(S, F.ϕ, data.x[rows], F.ω)
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
    train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer, TA<:ActivationFunction{TB}}

* `F` - The `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer, TA<:ActivationFunction{TB}}
    N = length(data);
    Σ_mean, acceptance_rate, loss = train_rwm!(F, Iterators.cycle([data]), N, N, Σ, options; show_progress=show_progress, record_loss=record_loss)
    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

* `F` - The `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `batch_size` - Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(data)
    Σ_mean, acceptance_rate, loss = train_rwm!(F, Iterators.cycle([data]), N, batch_size, Σ, options; show_progress=show_progress,record_loss=record_loss)
    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data_sets, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

* `F` - The `FourierModel` to be trained
* `data_sets`- An iterable of `DataSet` training data sets.  These are presumed
  to all be the same size.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record

"""
function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ScalarDataSet{TR,TB,TI}}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(first(data_sets))

    Σ_mean, acceptance_rate, loss = train_rwm!(F, data_sets, N, N, Σ, options; show_progress=show_progress, record_loss=record_loss)
    return Σ_mean, acceptance_rate, loss
end


"""
    train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data_sets, N::TI, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

Train the Fourier feature model using a random walk Metropolis exploration
strategy. Returns the entire trajectory of models during training.
### Fields
* `F₀` - The initial state of the `FourierModel` to be trained
* `data_sets`- An iterable of `DataSet` training data sets.  These are presumed
  to all be the same size.  They are cycled through each epoch.
* `N` - Length of each data set in `data_sets`.
* `batch_size` - Minibatching parameter, for subsampling at each epoch.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data_sets, N::TI, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}


    F = deepcopy(F₀)
    F_trajectory = tyepof(F)[]

    # extract values
    K = length(F)
    d = F.dx

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
    loss = Float64[]

    # fit initial coefficients
    rows = 1:N
    if (batch_size < N)
        rows = sample(1:N, batch_size, replace=false)
    end
    assemble_matrix!(S, F.ϕ, Iterators.first(data_sets).x[rows], F.ω)
    options.linear_solve!(F.β, S, Iterators.first(data_sets).y[rows], F.ω)

    p = Progress(options.n_epochs; enabled=show_progress)

    for (i, data) in enumerate(data_sets)
        if i> options.n_epochs
            break
        end

        accept_ = 0.0;
        loss_ = 0.;
        if (batch_size < N)
            rows = sample(1:N, batch_size, replace=false)
        end

        for j in 1:options.n_ω_steps
            # generate proposal
            @. ω_proposal = F.ω + rand((mv_normal,))
            assemble_matrix!(S, F.ϕ, data.x[rows], ω_proposal)
            options.linear_solve!(β_proposal, S, data.y[rows], ω_proposal)

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
        assemble_matrix!(S, F.ϕ, data.x[rows], F.ω)
        options.linear_solve!(F.β, S, data.y[rows], F.ω)

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

    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

* `F₀` - The initial state of the `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(data)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_rwm(F₀, Iterators.cycle([data]), N, N, Σ, options; show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

* `F₀` - The initial state of the `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `batch_size` - Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, batch_size::TI, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(data)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_rwm(F₀, Iterators.cycle([data]), N, batch_size, Σ, options; show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀, data_sets, Σ, options; show_progress=true, record_loss=true) 

* `F₀` - The initial state of the `FourierModel` to be trained
* `data_sets`- An iterable of `DataSet` training data sets.  These are presumed
  to all be the same size.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record

"""
function train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ScalarDataSet{TR,TB,TI}}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(first(data_sets))
    F_trajectory, Σ_mean, acceptance_rate, loss = train_rwm(F₀, data_sets, N, N, Σ, options; show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, Σ_mean, acceptance_rate, loss
end

