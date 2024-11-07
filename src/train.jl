"""
    train_arff!(F, data_sets, batch_size, solver, n_epochs; show_progress=true, record_loss=true) 

Perform RWM training in place on an ARFF model
### Fields
* `F` - Fourier feature model to be trained, in place
* `data_sets` - Training data sets 
* `batch_size` - Minibatch size
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff!(F::AbstractFourierModel, data_sets::TD, batch_size::TI, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TD,TI<:Integer}

    # extract values
    K = length(F)
    N = length(Iterators.first(data_sets))

    # initialize data structures
    S = zeros(typeof(F.β[1, 1]), batch_size, K)

    # track acceptance rate and loss over
    loss = Float64[]

    # fit initial coefficients
    rows = 1:N
    if (batch_size < N)
        rows = sample(1:N, batch_size, replace=false)
    end
    assemble_matrix!(S, F.ϕ, subsample(Iterators.first(data_sets).x, rows), F.ω)
    solver.linear_solve!(F.β, F.ω, subsample(Iterators.first(data_sets).x, rows), subsample(Iterators.first(data_sets).y_mat, rows), S, 0)
    # solver.linear_solve!(F, )

    pmeter = Progress(n_epochs; enabled=show_progress)

    for (epoch, data) in enumerate(data_sets)
        if epoch > n_epochs
            break
        end

        loss_ = 0
        if (batch_size < N)
            rows = sample(1:N, batch_size, replace=false)
        end
        # resample
        solver.resample!(F, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch)

        # perform mutation step
        solver.mutate!(F, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch);

        # record loss
        if record_loss
            loss_ = solver.loss(F, subsample(data.x, rows), subsample(data.y, rows))
            push!(loss, loss_)
        end

        next!(pmeter; showvalues=[(:loss, loss_)])
    end

    return loss
end

"""
    train_arff(F₀, data_sets, batch_size, solver, n_epochs; show_progress=true, record_loss=true) 

Perform RWM training an ARFF model, recording the result at all epochs.
### Fields
* `F₀` - Initial fourier feature model to be trained
* `data_sets` - Training data sets 
* `batch_size` - Minibatch size
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff(F₀::AbstractFourierModel, data_sets::TD, batch_size::TI, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TD,TI<:Integer}

    F = deepcopy(F₀)
    F_trajectory = typeof(F)[]

    # extract values
    K = length(F)
    N = length(Iterators.first(data_sets))

    # initialize data structures
    S = zeros(typeof(F.β[1, 1]), batch_size, K)

    # track acceptance rate and loss over
    loss = Float64[]

    # fit initial coefficients
    rows = 1:N
    if (batch_size < N)
        rows = sample(1:N, batch_size, replace=false)
    end
    
    pmeter = Progress(n_epochs; enabled=show_progress)

    for (epoch, data) in enumerate(data_sets)
        if epoch > n_epochs
            break
        end

        loss_ = 0
        if (batch_size < N)
            rows = sample(1:N, batch_size, replace=false)
        end
        # resample
        solver.resample!(F, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch)

        # perform mutation step
        solver.mutate!(F, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch)

        # perform full β update
        assemble_matrix!(S, F.ϕ, subsample(data.x, rows), F.ω)
        # solver.linear_solve!(F.β, S, subsample(data.y_mat, rows), F.ω)
        solver.linear_solve!(F.β, F.ω, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch)

        # record F_trajectory
        push!(F_trajectory, deepcopy(F))

        # record loss
        if record_loss
            loss_ = solver.loss(F, subsample(data.x, rows), subsample(data.y, rows))
            push!(loss, loss_)
        end

        next!(pmeter; showvalues=[(:loss, loss_)])
    end

    return F_trajectory, loss
end
