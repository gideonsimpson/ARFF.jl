function train_arff2!(F::AbstractFourierModel, data_sets::TD, batch_size::TI, solver::ARFFSolver, options::ARFFOptions; show_progress=true, record_loss=true) where {TD,TI<:Integer}

    # extract values
    K, dx, _ = size(F)
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
    solver.linear_solve!(F.β, S, subsample(Iterators.first(data_sets).y_mat, rows), F.ω)

    pmeter = Progress(options.n_epochs; enabled=show_progress)

    for (epoch, data) in enumerate(data_sets)
        if epoch > options.n_epochs
            break
        end

        loss_ = 0
        if (batch_size < N)
            rows = sample(1:N, batch_size, replace=false)
        end
        # resampler goes here
        solver.resample!(F, epoch);

        # perform mutation step
        solver.mutate!(F, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch);

        # perform full β update
        assemble_matrix!(S, F.ϕ, subsample(data.x, rows), F.ω);
        solver.linear_solve!(F.β, S, subsample(data.y_mat, rows), F.ω);

        # record loss
        if record_loss
            loss_ = options.loss(F, subsample(data.x, rows), subsample(data.y, rows))
            push!(loss, loss_)
        end

        next!(pmeter; showvalues=[(:loss, loss_)])
    end

    return loss
end
