"""
    train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB},TS<:AbstractRWMSampler}
    Σ::Matrix{TR}, options::ARFFOptions;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

TBW
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB},TS<:AbstractRWMSampler}
    
    N = length(data)
    
    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, ARFF.trivial_resample!, n_epochs, mse_loss)

    loss = train_arff!(F, Iterators.cycle([data]), N, solver, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.Σ_mean, rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB},TS<:AbstractRWMSampler}

TBW
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB},TS<:AbstractRWMSampler}

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, ARFF.trivial_resample!, n_epochs, mse_loss)

    loss = train_arff!(F, Iterators.cycle([data]), batch_size, solver, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.Σ_mean, rwm_sampler.acceptance_rate, loss
end
