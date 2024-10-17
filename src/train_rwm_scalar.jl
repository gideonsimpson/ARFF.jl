"""
    train_rwm!(F::ScalarFourierModel{TR,TY,TI,TA}, data::ScalarDataSet{TR,TY,TI}, rwm_sampler<:AbstractRWMSampler; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}

TBW
"""
function train_rwm!(F::ScalarFourierModel{TR,TY,TI,TA}, data::ScalarDataSet{TR,TY,TI}, rwm_sampler::TS, n_epochs::TI; 
        show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}
    
    N = length(data);

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n);

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, trivial_resample!, n_epochs, mse_loss)
    
    loss = train_arff!(F, Iterators.cycle([data]), N, solver, show_progress=show_progress, record_loss=record_loss);
    
    return rwm_sampler.Σ_mean, rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm!(F::ScalarFourierModel{TR,TY,TI,TA}, data::ScalarDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

TBW
"""
function train_rwm!(F::ScalarFourierModel{TR,TY,TI,TA}, data::ScalarDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, trivial_resample!, n_epochs, mse_loss)

    loss = train_arff!(F, Iterators.cycle([data]), batch_size, solver, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.Σ_mean, rwm_sampler.acceptance_rate, loss
end
