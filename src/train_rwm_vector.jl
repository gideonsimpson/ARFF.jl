"""
    train_rwm!(F, data, rwm_sampler; show_progress=true, record_loss=true)

Perform RWM training of an ARFF model
### Fields
* `F` - Fourier feature model to be trained, in place
* `data` - Training data 
* `rwm_sampler` - An RWM sampler data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AdaptiveRWMSampler}
    
    N = length(data)
    
    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, ARFF.trivial_resample!, mse_loss)

    loss = train_arff!(F, Iterators.cycle([data]), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm!(F, data, batch_size, rwm_sampler, n_epochs; show_progress=true, record_loss=true)

Perform RWM training of an ARFF model with mini batching
### Fields
* `F` - Fourier feature model to be trained, in place
* `data` - Training data 
* `batch_size` - Minibatch size
* `rwm_sampler` - An RWM sampler data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, ARFF.trivial_resample!, mse_loss)

    loss = train_arff!(F, Iterators.cycle([data]), batch_size, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm!(F, data, batch_size, rwm_sampler, n_epochs; show_progress=true, record_loss=true)

Perform RWM training of an ARFF model with an array of different training data
sets
### Fields
* `F` - Fourier feature model to be trained, in place
* `data_sets` - Training data sets 
* `rwm_sampler` - An RWM sampler data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm!(F::VectorFourierModel{TR,TY,TI,TA}, data_sets::Vector{VectorDataSet{TR,TY,TI}}, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

    N = length(first(data_sets))

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, trivial_resample!, mse_loss)

    loss = train_arff!(F, Iterators.cycle(data_sets), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm(F₀, data, rwm_sampler; show_progress=true, record_loss=true)

Perform RWM training of an ARFF model
### Fields
* `F₀` - Initial Fourier feature model to be trained
* `data` - Training data 
* `rwm_sampler` - An RWM sampler data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm(F₀::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

    N = length(data)

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, ARFF.trivial_resample!, mse_loss)

    F_trajectory, loss = train_arff(F₀, Iterators.cycle([data]), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm!(F, data, batch_size, rwm_sampler, n_epochs; show_progress=true, record_loss=true)

Perform RWM training of an ARFF model with mini batching
### Fields
* `F₀` - Initial Fourier feature model to be trained
* `data` - Training data 
* `batch_size` - Minibatch size
* `rwm_sampler` - An RWM sampler data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm(F₀::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, ARFF.trivial_resample!, mse_loss)

    F_trajectory, loss = train_arff(F₀, Iterators.cycle([data]), batch_size, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, rwm_sampler.acceptance_rate, loss
end

"""
    train_rwm!(F, data, batch_size, rwm_sampler, n_epochs; show_progress=true, record_loss=true)

Perform RWM training of an ARFF model with an array of different training data
sets
### Fields
* `F₀` - Initial Fourier feature model to be trained
* `data_sets` - Training data sets 
* `rwm_sampler` - An RWM sampler data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm(F₀::VectorFourierModel{TR,TY,TI,TA}, data_sets::Vector{VectorDataSet{TR,TY,TI}}, rwm_sampler::TS, n_epochs::TI;
    show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY},TS<:AbstractRWMSampler}

    N = length(first(data_sets))

    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, trivial_resample!, mse_loss)

    F_trajectory, loss = train_arff(F₀, Iterators.cycle(data_sets), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, rwm_sampler.acceptance_rate, loss
end
