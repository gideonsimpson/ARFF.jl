
"""
    train_rwm!(F, data, Σ, options; show_progress=true, record_loss=true) 

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F` - The `VectorFourierModel` to be trained
* `data`- A `DataSet` training data set
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    Σ::Matrix{TR}, options::ARFFOptions;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(data)
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    Σ_mean, acceptance_rate, loss = train_arff!(F, Iterators.cycle([data]), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)
    return Σ_mean, acceptance_rate, loss
end



"""
    train_rwm!(F, data, batch_size, Σ, options; show_progress=true, record_loss=true) 

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F` - The `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `batch_size` - Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, batch_size::TI,
    Σ::Matrix{TR}, options::ARFFOptions;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(data)
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    Σ_mean, acceptance_rate, loss = train_arff!(F, Iterators.cycle([data]), batch_size, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)
    return Σ_mean, acceptance_rate, loss
end


"""
    train_rwm!(F, data_sets, Σ, options; show_progress=true, record_loss=true) 

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F` - The `FourierModel` to be trained
* `data_sets`- A vector of `DataSet` training data sets.  These are presumed
  to all be the same size.  They will be cycled through each epoch.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data_sets::Vector{VectorDataSet{TR,TB,TI}}, 
    Σ::Matrix{TR}, options::ARFFOptions;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(first(data_sets))
    Σ_mean, acceptance_rate, loss = train_arff!(F, Iterators.cycle(data_sets), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)
    return Σ_mean, acceptance_rate, loss
end


"""
    train_rwm(F₀, data, Σ, options; show_progress=true, record_loss=true) 

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F₀` - The `FourierModel` to be trained
* `data`- The `DataSet` training data
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    N = length(data)
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_arff(F₀, Iterators.cycle([data]), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀, data, batch_size, Σ, options; show_progress=true, record_loss=true) 

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F₀` - The `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `batch_size` - Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI}, batch_size::TI,
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_arff(F₀, Iterators.cycle([data]), batch_size, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀, data_sets, Σ, options; show_progress=true, record_loss=true) 

Train the Fourier feature model using a random walk Metropolis exploration
strategy

### Fields
* `F₀` - The `FourierModel` to be trained
* `data_sets`- A vector of `DataSet` training data sets.  These are presumed
  to all be the same size.  They will be cycled through each epoch.
* `batch_size` - Minibatch size
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_rwm(F₀::VectorFourierModel{TR,TB,TI,TA}, data_sets::Vector{VectorDataSet{TR,TB,TI}},
    Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

    N = length(first(data_sets))
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_arff(F₀, Iterators.cycle(data_sets), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, Σ_mean, acceptance_rate, loss
end

