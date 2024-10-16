"""
    train_rwm!(F, data, Σ, options; show_progress=true, record_loss=true)

* `F` - The `FourierModel` to be trained
* `data`- A `DataSet` training data set
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record
"""
function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data::ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
  N = length(data);
  solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
  Σ_mean, acceptance_rate, loss = train_arff!(F, Iterators.cycle([data]), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)
  return Σ_mean, acceptance_rate, loss
end


"""
    train_rwm!(F, data, batch_size, Σ, options; show_progress=true, record_loss=true)

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
    N = length(data);
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    Σ_mean, acceptance_rate, loss = train_arff!(F, Iterators.cycle([data]), batch_size, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)

    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm!(F, data_sets, Σ, options; show_progress=true, record_loss=true)

* `F` - The `FourierModel` to be trained
* `data_sets`- A vector of `DataSet` training data sets.  These are presumed
  to all be the same size.  They will be cycled through each epoch.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record

"""
function train_rwm!(F::ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ScalarDataSet{TR,TB,TI}}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(first(data_sets))
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    Σ_mean, acceptance_rate, loss = train_arff!(F, Iterators.cycle(data_sets), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)

    return Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀, data, Σ, options; show_progress=true, record_loss=true)

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
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_arff(F₀, Iterators.cycle([data]), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀, data, batch_size, Σ, options; show_progress=true, record_loss=true)

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
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_arff(F₀, Iterators.cycle([data]), batch_size, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, Σ_mean, acceptance_rate, loss
end

"""
    train_rwm(F₀, data_sets, Σ, options; show_progress=true, record_loss=true) 

* `F₀` - The initial state of the `FourierModel` to be trained
* `data_sets`- A vector of `DataSet` training data sets.  These are presumed
  to all be the same size.  They will be cycled through each epoch.
* `Σ` - Initial covariance matrix for RWM proposals
* `options` - `ARFFOptions` structure specifcying the number epochs, proposal
  step size, etc.
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and
  record

"""
function train_rwm(F₀::ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ScalarDataSet{TR,TB,TI}}, Σ::Matrix{TR}, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(first(data_sets))
    solver = ARFFSolver(options.linear_solve!, trivial_mutate!, trivial_resample!)
    F_trajectory, Σ_mean, acceptance_rate, loss = train_arff(F₀, Iterators.cycle(data_sets), N, Σ, solver, options, show_progress=show_progress, record_loss=record_loss)

    return F_trajectory, Σ_mean, acceptance_rate, loss
end

