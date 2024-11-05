"""
    train_arff!(F, data, solver, n_epochs; show_progress=true, record_loss=true)

Perform ARFF training on a given data set
### Fields
* `F` - Fourier feature model to be trained, in place
* `data` - Training data set
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff!(F::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    N = length(data);
    loss = train_arff!(F, Iterators.cycle([data]), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss);
    return loss
end

"""
    train_arff!(F, data, solver, n_epochs; show_progress=true, record_loss=true)

Perform ARFF training on a given data set with minibatching
### Fields
* `F` - Fourier feature model to be trained, in place
* `data` - Training data set
* `batch_size` - Minibatch size
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff!(F::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, batch_size::TI, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    loss = train_arff!(F, Iterators.cycle([data]), batch_size, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)
    return loss
end

"""
    train_arff!(F, data, solver, n_epochs; show_progress=true, record_loss=true)

Perform ARFF training on a given data set with minibatching
### Fields
* `F` - Fourier feature model to be trained, in place
* `data_sets` - Array of training data sets
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff!(F::VectorFourierModel{TR,TY,TI,TA}, data::Vector{VectorDataSet{TR,TY,TI}}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    N = length(data)
    loss = train_arff!(F, Iterators.cycle(data_sets), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)
    return loss
end


"""
    train_arff(F₀, data, solver, n_epochs; show_progress=true, record_loss=true) 

Perform RWM training an ARFF model, recording the result at all epochs.
### Fields
* `F₀` - Initial fourier feature model to be trained
* `data` - Training data set
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff(F₀::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    N = length(data)
    F_trajectory, loss = train_arff(F₀, Iterators.cycle([data]), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, loss
end

"""
    train_arff(F₀, data, batch_size, solver, n_epochs; show_progress=true, record_loss=true) 

Perform RWM training an ARFF model, recording the result at all epochs.
### Fields
* `F₀` - Initial fourier feature model to be trained
* `data` - Training data set
* `batch_size` - Minibatch size
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff(F₀::VectorFourierModel{TR,TY,TI,TA}, data::VectorDataSet{TR,TY,TI}, batch_size::TI, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    F_trajectory, loss = train_arff(F₀, Iterators.cycle([data]), batch_size, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, loss
end

"""
    train_arff(F₀, data_sets, solver, n_epochs; show_progress=true, record_loss=true) 

Perform RWM training an ARFF model, recording the result at all epochs.
### Fields
* `F₀` - Initial fourier feature model to be trained
* `data_sets` - Array of training data sets
* `solver` - An `ARFFSolver` data structure
* `n_epochs` - Number of training epochs
* `show_progress=true` - Display training progress using `ProgressMeter`
* `record_loss=true` - Evaluate the specified loss function at each epoch and record
"""
function train_arff(F₀::VectorFourierModel{TR,TY,TI,TA}, data::Vector{VectorDataSet{TR,TY,TI}}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
    N = length(data)
    F_trajectory, loss = train_arff(F₀, Iterators.cycle(data_sets), N, solver, n_epochs, show_progress=show_progress, record_loss=record_loss)
    return F_trajectory, loss
end