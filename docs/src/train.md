# Training
```@contents
Pages = ["train.md"]
```

The key function is `train_rwm!`, which performs in place training on the model.
This is implemented to handle the training data in several ways:
* A single `DataSet` can be provided and used in every training epoch.
* A single `DataSet` and a minibatch size can be provided, and minibatchs will be generated at each epoch.
* An array of `DataSet` types can be provided, and they will be cycled through each epoch;

## In Place Training
For both the scalar and vector valued case, we have the following commands:
```@docs
    train_rwm!(F::ARFF.ScalarFourierModel{TR,TB,TI,TA}, data::ARFF.ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFF.ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer, TA<:ARFF.ActivationFunction{TB}}
    train_rwm!(F::ARFF.ScalarFourierModel{TR,TB,TI,TA}, data::ARFF.ScalarDataSet{TR,TB,TI}, batch_size::TI, Σ::Matrix{TR}, options::ARFF.ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TB}}    
    train_rwm!(F::ARFF.ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TB,TI}}, Σ::Matrix{TR}, options::ARFF.ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TB}}
```
Having created an initial `F` we can then call
```
Σ_mean, acceptance_rate, loss = train_rwm!(F, data, Σ0, opts);
```
The returned quantities are the mean adapted covariance matrix `Σ_mean`.  The
`acceptance_rate` is the mean acceptance rate at each epoch, averaged overa the
internal steps,  `K * n_ω_steps`.  The loss is the recorded training loss,
stored in `opts.loss`, at each epoch.

## Recording the Training Trajectory
We have also included routines which record the model at each epoch.  These are called in the same way as above:
```
F_trajectory, Σ_mean, acceptance_rate, loss = train_rwm(F0, data, Σ0, opts);
```

The functions are analogously named:
```@docs
    train_rwm(F::ARFF.ScalarFourierModel{TR,TB,TI,TA}, data::ARFF.ScalarDataSet{TR,TB,TI}, Σ::Matrix{TR}, options::ARFF.ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer, TA<:ARFF.ActivationFunction{TB}}
    train_rwm(F::ARFF.ScalarFourierModel{TR,TB,TI,TA}, data::ARFF.ScalarDataSet{TR,TB,TI}, batch_size::TI, Σ::Matrix{TR}, options::ARFF.ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TB}}    
    train_rwm(F::ARFF.ScalarFourierModel{TR,TB,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TB,TI}}, Σ::Matrix{TR}, options::ARFF.ARFFOptions; show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TB}}
```
This records the result at the end of each epoch.