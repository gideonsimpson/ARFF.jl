# Random Walk Metropolis Training
For convenience, we have implemented the RWM strategy from
[kammonen_adaptive_2020](@cite).  

## RWM Data Structure 
This makes use of an `AdaptiveRWMSampler` data structure, which can be constructed with one of the following two commands:
```@docs
ARFF.AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, δ::TR) where {TF<:ARFF.AbstractFourierModel,TS,TI<:Integer,TR<:AbstractFloat}
ARFF.AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, Σ0::TM, γ::TI, δ::TR, ω_max::TR) where {TF<:ARFF.AbstractFourierModel,TS,TI<:Integer,TM<:AbstractMatrix,TR<:AbstractFloat}
```
The `linear_solve!` argument should be of the same type as in [ARFFSolver](@ref
arffsolver), ``linear_solve!(β, ω, x, y, S, epoch)`; see [Linear Algebra] (@ref
linalg) for additional details.  It is not neccessary to separately construct an
`ARFFSolver` structure if using these.  These default to the mean squared error
loss function, [`mse_loss`](@ref loss).  As an exmaple, the following code is sufficient to construct
such a structure, assuming one has already constructed an initial
[`FourierModel`](@ref fourier), `F0`:
```@setup ex1
using ARFF
K = 64;
F0 = FourierModel([1.0 * randn() for _ in 1:K], [[randn()] for _ in 1:K])
```
```@example ex1
δ = 1.0 # rwm step size
n_rwm_steps = 10 # number of RWM steps within the mutation step
n_burn = 100 # number of training epochs to use for burn in.

# format the linear solver function to be compatible with ARFF.jl
linear_solver! = (β, ω, x, y, S, epoch) -> solve_normal!(β, S, y)

rwm_sampler = AdaptiveRWMSampler(F0, linear_solver!, n_rwm_steps, n_burn, δ);
```

## RWM Mutation
TBW

## RWM Training
RWM training is implemented to handle the training data in several ways:
* A single `DataSet` can be provided and used in every training epoch.
* A single `DataSet` and a minibatch size can be provided, and minibatchs will be generated at each epoch.
* An array of `DataSet` types can be provided, and they will be cycled through each epoch.

Additionally, we have both an in place training commands, along with commands which will record the training trajectory.  These can be used for both scalar and vector valued problems.

```@docs
ARFF.train_rwm!(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler}
ARFF.train_rwm!(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler}
ARFF.train_rwm!(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TY,TI}}, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler} 
```
Having created an initial `F` we can then call
```
Σ_mean, acceptance_rate, loss = train_rwm!(F, data, rwm_sampler, n_epochs);
```
The returned quantities are the mean adapted covariance matrix `Σ_mean`.  The `acceptance_rate` is the mean acceptance rate at each epoch, averaged overa the internal steps,  `K * n_rwm_steps`.  The `loss` is the recorded training loss at each epoch.


```@docs
ARFF.train_rwm(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler}
ARFF.train_rwm(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler}
ARFF.train_rwm(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TY,TI}}, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler} 
```
Having created an initial `F₀` we can then call
```
F_trajecotry, Σ_mean, acceptance_rate, loss = train_rwm(FF₀, data, rwm_sampler, n_epochs);
```
The returned quantities are the algorithmic trajectory, `F_trajecotry`, the mean
adapted covariance matrix `Σ_mean`.  The `acceptance_rate` is the mean
acceptance rate at each epoch, averaged overa the internal steps,  `K *
n_rwm_steps`.  The `loss` is the recorded training loss at each epoch.
