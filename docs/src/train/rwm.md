# Random Walk Metropolis Training
For convenience, we have implemented the RWM strategy from
[kammonen_adaptive_2020](@cite).  

## RWM Data Structures

### Classical RWM
For a classical RWM, we can employ a `RWMSampler` data structure, which can be constructed with one of the following commands:
```@docs
ARFF.RWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, δ::TR) where {TF<:ARFF.AbstractFourierModel,TS,TI<:Integer,TR<:AbstractFloat}
ARFF.RWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, Σ::TM, γ::TI, δ::TR, ω_max::TR) where {TF<:ARFF.AbstractFourierModel,TS,TI<:Integer,TM<:AbstractMatrix,TR<:AbstractFloat}
```
The `linear_solve!` argument should be of the same type as in [`ARFFSolver`](@ref), ``linear_solve!(β, ω, x, y, S, epoch)`; see [Linear Algebra](@ref
linalg) for additional details.  It is not neccessary to separately construct an
`ARFFSolver` structure if using these.  These default to the mean squared error
loss function, [`mse_loss`](@ref).  As an exmaple, the following code is sufficient to construct such a structure, assuming one has already constructed an initial
[`FourierModel`](@ref fourier), `F0`:
```@setup ex1
using ARFF
K = 64;
F0 = FourierModel([1.0 * randn() for _ in 1:K], [[randn()] for _ in 1:K])
```
```@example ex1
δ = 1.0 # rwm step size
n_rwm_steps = 10 # number of RWM steps within the mutation step

# format the linear solver function to be compatible with ARFF.jl
linear_solver! = (β, ω, x, y, S, epoch) -> solve_normal!(β, S, y)

rwm_sampler = RWMSampler(F0, linear_solver!, n_rwm_steps, δ);
```

### Adaptive RWM
It may be more useful to use an `AdaptiveRWMSampler` data structure; this will update proposal covariance matrix after `n_burn` epochs.  This is the version used in our examples.

The `AdaptiveRWMSampler` structure constructed with:
```@docs
ARFF.AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, δ::TR) where {TF<:ARFF.AbstractFourierModel,TS,TI<:Integer,TR<:AbstractFloat}
ARFF.AdaptiveRWMSampler(F::TF, linear_solve!::TS, n_rwm_steps::TI, n_burn::TI, Σ0::TM, γ::TI, δ::TR, ω_max::TR) where {TF<:ARFF.AbstractFourierModel,TS,TI<:Integer,TM<:AbstractMatrix,TR<:AbstractFloat}
```
As an example:
```@setup ex2
using ARFF
K = 64;
F0 = FourierModel([1.0 * randn() for _ in 1:K], [[randn()] for _ in 1:K]);
F = deepcopy(F0);
n_epochs = 100;
```
```@example ex2
δ = 1.0 # rwm step size
n_rwm_steps = 10 # number of RWM steps within the mutation step
n_burn = 100 # number of training epochs to use for burn in.

# format the linear solver function to be compatible with ARFF.jl
linear_solver! = (β, ω, x, y, S, epoch) -> solve_normal!(β, S, y)

rwm_sampler = AdaptiveRWMSampler(F0, linear_solver!, n_rwm_steps, n_burn, δ); nothing
```

## RWM Mutation
As noted in [General Training](@ref), the underlying training algorithm for ARFF
requires the specification of a mutation step.  When we use RWM for that mutation step, as is automatically done for the user in [RWM Training](@ref), and many of our examples, what we are actually doing is first constructing the mutation function, and then embedding it within an `ARFFSolver` data structure, as follows:
```@example ex2
mutate_rwm!(F, x, y, S, epoch) = ARFF.rwm!(F0, rwm_sampler, x, y, S, epoch);
solver = ARFFSolver(rwm_sampler.linear_solve!, mutate_rwm!, trivial_resample!, mse_loss); nothing
```
This example assumes, of course, that you have already constructed the `rwm_sample` object.

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
acceptance_rate, loss = train_rwm!(F, data, rwm_sampler, n_epochs);
```
The `acceptance_rate` is the mean acceptance rate at each epoch, averaged over the internal steps,  `K * n_rwm_steps`.  The `loss` is the recorded training loss at each epoch.  If using [`AdaptiveRWMSampler`](@ref), the tuned covariance can be obtained by examining
```
rwm_sampler.Σ_mean
```


```@docs
ARFF.train_rwm(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler}
ARFF.train_rwm(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, batch_size::TI, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler}
ARFF.train_rwm(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TY,TI}}, rwm_sampler::TS, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ARFF.ActivationFunction{TY},TS<:ARFF.AdaptiveRWMSampler} 
```
Having created an initial `F₀` we can then call
```
F_trajecotry, acceptance_rate, loss = train_rwm(FF₀, data, rwm_sampler, n_epochs);
```
One of the returned quantities is the algorithmic trajectory, `F_trajecotry`.
The `acceptance_rate` is the mean acceptance rate at each epoch, averaged overa
the internal steps,  `K * n_rwm_steps`.  The `loss` is the recorded training
loss at each epoch.
