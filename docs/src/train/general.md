# General Training

## [Data Structure](@id arffsolver)
In the most general case, when a user calls `train_arff!` or `train_arff`, they must provide an `ARFFSolver` data structure:
```@docs
ARFF.ARFFSolver
```
This includes stores the following functions:
* `linear_solve!(β, ω, x, y, S, epoch)`: `β` and `ω` are the coefficients and
  wave numbers;  `(x,y)` are arrays of training pairs; `S` is the design matrix;
  and `epoch` is the current epoch.  Not all arguments are neccessarily used,
  but this is how the solver will be called.
* `mutate!(F, x, y, S, epoch)`:  `F` is the Fourier model; `(x,y)` are arrays of
  training pairs; `S` is the design matrix; and `epoch` is the current epoch.
* `resample!(F, x, y, S, epoch)`: This follows the convetion of `mutate!`
* `loss(F, x, y)`: `F` is the Fourier model; and `(x,y)` are arrays of training
  pairs.

## Training
These are underlying, general, training functions.  The user may wish to access
these if they have custom mutation and resampling steps.  Otherwise, see, for
instance [Random Walk Metropolis Training](@ref) which performs the algorithm
described in [kammonen_adaptive_2020](@cite)
```@docs
ARFF.train_arff!(F::ARFF.AbstractFourierModel, data_sets::TD, batch_size::TI, solver::ARFF.ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TD,TI<:Integer}
ARFF.train_arff(F₀::ARFF.AbstractFourierModel, data_sets::TD, batch_size::TI, solver::ARFF.ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TD,TI<:Integer}
```

For convenience, we have also included the following versions of these functions:
```@docs
ARFF.train_arff!(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
ARFF.train_arff!(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, batch_size::TI, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
ARFF.train_arff!(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TY,TI}}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
ARFF.train_arff(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
ARFF.train_arff(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data::ARFF.ScalarDataSet{TR,TY,TI}, batch_size::TI, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
ARFF.train_arff(F₀::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data_sets::Vector{ARFF.ScalarDataSet{TR,TY,TI}}, solver::ARFFSolver, n_epochs::TI; show_progress=true, record_loss=true) where {TY<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TY}}
```
