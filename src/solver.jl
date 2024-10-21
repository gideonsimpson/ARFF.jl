"""
    ARFFSolver{TS, TE, TR, TI<:Integer, TL}

Data structure containing key parameters for ARFF training
### Fields
* `linear_solve!` - User specified solver for the normal equations
* `mutate!` - In place transformation for the mutation/exploration step
* `resample!` - In place transformation for the resampling step
* `n_epochs` - Total number of training epochs
* `loss` - Loss function
"""
struct ARFFSolver{TS, TE, TR, TI<:Integer, TL}
    linear_solve!::TS
    mutate!::TE
    resample!::TR
    n_epochs::TI
    loss::TL
end