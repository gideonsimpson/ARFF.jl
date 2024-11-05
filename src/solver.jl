"""
    ARFFSolver{TS, TM, TR, TL}

Data structure containing key parameters for ARFF training
### Fields
* `linear_solve!` - User specified solver for the normal equations
* `mutate!` - In place transformation for the mutation/exploration step
* `resample!` - In place transformation for the resampling step
* `loss` - Loss function
"""
struct ARFFSolver{TS, TM, TR, TL}
    linear_solve!::TS
    mutate!::TM
    resample!::TR
    loss::TL
end