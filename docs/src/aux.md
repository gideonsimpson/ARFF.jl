# Auxiliary Functions and Utilities
```@contents
Pages = ["aux.md"]
```

## [Linear Algebra] (@id linalg)
It is essential to be able solve for the updated ``\boldsymbol{\beta}`` when we update the ``\boldsymbol{\omega}``.  In a typical setting, this corresponds to solving
```math
({S^\ast}S + N \lambda I)\boldsymbol{\beta} = S^\ast \boldsymbol{y},
```
We have included two naive solvers for this problem:
```@docs
solve_normal!
solve_normal_svd!
```

Other formulations may be more appropriate.  Indeed, in [kiessling_wind_2021](@cite), the authors use the regularized loss function in the spirit of Sobolev:
```math
\|S\boldsymbol{\beta}-\boldsymbol{y}\|_2^2 + \lambda \sum_{k} (1+|\omega_k|^2)|\beta_k|^2.
```
When using the linear solvers as part of an [`ARFFSolver`](@ref) or [`AdaptiveRWMSampler`] (@ref), the expectation is that they can be called as `linear_solve!(β, ω, x, y, S, epoch)`.  Thus, you will see, in examples, things implemented as:
```
linear_solver! = (β, ω, x, y, S, epoch)-> solve_normal!(β, S, y);
```
and `linear_solver!` is then used in the rest of the code.  

In the case that one is solving a vector valued problem, the vector valued
``\beta``'s are obtained component by component against the vector valued
``y``'s. The following bit of code will perform the component by component solve in a way that is consistent with the rest of `ARFF.jl`,
```
β_ = similar(F0.β[:, 1]); # allocate workspace memory
function component_solver!(β, ω, x, y, S, epoch)
    for d_ in 1:dy
        solve_normal!(β_, S, @view(y[:, d_]))
        @. β[:, d_] = β_
    end
    β
end
```

## [Loss Functions] (@id loss)
```@docs
ARFF.mse_loss(F::ARFF.ScalarFourierModel{TR,TY,TI,TA}, data_x::AbstractVector{Vector{TR}}, data_y::AbstractVector{TY}) where {TR<:AbstractFloat,TY<:Number,TI<:Integer,TA<:ARFF.ActivationFunction{TY}}
```
As the entire framework is built around the mean square loss function, we have
included it for convenience.  Other loss functions can be implemented, but they
should have the calling sequence:
```
function loss_function(F, data_x, data_y)
    # compute loss 
    return loss
end
```

## Other Utilities
```@docs
optimal_γ
```
Following Remark 1 in [kammonen_adaptive_2020](@cite), the optimal ``\gamma``
corresponds to ``\gamma = 3d -2``, which is encoded in the above function.

```@docs
trivial_resample!
trivial_mutate!
```
These functions can be used within an [`ARFFSolver`](@ref) data structure.