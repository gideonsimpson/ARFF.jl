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
We thus require that the user provided linear solver take the form:
```
function linear_solver!(β, S, y, ω)
    # solve for β coefficients
    β
end
```
Obviously, if one does not need `ω` for your formulation, as is the case in the
original regularization of [kammonen_adaptive_2020](@cite), this argument is
just ignored.  For [`solve_normal!`](@ref) we would implement this as:
```
linear_solver! = (β, S, y, ω) -> solve_normal!(β, S, y)
```

In the case that one is solving a vector valued problem, the vector valued
``\beta``'s are obtained component by component against the vector valued
``y``'s. 

## [Loss Functions] (@id loss)
```@docs
ARFF.mse_loss
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

