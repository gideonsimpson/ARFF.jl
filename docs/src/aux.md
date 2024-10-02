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

## [Adding Bias] (@id bias)
It may be neccessary to modify an existing data set so as to include a constant bias term in a model.  If ``x\in \mathbb{R}^d``, then 
```math
x\mapsto (x,1)=\tilde{x}\in \mathbb{R}^{d+1}.
```
This is relevant when using generalized activation functions; see [Scalar Example with Generalized Activation Functions](@ref).  We provide tools for account for the constant in both the `DataSet` and `DataScalings` types:
```@docs
append_bias(data::ARFF.ScalarDataSet{TR,TY,TI}) where {TY<:Number,TR<:AbstractFloat,TI<:Integer}
append_bias(scalings::ARFF.ScalarDataScalings{TR,TY}) where {TY<:Number,TR<:AbstractFloat}
```
In this framework, we would compute and/or apply the scalings **before** adding the bias term to the data set, and then add the bias in to both the data and the scaling structures.  This avoids a potential divide by zero issue.
```@setup bias1
using ARFF
using Statistics
using Printf
```
```@example bias1
n_x = 10
x = [Float64[i] for i in 1:n_x]
f(x) = x[1]^2
y = f.(x)
data_ = DataSet(x, y)
scalings_ = get_scalings(data_);
data_scaled = scale_data!(data_, scalings_);
```
The means and variances of the scaled data set are as we would expect:
```@example bias1
println(mean(data_scaled.x));
println(var(data_scaled.x));
```
and the `scalings_` has the relevant information:
```@example bias1
println(scalings_.μx);
println(scalings_.σ2x);
```
Next, we can add in the bias into the data set and check that it is computing properly:
```@example bias1
data_bias = append_bias(data_scaled);
println(mean(data_bias.x));
println(var(data_bias.x));
```
and analogously with the scalings:
```@example bias1
scalings_bias = append_bias(scalings_);
println(scalings_bias.μx);
println(scalings_bias.σ2x);
```


## Other Utilities
```@docs
optimal_γ
```
Following Remark 1 in [kammonen_adaptive_2020](@cite), the optimal ``\gamma``
corresponds to ``\gamma = 3d -2``, which is encoded in the above function.

