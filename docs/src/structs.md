# Data Structures
```@contents
Pages = ["structs.md"]
```

## Data Sets 
Training (and testing data) are stored in `DataSet` structure.
```@docs
    DataSet
```
A training set can be constructed as follows
```@example 1
using ARFF
using Random 
Random.seed!(100); # for reproducibility

n = 10; # number of sample points
x = [rand(2) for _ in 1:n];
f(x) = exp(-x[1]*x[2]); # arbitrary function
y = f.(x);

data = DataSet(x,y);
println(data.x)
println(data.y)
```

Additionally, it is often helpful to scale the training data (setting means to zero and variances to unity).  These can be accomplished with the `DataScalings` structure and the associated functions.
```@docs
    DataScalings
    get_scalings
    scale_data!
    rescale_data!
```
For example, the following code obtains the means and variances in the data:
```@example 1
scalings = get_scalings(data);
println(scalings.μx);
println(scalings.σ2x);
println(scalings.μy);
println(scalings.σ2y);
```
Next, we can scale our data:
```@example 1
scale_data!(data, scalings);

using Statistics
println(mean(data.x));
println(var(data.x));
println(mean(data.y));
println(var(data.y));
```
If needed, we can then undo the scaling,
```@example 1
rescale_data!(data, scalings)

println(mean(data.x));
println(var(data.x));
println(mean(data.y));
println(var(data.y));
```
Note that these are in agreement with what was contained in our `scalings` structure.

## Fourier Feature Models
Fourier features approximates functions with ``K`` features according to
```math
f_{\rm true}(x) \approx f(x) = \sum_{k=1}^K \beta_k e^{i \omega_k \cdot x}
```
In the above expression, ``x,\omega_k\in \mathbb{R}^d``, while
``\beta_k\in\mathbb{C}``.  Consequently, our model is uniquely determined by the
coefficients and the wavenumbers.  

A Fourier features model can be instantiated with a `FourierModel` structure:
```@docs
    FourierModel
```
As an example,
```@example 1
Random.seed!(200); # for reproducibility

K = 10;
d = 2;

F = FourierModel([randn(ComplexF64) for _ in 1:K],  
    [randn(d) for _ in 1:K]);
```
This defines `F` with random wavenumbers and amplitudes.  Strictly speaking,
this is not required, but it is often helpful within the learning context that
we will apply this method.  Function evaluation has been overloaded for a `FourierModel`, allowing us to evaluate it at points:
```@example 1
x_test = [0., 1.];
println(F(x_test));
```
and it can perform a vectorized evaluation:
```@example 1
F.(data.x);
```


As is sometimes the case, we may scale our data, train in the scale coordinate,
and wish to evaluate new poitns in the original, unscaled, coordinate system.
We can accomplish that by passing a [`DataScalings`](@ref) argument in for evaluation:
```@example 1
F(x_test, scalings)
```

## Training Options
To train a `FourierModel`, the training options must first be specified by instantiating an `ARFFOptions` struct with the desired parameters.
```@docs
    ARFFOptions
```
The above arguments are passed in as follows:
```@example 1
δ = 1.; # rwm step size
n_epochs = 10^3; 
n_ω_steps = 10; 
n_burn = 100;
γ = 1;
ω_max =Inf;
adapt_covariance = true;

# solve the  normal equations
function reg_β_solver!(β, S, y, λ)
    N = length(y);
    β .= (S' * S + λ * N *I) \ (S' * y);
    β
end

linear_solve! = (β, S, y, ω)-> reg_β_solver!(β, S, y, λ);

opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max, adapt_covariance, 
    linear_solve!, ARFF.mse_loss);
```
The linear solver type is typed as:
```
function linear_solve!(β, S, y, ω)
    # in place solver code...
end
```
which is to say, it can depend on the design matrix ``S``, the response
variables, ``y``, and, potentially, the current set of wave numbers, ``ω``.
The `loss` function is to be typed as:
```
function loss(F, x, y)
    # compute and return loss function...
end
```
The `γ` parameter can be set using [`optimal_γ`](@ref), which, in a certain
regime, is the optimal value for given dimension, `d` when ``x\in\mathbb{R}^d``.