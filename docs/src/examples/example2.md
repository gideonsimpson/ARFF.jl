# Scalar Example with Generalized Activation Functions

This example demonstates learning with generalized activation functions.  For this problem, 
```math
f(x) = e^{-x^2/2}.
```
and we will work with sigmoid activation functions.  For this problem, we will augment our data to allow for a bias term, training, with an abuse of
```math
f(x) \approx F(\tilde{x}=(x,1)) = \sum_{k} \beta_k \varphi(\tilde{x};\omega_k).
```
The ``\tilde{x}`` is the variable padded with a unit value.  Thus, the problem will be studied as though it were over ``\mathbb{R}^2``.

## Generate Training Data
First, we will generate and visualize the training data:
```@example ex2
using Random
using Plots
using LinearAlgebra
using ARFF

f(x) = exp(-0.5 * (x^2));

n_x = 100; # number of training points
d = 2;
Random.seed!(100); # for reproducibility
x = [[4*rand(),1] for _ in 1:n_x]; # generate n_x sample points, storying them as an array of 1D points
y = [f(x_[1]) for x_ in x];

# store data in DataSet structure
data = DataSet(x,y);

scatter([x_[1] for x_ in x], y, label="Sample Points")
xx = LinRange(0, 4, 100);
plot!(xx, f.(xx), label="f(x)")
xlabel!("x")
```

## Initialize Fourier Model with Generalized Features
Next, we need to initialize our [`FourierModel`](@ref)
```@example ex2
K = 2^7;
Random.seed!(200); # for reproducibility
F0 = FourierModel([1. *randn() for _ in 1:K],  
    [randn(d) for _ in 1:K],SigmoidActivation); nothing
```

## Set Parameters and Train
```@example ex2
δ = 0.1; # rwm step size
λ = 1e-6; # regularization
n_epochs = 10^3; # total number of iterations
n_ω_steps = 10; # number of steps between full β updates
n_burn = n_epochs ÷ 10;
γ = optimal_γ(d);
ω_max =Inf;
adapt_covariance = true;

Σ0 = Float64[1 0; 0 1];


β_solver! = (β, S, y, ω)-> solve_normal!(β, S, y, λ=λ);

opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max,adapt_covariance, 
    β_solver!, ARFF.mse_loss);

Random.seed!(1000); # for reproducibility
F = deepcopy(F0);
Σ_mean, acceptance_rate, loss= train_rwm!(F, data, Σ0, opts, show_progress=false); nothing 
```
## Evaluate Results
Looking at the trianing loss, we see the model appears to be well trained for the selected width, ``K``:
```@example ex2
plot(1:length(loss), loss, yscale=:log10, label="")
xlabel!("Epoch")
ylabel!("Loss")
```
Next, we can verify that we have a high quality approximation of the true ``f(x)``:
```@example ex2
xx = LinRange(0, 4, 500);
scatter([x_[1] for x_ in x], y, label="Sample Points", legend=:right)
plot!(xx, f.(xx), label = "Truth" )
plot!(xx, [F([x_, 1]) for x_ in xx],label="Learned Model")
xlabel!("x")
```
We can also verify that the training data is well fit:
```@example ex2
scatter(data.y,F.(data.x),label="Data")
xx = LinRange(0,1,100);
plot!(xx, xx, ls=:dash, label="")
xlabel!("Truth")
ylabel!("Prediction")
```
