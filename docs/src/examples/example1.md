# Scalar Example

This example is taken from Section 5 of [kammonen_adaptive_2020](@cite).
Consider trying to learn the scalar mapping
```math
f(x) = \mathrm{Si}\left(\frac{x}{a}\right)e^{-\frac{x^2}{2}}, \quad a>0,
```
where ``\mathrm{Si}`` is the [Sine
integral](https://en.wikipedia.org/wiki/Trigonometric_integral).  To make the
problem somewhat challenging, we take ``a=10^{-3}``.

## Generate Training Data
First, we will generate and visualize the training data:
```@example ex1
using SpecialFunctions
using Random
using Plots
using LinearAlgebra
using ARFF

a = 1e-3;
f(x) = sinint(x/a) * exp(-0.5 * (x^2));

n_x = 500; # number of training points
d = 1;
Random.seed!(100); # for reproducibility
x = [0.1*rand(d) for _ in 1:n_x];
y = [f(x_[1]) for x_ in x];

# store data in DataSet structure
data = DataSet(x,complex.(y));

scatter([x_[1] for x_ in x], y, label="Sample Points")
xx = LinRange(0, 0.1, 500);
plot!(xx, f.(xx), label="f(x)")
xlabel!("x")
```
**Note**: When the domain of our target function is ``\mathbb{R}^1``, the
``x``-data must still be stored as an array of arrays of length one, not an
array of scalars.  We also do not require [`DataScalings`](@ref scalings) for this
problem.

## Initialize Fourier Model
Next, we need to initialize our [`FourierModel`](@ref)
```@example ex1
K = 2^7;
Random.seed!(200); # for reproducibility
F0 = FourierModel([1. *randn(ComplexF64) for _ in 1:K],  
    [randn(d) for _ in 1:K]); nothing
```
This defaults to complex exponential activation functions.

## Set Parameters and Train
```@example ex1
δ = 10.; # rwm step size
λ = 1e-6; # regularization
n_epochs = 10^3; # number of epochs
n_rwm_steps = 10; # number of RWM steps during mutation
n_burn = n_epochs ÷ 10; # use 10% of the run for burn in

linear_solver! = (β, ω, x, y, S, epoch)-> solve_normal!(β, S, y, λ=λ);

rwm_sampler = AdaptiveRWMSampler(F0, linear_solver!, n_rwm_steps, n_burn, δ);

Random.seed!(1000); # for reproducibility
F = deepcopy(F0);
acceptance_rate, loss= train_rwm!(F, data, rwm_sampler, n_epochs, show_progress=false); nothing 
```
## Evaluate Results
Looking at the trianing loss, we see the model appears to be well trained for the selected width, ``K``:
```@example ex1
plot(1:length(loss), loss, yscale=:log10, label="")
xlabel!("Epoch")
ylabel!("Loss")
```
Next, we can verify that we have a high quality approximation of the true ``f(x)``:
```@example ex1
xx = LinRange(0, .1, 500);
scatter([x_[1] for x_ in x], y, label="Sample Points", legend=:right)
plot!(xx, f.(xx), label = "Truth" )
plot!(xx, real.([F([x_]) for x_ in xx]),label="Learned Model (Real Part)")
plot!(xx, imag.([F([x_]) for x_ in xx]),label="Learned Model (Imaginary Part)" )
xlabel!("x")
```
We can also verify that the training data is well fit:
```@example ex1
scatter(real.(data.y),real.(F.(data.x)),label="Training Data")
xx = LinRange(0,2,100);
plot!(xx, xx, ls=:dash, label="")
xlabel!("Truth")
ylabel!("Prediction")
```
