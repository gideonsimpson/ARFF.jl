# Example with Generalized Activation Functions and Scalings

This example demonstates learning with generalized activation functions, a bias term, and scalings.  For this problem, 
```math
f(x) = \exp\left(-\frac{1}{2}\left(x_1^2 + \tfrac{1}{100}x_2^2 \right)\right).
```
As was the case in [Scalar Example with Generalized Activation Functions](@ref),
we will make use of sigmoid activations and augment the problem with a bias.
What is a bit different here is that we will also scale the data.

## Generate Training Data
```@setup ex3
using Random
using Distributions
using Plots
using LinearAlgebra
using ARFF
```
First, we will generate and visualize the training data:
```@example ex3
f(x) = exp(-0.5 * (x[1]^2+ x[2]^2/100));

n_x = 1000; # number of training points

Random.seed!(100); # for reproducibility
# generate n_x sample points
d_ = 2;
x = [[rand(Uniform(-5.0, 5.0)), rand(Uniform(-50.0, 50.0))] for _ in 1:n_x];
y = f.(x);

# store data in DataSet structure
data_ = DataSet(x,y);
# scale the data
scalings_ = get_scalings(data_);
scale_data!(data_, scalings_)
# append the bias term to our data and scalings
data = append_bias(data_);
scalings = append_bias(scalings_);
d  = d_ + 1; nothing
```

## Initialize Fourier Model with Generalized Features
Next, we need to initialize our [`FourierModel`](@ref)
```@example ex3
K = 2^7;
Random.seed!(200); # for reproducibility
F0 = FourierModel([1. *randn() for _ in 1:K],  
    [randn(d) for _ in 1:K],SigmoidActivation); nothing
```

## Set Parameters and Train
```@example ex3
δ = 0.1; # rwm step size
λ = 1e-6; # regularization
n_epochs = 10^3; # total number of iterations
n_rwm_steps = 10; # number of steps between full β updates
n_burn = n_epochs ÷ 10;

linear_solver! = (β, ω, x, y, S, epoch)-> solve_normal!(β, S, y, λ=λ);

rwm_sampler = AdaptiveRWMSampler(F0, linear_solver!, n_rwm_steps, n_burn, δ);

Random.seed!(1000); # for reproducibility
F = deepcopy(F0);
Σ_mean, acceptance_rate, loss= train_rwm!(F, data, rwm_sampler, n_epochs, show_progress=false); nothing 
```

## Evaluate Results
Looking at the trianing loss, we see the model appears to be well trained for the selected width, ``K``:
```@example ex3
plot(1:length(loss), loss, yscale=:log10, label="")
xlabel!("Epoch")
ylabel!("Loss")
```
We verify that the training data is well fit:
```@example ex3
scatter(data.y,F.(data.x),label="Data")
xx = LinRange(minimum(data.y),maximum(data.y),100);
plot!(xx, xx, ls=:dash, label="")
xlabel!("Truth")
ylabel!("Prediction")
```
Lastly, a direct comparison of the learned approximation of ``f``:
```@example ex3
xx = LinRange(-2, 2, 100)
yy = LinRange(-20, 20, 100)

p1 = contourf(xx, yy, [f([x_, y_]) for y_ in yy, x_ in xx],levels=LinRange(-0.1,1.1,20), colorbar=:false)
xlabel!(p1, "x")
ylabel!(p1, "y")
title!(p1, "Truth")
p2 = contourf(xx, yy, [F([x_, y_, 1], scalings) for y_ in yy, x_ in xx], levels=LinRange(-0.1, 1.1, 20), colorbar=:false)
xlabel!(p2, "x")
title!(p2, "Learned Model")
plot(p1, p2, layout=(1, 2))
```