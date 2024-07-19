# Vector Valued Example

This example demonstarates learning a vector valued problem with
```math
f(x) = \begin{pmatrix}x_1 x_2\\x_1^2 - x_2^2\end{pmatrix}
```

```@example ex3
using Random
using Plots
using LinearAlgebra
using ARFF

# define components of vector valued function
f1(x) = x[1]*x[2];
f2(x) = x[1]^2 - x[2]^2;

xx = LinRange(-2, 2, 101);
yy = LinRange(-2, 2, 101);

z1 = [f1([x_, y_]) for y_ in yy, x_ in xx];
z2 = [f2([x_, y_]) for y_ in yy, x_ in xx];

p1 = contourf(xx, yy, z1)
xlabel!("x")
ylabel!("y")
p2 = contourf(xx, yy, z2)
xlabel!("x")
plot(p1, p2, layout=(1, 2), plot_title="Truth")
```
## Generate Training Data
First, we generate our training data
```@example ex3
N = 10^3;
d=2;
Random.seed!(100)
x_data = [randn(2) for _ in 1:N];
y_data = [[f1(x_), f2(x_)] for x_ in x_data];
data = DataSet(x_data, complex.(y_data));
```
## Initialize Fourier Model
Next, we need to initialize our [`FourierModel`](@ref)
```@example ex3
K = 2^6;
Random.seed!(200)
d = 2;
F0 = FourierModel([1.0 * randn(d) for _ in 1:K], 
    [1.0 * randn(d) for _ in 1:K]);
```
## Set Parameters and Train
```@example ex3
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

Random.seed!(1000);
F = deepcopy(F0);
Σ_mean, acceptance_rate, loss = train_rwm!(F, data, Σ0, opts, show_progress=false); nothing
```
## Evaluate Results
Looking at the trianing loss, we see the model appears to be well trained for the selected width, ``K``:
```@example ex3
plot(1:length(loss), loss, yscale=:log10, label="")
xlabel!("Epoch")
ylabel!("Loss")
```
Next, we look at how the components of the trained model look:
```@example ex3
z1 = [real(F([x_, y_])[1]) for y_ in yy, x_ in xx];
z2 = [real(F([x_, y_])[2]) for y_ in yy, x_ in xx];

p1 = contourf(xx, yy, z1)
xlabel!("x")
ylabel!("y")
p2 = contourf(xx, yy, z2)
xlabel!("x")
plot(p1, p2, layout=(1, 2),plot_title="Trained Model")
```