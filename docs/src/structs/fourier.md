# [Fourier Feature Models] (@id fourier)
The generalized fourier features model approximates functions with ``K`` features, with acvtivation function ``\varphi`` as
```math
f^\dagger(x) \approx f(x) = \sum_{k=1}^K \beta_k \varphi(x;\omega_k).
```
In the above expression, ``x,\omega_k\in \mathbb{R}^d``, while
``\beta_k\in\mathbb{R}^{d'}`` or ``\beta_k\in\mathbb{C}^{d'}``, depending on the choice of activation function.  Consequently, our model is uniquely determined by the
coefficients and the wavenumbers.  

These are stored in either a scalar or vector valued data structure:
```@docs
ARFF.ScalarFourierModel
ARFF.VectorFourierModel
```
## Constructing a Fourier Features Model
A Fourier features model can be instantiated with:
```@docs
    FourierModel
```
As an example, the scalar valued Fourier model with complex exponentials can be
instantiated as:
```@example 1
using ARFF
using Random
Random.seed!(200); # for reproducibility

K = 10;
d = 2;
# convenience
F = FourierModel(randn(ComplexF64, K),  [randn(d) for _ in 1:K]);
```
This defines `F` with random wavenumbers and amplitudes.  Strictly speaking,
this is not required, but it is often helpful within the learning context that
we will apply this method.  By not specifying an activation function, this
defaults to `FourierActivation` function, ```e^{i \omega \cdot x}```, and presumes all ```y``` related values are of the same complex type.


Function evaluation has been overloaded for a `FourierModel`, allowing us to evaluate it at points:
```@example 1
x_test = [0., 1.];
println(F(x_test));
```
and it can perform a vectorized evaluation:
```@example 1
n = 10; # number of sample points
x = [rand(2) for _ in 1:n];
f(x) = exp(-x[1]*x[2]); # arbitrary function
y = complex.(f.(x)); # make y valued data complex for type consistency

data = DataSet(x,y);

F.(data.x);
```

As is sometimes the case, we may scale our data, train in the scale coordinate,
and wish to evaluate new poitns in the original, unscaled, coordinate system.
We can accomplish that by passing a [`DataScalings`](@ref scalings) argument in for evaluation:
```@example 1
scalings = get_scalings(data);

F(x_test, scalings)
```

If one wishes to use a different activation function, we pass that as an argument in the construction:
```@example 1
G = FourierModel(randn(K),  [randn(d) for _ in 1:K],SigmoidActivation);
println(G([0.1, 0.2]))
```
