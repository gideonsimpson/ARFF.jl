# Activation Functions
While the theory has been developed for the Fourier activation functions of type
```e^{i \omega \cdot x}```, this module permits for general activation funcitons
to be encoded in an `ActivationFunction` data structure.  
```@docs
ActivationFunction
```
This structure encodes both the function, along with the return data type (real or complex) to ensure type consistency for performance.  The following example would define the ReLU activation function:
```@example 1
using ARFF
relu(z) = z*(z>0);
ReLUActivation = ActivationFunction{Float64}(relu)
```
This can then be passed in as an argument when defining a Fourier feature model.


The `ActivationFunction` type can be evaluated:
```@example 1
println(ReLUActivation([-1.], [1.]));
println(ReLUActivation([1.], [1.]));
```

**NOTE** It is essential that the data type, `TN` used when defining
`ActivationFunction{TN}(myactfunc)` must be the same as the numerical type of
`y` used in the construction of a data set.

Some convenience functions are a part of the package:
* `FourierActivation` - The complex exponential for type `ComplexF64`
* `SigmoidActivation` - The sigmoid function for type `Float64`
* `ArcTanActivation` - The arctan function for type `Float64`

