# [Data Sets](@id dataset)
Training (and testing data) are stored in a `DataSet` structure (with either scalar or vector valued ``y``) 
```@docs
ARFF.ScalarDataSet
ARFF.VectorDataSet
```

## Constructing Data Sets
Data sets can be constructed by passing in the ``(x_i,y_i)`` information:
```@docs
    DataSet
```
For instance, a scalar valued data set can be generated with:
A training set can be constructed as follows:
```@example 1
using ARFF
using Random 
Random.seed!(100); # for reproducibility

n = 10; # number of sample points
x = [rand(2) for _ in 1:n];
f(x) = exp(-x[1]*x[2]); # arbitrary function
y = complex.(f.(x)); # make y valued data complex for type consistency

data = DataSet(x,y);
println(data.x)
println(data.y)
```
Vector valued data can be similarly constructed:
```@example 2
using ARFF
using Random 
Random.seed!(100); # for reproducibility

n = 10; # number of sample points
x = [rand(2) for _ in 1:n];
f(x) = [x[1]+x[2]; exp(-x[1]*x[2])]; # arbitrary function
y = f.(x);

data = DataSet(x,y);
println(data.x)
println(data.y)
```
## [Scaling Data Sets] (@id scalings)
It is often helpful to scale the training data (setting means to zero and variances to unity).  These can be accomplished with the scalings structure and the associated functions.
```@docs
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