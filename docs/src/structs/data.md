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
