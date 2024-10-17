


"""
    ARFFOptions{TI<:Integer,TF<:AbstractFloat,TB<:Bool,TL,TS}

Data structure containing some global training options and parameters
### Fields
* `n_epochs` - Number of training epochs
* `loss` - User specified loss function
"""
struct ARFFOptions{TI<:Integer,TL}
    n_epochs::TI # M in text, total number of iterations
    loss::TL
end
