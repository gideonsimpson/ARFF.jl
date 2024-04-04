# Training

The key function is `train_rwm!`, which performs in place training on the model.
This is implemented to handle the training data in several ways:
* A single `DataSet` can be provided and used in every training epoch
* An array of `DataSet` types can be provided, and they will be cycled through each epoch
* A single `DataSet` and a minibatch size can be provided, and minibatchs will be generated at each epoch

```@docs
    train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}
    train_rwm!(F::FourierModel{TB,TR,TW}, batched_data::Vector{DataSet{TB,TR,TW}}, Σ::TM, options::ARFFOptions) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}
    train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, batch_size::TI, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix,TI<:Integer}
```