# Data Structures
```@contents
Pages = ["structs.md"]
```

## Data Sets
Training (and testing data) are stored in `DataSet` structure.
```@docs
    DataSet
```

Additionally, it is often helpful to scale the training data (setting means to zero and variances to unity).  These can be accomplished with the `DataScalings` structure and the associated functions.
```@docs
    DataScalings
    get_scalings
    scale_data!
    rescale_data!
```

## Fourier Feature Models
```@docs
    FourierModel
```

## Training Options
```@docs
    ARFFOptions
```