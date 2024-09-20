
"""
    convert_problem(model::FourierModel)

Convert a Fourier Problem (non-Flux compatible) to a Flux Model
"""
function convert_problem(model::ScalarFourierModel)
    W = transpose(hcat(model.ω...))
    b = reshape(copy(model.β), 1, length(model.β))
    return Chain(
        Dense(W, false, fourier_features),
        Dense(b, false, identity)
    )
end

"""
    convert_dataset(data::DataSet)

Convert a dataset designed to work with FourierModel to work with RFF
"""
function convert_dataset(data::ScalarDataSet)
    return collect(zip(data.x, data.y))
end

