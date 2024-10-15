let
    n_x = 10
    x = [Float64[i] for i in 1:n_x]
    f(x) = [x[1], x[1]^2]
    y = f.(x)
    data = DataSet(x, y)
    scalings = get_scalings(data)
    scale_data!(data, scalings)
    rescale_data!(data, scalings)
    σ2y = var(data.y_mat, dims=1)
    σ2y[1] ≈ 9.166666666666666
end