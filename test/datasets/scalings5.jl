let
    n_x = 10
    x = [Float64[i] for i in 1:n_x]
    f(x) = [x[1], x[1]^2]
    y = f.(x)
    data = DataSet(x, y)
    scalings = get_scalings(data)
    scale_data!(data, scalings)
    σ2y = [var(data.yt[i]) for i in 1:data.dy]
    σ2y[1] ≈ 1.
end