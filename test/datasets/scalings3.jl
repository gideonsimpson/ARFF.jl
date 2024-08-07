let 
    n_x = 10
    x = [Float64[i] for i in 1:n_x]
    f(x) = x[1]^2
    y = complex.(f.(x))
    data = DataSet(x, y)
    scalings = get_scalings(data);
    scale_data!(data, scalings);
    rescale_data!(data, scalings);
    scalings.μy ≈ 38.5 + 0.0im
end