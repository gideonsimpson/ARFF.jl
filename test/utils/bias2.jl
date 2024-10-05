let
    n_x = 10
    x = [Float64[i] for i in 1:n_x]
    f(x) = x[1]^2
    y = f.(x)
    data_ = DataSet(x, y);
    scalings_ = get_scalings(data_)
    scale_data!(data_, scalings_)
    data = append_bias(data_);
    scalings = append_bias(scalings_)
    norm(scalings.σ2x) ≈ 9.221050795748702
end