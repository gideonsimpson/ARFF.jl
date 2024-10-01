let
    n_x = 10
    x = [Float64[i] for i in 1:n_x]
    f(x) = x[1]^2
    y = f.(x)
    data_ = DataSet(x, y)
    data = append_bias(data_);
    data.x[5][2] ≈ 1.0;
end