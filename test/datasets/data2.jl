let
    n_x = 10
    x = [Float64[i] for i in 1:n_x]
    f(x) = [x[1], x[1]^2]
    y = f.(x)
    data = DataSet(x, y)
    data.y_mat[end, 2] ≈ 100
end