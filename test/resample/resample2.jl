let 
    f(x) =x

    n_x = 10 # number of training points
    d = 1
    Random.seed!(100) # for reproducibility
    x = [0.1 * rand(d) for _ in 1:n_x]
    y = [f(x_[1]) for x_ in x]

    # store data in DataSet structure
    data = DataSet(x, complex.(y))
    K = 4

    # should preferentially sample from the fourth mode
    F = FourierModel([0.01, 0.01, 0.01, 1.], [[1.], [2.], [3.], [4.] ])

    linear_solver! = (β, ω, x, y, S, epoch) -> solve_normal!(β, S, y)

    R = 1.0

    S = zeros(typeof(F.β[1, 1]), n_x, K)

    resample!(F, data.x, data.y, S, 0, linear_solver!, R=R)

    mean(F.ω)[1] ≈ 4.0
    
end