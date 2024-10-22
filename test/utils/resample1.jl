let 
    a = 1e-3
    f(x) = sinint(x / a) * exp(-0.5 * (x^2))

    n_x = 500 # number of training points
    d = 1
    Random.seed!(100) # for reproducibility
    x = [0.1 * rand(d) for _ in 1:n_x]
    y = [f(x_[1]) for x_ in x]

    # store data in DataSet structure
    data = DataSet(x, complex.(y))
    K = 2^7
    Random.seed!(200) # for reproducibility
    F = FourierModel([1.0 * randn() for _ in 1:K], [randn(d) for _ in 1:K])
    δ = 10.0 # rwm step size
    λ = 1e-8 # regularization

    linear_solver! = (β, S, y, ω) -> solve_normal!(β, S, y, λ=λ)

    R = 1.0

    S = zeros(typeof(F.β[1, 1]), n_x, K)
    ARFF.assemble_matrix!(S, F.ϕ, data.x, F.ω)

    resample!(F, linear_solver!, data.x, data.y, S, R)

    F([0.0]) ≈ 1.4839845439477237 + 1.7936518653449696e-5im
    
end