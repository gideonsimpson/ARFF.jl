let
    f(x) = exp(-0.5 * (x^2))
    n_x = 100 # number of sample points
    d = 2
    Random.seed!(100)
    # generate n_x sample points, storying them as an array of points and pad
    # for bias term
    x = [[4 * rand(), 1] for _ in 1:n_x]
    y = [f(x_[1]) for x_ in x]


    # store data in DataSet structure
    data = DataSet(x, y)
    K = 2^4

    Random.seed!(200)
    F0 = FourierModel([1.0 * randn() for _ in 1:K],
        [1.0 * randn(d) for _ in 1:K],
        SigmoidActivation)

    δ = 0.1 # rwm step size
    λ = 1e-6 # regularization
    n_epochs = 10^3 # number of epochs
    n_ω_steps = 10 # number of steps between full β updates
    n_burn = n_epochs ÷ 10 # use 10% of the run for burn in
    γ = optimal_γ(d)
    ω_max = Inf # no cut off
    adapt_covariance = true

    Σ0 = Float64[1 0; 0 1]

    β_solver! = (β, S, y, ω) -> solve_normal!(β, S, y, λ=λ)

    opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max, adapt_covariance,
        β_solver!, ARFF.mse_loss)

    Random.seed!(1000) # for reproducibility
    F = deepcopy(F0)
    Σ_mean, acceptance_rate, loss = train_rwm!(F, data, Σ0, opts, show_progress=false)
    abs(F([0.0, 1]) - f(0.0)) < 1e-2
end