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
    F0 = FourierModel([1.0 * randn() for _ in 1:K], [randn(d) for _ in 1:K])
    δ = 10.0 # rwm step size
    λ = 1e-8 # regularization
    n_epochs = 10^3 # number of epochs
    batch_size = 50
    n_ω_steps = 10 # number of steps between full β updates
    n_burn = n_epochs ÷ 10 # use 10% of the run for burn in
    γ = optimal_γ(d)
    ω_max = Inf # no cut off
    adapt_covariance = true

    Σ0 = ones(1, 1)

    β_solver! = (β, S, y, ω) -> solve_normal!(β, S, y, λ=λ)

    opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max, adapt_covariance,
        β_solver!, ARFF.mse_loss)

    Random.seed!(1000) # for reproducibility
    F = deepcopy(F0)
    Σ_mean, acceptance_rate, loss = train_rwm!(F, data, batch_size, Σ0, opts, show_progress=false)
    abs(F([0.02]) - f(0.02)) < 1e-3
end