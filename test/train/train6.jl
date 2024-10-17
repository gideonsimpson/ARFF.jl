let
    f(x) = exp(-0.5 * (x^2))
    n_x = 100 # number of sample points
    d = 2
    Random.seed!(100)
    # generate n_x sample points, storying them as an array of points and pad
    # for bias term
    x = [[4 * rand()] for _ in 1:n_x]
    y = [f(x_[1]) for x_ in x]


    # store data in DataSet structure
    data_ = DataSet(x, y)
    # add bias term
    data = append_bias(data_);
    
    K = 2^4

    Random.seed!(200)
    F0 = FourierModel([1.0 * randn() for _ in 1:K],
        [1.0 * randn(d) for _ in 1:K],
        SigmoidActivation)

    δ = 0.1 # rwm step size
    λ = 1e-6 # regularization
    n_epochs = 10^3 # number of epochs
    n_rwm_steps = 10 # number of steps between full β updates
    n_burn = n_epochs ÷ 10 # use 10% of the run for burn in
    γ = optimal_γ(d)
    ω_max = Inf # no cut off
    adapt_covariance = true

    Σ0 = Float64[1 0; 0 1]

    linear_solver! = (β, S, y, ω) -> solve_normal!(β, S, y, λ=λ)

    rwm_sampler = AdaptiveRWMSampler(F0, linear_solver!, n_rwm_steps, n_burn, Σ0, γ, δ, ω_max)

    Random.seed!(1000) # for reproducibility
    F = deepcopy(F0)
    Σ_mean, acceptance_rate, loss = train_rwm!(F, data, rwm_sampler, n_epochs, show_progress=false)
    abs(F([0.0, 1]) - f(0.0)) < 1e-2
end