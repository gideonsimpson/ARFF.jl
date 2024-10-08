let
    f(x) = [x[1] * x[2], x[1]^2 - x[2]^2]
    N = 10^3
    d = 2

    Random.seed!(100)
    x_data = [randn(2) for _ in 1:N]
    y_data = f.(x_data)
    data = DataSet(x_data, complex.(y_data))

    K = 2^6
    Random.seed!(200)
    d = 2
    F = FourierModel([1.0 * randn(d) for _ in 1:K], [1.0 * randn(d) for _ in 1:K])


    d = 2
    δ = 0.5 # rwm step size
    Σ0 = Float64[1 0; 0 1]

    n_epochs = 1 * 10^2 # total number of iterations
    n_ω_steps = 10 # number of steps between full β updates
    n_burn = n_epochs ÷ 10
    batch_size = 100
    γ = optimal_γ(d)
    ω_max = Inf
    adapt_covariance = true

    β_solver! = (β, S, y, ω) -> solve_normal!(β, S, y)

    opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max,
        adapt_covariance, β_solver!, ARFF.mse_loss)

    Random.seed!(1000)
    Σ_mean, acceptance_rate, loss = train_rwm!(F, data, batch_size, Σ0, opts, show_progress=false)

    norm(F([1.0, 1.0]) - [1.0, 0.0]) < 1e-2
end