let
    f(x) = [x[1] * x[2], x[1]^2 - x[2]^2];
    N = 10^3
    d = 2;
    
    Random.seed!(100)
    x_data = [randn(2) for _ in 1:N]
    y_data = f.(x_data);
    data = DataSet(x_data, complex.(y_data));

    K = 2^6
    Random.seed!(200)
    d = 2
    F0 = FourierModel([1.0 * randn(d) for _ in 1:K], [1.0 * randn(d) for _ in 1:K])


    d = 2
    δ = 0.5 # rwm step size
    Σ0 =Float64[1 0; 0 1];

    n_epochs = 1 * 10^2 # total number of iterations
    n_rwm_steps = 10 # number of steps between full β updates
    n_burn = n_epochs ÷ 10
    γ = optimal_γ(d)
    ω_max = Inf

    # allocate memory
    β_ = similar(F0.β[:, 1])
    
    function component_solver!(β, ω, x, y, S, epoch)
        for d_ in 1:d
            solve_normal!(β_, S, @view(y[:, d_]))
            @. β[:, d_] = β_
        end
        β
    end

    rwm_sampler = AdaptiveRWMSampler(F0, component_solver!, n_rwm_steps, n_burn, δ)

    Random.seed!(1000)
    F = deepcopy(F0)
    Σ_mean, acceptance_rate, loss = train_rwm!(F, data, rwm_sampler, n_epochs, show_progress=false)

    norm(F([1.0, 1.0]) - [1.0, 0.0]) < 1e-2
end