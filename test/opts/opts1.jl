let
    d = 3;
    δ = 1.0 # rwm step size
    λ = 1e-8 # regularization
    n_epochs = 10^3 # total number of iterations
    n_ω_steps = 10 # number of steps between full β updates
    n_burn = n_epochs ÷ 10
    γ = optimal_γ(d)
    ω_max = Inf
    adapt_covariance = true
    β_solver! = (β, S, y, ω) -> solve_normal!(β, S, y, λ=λ);

    opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max, adapt_covariance,
        β_solver!, ARFF.mse_loss);
    opts.δ ≈ δ
end