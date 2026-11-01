let 
    using Random
    using LinearAlgebra
    using Statistics
    using SpecialFunctions


    # ============================================================
    # Test problem
    # ============================================================

    a = 1e-3
    f(x) = sinint(x / a) * exp(-0.5 * x^2)

    n_x = 100
    d = 1
    K = 64

    λ = 1e-6
    δ = 0.05

    n_epochs = 50
    n_rwm_steps = 1
    n_burn = n_epochs ÷ 10


    # ============================================================
    # Fixed training data
    # ============================================================

    Random.seed!(100)

    x = [0.1 * rand(d) for _ in 1:n_x]
    y = [f(x_[1]) for x_ in x]

    data = DataSet(x, complex.(y))


    # ============================================================
    # Common initial Fourier model
    # ============================================================

    Random.seed!(200)

    F0 = FourierModel(
        [randn(ComplexF64) for _ in 1:K],
        [randn(d) for _ in 1:K],
    )

    F_continuous = deepcopy(F0)
    F_split = deepcopy(F0)


    # ============================================================
    # Linear solver
    # ============================================================

    linear_solver! = (β, ω, x, y, S, epoch) ->
        solve_normal!(β, S, y; λ=λ)


    # ============================================================
    # Independent, identically initialized samplers
    # ============================================================

    sampler_continuous = AdaptiveRWMSampler(
        F_continuous,
        linear_solver!,
        n_rwm_steps,
        n_burn,
        δ,
    )

    sampler_split = AdaptiveRWMSampler(
        F_split,
        linear_solver!,
        n_rwm_steps,
        n_burn,
        δ,
    )


    # ============================================================
    # A. Uninterrupted 2E-epoch training
    # ============================================================

    Random.seed!(300)

    acc_continuous, loss_continuous = train_rwm!(
        F_continuous,
        data,
        sampler_continuous,
        2 * n_epochs;
        show_progress=false,
    )


    # ============================================================
    # B. E epochs followed by E epochs of true continuation
    #
    # Do not reseed between these calls.
    # ============================================================

    Random.seed!(300)

    acc_split_1, loss_split_1 = train_rwm!(
        F_split,
        data,
        sampler_split,
        n_epochs;
        show_progress=false,
    )

    acc_split_2, loss_split_2 = train_rwm!(
        F_split,
        data,
        sampler_split,
        n_epochs;
        show_progress=false,
        epoch_offset=n_epochs,
    )


    # ============================================================
    # Checks
    # ============================================================

    model_match =
        F_continuous.ω == F_split.ω &&
        F_continuous.β == F_split.β

    sampler_match =
        sampler_continuous.ω_mean == sampler_split.ω_mean &&
        sampler_continuous.Σ_mean == sampler_split.Σ_mean &&
        sampler_continuous.mv_normal.Σ == sampler_split.mv_normal.Σ

    acceptance_match =
        sampler_continuous.acceptance_rate ==
            sampler_split.acceptance_rate &&
        acc_continuous == acc_split_2

    loss_match =
        loss_continuous ==
            vcat(loss_split_1, loss_split_2)

    length_match =
        length(loss_continuous) == 2 * n_epochs &&
        length(sampler_continuous.acceptance_rate) == 2 * n_epochs

    covariance_match =
        sampler_split.mv_normal.Σ ==
            sampler_split.Σ_mean


    # ============================================================
    # Diagnostics
    # ============================================================

    # println("epoch-offset continuation test")
    # println("  model match       = ", model_match)
    # println("  sampler match     = ", sampler_match)
    # println("  acceptance match  = ", acceptance_match)
    # println("  loss match        = ", loss_match)
    # println("  length match      = ", length_match)
    # println("  covariance match  = ", covariance_match)


    # ============================================================
    # Return Boolean for
    #
    #     @test include("train/train7.jl")
    # ============================================================

    model_match &&
    sampler_match &&
    acceptance_match &&
    loss_match &&
    length_match &&
    covariance_match
        
end