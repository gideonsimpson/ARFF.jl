function resample!(F, solver, x, y, S, epoch, R)
    K = length(F.ω);
    idx_original = 1:K;

    a = sum(abs.(F.β));

    p_hat = abs.(F.β)./a;
    K_ess = 1/sum(p_hat.^2);

    ω_resampled = deepcopy(F.ω)
    if K_ess ≤ R * K
        dist = Categorical(p_hat)

        idx_resampled = [idx_original[rand(dist)] for _ in 1:K];
        ω_resampled = F.ω[idx_resampled];

        assemble_matrix!(S, F.ϕ, x, ω_resampled);
        solver.linear_solve!(F.β, S, y, ω_resampled);

        F = FourierModel(F.β, ω_resampled, F.ϕ);
    end

    F
end