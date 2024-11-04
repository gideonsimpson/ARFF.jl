# function resample!(F, linear_solver!, x, y, S, R)
#     K = length(F.ω);
#     idx_original = 1:K;

#     a = sum(abs.(F.β));

#     p_hat = abs.(F.β)./a;
#     K_ess = 1/sum(p_hat.^2);

#     ω_resampled = deepcopy(F.ω)
#     if K_ess ≤ R * K
#         dist = Categorical(p_hat)

#         idx_resampled = [idx_original[rand(dist)] for _ in 1:K];
#         ω_resampled = F.ω[idx_resampled];

#         assemble_matrix!(S, F.ϕ, x, ω_resampled);
#         linear_solver!(F.β, S, y, ω_resampled);

#         F = FourierModel(F.β, ω_resampled, F.ϕ);
#     end

#     F
# end

function resample!(F, x, y, S, epoch, linear_solver!; R=1.0)
    K = length(F.ω)

    # normalization
    a = sum([norm((F.β)[r, :]) for r in 1:K])

    p_hat = [norm((F.β)[r,:]) ./ a for r in 1:K]
    K_ess = 1 / sum(p_hat .^ 2)

    if K_ess ≤ R * K
        dist = Categorical(p_hat)
        idx_resampled = rand(dist, K)

        # idx_resampled = [idx_original[rand(dist)] for _ in 1:K]
        @. F.ω = F.ω[idx_resampled]

        ARFF.assemble_matrix!(S, F.ϕ, x, F.ω)
        # solver.linear_solve!(F.β, F.ω, subsample(data.x, rows), subsample(data.y_mat, rows), S, epoch)

        linear_solver!(F.β, F.ω, x, y, S, epoch)
    end

    F
end