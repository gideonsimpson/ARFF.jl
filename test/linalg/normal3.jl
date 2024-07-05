let
    N = 10;
    K = 5;
    λ = 1e-8;

    S = zeros(N, K)

    x = [Float64[i] for i in 1:N]
    y = [cos(i) for i in 1:N]
    ω = [Float64[k] for k in 1:K]
    ARFF.assemble_matrix!(S, SigmoidActivation, x, ω);

    β = zeros(K)
    solve_normal!(β, S, y, λ = λ)
    β[1] ≈ 28.556041327809307
end