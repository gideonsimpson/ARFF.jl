let
    N = 10
    K = 5
    λ = 1e-8

    S = zeros(ComplexF64, N, K)

    x = [Float64[i] for i in 1:N]
    y = ComplexF64[cos(i) for i in 1:N]
    ω = [Float64[k] for k in 1:K]
    ARFF.assemble_matrix!(S,FourierActivation, x, ω)

    β = zeros(ComplexF64, K)
    solve_normal_svd!(β, S, y, λ=λ)
    β[1] ≈ 0.5001053314041853 + 0.023800929594537806im
end