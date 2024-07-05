let 
   N = 10;
   K = 5;
   
   S = zeros(N,K);

    x = [Float64[i] for i in 1:N]
    ω = [Float64[k] for k in 1:K]
    ARFF.assemble_matrix!(S, ActivationFunction{Float64}(ARFF.sigmoid), x, ω);
    S[1, 1] ≈ 0.7310585786300049
end