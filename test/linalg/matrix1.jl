let 
   N = 10;
   K = 5;
   
   S = zeros(ComplexF64, N,K);

    x = [Float64[i] for i in 1:N]
    ω = [Float64[k] for k in 1:K]
    ARFF.assemble_matrix!(S,FourierActivation, x, ω);
    S[1, 1] ≈ 0.5403023058681398 + 0.8414709848078965im;
end