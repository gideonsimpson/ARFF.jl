let 
    ϕ = ActivationFunction{Float64}(ARFF.sigmoid);
    x = [0.1, 0.2];
    ω = [1., 2.];
    ϕ(x, ω) ≈ 0.6224593312018546
end