let 
    F = FourierModel([1.0], [[1.,2.]], ActivationFunction{Float64}(ARFF.sigmoid));
    F([0.1, 0.2]) ≈ 0.6224593312018546
end