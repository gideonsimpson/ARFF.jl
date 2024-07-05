let 
    F = FourierModel(ComplexF64[1.0], [[1.,2.]], ActivationFunction{ComplexF64}(ARFF.fourier));
    F([0.1, 0.2]) ≈ 0.8775825618903728 + 0.479425538604203im;
end