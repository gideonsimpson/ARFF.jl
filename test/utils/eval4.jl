let 
    F = FourierModel([[1.0, 2.0]], [[1.0, 2.0]], SigmoidActivation)
    F([0.1, 0.2]) ≈ [0.6224593312018546, 1.2449186624037092]
end