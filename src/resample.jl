"""
    resample!(F, x, y, S, epoch, linear_solver!; R=1.0)

Function that performs the resampling step of the training algorithm.
### Fields
* `F` - A `FourierModel` structure
* `x` - the x coordinates in training data
* `y` - the y coordinates in training data
* `S` - the design matrix
* `epoch` - the current epoch of the training process
* `linear_solver!` - an in place solver for the linear system
* `R = 1.0` - Effective sample size threshold
"""
function resample!(F, x, y, S, epoch, linear_solver!; R=1.0)
    K = length(F)

    # normalization
    a = sum([norm((F.β)[r, :]) for r in 1:K])

    p_hat = [norm((F.β)[r,:]) ./ a for r in 1:K]
    K_ess = 1 / sum(p_hat .^ 2)

    if K_ess ≤ R * K
        dist = Categorical(p_hat)
        idx_resampled = rand(dist, K)

        @. F.ω = F.ω[idx_resampled]

        assemble_matrix!(S, F.ϕ, x, F.ω)

        linear_solver!(F.β, F.ω, x, y, S, epoch)
    end

    F
end