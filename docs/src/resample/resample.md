# Resampling
It has been shown in [kammonen_adaptive_2024](@cite) that the performance of the training algorithm can be significantly improved if a resampling step is done at each epoch. This resampling is done by generating a Probability Mass Function $\hat{p} = (\hat{p}_1, ... , \hat{p}_K)$ with values ``\propto |\beta_k|``:
```math
\hat{p}_k = \frac{\|\beta_k\|}{\sum_j\|\beta_j\|}.
```
This is then used to estimate the Effective Sample Size (ESS):
```math
\mathrm{ESS}= \left\{\sum_{k} \hat{p}_k^2\right\}^{-1}
```
We then compare the ESS to ``R\cdot K``, where ``R\in [0,1]``, is a tolerance parameter.  When $R = 1$, the resampling will be done on each epoch of the training process and when $R = 0$, there will be no resampling; this is equivalent to calling [`trivial_resample!`](@ref). ``R`` can be tuned for individual problems.

When the ESS falls below ``R\cdot K``, we proceed as follows.  We resample $K$ independent index values $\{i_1, ..., i_K\}$ from $\{1, 2, ..., K\}$ using the PMF $\hat{p}$. Using These resampled indicies then determine the new wavenumbers $\boldsymbol{\omega} = \{\omega_{i_1}, ..., \omega_{i_K}\}$, and we solve the linear system (again) to obtain the coefficient vector $\hat{\boldsymbol{\beta}} = (\hat{\beta}_1, ..., \hat{\beta}_K)$.    


The resampler can be called in the following way:

```@docs
resample!
```
The linear solver you provide msut be compatible with ARFF.jl, and must accept the following arugments `linear_solver!(β, ω, x, y, S, epoch)`.

