# Resampling
It has been shown in [kammonen_adaptive_2024](@cite) that the performance of the training algorithm can be significantly improved if a resampling step is done at each epoch. This resampling is done by generating a Probability Mass Function $\hat{p} = (\hat{p}_1, ... , \hat{p}_K)$ that is calculuated by considering the magnetude of the coefficients $\beta_k$. More specifically,

$$ \hat{p}_k = \frac{||\beta_k||}/\sum{||\beta_k||} $$

We then resample $K$ independent index values $\{i_1, ..., i_K\}$ from $\{1, 2, ..., K\}$ using the PMF $\hat{p}$. Using these resampled indicies, we construct a resampled wavenumber vector $\boldsymbol{\omega} = \{\omega_{i_1}, ..., \omega_{i_K}\}$ and resolve the linear system in order to obtain a resampled coefficient vector $\hat{\boldsymbol{\beta}} = (\hat{\beta}_1, ..., \hat{\beta}_K)$.  
The resampler can be called in the following way:

```@docs
resample!
```

Note that the resampling rule $R \in [0,1]$ will determine how often the resampling is done. When $R = 1$, the resampling will be done on each epoch of the training process and when $R = 0$, the resampling step will be bipassed on every epoch. This parameter can be tuned for individual problems.