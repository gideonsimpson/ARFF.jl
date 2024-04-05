# ARFF.jl Documentation

Documentation for the adaptive random fourier features (ARFF) package.  This package is built around the methodology presented in [kammonen_adaptive_2020](@cite).

Using the package involves three steps:
* Formatting your training data into a [`DataSet`](@ref) structure
* Initializing a [`FourierModel`](@ref) structure
* Training

## Overview (@id overview1)
The essential idea of ARFF is to make an approximation of a true function,
``f_{\rm true}:\mathbb{R}^d\to \mathbb{C}``, as
```math
f_{\rm true}(x) \approx f(x) = \sum_{k=1}^K \beta_k e^{i \omega_k \cdot x}
```
where ``x,\omega_k\in \mathbb{R}^d``, while ``\beta_k\in\mathbb{C}``.  In the
naive random Fourier featuer setting, the ``\omega_k`` are sampled from some
known distribution ``\mu``, and the ``\beta_k`` are obtained by classical least
squares regression or ridge regression,
```math
(S^\ast S + N \lambda I)\beta = S^\ast y,
```
where the design matrix, ``S``, is ``N\times K``, with entries
```math
S_{jk} = e^{ i \omega_k \cdot x_j},
```
We presume that we have training data of size ``N``, ``\{(x_j,y_j)\}_{j=1}^N``.
Other solutions are possible.

To make the algorithm adaptive, that is to say, to sample the frequencies from
an _optimal_ distribution, we use a Random Walk Metropolis scheme described in
[kammonen_adaptive_2020](@cite).  The strategy is as follows:
1. Perturb the vector ``\boldsymbol{\omega}`` of wave numbers with a Gaussian,
```math
\boldsymbol{\omega}' = \boldsymbol{\omega} + \delta \boldsymbol{\xi}, \quad \boldsymbol{\xi}\sim N(0, \Sigma)
```
where ``\delta>0`` is a proposal step size and  ``\Sigma`` is a covariance matrix. 
2. Compute the proposed amplitudes, ``\boldsymbol{\beta}`` for the perturbed wave numbers, by building up the new design matrix and solving the linear system. 
3. Accepting/rejecting wave vector ``\omega_k`` with probability
```math
\min\left\{1, \frac{|\beta'_k|^\gamma}{|\beta_k|^\gamma}\right\}
```
where ``\gamma>0`` is a tuning parameter that plays a role anlogous to inverse
temperature.


## Acknowledgements

Contributors to this project include:
* Gideon Simpson 
* Petr Plechac
* Jerome Troy
* Liam Doherty
* Hunter Wages

This work was supported in part by the ARO Cooperative Agreement W911NF2220234.

## References
```@bibliography
*
```