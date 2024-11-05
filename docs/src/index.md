# ARFF.jl Documentation

Documentation for the adaptive random fourier features (ARFF) package.  This package is built around the methodology presented in [kammonen_adaptive_2020](@cite).

Using the package involves three steps:
* Formatting your training data into a [`DataSet`](@ref dataset) structure
* Initializing a [`FourierModel`](@ref fourier)  structure
* Training

## Overview 
The essential idea of ARFF is to make an approximation of a true function,
``f^\dagger:\mathbb{R}^d\to \mathbb{C}``, as
```math
f^\dagger(x) \approx f(x) = \sum_{k=1}^K \beta_k e^{i \omega_k \cdot x}
```
where ``x,\omega_k\in \mathbb{R}^d``, while ``\beta_k\in\mathbb{C}``.  In the
naive random Fourier featuer setting, the ``\omega_k`` are sampled from some
known distribution ``\mu``, and the ``\beta_k`` are obtained by classical least
squares regression or ridge regression,
```math
({S^\ast}S + N \lambda I)\boldsymbol{\beta} = S^\ast \boldsymbol{y},
```
where the design matrix, ``S``, is ``N\times K``, with entries
```math
S_{jk} = e^{ i \omega_k \cdot x_j},
```
We presume that we have training data of size ``N``, ``\{(x_j,y_j)\}_{j=1}^N``.
Other solutions are possible.

This package generalizes the method to allow for both vector valued functions
and permit activation functions other than the complex exponential:
```math 
f(x) = \sum_{k=1}^K \beta_k \varphi(x;\omega_k),
```
where now ``\beta_k \in \mathbb{R}^{d'}`` or ``\beta_k \in \mathbb{C}^{d'}``,
where ``d'`` need not be the same as ``d``.

### Adaptivity
To make the algorithm adaptive, that is to say, to sample the frequencies from
an _optimal_ distribution, we use a Random Walk Metropolis scheme described in
[kammonen_adaptive_2020](@cite).  The goal is to sample from the variance
minimizing distribution, known to be ``\propto |\hat{f}(\omega)|``.


The strategy is as follows:

#### Generate Proposal
Perturb the vector ``\boldsymbol{\omega}`` of wave numbers with a Gaussian,
```math
\boldsymbol{\omega}' = \boldsymbol{\omega} + \delta \boldsymbol{\xi}, \quad \boldsymbol{\xi}\sim N(0, \Sigma)
```
where ``\delta>0`` is a proposal step size and  ``\Sigma`` is a covariance
matrix. 

#### Update Amplitudes
Compute the proposed amplitudes, ``\boldsymbol{\beta}`` for the perturbed wave
numbers, by building up the new design matrix and solving the linear system. 

#### Accept/Reject
Accept/reject each wave vector ``\omega_k`` with probability
```math
\min\left\{1, \frac{|\beta'_k|^\gamma}{|\beta_k|^\gamma}\right\}
```
where ``\gamma>0`` is a tuning parameter that plays a role anlogous to inverse
temperature.

### Training
Having described a single the RWM step, the core of ARFF training requires  a
total number of epochs (`n_epochs`) and number of RWM steps (`n_rwm_steps`).  The
core of the training loop consists of:
```
for i in 1:n_epochs
    # solve for β with current ω
    for j in 1:n_ω_steps
        # generate an RWM proposal
            for k in 1:K
                # accept/reject each ω_k
            end
    end
end
```
In the current version of `ARFF.jl`, the RWM piece is encapsulated inside of a
`mutate!` operation, which can be modified as the user desires.

## Notes on Current Training Implementation
This package has undergone several revisions.  In the current form, one can specify two actions:
* `resample!` - The resampling step resamples amongs the current set of wave numbers.  Resampling was introduced to the training method for ARFF in [kammonen_adaptive_2024](@cite).
* `mutate!` - The mutation step is what discovers new weave numbers.  In the
current implementation, this corresponds to the RWM step. 
These two procedures are embedded within a [`ARFFSolver`](@ref) structure.
Furthermore, it is assumed that both of these procedures are implemented so as
to conclude by calling the linear solver to obtain the updated
``\boldsymbol{\beta}``. 

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