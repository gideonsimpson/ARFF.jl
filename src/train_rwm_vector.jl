"""
    train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    Σ::Matrix{TR}, options::ARFFOptions;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}

TBW
"""
function train_rwm!(F::VectorFourierModel{TR,TB,TI,TA}, data::VectorDataSet{TR,TB,TI},
    Σ::Matrix{TR}, options::ARFFOptions;
    show_progress=true, record_loss=true) where {TB<:Number,TR<:AbstractFloat,TI<:Integer,TA<:ActivationFunction{TB}}
    N = length(data)

    rwm_sampler = ARFF.AdaptiveRWMSampler(F, options.linear_solve!, options.n_ω_steps, options.n_burn, options.n_epochs, Σ, options.γ, options.δ, options.ω_max)
    mutate_rwm!(F, x, y, S, n) = ARFF.rwm!(F, rwm_sampler, x, y, S, n)

    solver = ARFFSolver(options.linear_solve!, mutate_rwm!, ARFF.trivial_resample!)

    loss = train_arff2!(F, Iterators.cycle([data]), N, solver, options, show_progress=show_progress, record_loss=record_loss)

    return rwm_sampler.Σ_mean, rwm_sampler.acceptance_rate, loss
end
