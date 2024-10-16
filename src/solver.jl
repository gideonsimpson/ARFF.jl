struct ARFFSolver{TS, TE, TR, TI<:Integer, TL}
    linear_solve!::TS
    mutate!::TE
    resample!::TR
    n_epochs::TI
    loss::TL
end