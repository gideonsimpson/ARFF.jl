
"""
    assemble_matrix!(S::TS, x_data::Vector{TX}, ω_vals::Vector{TX}) where {TS<:Matrix,TF<:AbstractFloat,TX<:AbstractArray{TF}}

Assemble the design matrix using the current ω values and x measurement positions
"""
function assemble_matrix!(S::TS, x_data::Vector{TX}, ω_vals::Vector{TX}) where {TS<:Matrix,TF<:AbstractFloat,TX<:AbstractArray{TF}}
    N = length(x_data)
    K = length(ω_vals)

    Threads.@threads for k in 1:K
        for n in 1:N
            S[n, k] = exp(im * (x_data[n] ⋅ ω_vals[k]))
        end
    end
    S
end


"""
    solve_normal_equations!(β::Vector{TY}, S::TS, y_data::Vector{TY}, λ::TL) where {TS<:Matrix,TY<:Complex, TL<:AbstractFloat}

Solve the regularized linear system using the normal equations.
"""
function solve_normal_equations!(β::Vector{TY}, S::TS, y_data::Vector{TY}, λ::TL) where {TS<:Matrix,TY<:Complex,TL<:AbstractFloat}
    N = length(y_data)
    
    β .= (S' * S + λ * N * I) \ (S' * y_data)

    β
end


"""
    solve_svd!β::Vector{TY}, S::TS, y_data::Vector{TY}, λ::TL) where {TS<:Matrix,TY<:Complex,TL<:AbstractFloat}

Solve the regularized linear system using the SVD.
"""
function solve_svd!(β::Vector{TY}, S::TS, y_data::Vector{TY}, λ::TL) where {TS<:Matrix,TY<:Complex,TL<:AbstractFloat}

    F = svd(S)
    N = length(y_data)

    β .= F.V * diagm(@. F.S / (F.S^2 + λ * N)) * F.U' * y_data;

    β
end