
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
    assemble_matrix!(S::TS, features::Function x_data::Vector{TX}, ω_vals::Vector{TX}) where {TS<:Matrix,TF<:AbstractFloat,TX<:AbstractArray{TF}}

Assemble the design matrix using the current ω values and x measurement positions with defined features function.
"""
function assemble_matrix!(S::TS, features::Function, x_data::Vector{TX}, ω_vals::Vector{TX}) where {TS<:Matrix,TF<:AbstractFloat,TX<:AbstractArray{TF}}
    N = length(x_data)
    K = length(ω_vals)

    Threads.@threads for k in 1:K
        for n in 1:N
            S[n, k] = features(ω_vals[k],x_data[n])
        end
    end
    S
end


"""
    solve_normal!(β, S, y_data; λ = 1e-8)

Solve the regularized linear system using the normal equations.
### Fields
* `β` - The vector of coefficients that will be obtained
* `S` - The design matrix
* `y_data` - y coordinates
* `λ = 1e-8` - Regularization parameter
"""
function solve_normal!(β::Vector{TY}, S::TS, y_data::Vector{TY}; λ=1e-8) where {TS<:Matrix,TY<:Complex}
    N = length(y_data)
    
    β .= (S' * S + λ * N * I) \ (S' * y_data)

    β
end


"""
    solve_normal_svd!(β, S, y_data; λ = 1e-8)

Solve the regularized linear system using the SVD.
### Fields
* `β` - The vector of coefficients that will be obtained
* `S` - The design matrix
* `y_data` - y coordinates
* `λ = 1e-8` - Regularization parameter
"""
function solve_normal_svd!(β::Vector{TY}, S::TS, y_data::Vector{TY}; λ=1e-8) where {TS<:Matrix,TY<:Complex}

    F = svd(S)
    N = length(y_data)

    β .= F.V * diagm(@. F.S / (F.S^2 + λ * N)) * F.U' * y_data;

    β
end