"""
    assemble_matrix!(S::Matrix{TB}, ϕ::ActivationFunction{TB}, 
    x_data::Vector{Vector{TF}}, ω_vals::Vector{Vector{TF}}) where {TB<:Number,TF<:AbstractFloat}

Assemble the design matrix using the current ω values and x measurement positions with defined features function.
"""
function assemble_matrix!(S::Matrix{TB}, ϕ::ActivationFunction{TB}, 
    x_data::AbstractVector{TX}, ω_vals::Vector{Vector{TF}}) where {TB<:Number,TF<:AbstractFloat,TX<:AbstractVector{TF}}
    N = length(x_data)
    K = length(ω_vals)

    Threads.@threads for k in 1:K
        for n in 1:N
            S[n, k] = ϕ(ω_vals[k], x_data[n])
        end
    end
    S
end

# function assemble_matrix!(S, ϕ, x_data, ω_vals)
#     N = length(x_data)
#     K = length(ω_vals)

#     Threads.@threads for k in 1:K
#         for n in 1:N
#             S[n, k] = ϕ(ω_vals[k], x_data[n])
#         end
#     end
#     S
# end


"""
    solve_normal!(β, S, y_data; λ = 1e-8)

Solve the regularized linear system using the normal equations.
### Fields
* `β` - The vector of coefficients that will be obtained
* `S` - The design matrix
* `y_data` - y coordinates
* `λ = 1e-8` - Regularization parameter
"""
function solve_normal!(β::AbstractVecOrMat{TY}, S::Matrix{TY}, y_data::AbstractVecOrMat{TY}; λ=1e-8) where {TY<:Number}
    N = length(y_data)
    
    β .= (S' * S + λ * N * I) \ (S' * y_data)

    β
end

# function solve_normal!(β::Matrix{TY}, S::Matrix{TY}, y_data::Matrix{TY}; λ=1e-8) where {TY<:Number}
#     N = length(y_data)

#     β .= (S' * S + λ * N * I) \ (S' * y_data)

#     β
# end


"""
    solve_normal_svd!(β, S, y_data; λ = 1e-8)

Solve the regularized linear system using the SVD.
### Fields
* `β` - The vector of coefficients that will be obtained
* `S` - The design matrix
* `y_data` - y coordinates
* `λ = 1e-8` - Regularization parameter
"""
function solve_normal_svd!(β::Vector{TY}, S::Matrix{TY}, y_data::Vector{TY}; λ=1e-8) where {TY<:Number}

    F = svd(S)
    N = length(y_data)

    β .= F.V * diagm(@. F.S / (F.S^2 + λ * N)) * F.U' * y_data;

    β
end