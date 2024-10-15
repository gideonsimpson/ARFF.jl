function likelihood(β_new::TY, β_old::TY, γ::TG) where {TY<:Number,TG<:Integer}

    return (abs(β_new) / abs(β_old))^γ

end

function likelihood(β_new::AbstractVector{TY}, β_old::AbstractVector{TY}, γ::TG) where {TY<:Number,TG<:Integer}

    return (norm(β_new) / norm(β_old))^γ

end