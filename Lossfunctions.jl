# This module contains a description of loss functions, including regularization terms
module Lossfunctions

using Datasets 

export AbstractCost, Loss, Cost, RegularizationTerm
export MSE
export L2Regularization
export ∇, ∇_θ

# pass both loss and regularization term somehow to the program

abstract type AbstractCost end 
# For all AbstractCost, one must be able to evaluate the cost in 
# (yₘ, y, θ) with θ some parameters 
# and also the gradients 
# ∇(::AbstractCost, yₘ, y[, θ]) # Optional θ
# ∇_θ(::AbstractCost, yₘ, y, θ)

abstract type Loss <: AbstractCost end # Loss is a cost function dependent on yₘ and y
(loss::Loss)(network, x, y) = loss(network(x),y)
(loss::Loss)(network, s::Sample) = loss(network(s.x),s.y)
(loss::Loss)(network, d::Dataset) = sum(loss(network(s.x),s.y) for s in d.samples)
∇_θ(::Loss, yₘ, y, θ) = zeros(size(θ))

# singleton MSE struct
struct MSE <: Loss end
(::MSE)(yₘ, y, θ=nothing) = (yₘ.-y).^2
∇(::MSE, yₘ, y, θ=nothing) = 2 .*(yₘ.-y)

abstract type RegularizationTerm <: AbstractCost end

# Implements α/2 ‖θ‖²
struct L2Regularization <: RegularizationTerm 
    α::Float64
    function L2Regularization(α) 
        α > 0 || error("α must be a positive real number.")
        return  new(α)
    end
end 
(l2reg::L2Regularization)(yₘ, y, θ) = (l2reg.α/2) * sum(θ.*θ)
∇(l2reg::L2Regularization, yₘ, y, θ) = zeros(size(yₘ))
∇_θ(l2reg::L2Regularization, yₘ, y, θ) = l2reg.α*θ

# sum of loss functions
struct Cost <: AbstractCost
    y_term :: Loss
    θ_term :: RegularizationTerm
end
(cost::Cost)(args...) = cost.y_term(args...) + cost.θ_term(args...)
∇(cost::Cost, yₘ, y, θ) = ∇(cost.y_term, yₘ, y)
∇_θ(cost::Cost, yₘ, y, θ) = ∇_θ(cost.θ_term, yₘ, y, θ)


end