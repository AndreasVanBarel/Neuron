# This module contains a description of loss functions, including regularization terms
module Lossfunctions

# AbstractCost: Supertype of all costs 
# Cost: total cost, contains multiple Loss and RegularizationTerm
# Loss: Cost function dependent on yₘ and y only
# RegularizationTerm: Cost function dependent only on θ

export AbstractCost, Cost, Loss, RegularizationTerm
export ∇, ∇_θ
export MSE, CrossEntropy
export L2Regularization

import Nablas: ∇, ∇_θ

abstract type AbstractCost end 
# For all AbstractCost, one can evaluate the cost in 
# (yₘ, y, θ) with θ some parameters 
# and also the gradients 
# ∇(::AbstractCost, yₘ, y, θ)
# ∇_θ(::AbstractCost, yₘ, y, θ)

abstract type Loss <: AbstractCost end # Loss is a cost function dependent on yₘ and y
(loss::Loss)(yₘ, y, θ) = loss(yₘ, y)
∇(loss::Loss, yₘ, y, θ) = ∇(loss, yₘ, y)
∇_θ(::Loss, yₘ, y, θ) = zeros(size(θ))

abstract type RegularizationTerm <: AbstractCost end
(reg::RegularizationTerm)(yₘ, y, θ) = reg(θ)
∇(::RegularizationTerm, yₘ, y, θ) = zeros(size(yₘ))
∇_θ(reg::RegularizationTerm, yₘ, y, θ) = ∇_θ(reg, θ)

# sum of loss functions
struct Cost <: AbstractCost
    terms::Vector{AbstractCost}
end
Cost(args...) = Cost(collect(args))
(cost::Cost)(args...) = sum(term(args...) for term in cost.terms)
∇(cost::Cost, args...) = sum(∇(term, args...) for  term in cost.terms)
∇_θ(cost::Cost, args...) = sum(∇_θ(term, args...) for  term in cost.terms)

### SPECIFIC LOSS FUNCTIONS ###
# MSE loss
struct MSE <: Loss end
(::MSE)(yₘ, y) = sum((yₘ.-y).^2)
∇(::MSE, yₘ, y) = 2 .*(yₘ.-y)

# Cross entropy 
struct CrossEntropy <: Loss end 
(::CrossEntropy)(yₘ, y) = -sum(y.*log.(yₘ))
∇(::CrossEntropy, yₘ, y) = -y./yₘ

### SPECIFIC REGULARIZATION TERMS ###
# α/2 ‖θ‖² Tikhonov regularization
struct L2Regularization <: RegularizationTerm 
    α::Float64
    function L2Regularization(α) 
        α > 0 || error("α must be a positive real number.")
        return  new(α)
    end
end 
(l2reg::L2Regularization)(θ) = (l2reg.α/2) * sum(θ.*θ)
∇_θ(l2reg::L2Regularization, θ) = l2reg.α*θ

end