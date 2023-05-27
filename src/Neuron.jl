module Neuron

# Datasets
include("Nablas.jl")
using .Nablas

# Networks
include("Networks.jl")
using .Networks

export Layer, get_θ, set_θ!, size_θ
export LU, ReLU, Softmax, ConstUnit
export Model, Network
export NetworkWithData, allocate
export gradient

# Lossfunctions
include("Lossfunctions.jl")
using .Lossfunctions

export AbstractCost, Cost, Loss, RegularizationTerm
export ∇, ∇_θ
export MSE, CrossEntropy
export L2Regularization

# Datasets
include("Datasets.jl")
using .Datasets

export Sample 
export Dataset

# Training
include("Training.jl")
using .Training

export Optimizer, optimize
export SGD, AdaGrad, RMSProp, Adam

end
