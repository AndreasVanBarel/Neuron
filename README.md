# Neuron.jl
A toy CPU based neural network library written entirely from scratch in Julia. It uses only `LinearAlgebra` and `Statistics`, which are part of Julia's standard library.

For serious deep learning and artificial intelligence applications in Julia, see [Flux.jl](https://github.com/FluxML/Flux.jl) instead.

## Features

- Networks with general connectivity between the layers: layers can have multiple inputs
- Layers: LU (Linear unit), ReLU (Rectified linear unit), Softmax, ConstUnit, ...
- Backpropagation explicitly defined, no automatic differentiation
- Costs: MSE (Mean square error), CrossEntropy, L2Regularization
- Costs can be composed of an arbitrary number of terms
- Training schemes: SGD, AdaGrad, RMSProp, Adam
- Easy extensibility of layer types, costs and training schemes
- Parallelized processing of each batch (automatically done when starting the julia process with multiple threads)

## Installation

This is installed using the standard tools of the [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/):

```julia
pkg> add https://github.com/AndreasVanBarel/Neuron.jl.git
```
You get the `pkg>` prompt by hitting `]` as the first character of the line.


## Example 
In the following example we attempt to classify digits using the MNIST dataset.
The example can also be found in `examples\digits.jl`. It assumes the package `MLDatasets` containing the MNIST dataset is available.

```julia
using Neuron
using Statistics
using MLDatasets # Assumes this is installed

# Specify network and cost function
network = Network(ReLU(784,256), ReLU(256,10), Softmax())
cost = CrossEntropy()

# Load data
onehot(i,n) = (v = zeros(n); v[i] = 1.0; return v)

d = MNIST(split=:train)[:];
training_data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)];

d = MNIST(split=:test)[:];
test_data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)];

# Specify network training procedure
epochs = 3
batch_size = 64
optimizer = Adam(network)

# Define loss and accuracy
loss(network, dataset) = mean(cost(network(s.x),s.y) for s in dataset)
accuracy(network, dataset) = mean(argmax(network(s.x)) == argmax(s.y) for s in dataset)

# train network
println("Initially, the test loss is $(loss(network, test_data)) and the test accuracy is $(accuracy(network, test_data))")
for epoch = 1:epochs
    println("Epoch $epoch started...")
    optimize(optimizer, cost, training_data; batch_size=batch_size)      
    L = loss(network, test_data)
    A = accuracy(network, test_data)
    println("After epoch $epoch, the test loss is $L and the test accuracy is $A")
end 
```

After one epoch of training, the accuracy will likely be above 90. Each epoch should take a few minutes at most on a standard CPU.
