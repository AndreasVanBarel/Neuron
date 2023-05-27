# In this script we attempt to learn to classify digits using the MNIST dataset

using Neuron
using Statistics

# Assumes these packages are installed
using MLDatasets
using ProgressMeter
using Plots

function onehot(i,n)
    v = zeros(n)
    v[i] = 1.0
    return v
end

function get_training_data()
    d = MNIST(split=:train)[:]
    data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)]
    return data
end 
training_data = get_training_data();

function get_test_data()
    d = MNIST(split=:test)[:]
    data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)]
    return data
end 
test_data = get_test_data();

cost = CrossEntropy()
network = Network(ReLU(784,256), ReLU(256,10), Softmax())

perf_log = [mean(cost(network(s.x),s.y) for s in training_data)] # initial cost

epochs = 2
batch_size = 64
optimizer = Adam(network)

function train()
    for epoch = 1:epochs
        println("Epoch $epoch started...")
        optimize(optimizer, cost, training_data; batch_size=batch_size)      
        L = mean(cost(optimizer.network(s.x),s.y) for s in training_data)
        push!(perf_log, L)
        println("After epoch $epoch, the loss is $L")
    end 
end
train()

# Check accuracy 
function accuracy(network, dataset)
    correct = sum(argmax(network(s.x)) == argmax(s.y) for s in dataset)
    return correct/length(dataset)
end
accuracy(network, training_data)
accuracy(network, test_data)

# Check a single sample
i=3 # sample index
network(training_data[i].x)
training_data[i].y

# Plot performance as function of epoch
plot(1:length(perf_log), perf_log, xscale=:log10)

# saving parameters of the model
using JLD
θ = collect.(get_θ(network))
save("theta3.jld", "θ", θ)

# loading parameters of the model
θ = load("theta3.jld")["θ"]
set_θ!(network, θ)