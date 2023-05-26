# In this script we attempt to learn to classify digits using the MNIST dataset

using Statistics
using Networks 
using MLDatasets
using Lossfunctions
using Datasets
using Training
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

losses = [cost(network(s.x),s.y) for s in training_data]
mean(losses)

steps = 60_000
epochs = 1
batch_size = 64
perf_log = Float64[]

function train(network)
    for epoch = 1:epochs
        println("$epoch started")
        adam!(network, cost, training_data[1:steps]; batch_size = batch_size, learning_rate = i->1e-3)
        # L = mean(cost(network(s.x),s.y) for s in training_data)
        # push!(perf_log, L)
        # println("After epoch $epoch, the loss is $L")
    end 
end
VSCodeServer.@profview train(network)
@time train(network)
train(network)


optimizer = Adam(network)
try
    optimize(optimizer, cost, training_data[1:steps]; batch_size=batch_size)
catch e 
    global exc = e 
end

@time optimize(optimizer, cost, training_data[1:steps]; batch_size=batch_size)
VSCodeServer.@profview optimize(optimizer, cost, training_data[1:steps]; batch_size=batch_size)
















plot(1:length(perf_log), perf_log, xscale=:log10)
mean(cost(network(s.x),s.y) for s in training_data)
network.(getproperty.(training_data,:x))

# Check a single sample
i=3 # sample index
network(training_data[i].x)
training_data[i].y

# Check accuracy 
function accuracy(network, dataset)
    correct = sum(argmax(network(s.x)) == argmax(s.y) for s in dataset)
    return correct/length(dataset)
end
accuracy(network, training_data)
accuracy(network, test_data)

# saving parameters of the model
using JLD
θ = collect.(get_θ(network))
save("theta3.jld", "θ", θ)

# loading parameters of the model
θ = load("theta3.jld")["θ"]
set_θ!(network, θ)