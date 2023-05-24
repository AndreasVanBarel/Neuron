# In this script we attempt to learn to classify digits

using Networks 
using Training
using MLDatasets
using Plots
using Lossfunctions
using Statistics
using Datasets

network = Network(ReLU(784,256), ReLU(256,10), Softmax())

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
training_data = get_training_data()

function get_test_data()
    d = MNIST(split=:test)[:]
    data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)]
    return data
end 
test_data = get_test_data()

cost = CrossEntropy()

losses = [cost(network(s.x),s.y) for s in training_data]
mean(losses)

steps = 100000
epochs = 5
perf_log = Float64[]

for epoch = 1:epochs
    adam!(network, cost, training_data, steps; learning_rate = i->1e-3)
    L = mean(cost(network(s.x),s.y) for s in training_data)
    push!(perf_log, L)
    println("After epoch $epoch, the loss is $L")
    # println("$epoch finished")
end 

plot(1:length(perf_log), perf_log, xscale=:log10)
mean(cost(network(s.x),s.y) for s in training_data)
network.(getproperty.(training_data,:x))

# Check a single sample
i=1
network(training_data[i].x)
training_data[i].y

# Check accuracy 
function accuracy(network, dataset)
    correct = sum(argmax(network(s.x)) == argmax(s.y) for s in dataset)
    return correct/length(dataset)
end
accuracy(network, training_data)