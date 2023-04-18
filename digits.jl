# In this script we attempt to learn to classify digits

using Networks 
using Training
using MLDatasets
using Plots

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

Training.loss(network, training_data)

steps = 1000
epochs = 70
perf_log = Float64[]

for epoch = 1:epochs
    train!(network, training_data, steps)
    L = loss(network, test_data)
    push!(perf_log, L)
    println("After epoch $epoch, the loss is $L")
end

plot(1:length(perf_log), perf_log, xscale=:log10)
loss(network, training_data)
network.(getproperty.(training_data,:x))