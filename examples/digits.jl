# In this script we attempt to classify the digits of the MNIST dataset

using Neuron
using Statistics
using MLDatasets # Assumes this is installed

# Specify network and cost function
network = Network(ReLU(784,256), ReLU(256,10), Softmax())
cost = CrossEntropy()

# Load data
onehot(i,n) = (v = zeros(n); v[i] = 1.0; return v)
function get_training_data()
    d = MNIST(split=:train)[:]
    data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)]
end 
training_data = get_training_data();
function get_test_data()
    d = MNIST(split=:test)[:]
    data = [Sample(vec(d.features[:,:,i]), onehot(d.targets[i]+1, 10)) for i in 1:size(d.features,3)]
end 
test_data = get_test_data();

# Specify network training procedure
epochs = 1
batch_size = 64
optimizer = Adam(network)

# Define loss and accuracy for the 
loss(network, dataset) = mean(cost(network(s.x),s.y) for s in dataset)
accuracy(network, dataset) = mean(argmax(network(s.x)) == argmax(s.y) for s in dataset)
L_log = [loss(network, training_data)] # initial loss 
A_log = [accuracy(network, training_data)] # initial accuracy 
println("Initially, the test loss is $(L_log[1]) and the test accuracy is $(A_log[1])")

# train network
function train()
    for epoch = 1:epochs
        println("Epoch $epoch started...")
        optimize(optimizer, cost, training_data; batch_size=batch_size)      
        push!(L_log, loss(network, training_data))
        push!(A_log, accuracy(network, training_data))
        println("After epoch $epoch, the test loss is $(L_log[end]) and the test accuracy is $(A_log[end])")
    end 
end
train()

# Check a single sample
i=1 # sample index
network(training_data[i].x)
training_data[i].y

# Plot performance as function of epoch
using Plots
plot(1:length(L_log), perf_log, xscale=:log10)

# saving parameters of the model
using JLD
θ = collect.(get_θ(network))
save("theta.jld", "θ", θ)

# loading parameters of the model
θ = load("theta.jld")["θ"]
set_θ!(network, θ)