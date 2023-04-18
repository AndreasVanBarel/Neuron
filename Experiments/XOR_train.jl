# In this script we attempt to learn the XOR function

using Networks 

struct Sample 
    x
    y
end

training_data = [Sample([0,0],[1,0]), Sample([0,1],[0,1]), Sample([1,0],[0,1]), Sample([1,1],[1,0])]
training_data = [Sample([0,0],0), Sample([0,1],1), Sample([1,0],1), Sample([1,1],0)]

network = Network(ReLU(rand(3,2), rand(3)), ReLU(rand(2,3), rand(2)), Softmax())
network = Network(ReLU(rand(2,2), rand(2)))
network = Network(ReLU(rand(2,2), rand(2)), LU(rand(1,2), rand(1)))

network = Network(ReLU([1 1; 1 1], [0,-1]), LU([1 -2], [0]))
θ = get_θ(network)
Δθ = [randn(size(p))*0.1 for p in θ]
set_θ!(network, θ.+Δθ)

network([1,1])

# Single training iteration

# # take random element from training_data 
# α = 1 # initial learning rate 
# m = length(training_data)

# sample = training_data[rand(1:m)]
# x,y = sample.x, sample.y

# ne = network(x, save)
# ng = gradient(ne, y)
# grad = ∇_θ(ng)

# θ = get_θ(network)
# Δθ = -grad.*α

# set_θ!(network, θ.+Δθ)  

function loss(network::Network, data)
    J = 0
    for sample in data 
        x,y = sample.x, sample.y
        yₘ = network(x)
        J += sum((y.-yₘ).^2)
    end
    J/=length(data)
end

loss(network, training_data)

# Do n training iterations
function train(network::Network, training_data, iterations::Int; perf_log=nothing)

    α_init = 0.1 # initial learning rate 
    α_min = 1e-4

    α(i::Int) = max(α_init / i, α_min) # learning rate during iteration i

    m = length(training_data)

    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[rand(1:m)]
        x,y = sample.x, sample.y

        ne = network(x, save)
        ng = gradient(ne, y)
        grad = ∇_θ(ng)

        θ = get_θ(network)
        Δθ = -grad.*α(i)
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = loss(network, training_data); end
    end
end

steps = 10000
perf_log = Vector{Float64}(undef,steps)
train(network, training_data, steps; perf_log=perf_log)

plot(1:steps, perf_log, xscale=:log10)
loss(network, training_data)
network.(getproperty.(training_data,:x))