# In this script we attempt to learn the XOR function

using Networks 

struct Sample 
    x
    y
end

Wtrue = [2.0 -1.0]
btrue = 1.0

training_data = [Sample([x1,x2], Wtrue*[x1,x2].+btrue) for x1 in -1:1 for x2 in -1:1]

network = Network(ReLU(rand(1,2), rand(1)))
network = Network(ReLU(rand(1,2), rand(1)), LU(rand(1,1), rand(1)))

network([1,1])

function performance(network::Network, data)
    J = 0
    for sample in data 
        x,y = sample.x, sample.y
        yₘ = network(x)
        J += sum((y.-yₘ).^2)
    end
    J/=length(data)
end

performance(network, training_data)

# Do n training iterations
function train(network::Network, training_data, iterations::Int; perf_log=nothing)

    α_init = 0.1 # initial learning rate 
    α_min = 1e-3

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

        if !isnothing(perf_log); perf_log[i] = performance(network, training_data); end
    end
end

steps = 10000
perf_log = Vector{Float64}(undef,steps)
train(network, training_data, steps; perf_log=perf_log)

plot(1:steps, perf_log, yscale=:log10)
performance(network, training_data)
network.(getproperty.(training_data,:x))

######

# Investigate gradient for specific samples
sample = training_data[1]
x,y = sample.x, sample.y

ne = network(x, save)
ng = gradient(ne, y)
grad = ∇_θ(ng)

θ = get_θ(network)
