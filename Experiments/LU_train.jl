# In this script we attempt to learn the XOR function

using Networks 

struct Sample 
    x
    y
end

Wtrue = [2.0 -1.0]
btrue = 1.0

training_data = [Sample([x1,x2], Wtrue*[x1,x2].+btrue) for x1 in -1:1 for x2 in -1:1]

network = Network(LU(rand(1,2), rand(1)))

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
    m = length(training_data)

    for i = 1:iterations
        α = α_init / i # learning rate during iteration i
        # α = α_init 
        # take random element from training_data 
        sample = training_data[rand(1:m)]
        x,y = sample.x, sample.y

        ne = network(x, save)
        ng = gradient(ne, y)
        grad = ∇_θ(ng)

        θ = get_θ(network)
        Δθ = -grad.*α 
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = performance(network, training_data); end
    end
end

steps = 1000
perf_log = Vector{Float64}(undef,steps)
train(network, training_data, steps; perf_log=perf_log)

plot(1:steps, perf_log, xscale=:log10, yscale=:log10)
performance(network, training_data)
network.(getproperty.(training_data,:x))

