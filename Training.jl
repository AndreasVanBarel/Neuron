module Training 

export loss
export train! 
export Sample 

using Networks 
using Datasets
using Lossfunctions

function train!(network::Network, cost, training_data::Vector{Sample}, iterations::Int; method=:SGD, perf_log=nothing, test_data=training_data)
    if method==:SGD 
        SGD!(network, cost, training_data, iterations; perf_log=perf_log, test_data=test_data)
    end
end

# Do n training iterations
function SGD!(network::Network, cost, training_data, iterations::Int; perf_log=nothing, test_data=training_data)

    α_init = 1e-3 # initial learning rate 
    α_min = 1e-3

    α(i::Int) = max(α_init / i, α_min) # learning rate during iteration i

    m = length(training_data)

    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[rand(1:m)]
        x,y = sample.x, sample.y

        ne = network(x, save)
        yₘ = ne.outputs[end] # output of final layer

        # Evaluate the gradient of the cost function
        J = cost(yₘ, y)
        dJdyₘ = ∇(cost, yₘ, y)'

        ng = gradient(ne, dJdyₘ)
        grad = ∇_θ(ng)

        θ = get_θ(network)
        Δθ = -grad.*α(i)
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, test_data); end
    end
end


end