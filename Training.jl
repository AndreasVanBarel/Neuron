module Training 

export loss
export train! 
export Sample 

using Networks 


# A single input-output pair (x,y)
struct Sample 
    x
    y
end

# struct DataSet 
#     samples::Vector{Sample}
# end

# Calculates the loss function for the given samples
# MSE loss of the network for the given data
function loss(network::Network, data::Vector{Sample})
    J = 0
    for sample in data 
        x,y = sample.x, sample.y
        yₘ = network(x)
        J += sum((y.-yₘ).^2)
    end
    J/=length(data)
end

function train!(network::Network, training_data::Vector{Sample}, iterations::Int; method=:SGD, perf_log=nothing, test_data=training_data)
    if method==:SGD 
        SGD!(network, training_data, iterations; perf_log=perf_log, test_data=test_data)
    end
end

# Do n training iterations
function SGD!(network::Network, training_data, iterations::Int; perf_log=nothing, test_data=training_data)

    α_init = 1e-3 # initial learning rate 
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

        if !isnothing(perf_log); perf_log[i] = loss(network, test_data); end
    end
end



# Design considerations
# Sample struct contains an individual sample
# A Vector{Sample} constitutes a DataSet (might make an explicit struct encapsulating this later)

end