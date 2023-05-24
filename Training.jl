module Training 

export sgd!, adagrad!, rmsprop!, adam!
export linear_decay, reciprocal_decay
export compute_gradient

using Networks 
using Datasets
using Lossfunctions

function compute_gradient(network::NetworkWithData, cost, sample)
    x,y = sample.x, sample.y

    yₘ = network(x) # output of final layer

    # Evaluate the gradient of the cost function
    J = cost(yₘ, y)
    dJdyₘ = ∇(cost, yₘ, y)'

    grad = gradient(network, dJdyₘ)
    return grad
end

# NOTE: Batch size not yet implemented
# Stochastic Gradient Descent
function sgd!(network::Network, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = linear_decay, 
    perf_log=nothing, 
    test_data=training_data)

    α(i::Int) = learning_rate(i) # learning rate during iteration i

    m = length(training_data)

    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[rand(1:m)]
        grad = compute_gradient(network, cost, sample)

        Δθ = -grad.*α(i)

        θ = get_θ(network)
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, test_data); end
    end
end

# NOTE: Batch size not yet implemented
# AdaGrad
function adagrad!(network::Network, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = i->1, 
    perf_log=nothing, 
    test_data=training_data)

    rmsprop!(network, cost, training_data, iterations; 
        batch_size = batch_size, learning_rate = learning_rate, perf_log = perf_log, test_data = test_data, ρ = 1, ρ2 = 0)
end

# NOTE: Batch size not yet implemented
# RMSProp
function rmsprop!(network::Network, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = i->1e-3, 
    perf_log=nothing, 
    test_data=training_data,
    ρ = 0.9, ρ2 = nothing) 

    if ρ2 == nothing; ρ2 = ρ; end

    α(i::Int) = learning_rate(i) # learning rate during iteration i
    δ = 1e-7 # for numerical stability

    m = length(training_data)

    r = compute_gradient(network, cost, sample).*0 # Accumulation of the squared gradient #NOTE: Should implement a function in Networks to return an empty (uninitialized or zero) gradient corresponding to a given network. 
    
    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[rand(1:m)]
        g = compute_gradient(network, cost, sample)

        r = [ρ.*r[i] .+ (1-ρ2).*g[i].*g[i] for i in eachindex(r)]

        Δθ = [-g[i].*α(i)./(δ.+sqrt.(r[i])) for i in eachindex(g)]

        θ = get_θ(network)
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, test_data); end
    end
end

# NOTE: Batch size not yet implemented
# Adam # Adaptive moments
function adam!(network::Network, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = i->1e-3, 
    perf_log=nothing, 
    test_data=training_data,
    ρ1 = 0.9, ρ2 = 0.999)

    α(i::Int) = learning_rate(i) # learning rate during iteration i
    δ = 1e-8 # for numerical stability

    m = length(training_data)

    r = s = compute_gradient(network, cost, training_data[1]).*0 # Accumulation of the squared gradient #NOTE: Should implement a function in Networks to return an empty (uninitialized or zero) gradient corresponding to a given network. 
    
    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[rand(1:m)]
        g = compute_gradient(network, cost, sample)

        s = [ρ1.*s[i] .+ (1-ρ1).*g[i] for i in eachindex(r)] # update biased first moment estimate
        r = [ρ2.*r[i] .+ (1-ρ2).*g[i].*g[i] for i in eachindex(r)] # update biased second moment estimate

        s = s./(1-ρ1^i) # correct bias in first moment
        r = r./(1-ρ2^i) # correct bias in second moment

        Δθ = [-s[i].*α(i)./(δ.+sqrt.(r[i])) for i in eachindex(g)]

        θ = get_θ(network)
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, test_data); end
    end
end

### Learning rate schedules
# lineardecay for i=0,...,n between α_0 and α_n, and for i>n, returns α_n.
function linear_decay(i; α_0=1e-3, α_n=1e-4, n=100)
    if i <= n
        t = i/n
        α = (1-t)*α_0 + t*α_n
    else 
        α = α_n
    end
    return α
end

# α_min is the minimal learning rate
# half_i is such that decay(half_i) = 0.5*decay(0)
function reciprocal_decay(i; α_0 = 1e-3, α_min = 1e-4, half_i=1, p = 1)
    # scale = i_offset/(i+i_offset)
    scale = 1/((i/half_i)^p+1)
    α = max(α_0 * scale, α_min) # learning rate during iteration i
    return α
end

end