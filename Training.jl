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
function sgd!(network::NetworkWithData, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = linear_decay, 
    perf_log=nothing)

    α(i::Int) = learning_rate(i) # learning rate during iteration i

    m = length(training_data)

    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[i]
        grad = compute_gradient(network, cost, sample)

        Δθ = -grad.*α(i)

        θ = get_θ(network)
        set_θ!(network, θ.+Δθ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, sample); end
    end
end

# NOTE: Batch size not yet implemented
# AdaGrad
function adagrad!(network::NetworkWithData, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = i->1, 
    perf_log=nothing)

    rmsprop!(network, cost, training_data, iterations; 
        batch_size = batch_size, learning_rate = learning_rate, perf_log = perf_log, ρ = 1, ρ2 = 0)
end

# NOTE: Batch size not yet implemented
# RMSProp
function rmsprop!(network::NetworkWithData, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = i->1e-3, 
    perf_log=nothing, 
    ρ = 0.9, ρ2 = nothing) 

    if ρ2 === nothing; ρ2 = ρ; end

    α(i::Int) = learning_rate(i) # learning rate during iteration i
    δ = 1e-7 # for numerical stability

    m = length(training_data)

    θ = get_θ(network) # Preallocation for all θ
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    
    #NOTE: Should implement a function in Networks to return an empty (uninitialized or zero) gradient corresponding to a given network. 
    
    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[i]
        g = compute_gradient(network, cost, sample)

        [r[k] .= ρ.*r[k] .+ (1-ρ2).*g[k].*g[k] for k in eachindex(r)]
        [θ[k] .+= -g[k].*α(i)./(δ.+sqrt.(r[k])) for k in eachindex(g)]

        set_θ!(network, θ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, sample); end
    end
end

# NOTE: Batch size not yet implemented
# Adam # Adaptive moments
function adam!(network::NetworkWithData, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = i->1e-3, 
    perf_log=nothing, 
    ρ1 = 0.9, ρ2 = 0.999)

    α(i::Int) = learning_rate(i) # learning rate during iteration i
    δ = 1e-8 # for numerical stability

    m = length(training_data)

    θ = get_θ(network) # Preallocation for all θ
    s = get_θ(network).*0 # Accumulation of the gradient
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    #NOTE: Should implement a function in Networks to return an empty (uninitialized or zero) gradient corresponding to a given network. 
    
    for i = 1:iterations
        # take random element from training_data 
        sample = training_data[i]
        g = compute_gradient(network, cost, sample)

        for k in eachindex(s)
            s[k].=ρ1.*s[k] .+ (1-ρ1).*g[k] # update biased first moment estimate
            s[k].=s[k]./(1-ρ1^i) # correct bias in first moment
            r[k].=ρ2.*r[k] .+ (1-ρ2).*g[k].*g[k] # update biased second moment estimate
            r[k].=r[k]./(1-ρ2^i) # correct bias in second moment
            θ[k] .+= -s[k].*α(i)./(δ.+sqrt.(r[k]))
        end

        set_θ!(network, θ)  

        if !isnothing(perf_log); perf_log[i] = cost(network, sample); end
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