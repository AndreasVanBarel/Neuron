module Training 

export sgd!, adagrad!, rmsprop!, adam!
export linear_decay, reciprocal_decay
export compute_gradient
export Optimizer, Adam, optimize

using Networks
using Datasets
using Lossfunctions
using Statistics
using LinearAlgebra

# Computation of the gradient for a single sample
function compute_gradient(nwd::NetworkWithData, cost, sample::Sample)
    x,y = sample.x, sample.y

    yₘ = nwd(x) # output of final layer

    # Evaluate the gradient of the cost function
    J = cost(yₘ, y) 
    dJdyₘ = ∇(cost, yₘ, y)

    grad = gradient(nwd, dJdyₘ)
    return grad
end

# Parallelized computation of the gradient for a given batch
function compute_gradient(nwds::Vector{NetworkWithData}, cost, batch::Vector{Sample})
    @assert length(nwds) == length(batch) "Allocated memory and batch size do not correspond"

    batch_size = length(batch)
    grads = Vector{Any}(undef,batch_size)

    Threads.@threads for b = 1:batch_size
        sample = batch[b]
        nwd = nwds[b]
        grads[b] = compute_gradient(nwd, cost, sample)
    end

    grad = grads[1]
    for b = 2:batch_size 
        [grad[i].+= grads[b][i] for i in eachindex(grad)]
    end
    [grad[i]./= length(grads) for i in eachindex(grad)]

    return grad
end

# Stochastic Gradient Descent
function sgd!(network::NetworkWithData, cost::AbstractCost, training_data, iterations::Int;
    batch_size = 1,
    learning_rate = linear_decay, 
    perf_log=nothing)

    iterations = floor(Int,length(training_data)/batch_size) # calculate number of iterations 
    nwds = [allocate(network, training_data[1].x) for _ in 1:batch_size] # Generate NetworkWithData objects (see also [NOTE 1] above)
    permutation = random_permutation(length(training_data)) # shuffle training data
    α(i::Int) = learning_rate(i) # learning rate during iteration i

    θ = get_θ(network) # Preallocation for all θ

    for i = 1:iterations
        # take batch from training_data 
        batch = [training_data[permutation[(i-1)*batch_size+b]] for b in 1:batch_size]
        g = compute_gradient(nwds, cost, batch)

        [θ[k] .+= .-g[k].*α(i) for k in eachindex(g)]

        set_θ!(network, θ)  

        if !isnothing(perf_log); perf_log[i] = mean(cost.(network, batch)); end
    end
end

# AdaGrad
function adagrad!(network::NetworkWithData, cost::AbstractCost, training_data;
    batch_size = 1,
    learning_rate = i->1, 
    perf_log=nothing)

    rmsprop!(network, cost, training_data; 
        batch_size = batch_size, learning_rate = learning_rate, perf_log = perf_log, ρ = 1, ρ2 = 0)
end

# RMSProp
function rmsprop!(network::Network, cost::AbstractCost, training_data;
    batch_size = 1,
    learning_rate = i->1e-3, 
    perf_log=nothing, 
    ρ = 0.9, ρ2 = nothing) 
    if ρ2 === nothing; ρ2 = ρ; end

    iterations = floor(Int,length(training_data)/batch_size) # calculate number of iterations 
    nwds = [allocate(network, training_data[1].x) for _ in 1:batch_size] # Generate NetworkWithData objects (see also [NOTE 1] above)
    permutation = random_permutation(length(training_data)) # shuffle training data
    α(i::Int) = learning_rate(i) # learning rate during iteration i

    δ = 1e-7 # for numerical stability

    θ = get_θ(network) # Preallocation for all θ
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    
    for i = 1:iterations
        # take batch from training_data 
        batch = [training_data[permutation[(i-1)*batch_size+b]] for b in 1:batch_size]
        g = compute_gradient(nwds, cost, batch)

        [r[k] .= ρ.*r[k] .+ (1-ρ2).*g[k].*g[k] for k in eachindex(r)]
        [θ[k] .+= -g[k].*α(i)./(δ.+sqrt.(r[k])) for k in eachindex(g)]

        set_θ!(network, θ)  

        if !isnothing(perf_log); perf_log[i] = mean(cost.(network, batch)); end
    end
end

# [NOTE 1]: Should implement a function in Networks to return an empty (uninitialized or zero) gradient (or parameter) vector corresponding to a given network. 

abstract type Optimizer end

# Adam
mutable struct Adam <: Optimizer
    const network::Network
    const α::Function # Provides α as a function of iteration i
    const ρ1::Float64 
    const ρ2::Float64
    const δ::Float64 # Provides δ as a function of iteration i
    i::Int # Number of times called, i.e., number of iterations passed
    const s # Accumulation of the gradient
    const r # Accumulation of the squared gradient 
end 

function Base.show(io::IO, adam::Adam)
    print(io, "Adam optimizer for $(adam.network)")
end

function Adam(network, α::Function=i->1e-3, ρ1=0.9, ρ2=0.999, δ=1e-8)
    s = get_θ(network).*0 # Accumulation of the gradient
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    Adam(network, α, ρ1, ρ2, δ, 0, s, r)
end
# Adam(network, α::Number=1e-3, args...) = Adam(network, i->α, args...)

function (adam::Adam)(g)
    adam.i = adam.i+1
    α = adam.α; ρ1 = adam.ρ1; ρ2 = adam.ρ2; δ = adam.δ; i = adam.i
    s = adam.s 
    r = adam.r 
    θ = get_θ(adam.network)
    for k in eachindex(s)
        s[k].=ρ1.*s[k] .+ (1-ρ1).*g[k] # update biased first moment estimate
        r[k].=ρ2.*r[k] .+ (1-ρ2).*g[k].*g[k] # update biased second moment estimate
        # s_corrected[k].=s[k]./(1-ρ1^i) # correct bias in first moment
        # r_corrected[k].=r[k]./(1-ρ2^i) # correct bias in second moment
        θ[k] .+= .-(s[k]./(1-ρ1^i)).*α(i)./(δ.+sqrt.(r[k]./(1-ρ2^i))) #updates θ ← θ + Δθ

        if any(isinf.(r[k]))
            println("Problem: Infinity detected in r[k]...")
            println("at layer k=$k, at iteration i=$i")
            throw(DebugException(g[k], s[k], r[k], θ[k]))
        end
    end

    set_θ!(adam.network, θ) # Probably superfluous
end

function optimize(optimizer::Optimizer, cost::AbstractCost, training_data; batch_size=1, perf_log=nothing)
    network = optimizer.network
    iterations, nwds, permutation = setup(network, training_data, batch_size)

    for i = 1:iterations
        # take batch from training_data 
        batch = [training_data[permutation[(i-1)*batch_size+b]] for b in 1:batch_size]

        gradient = compute_gradient(nwds, cost, batch)
        optimizer(gradient)

        if !isnothing(perf_log); push!(perf_log, mean(cost.(network, batch))); end
    end
end

function setup(network, training_data, batch_size)
    iterations = floor(Int,length(training_data)/batch_size) # calculate number of iterations 
    nwds = [allocate(network, training_data[1].x) for _ in 1:batch_size] # Generate NetworkWithData objects (see also [NOTE 1] above)
    permutation = random_permutation(length(training_data)) # shuffle training data
    return iterations, nwds, permutation
end

function adam!(network::Network, cost::AbstractCost, training_data;
    batch_size = 1,
    learning_rate = i->1e-3, 
    perf_log=nothing, 
    ρ1 = 0.9, ρ2 = 0.999)

    iterations, nwds, permutation = setup(network, training_data, batch_size)

    α(i::Int) = learning_rate(i) # learning rate during iteration i

    δ = 1e-8 # for numerical stability

    θ = get_θ(network) # Gets pointers to all θ in the network
    s = get_θ(network).*0 # Accumulation of the gradient
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    # See also [NOTE 1] above

    for i = 1:iterations
        # take batch from training_data 
        batch = [training_data[permutation[(i-1)*batch_size+b]] for b in 1:batch_size]

        g = compute_gradient(nwds, cost, batch)

        for k in eachindex(s)
            s[k].=ρ1.*s[k] .+ (1-ρ1).*g[k] # update biased first moment estimate
            r[k].=ρ2.*r[k] .+ (1-ρ2).*g[k].*g[k] # update biased second moment estimate
            # s_corrected[k].=s[k]./(1-ρ1^i) # correct bias in first moment
            # r_corrected[k].=r[k]./(1-ρ2^i) # correct bias in second moment
            θ[k] .+= .-(s[k]./(1-ρ1^i)).*α(i)./(δ.+sqrt.(r[k]./(1-ρ2^i))) #updates θ ← θ + Δθ
        end

        set_θ!(network, θ) 
        #note that in fact the network already points to θ for its parameters
        #in case something gets allocated spuriously anyway, this is safe.

        if !isnothing(perf_log); perf_log[i] = mean(cost.(network, batch)); end
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

### A Random permutation generator 
function random_permutation!(E::AbstractVector{T}) where T
    # Given E = [e1,...,en], we pick a random one out (by generating a random number between 1 and n), then swapping it with the last, i.e., nth element producing the array E = [e1,...,en,...,e(n-1)]. Then one can do the same for n-1 and so on until reaching 0.
    
    n = length(E)
    
    for r = length(E):-1:2 # r is the number of remaining elements
        index = rand(1:r) # Potential optimization possibility here; unsure if rand(1:r) is the fastest way to generate a random number between 1 and r.
        E[r],E[index] = E[index],E[r] # switches E[r] and E[index]
    end
    
    return E
end
    
random_permutation(E::AbstractVector{T}) where T = random_permutation!(collect(E))
random_permutation(n::Int) = random_permutation!(collect(1:n))

end