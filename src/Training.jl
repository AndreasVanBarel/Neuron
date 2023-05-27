module Training 

export Optimizer, optimize
export SGD, AdaGrad, RMSProp, Adam
export linear_decay, reciprocal_decay
export compute_gradient

using ..Networks
using ..Datasets
using ..Lossfunctions
using Statistics
using LinearAlgebra


# [NOTE 1]: Should implement a function in Networks to return an empty (uninitialized or zero) gradient (or parameter) vector corresponding to a given network. 

##########################
## Gradient computation ##
##########################

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

################
## Optimizers ##
################

abstract type Optimizer end

#### Stochastic Gradient Descent
mutable struct SGD <: Optimizer 
    i::Int # Number of times called, i.e., number of iterations passed
    const network::Network
    const α::Function # Provides α as a function of iteration i
end
Base.show(io::IO, sgd::SGD) = print(io, "SGD optimizer for $(sgd.network)")

SGD(network; α::Function=i->1e-3) = SGD(0, network, α)

function (sgd::SGD)(g)
    sgd.i += 1
    α = sgd.α; i = sgd.i
    θ = get_θ(sgd.network)
    for k in eachindex(s)
        θ[k] .+= .-g[k].*α(i) #updates θ ← θ + Δθ
    end
    set_θ!(sgd.network, θ) # Probably superfluous
end

#### AdaGrad
mutable struct AdaGrad <: Optimizer 
    i::Int # Number of times called, i.e., number of iterations passed
    const network::Network
    const α::Function # Provides α as a function of iteration i
    const δ::Float64
    const r # Accumulation of the squared gradient 
end
Base.show(io::IO, adagrad::AdaGrad) = print(io, "AdaGrad optimizer for $(adagrad.network)")

function AdaGrad(network; α::Function=i->1e-3, δ=1e-7) 
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    AdaGrad(0, network, α, δ, r)
end

function (adagrad::AdaGrad)(g)
    adagrad.i += 1
    α = adagrad.α; δ = adagrad.δ; i = adagrad.i
    r = adagrad.r
    θ = get_θ(adagrad.network)
    for k in eachindex(s)
        r[k] .= r[k] .+ g[k].*g[k]
        θ[k] .+= -g[k].*α(i)./(δ.+sqrt.(r[k])) #updates θ ← θ + Δθ
    end
    set_θ!(adagrad.network, θ) # Probably superfluous
end

#### RMSProp
mutable struct RMSProp <: Optimizer 
    i::Int # Number of times called, i.e., number of iterations passed
    const network::Network
    const α::Function # Provides α as a function of iteration i
    const ρ::Float64 
    const δ::Float64
    const r # Accumulation of the squared gradient 
end
Base.show(io::IO, rmsprop::RMSProp) = print(io, "RMSProp optimizer for $(rmsprop.network)")

function RMSProp(network, α::Function=i->1e-3, ρ=0.9, δ=1e-6) 
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    RMSProp(0, network, α, ρ, δ, r)
end

function (rmsprop::RMSProp)(g)
    rmsprop.i += 1
    α = rmsprop.α; ρ = rmsprop.ρ; δ = rmsprop.δ; i = rmsprop.i
    r = rmsprop.r
    θ = get_θ(rmsprop.network)
    for k in eachindex(s)
        r[k] .= ρ.*r[k] .+ (1-ρ).*g[k].*g[k]
        θ[k] .+= -g[k].*α(i)./(δ.+sqrt.(r[k])) #updates θ ← θ + Δθ
    end
    set_θ!(rmsprop.network, θ) # Probably superfluous
end

#### Adam
mutable struct Adam <: Optimizer
    i::Int # Number of times called, i.e., number of iterations passed
    const network::Network
    const α::Function # Provides α as a function of iteration i
    const ρ1::Float64 
    const ρ2::Float64
    const δ::Float64 
    const s # Accumulation of the gradient
    const r # Accumulation of the squared gradient 
end 
Base.show(io::IO, adam::Adam) = print(io, "Adam optimizer for $(adam.network)")

function Adam(network; α::Function=i->1e-3, ρ1=0.9, ρ2=0.999, δ=1e-8)
    s = get_θ(network).*0 # Accumulation of the gradient
    r = get_θ(network).*0 # Accumulation of the squared gradient 
    Adam(0, network, α, ρ1, ρ2, δ, s, r)
end

function (adam::Adam)(g)
    adam.i += 1
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
    end
    set_θ!(adam.network, θ) # Probably superfluous
end

##########################
## General optimization ##
##########################

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

#############################
## Learning rate schedules ##
#############################

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

#############
## Utility ##
#############

#### A Random permutation generator 
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