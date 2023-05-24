# This module defines layers and their combination into networks
module Networks

export Layer, get_θ, set_θ!
export LU, ReLU, Softmax, ConstUnit
export Model, Network
export eval_network
export NetworkWithData, allocate!, allocate, size_params
export NetworkEvaluation, allocate!
export save 
#TODO Fix this save, it's ugly, replace with parameter ..., save=true)
export NetworkGradient, backprop, gradient, ∇_θ

import Base.show 
import Nablas.∇_θ

# ToDo: For consistency, replace the naming "output" and "outputs" to "y" for layer outputs 
# ToDo: Also replace the input by "x"

# Glorot Uniform initialization
# Generates m × n matrix with each element picked from uniform distribution between -sqrt(6/(m+n)) and sqrt(6/(m+n))
glorot_uniform(m,n) = (2 .*rand(m,n) .-1).*sqrt(6/(m+n))

############
## Layers ##
############

abstract type Layer end

evaluate(L::Layer, input) = L(inputs...)
(L::Layer)(inputs...) = error("Layer has no implementation") # fallback
(L::Layer)(θ, input) = error("Layer has no implementation") # fallback
# function (L::Layer)(input, θ) # fallback evaluation with given θ (inefficient)
#     θₒ = get_θ(L)
#     set_θ!(L,θ)
#     output = L(input)
#     set_θ!(L,θ₀) 
#     return output
# end  

# get_θ returns all parameters into a single Array
get_θ(L::Layer) = Float64[] # fallback empty Array
# set_θ! sets all parameters from a single provided Array
set_θ!(L::Layer, θ) = θ == Float64[] ? L : error("The parameter format does not match the required format")

#### Outputs a constant value
mutable struct ConstUnit <: Layer
    const_value
end 
(c::ConstUnit)() = c.const_value
get_θ(c::ConstUnit) = c.const_value
set_θ!(c::ConstUnit, θ) = (const_value = θ; return c)

function backprop(c::ConstUnit, y, dJdy) 
    dJdθ = dJdy
    return dJdθ
end

size_params(c::ConstUnit) = size(c.const_value)
# size_input(c::ConstUnit) = 0
# size_output(c::ConstUnit) = size(c.const_value)

#### Wx+b
mutable struct LU <: Layer #linear unit
    W::Matrix{Float64} # weight matrix 
    b::Vector{Float64} # bias vector
end
function (lu::LU)(x)
    return lu.W*x.+lu.b
end
get_θ(lu::LU) = [lu.W lu.b]
set_θ!(lu::LU, Wb) = set_θ!(lu, Wb[:,1:end-1], Wb[:,end])
function set_θ!(lu::LU, W, b)
    lu.W = W 
    lu.b = b
    return lu
end

# Construct a dense ReLU mapping i inputs to k outputs and fill the parameters with random values
function LU(i::Int, k::Int; init_W=glorot_uniform(k,i), init_b=zeros(k)) 
    LU(init_W, init_b)
end

size_params(lu::LU) = size(lu.W) .+ (0,1)
# size_input(lu::LU) = size(lu.W)[2]
# size_output(lu::LU) = size(lu.W)[1] # Also equals length(lu.b)

# Backpropagates the gradient of the cost function w.r.t. the layer output towards a gradient of the cost function w.r.t. the layer input and the layer parameters.
# x     Layer input 
# y     Layer output 
# dJdy  Gradient of cost w.r.t. y
# Note: for layers with multiple inputs, the header would be backprop(layer::Layer, x1, ..., xn, y, dJdy)
#       and the result would be dJdx1,...,dJdxn,dJdθ
function backprop(lu::LU, x, y, dJdy) 
    # dJ/dx = dJ/dy * dy/dx = dJ/dy*W with rows in W where y is zero set to 0, i.e., y[i] = 0 ⟹ W[i,:] = 0.
    # Equivalently, and more efficiently, we can set y[i] ⟹ dJdy[i] = 0.
    # dJ/db = dJ/dy * dy/db = dJ/dy

    dJdx = dJdy*lu.W
    dJdW = x*dJdy #outer product
    dJdb = dJdy

    dJdθ = [dJdW; dJdb]
    return dJdx, dJdθ
end

#### max(0,Wx+b)
mutable struct ReLU <: Layer #Rectified linear unit
    W::Matrix{Float64} # weight matrix 
    b::Vector{Float64} # bias vector
end
function (relu::ReLU)(x)
    return max.(0,relu.W*x.+relu.b)
end
get_θ(relu::ReLU) = [relu.W relu.b]
set_θ!(relu::ReLU, Wb) = set_θ!(relu, Wb[:,1:end-1], Wb[:,end])
function set_θ!(relu::ReLU, W, b)
    relu.W = W 
    relu.b = b
    return relu
end

# Construct a dense ReLU mapping i inputs to k outputs and fill the parameters with random values
function ReLU(i::Int, k::Int; init_W=glorot_uniform(k,i), init_b=zeros(k)) 
    ReLU(init_W, init_b)
end

size_params(relu::ReLU) = size(relu.W) .+ (0,1)

# size_input(relu::ReLU) = size(relu.W)[2]
# size_output(relu::ReLU) = size(relu.W)[1] # Also equals length(relu.b)

# Backpropagates the gradient of the cost function w.r.t. the layer output towards a gradient of the cost function w.r.t. the layer input and the layer parameters.
# x     Layer input 
# y     Layer output 
# dJdy  Gradient of cost w.r.t. y
# Note: for layers with multiple inputs, the header would be backprop(layer::Layer, x1, ..., xn, y, dJdy)
#       and the result would be dJdx1,...,dJdxn,dJdθ
function backprop(relu::ReLU, x, y, dJdy) 
    # dJ/dx = dJ/dy * dy/dx = dJ/dy*W with rows in W where y is zero set to 0, i.e., y[i] = 0 ⟹ W[i,:] = 0.
    # Equivalently, and more efficiently, we can set y[i] ⟹ dJdy[i] = 0.
    # dJ/db = dJ/dy * dy/db = dJ/dy*1_{y≠0}
    dJdy_0 = similar(dJdy) # We don't wish to overwrite dJdy
    for i in eachindex(dJdy)
        dJdy_0[i] = y[i] == 0 ? 0 : dJdy[i]
    end

    dJdx = dJdy_0*relu.W
    dJdW = x*dJdy_0 #outer product
    dJdb = dJdy_0

    dJdθ = [dJdW; dJdb]
    return dJdx, dJdθ
end

#### Softmax
struct Softmax <: Layer end # Has no parameters
function (s::Softmax)(z)
    max_z = maximum(z)
    exps = exp.(z.-max_z)
    return exps./sum(exps)
end

size_params(s::Softmax) = (0,1)
# size_input(s::Softmax) = ?
# size_output(s::Softmax) = ?

# Backpropagates the gradient of the cost function w.r.t. the layer output towards a gradient of the cost function w.r.t. the layer input and the layer parameters.
# x     Layer input 
# y     Layer output 
# dJdy  Gradient of cost w.r.t. y
function backprop(::Softmax, x, y, dJdy) 
    # Let the softmax function be represented by y=f(x), then dy/dx is
    # 1/s^2 * ( -v*v'  +  diag(v)*s )
    # where v = exp.(x) and s = sum(v)
    # Therefore, dJdy * dydx is 
    # 1/s^2 * ( -dJdy*v*v' + dJdy.*v*s ) = -dJdy*v*v'/s^2 + dJdy.*v/s
    x = x.-maximum(x)
    v = exp.(x)
    s = sum(v)
    dJdx = -(dJdy*v/s^2)*v' + dJdy.*v'/s
    return dJdx, Float64[]'
end

################
### Networks ###
################

abstract type Model end 

struct Network <: Model 
    layers::Vector{Layer} #Contains a list of all layers in the network. The output layer should probably be layers[end]
    connections::Vector{Vector{Int32}}
    # Has a length equal to the number of layers. 
    # connections[i] gives a vector{Int32} of input layer indices for layer i.
    # Since the network is not recursive, it must hold that connections[i].<i
end

# construct a simple sequential network 
function Network(layers::Vararg{Layer})
    connections = [[i-1] for i in 1:length(layers)]
    return Network(collect(Layer,layers), connections)
end 

function Base.show(io::IO, n::Network)
    print(io, "Network with layers (")
    for (i,layer) in enumerate(n.layers)
        print(io, typeof(layer))
        i < length(n.layers) && print(io, ", ")
    end
    print(io, ")")
end

get_θ(n::Network) = get_θ.(n.layers)
function set_θ!(n::Network, θs) 
    set_θ!.(n.layers, θs)
    return n
end

############################
### Evaluating a network ###
############################

### Simple evaluation without saving any intermediate variables
(n::Network)(input, i_output) = eval_network(n, input, i_output)
(n::Network)(input) = n(input, length(n.layers)) # By default, considers the last layer of the network to be the output layer

# evaluating the network without saving to intermediate results
function eval_network(network::Network, input, i_layer::Integer) #eval_layer fills the output for layer i_layer in results
    if i_layer == 0 # should return the network input 
        return input
    end
    inputs = [eval_network(network, input, i) for i in network.connections[i_layer]]
    output = network.layers[i_layer](inputs...)
    return output
end

### Evaluation with saving intermediate data, allowing backpropagation and gradient calculation.
mutable struct NetworkWithData
    const network::Network # reference to the network this evaluation is for
    input # The provided input
    const outputs::Vector{Any} # evaluations of the layers of the network, some of which may remain empty if not needed    
    dJdx # The gradient w.r.t. the input of the network
    const dJdy::Vector{Any} # dJdy[i] provides dJ/dLᵢ with yᵢ the output of the i-th layer of the network. Some of these may remain empty if not needed.
    const dJdθ::Vector{Any} # dJdθ[i] provides dJ/dθᵢ with θᵢ the parameters of the i-th layer of the network. Some of these may remain empty if not needed.
end

# Creates an empty, unallocated NetworkWithData object for the given network
function NetworkWithData(network::Network) 
    m = length(network.layers)
    NetworkWithData(network, nothing, Vector{Any}(nothing, m), nothing, Vector{Any}(nothing,m), Vector{Any}(nothing,m))
end

# Creates a new NetworkWithData object and allocates appropriate memory for it
function allocate(network::Network, input, i_layer::Integer=length(network.layers))
    nwd = NetworkWithData(network) 
    allocate!(nwd, input, i_layer)
    return nwd
end

# Allocates space in the networkWithData object for inputs such as the provided one.
# i_layer is the layer of the network serving as the output layer
function allocate!(nwd::NetworkWithData, input, i_layer::Integer=length(nwd.network.layers), first::Bool=true) #eval_layer fills the output for layer i_layer in ne

    if first
        nwd.outputs.=nothing #de-allocate the outputs
        nwd.input = input
        nwd.dJdx = Array{Float64}(undef,size(input'))
    end

    if i_layer == 0 # should return the network input 
        return input
    end

    if !isnothing(nwd.outputs[i_layer]) # already allocated
        return nwd.outputs[i_layer]
    end

    network = nwd.network # The network in which evaluations will occur
    inputs = [allocate!(nwd, input, i, false) for i in network.connections[i_layer]]
    output = network.layers[i_layer](inputs...)

    # Allocate for layer i_layer
    nwd.outputs[i_layer] = output
    nwd.dJdy[i_layer] = similar(output')
    nwd.dJdθ[i_layer] = Array{Float64}(undef, reverse(size_params(network.layers[i_layer])))
    return output
end

# Evaluation of a NetworkWithData object stores the intermediate results.
function (nwd::NetworkWithData)(input, i_layer=length(nwd.network.layers), done=falses(length(nwd.network.layers))) 
    if i_layer == 0 # should return the network input 
        nwd.input = input # input should always just be a pointer to whatever input is given, not a copy.
        return input
    end
    if done[i_layer] # already calculated
        return nwd.outputs[i_layer] # returns a pointer to output of layer i_layer
    else
        done[i_layer] = true
    end

    network = nwd.network # The network in which evaluations will occur

    inputs = [nwd(input, i, done) for i in network.connections[i_layer]]

    output = network.layers[i_layer](inputs...)
    nwd.outputs[i_layer] .= output
    return output
end

#######################
### Backpropagation ###
#######################

# Calculates the gradient of the network w.r.t. the parameters and the input
function gradient(nwd::NetworkWithData, dJdyₘ)
    network = nwd.network
    m = length(network.layers)

    nwd.dJdy[end] = dJdyₘ 

    [a.=0 for a in nwd.dJdθ]
    [a.=0 for a in nwd.dJdy[1:end-1]]
    nwd.dJdx.=0

    propagate_gradient!(nwd, m)

    return collect.(adjoint.(nwd.dJdθ))
end

# Function propagates gradients starting at the output of layer i_layer towards its parameters, all parent layers and their parameters and the network inputs.
function propagate_gradient!(nwd::NetworkWithData, i_layer::Integer)
    #### Backpropagates gradients at the output of layer i_layer across that layer towards its inputs and parameters 

    network = nwd.network
    layer = network.layers[i_layer]

    # loop over and gather all layer inputs
    connections = network.connections[i_layer] # input layers
    xs = [i==0 ? nwd.input : nwd.outputs[i] for i in connections]
    y = nwd.outputs[i_layer]
    dJdy = nwd.dJdy[i_layer]
    result = backprop(layer, xs..., y, dJdy) # contains dJdx₁,...,dJdxₙ,dJdθ

    #Adds term to a[i] or sets a[i] af a[i] contains nothing
    function add!(a::AbstractArray, term) 
        # println(size(a))
        # println(size(term))
        a.+=term
    end

    # println(i_layer)

    for (k,i) in enumerate(connections) # the k-th input is layer i
        if i==0 
            # println("dJdx")
            add!(nwd.dJdx, result[k])
        else
            # println("dJdy")
            add!(nwd.dJdy[i], result[k])
        end
    end
    add!(nwd.dJdθ[i_layer], result[end])

    #### recursively propagates gradients further backwards through the network

    for i in connections
        i>0 && propagate_gradient!(nwd, i) #for i<=0 we have base inputs; the propagation should end there.
    end
end

# Produces the gradient w.r.t. θ
function ∇_θ(nwd::NetworkWithData) 
    return collect.(adjoint.(nwd.dJdθ))
end




# Thoughts on the Architecture of the Network object 
#   Representation of the network 
#       Have layers be their own thing, such that they can be evaluated separately (not referencing other layers as input)
#       Have the connections between layers stored in the Network object 
#   Second layer has as input the first layer etc 
#

#   Handling of evaluation 
#       Having the output layer recursively reference and invoke evaluation of previous layers
#       Saving the last evaluation value in the layer? 
#   Handling of gradient calculation (backpropagation)
#       Using last evaluation data, stored in some way, to effect backpropagation
#
#   Choice of where and how to store the network evaluation, if requested to do so. Either
#       (1) Network object saves the last evaluation and gradient information, such that it can be extracted in some way. 
#       (2) The evaluated data is stored in a new accompanying object, called, e.g., NetworkEvaluation.
#   Probably best to pick (2):
#       Pros:   Can then be utilized only if storing the intermediate evaluation states is desired. 
#               Actually parallelizable, since a single Network object can spawn multiple data evaluation objects in parallel.
#               The Network object is then accessed in read-only.
#       Cons:   Potentially additional complexity and duplication of certain implementation logic. 

end