module Networks

export Layer, get_θ, set_θ!
export LU, ReLU, Softmax, ConstUnit
export Model, Network
export eval_network
export NetworkEvaluation, eval_network!
export save

export NetworkGradient, backprop, gradient, ∇_θ

import Base.show 

# ToDo: For consistency, replace the naming "output" and "outputs" to "y" for layer outputs 
# ToDo: Also replace the input by "x"

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
get_θ(L::Layer) = [] # fallback empty Array
# set_θ! sets all parameters from a single provided Array
set_θ!(L::Layer, θ) = θ == [] ? L : error("The parameter format does not match the required format")

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
function LU(i::Int, k::Int; f=randn) 
    LU(f(k,i), f(k))
end

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
function ReLU(i::Int, k::Int; f=randn) 
    ReLU(f(k,i), f(k))
end

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
    return dJdx, []'
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
struct Save end
save = Save() # To be passed when evaluating a network if the intermediate states and other information should be returned instead of just the final value.

function (n::Network)(input, i_output, ::Save) # input is the network input, i_output is the layer of the network that serves as the output layer
    results = NetworkEvaluation(n, input) # Generate new empty NetworkEvaluation object
    eval_network!(results, i_output)
    return results
end
(n::Network)(input, s::Save) = n(input, length(n.layers), s) # By default, considers the last layer of the network to be the output layer
(n::Network)(input, i_output) = eval_network(n, input, i_output)
(n::Network)(input) = n(input, length(n.layers)) # By default, considers the last layer of the network to be the output layer

struct NetworkEvaluation 
    network::Network # reference to the network this evaluation is for
    input # The provided input
    outputs::Vector{Any} # evaluations of the layers of the network, some of which may remain empty if not needed
end

NetworkEvaluation(network::Network, input) = NetworkEvaluation(network, input, Vector{Any}(nothing, length(network.layers)))

# evaluating the network without saving to intermediate results
function eval_network(network::Network, input, i_layer::Integer) #eval_layer fills the output for layer i_layer in results
    if i_layer == 0 # should return the network input 
        return input
    end
    inputs = [eval_network(network, input, i) for i in network.connections[i_layer]]
    output = network.layers[i_layer](inputs...)
    return output
end

# evaluating the network and saving intermediate results
function eval_network!(ne::NetworkEvaluation, i_layer::Integer) #eval_layer fills the output for layer i_layer in ne
    if i_layer == 0 # should return the network input 
        return ne.input
    end
    if !isnothing(ne.outputs[i_layer]) # already calculated
        return ne.outputs[i_layer]
    end

    network = ne.network # The network in which evaluations will occur

    inputs = [eval_network!(ne, i) for i in network.connections[i_layer]]

    output = network.layers[i_layer](inputs...)
    ne.outputs[i_layer] = output
    return output
end


#######################
### Backpropagation ###
#######################

# We assume for now a MSE cost function

struct NetworkGradient
    network::Network # reference to the network this evaluation is for
    evaluation::NetworkEvaluation
    y # The desired output as given by the training data
    dJdx # The gradient w.r.t. the input of the network
    dJdy::Vector{Any} # dJdy[i] provides dJ/dLᵢ with yᵢ the output of the i-th layer of the network. Some of these may remain empty if not needed.
    dJdθ::Vector{Any} # dJdθ[i] provides dJ/dθᵢ with θᵢ the parameters of the i-th layer of the network. Some of these may remain empty if not needed.
end

# Calculates the gradient of the network w.r.t. the parameters and the input
function gradient(ne::NetworkEvaluation, y)
    network = ne.network
    m = length(network.layers)

    # Construct a NetworkGradient object 
    ng = NetworkGradient(network, ne, y, zeros(size(ne.input')), Vector{Any}(nothing,m), Vector{Any}(nothing,m))

    yₘ = ne.outputs[end] # output of final layer

    # Evaluate the gradient of the MSE cost function
    J = sum((yₘ.-y).^2) # Cost function 
    dJdyₘ = (yₘ.-y)' # dJ/dyₘ

    ng.dJdy[end] = dJdyₘ 

    propagate_gradient!(ng, m)

    return ng
end

# Function propagates gradients starting at the output of layer i_layer towards its parameters, all parent layers and their parameters and the network inputs.
function propagate_gradient!(ng::NetworkGradient, i_layer::Integer)
    #### Backpropagates gradients at the output of layer i_layer across that layer towards its inputs and parameters 

    network = ng.network
    ne = ng.evaluation
    layer = network.layers[i_layer]

    # loop over and gather all layer inputs
    connections = network.connections[i_layer] # input layers
    xs = [i==0 ? ne.input : ne.outputs[i] for i in connections]
    y = ne.outputs[i_layer]
    dJdy = ng.dJdy[i_layer]
    result = backprop(layer, xs..., y, dJdy) # contains dJdx₁,...,dJdxₙ,dJdθ

    #Adds term to a[i] or sets a[i] af a[i] contains nothing
    @inline add!(a::Array, i, term) = isnothing(a[i]) ? a[i]=term : a[i].+=term

    for (k,i) in enumerate(connections) # the k-th input is layer i
        if i==0 
            ng.dJdx .= result[k]
        else
            add!(ng.dJdy, i, result[k])
        end
    end
    add!(ng.dJdθ, i_layer, result[end])

    #### recursively propagates gradients further backwards through the network

    for i in connections
        i>0 && propagate_gradient!(ng, i) #for i<=0 we have base inputs; the propagation should end there.
    end
end

# Produces the gradient w.r.t. θ
function ∇_θ(ng::NetworkGradient) 
    return collect.(adjoint.(ng.dJdθ))
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