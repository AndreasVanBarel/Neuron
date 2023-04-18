using Networks
import LinearAlgebra.norm

relus = [ReLU(rand(3,3), rand(3)) for i in 1:3]
network = Network(relus..., Softmax())

input = rand(3)
out = network(input)
ne = network(input, save)

ng = gradient(ne, out*0)
grad = ∇_θ(ng)

#########
# Backprop testing
# Generates an approximation of dy/dx with y=f(x)
function finite_diff(f,x,y)
    ϵ = 1e-8
    dydx = Matrix{Float64}(undef,length(y), length(x))
    for i in eachindex(x)
        Δx = zeros(size(x))
        Δx[i] = ϵ
        dydx[:,i] = (f(x+Δx) - y)./ϵ
    end
    return dydx
end
function backprop_test(f,x,y)
    dydx = Matrix{Float64}(undef,length(y), length(x))
    for i in eachindex(y)
        dJdy = zeros(size(y))'
        dJdy[i] = 1 
        dydx[i,:] = backprop(f,x,y,dJdy)[1]
    end
    return dydx
end
function finite_diffθ(f,x,y)
    ϵ = 1e-8
    θ = get_θ(f)
    dydθ = Matrix{Float64}(undef,length(y), length(θ))
    for i in eachindex(θ)
        Δθ = zeros(size(θ))
        Δθ[i] = ϵ
        set_θ!(f,θ+Δθ)
        dydθ[:,i] = (f(x) - y)./ϵ
    end
    set_θ!(f,θ) # Restore the initial parameters
    return dydθ
end
function backprop_testθ(f,x,y)
    θ = get_θ(f)
    dydθ = Matrix{Float64}(undef,length(y), length(θ))
    for i in eachindex(y)
        dJdy = zeros(size(y))'
        dJdy[i] = 1 
        dydθ[i,:] = reshape(backprop(f,x,y,dJdy)[2]', length(θ))
    end
    return dydθ
end

# ReLU
W = rand(2,3); b = -rand(2)
relu = ReLU(W,b)
x = rand(3); y = relu(x)

dydx_fd = finite_diff(relu, x, y)
dydx_bp = backprop_test(relu, x, y)
@test norm(dydx_fd-dydx_bp) < 1e-7

# ReLU parameters
dydθ_fd = finite_diffθ(relu, x, y)
dydθ_bp = backprop_testθ(relu, x, y)
@test norm(dydx_fd-dydx_bp) < 1e-7

# Softmax
softmax = Softmax()

x = rand(3); y = softmax(x)

dydx_fd = finite_diff(softmax, x, y)
dydx_bp = backprop_test(softmax, x, y)
@test norm(dydx_fd-dydx_bp) < 1e-7


