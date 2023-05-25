# Just some random tests etc 

### Produces a 28x28 image resembling the MNIST dataset training images.
m = InputDraw.distance_matrix(line,28,28)
# Show a sample
heatmap(rotl90(m))

f(x) = min(1,max(1.8-20x,0))
heatmap(rotl90(f.(m)))

heatmap(rotl90(x_test[:,:,1]))



# Glorot Uniform initialization
# Generates uniform distribution between -sqrt(6/(x+y)) and sqrt(6/(x+y))
glorot_uniform(x,y) = (2 .*rand(x,y) .-1).*sqrt(6/(x+y))


####
sample = training_data[1]
x,y = sample.x, sample.y

yₘ = network(x) # output of final layer

# Evaluate the gradient of the cost function
J = cost(yₘ, y)
dJdyₘ = ∇(cost, yₘ, y)

nwd = allocate(network, x)

θ = get_θ(nwd)
inspect(θ)
grad = gradient(nwd,dJdyₘ)
inspect(grad)