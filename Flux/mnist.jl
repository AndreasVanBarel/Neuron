using Flux, MLDatasets, CUDA, FileIO
using Flux: train!, onehotbatch
using MLUtils: flatten
using ProgressMeter

x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_test, y_test = MLDatasets.MNIST(split=:test)[:]
# Note that y_train and y_test are already in one-hot encoding

y_train_oh = Flux.onehotbatch(y_train, 0:9)
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 32, relu), 
    Dense(32, 10, relu), 
    softmax
) |> gpu

model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10, relu), 
    softmax
) |> gpu

# The model encapsulates parameters, randomly initialised. Its initial output is:
out1 = model(flatten(x_train) |> gpu) |> cpu     
 
# To train the model, we use batches of 64 samples
loader = Flux.DataLoader((x_train, y_train_oh) |> gpu, batchsize=64, shuffle=true);
# 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

optim = Flux.setup(Flux.Adam(1e-3), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
losses = []
@showprogress for epoch in 1:10
    for (x, y) in loader
        x = flatten(x)
        y = flatten(y)
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end


out_train = model(flatten(x_train) |> gpu) |> cpu  # each column gives probability of that outcome
out_test = model(flatten(x_test) |> gpu) |> cpu  # each column gives probability of that outcome

mean(getindex.(argmax(out_train,dims=1),1).-1 .== y_train')  # accuracy training set
mean(getindex.(argmax(out_test,dims=1),1).-1 .== y_test')  # accuracy test set

## Plotting convergence of loss to zero
using Plots  

n = length(loader)
epoch_means = mean.(Iterators.partition(losses, n))
plot(losses; xaxis=(:log10, "iteration"),
    yaxis="loss", label="per batch")
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200)

# Show a sample
heatmap(rotl90(x_train[:,:,132]))

## Input a new sample from drawing
using InputDraw 
lines = make_canvas()

line = get_last_line(lines)
m = InputDraw.distance_matrix(line,28,28)

f(x) = min(1,max(1.8-20x,0))
f(x) = min(1,max(1.25-20x,0))

x_draw = f.(m)
heatmap(rotl90(x_draw))

result = model(reshape(x_draw,length(x_draw)) |> gpu) |> cpu

# For investigating model parameters
function get_params(model)
    p = Flux.params(model).params
    p = collect(p)
    [p[i] |> cpu for i in 1:length(p)]
end
p = get_params(model)
W = p[3]
feature = 2
heatmap(rotl90(reshape(W[feature,:],28,28)))
  