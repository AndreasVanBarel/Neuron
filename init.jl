# run at start
# cd("C:\\Users\\Andreas\\Neuron")
push!(LOAD_PATH, pwd())
using Pkg
Pkg.activate("./env")  

using Statistics # stuff such as mean, var, cov, etc
#using LinearAlgebra
using Printf
using Revise

println("initalization complete.")