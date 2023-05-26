# run at start
working_directory = @__DIR__
cd(working_directory)
if working_directory ∉ LOAD_PATH
    push!(LOAD_PATH, working_directory) 
end

using Pkg
Pkg.activate("./env")  

using Statistics # stuff such as mean, var, cov, etc
#using LinearAlgebra
using Printf
using Revise

println("initalization complete.")