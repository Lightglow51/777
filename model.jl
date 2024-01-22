using Flux, CSV, JLD2, DataFrames, Plots

model = Chain(Dense(2 => 1, tanh; bias = false), only)
