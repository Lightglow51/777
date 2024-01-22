include("model.jl")


losses = push!([], CSV.read("losses.csv", DataFrame).loss)

X_test = CSV.read("data/test.csv", DataFrame)

Flux.loadmodel!(model, JLD2.load("model.jld2", "model_state"))

data = [([x, y], z) for (x, y, z) in DataFrames.eachrow(X_test)]

push!(losses[1], Flux.Losses.mse(model.(first.(data)), last.(data)))
CSV.write("losses.csv", DataFrame(losses, DataFrames.Index([:loss])))

plot(losses)
savefig("loss.png")
