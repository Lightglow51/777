include("model.jl")

X = CSV.read("data/train.csv", DataFrame)

data = [([x, y], z) for (x, y, z) in DataFrames.eachrow(X)]

model_state = JLD2.load("model.jld2", "model_state");
Flux.loadmodel!(model, model_state);

opt = Flux.setup(Flux.Descent(), model)

for epoch in 1:20
    Flux.train!((m, x, y) -> (m(x) - y)^2, model, data, opt)
end

model_state = Flux.state(model)
jldsave("model.jld2"; model_state)
