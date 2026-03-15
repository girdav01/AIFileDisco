using Flux
using DataFrames
using CSV

function build_model(input_dim, hidden_dim, output_dim)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim),
        softmax
    )
end
