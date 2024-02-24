push!(LOAD_PATH,"../src/")
using ARFF
using Documenter
makedocs(sitename = "ARFF.jl",
         modules  = [ARFF],
         pages=[
                "Home" => "index.md"
    ], checkdocs=:none)
deploydocs(;
    repo="github.com/gideonsimpson/ARFF.jl",
)