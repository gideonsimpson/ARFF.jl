push!(LOAD_PATH,"../src/")
using BasicMD
using Documenter
makedocs(
         sitename = "ARFF.jl",
         modules  = [ARFF],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/gideonsimpson/ARFF.jl",
)