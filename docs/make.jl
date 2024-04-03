push!(LOAD_PATH,"../src/")
using ARFF
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(checkdocs=:none,
    sitename = "ARFF.jl",
    modules  = [ARFF],
    format=Documenter.HTML(
        # ...
        assets=String["assets/citations.css"],
    ),
    plugins=[bib],
    pages=[
        "Home" => "index.md",
        "Structures" => "structs.md",
        "Training" => "train.md",
        "Auxiliary Functions"=>"aux.md",
        "Utilities"=>"utils.md",
        "Examples"=>["examples/example1.md"]
    ])
deploydocs(;
    repo="github.com/gideonsimpson/ARFF.jl",
)