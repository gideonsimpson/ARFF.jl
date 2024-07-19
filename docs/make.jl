if abspath(PROGRAM_FILE) == @__FILE__
    # When running the `make.jl` file as a script, automatically activate the
    # `docs` environment and dev-install the main package into that environment
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(path=joinpath(@__DIR__, ".."))
    Pkg.instantiate()
end

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
        "Structures" => ["structs/data.md", "structs/activation.md", "structs/fourier.md"],
        "Training" => "train.md",
        "Auxiliary Functions and Utilities"=>"aux.md",
        "Examples" => ["examples/example1.md", "examples/example2.md"]
    ])
deploydocs(;
    repo="github.com/gideonsimpson/ARFF.jl",
)