using Documenter, HighDimensionalModels

makedocs(
    sitename = "HighDimensionalModels.jl",
    authors = "Mladen Kolar",
    modules = [HighDimensionalModels],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlakolar.github.io/HighDimensionalModels.jl/stable/",
        mathengine = Documenter.KaTeX(),
    ),
    pages = [
        "Home" => "index.md",
        "The lasso" => "lasso.md",
        "The group lasso" => "group_lasso.md",
    ],
    checkdocs = :exports,
)

deploydocs(repo = "github.com/mlakolar/HighDimensionalModels.jl.git", devbranch = "master", push_preview = false)
