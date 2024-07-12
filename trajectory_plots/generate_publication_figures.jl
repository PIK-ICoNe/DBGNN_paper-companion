using Pkg
Pkg.activate(joinpath(@__DIR__, "./"))

layout_fontsize = "3images_per_column"

oscillatory = false
include("grid.jl")

oscillatory = true
include("grid.jl")

##

oscillatory = false
include("pathgraph.jl")

oscillatory = true
include("pathgraph.jl")

##

include("publication_movie_ladder.jl")
include("publication_movie_ws.jl")
