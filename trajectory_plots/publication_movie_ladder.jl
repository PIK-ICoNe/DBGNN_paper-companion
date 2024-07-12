include("src/DB-dynamics.jl")
include("src/choices.jl")

include("src/movie.jl")
##

g, gf = gs.ladder_graph # topology
oscillatory = true # oscillatory initialization
dim = 2 # Features per node/edge
Δ_init = 0.1 # Width of initalization
l= 0. # leaky ReLu rate

layout_fontsize = "2images_per_column"

include("src/prepare.jl")

##

s = 11

B, β, T, pert_node, W, W_p, folder = prepare_data(s; dim, g, Δ_init, oscillatory, gf, l)

folder = "plot/movie/ladder"



T = 1200
framerate = 60
all = false
edge_pert = true
movies = true
pert_node = 1

pf = "$(s)"

##

x_DBGNN, e_DBGNN = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, all, edge_pert)
x, e = x_DBGNN, e_DBGNN

make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_DBGNN_")
movies && make_movie(x, e, g, joinpath(folder, "$(pf)_DBGNN_Movie.mp4"); framerate)

##

x_MPNN, e_MPNN = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, all, edge_pert, step=MPNN_step!)
x, e = x_MPNN, e_MPNN

make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_MPNN_")
movies && make_movie(x, e, g, joinpath(folder, "$(pf)_MPNN_Movie.mp4"); framerate)
