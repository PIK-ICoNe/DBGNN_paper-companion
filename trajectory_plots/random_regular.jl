using Pkg
Pkg.activate(joinpath(@__DIR__, "./"))
include("src/DB-dynamics.jl")
include("src/choices.jl")

##

g, gf = gs.random_regular # topology
oscillatory = false # oscillatory initialization
dim = 48 # Features per node/edge
Δ_init = 0.1 # Width of initalization
l= 0.0 # leaky ReLu rate

include("src/prepare.jl")

B, β, T, pert_node, W, W_p, folder = prepare_data(42; dim, g, Δ_init, oscillatory, gf, l)


##

T = 5000
folder_T = joinpath(folder, "T_$T")

x,e = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node)

dists, max_responses, max_dist, max_dist_feature = make_tanh_trajectory_plots(; x, g, pert_node, folder=folder_T)

# make_log_trajectory_plots(; x, g, pert_node, folder=folder_T)

##

folder_MPNN = joinpath(folder, "MPNN_T_$T")

x,e = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, step=MPNN_step!)

dists, max_responses, max_dist, max_dist_feature = make_tanh_trajectory_plots(; x, g, pert_node, folder=folder_MPNN)
