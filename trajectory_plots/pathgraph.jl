include("src/DB-dynamics.jl")
include("src/choices.jl")

include("src/movie.jl")
using Serialization

##

g, gf = gs.path_graph # topology

if oscillatory
    osc = "oscillatory"
    dims = [2, 4, 12, 24]
else
    osc = "non_oscillatory"
    dims = [1, 2, 4, 12, 24]
end
Δ_init = 0.1 # Width of initalization
l= 0.1 # leaky ReLu rate
dim = 1

f = open("path_plot_$(osc).serial", "w")

include("src/prepare.jl")


for dim in dims

    for s in 1:20

        local l = 1.

        B, β, T, pert_node, W, W_p, folder = prepare_data(s; dim, g, Δ_init, oscillatory, gf, l)

        folder = "plot/path/$(osc)_$(dim)d"
        jp(fn) = joinpath(folder, fn)


        T = 400
        framerate = 90
        all = false
        edge_pert = false
        movies = false
        pert_node = 1:5

        pf = "$(s)"

        local l = 1.

        name = "DBLin"

        x, e = make_trajectory(; g, T, W, W_p, β, B, l = 1., dim, pert_node, all, edge_pert)

        serialize(f, (;s, dim, x, e, g, l, name, pert_node, pf))

        make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_$(name)_")
        movies && make_movie(x, e, g, joinpath(folder, "$(pf)_$(name)_Movie.mp4" |> jp); framerate)

        name = "MPNNLin"

        x, e = make_trajectory(; g, T, W, W_p, β, B, l = 1., dim, pert_node, step=MPNN_step!, all, edge_pert)

        serialize(f, (;s, dim, x, e, g, l, name, pert_node, pf))

        make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_$(name)_")
        movies && make_movie(x, e, g, joinpath(folder, "$(pf)_$(name)_Movie.mp4" |> jp); framerate)

        local l = 0.1

        name = "DBGNN"

        x, e = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, all, edge_pert)

        serialize(f, (;s, dim, x, e, g, l, name, pert_node, pf))

        make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_$(name)_")
        movies && make_movie(x, e, g, joinpath(folder, "$(pf)_$(name)_Movie.mp4" |> jp); framerate)

        name = "MPNN_edge_nl"

        x, e = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, step=MPNN_step_relu!, all, edge_pert)

        serialize(f, (;s, dim, x, e, g, l, name, pert_node, pf))

        make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_$(name)_")
        movies && make_movie(x, e, g, joinpath(folder, "$(pf)_$(name)_Movie.mp4" |> jp); framerate)

        name = "MPNN_basic"

        x, e  = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, step=MPNN_step!, all, edge_pert)

        serialize(f, (;s, dim, x, e, g, l, name, pert_node, pf))

        make_average_feature_plot(; x, g, pert_node, folder, prefix = "$(pf)_$(name)_")
        movies && make_movie(x, e, g, joinpath(folder, "$(pf)_$(name)_Movie.mp4" |> jp); framerate)

    end
end

close(f)
