

using GLMakie
using GraphMakie,NetworkLayout

function make_movie(x, e, g, filename;
        T = size(x)[1],
        framerate = T ÷ 60,
        squeeze_f = tanh,
        edge_width = 2,
        node_size = 20)

    ecolor = Observable(ones(ne(g)))
    ncolor = Observable(ones(nv(g)))
    # Shell(; nlist=[1:3:nv(g), 2:3:nv(g), 3:3:nv(g)])
    fig, ax, p = graphplot(g; layout=Stress(),
                       node_color = ncolor,
                       edge_color = ecolor,
                       edge_width,
                       node_size,
                       edge_attr=(; colorrange = (-1., 1.)),
                       node_attr=(; colorrange = (-1., 1.)),
                       curve_distance_usage=false,
                       arrow_show=false,
                       )

    hidespines!(ax); hidedecorations!(ax) # just graph, no axis

    ##
    frameiter = 1:T

    mean_xf = mean(x[frameiter, :, :], dims=3)[:, :, 1] .|> tanh
    mean_ef = mean(e[frameiter, :, :], dims=3)[:, :, 1] .|> tanh

    ##

    record(fig, filename, frameiter; framerate) do t
        ncolor[] = mean_xf[t, :]
        ecolor[] = mean_ef[t, :]
        notify(ecolor)
        notify(ncolor)
    end
end
