using Graphs
using Plots
using LinearAlgebra
using Statistics
using LaTeXStrings

using Random

Random.seed!(42)

relu(x, l) = x > 0. ? x : l .* x
norm_xe!(x, e) = (s = 0.5 .* (mean(abs.(x)) + mean(abs.(e))); (x ./= s, e ./= s))

@views function DB_step!(xt1, et1, x, e, W, W_p, β, g, l)
    xt1 .= x .- x * β
    for (i, ed) in enumerate(edges(g))
        et1[i, :] .= relu.(e[i, :] .+ β * e[i, :] .+ W * (x[src(ed), :] .- x[dst(ed), :]), l)
        xt1[src(ed), :] .+= W_p * e[i, :]
    end
    xt1 .= relu.(xt1, l)
    et1 .= relu.(et1, l)
    norm_xe!(xt1, et1)
end

function MPNN_step_relu!(xt1, et1, x, e, W, W_p, β, g, l)
    xt1 .= x .- x * β
    for (i, ed) in enumerate(edges(g))
        em = relu.(W * (x[src(ed), :] .- x[dst(ed), :]), l)
        xt1[src(ed), :] .+= W_p * em
    end
    xt1 .= relu.(xt1, l)
    norm_xe!(xt1, et1)
end

function MPNN_step!(xt1, et1, x, e, W, W_p, β, g, l)
    xt1 .= x .- x * β
    for (i, ed) in enumerate(edges(g))
        em = W * (x[src(ed), :] .- x[dst(ed), :])
        xt1[src(ed), :] .+= W_p * em
    end
    xt1 .= relu.(xt1, l)
    norm_xe!(xt1, et1)
end

function GCN_step!(xt1, et1, x, e, W, W_p, β, B, l)
    D = Diagonal(B * B')
    D12 = sqrt(D)
    xt1 .= relu.(D12 * (B * B' - D) * D12 * x * W_p, l)
    norm_xe!(xt1, [1.])
end

node_in_edge(node) = edge -> (src(edge) == node) || (dst(edge) == node)

function make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, step=DB_step!, all = false, edge_pert=true)
    x = zeros(T, nv(g), dim)
    e = zeros(T, ne(g), dim)
    for pn in pert_node
        x[1, pn, :] = rand(dim)
        if edge_pert
            e_pert = rand(findall(node_in_edge(pn), collect(edges(g))))
            e[1, e_pert, :] = rand(dim)
        end
    end

    if all
        x[1, :, :] .= rand(size(B)[1], dim)
    end
    for t in 2:T
        @views step(x[t, :, :], e[t, :, :], x[t-1, :, :], e[t-1, :, :], W, W_p, β, g, l)
    end
    x, e
end

function make_log_trajectory_plots(; x, g, pert_node, folder)
    dists = bellman_ford_shortest_paths(g, pert_node).dists
    # @show maximum(dists)
    sort_idx = sortperm(dists)

    max_responses = [findmax(x[end, :, i])[2] for i in 1:dim]
    max_dist, max_dist_feature = findmax(dists[max_responses])
    cap(x) = x < -10. ? NaN : x
    hs_log(x) = x <= 0. ? -Inf : log(x)
    hlc(x) = hs_log(x) |> cap

    mkpath(joinpath(folder, "log"))

    for f in 1:dim
        cbar_t = LaTeXString("\n\$log(x^$(f)_n(t))\$")
        heatmap(hlc.(x[1:end, sort_idx, f]'),
        xlabel="t in steps",
        ylabel="Nodes (ascending d(n, 1))",
        colorbar_title=cbar_t,
        right_margin = 10Plots.mm,
        titlefontsize = 10,
        top_margin = 5Plots.mm,
        title="Feature $f\nResponse to random signal at node $pert_node.\nMaximum response at distance $(dists[max_responses[f]])")
        savefig(joinpath(folder, "log", "spreading_from_node_$(pert_node)_feature_$(f).png"))
    end

    dists, max_responses, max_dist, max_dist_feature
end


function make_tanh_trajectory_plots(; x, g, pert_node, folder, prefix="")
    dists = bellman_ford_shortest_paths(g, pert_node).dists
    # @show maximum(dists)
    sort_idx = sortperm(dists)

    max_responses = [findmax(x[end, :, i])[2] for i in 1:dim]
    max_dist, max_dist_feature = findmax(dists[max_responses])

    mkpath(folder)

    for f in 1:dim
        cbar_t = LaTeXString("\n\$tanh(x^$(f)_n(t))\$")
        heatmap(tanh.(x[1:end, sort_idx, f]'),
        xlabel="t in steps",
        ylabel="Nodes (ascending d(n, 1))",
        colorbar_title=cbar_t,
        right_margin = 10Plots.mm,
        titlefontsize = 10,
        top_margin = 5Plots.mm,
        title="Feature $f\nResponse to random signal at node $pert_node.\nMaximum response at distance $(dists[max_responses[f]])")
        savefig(joinpath(folder, prefix * "spreading_from_node_$(pert_node)_feature_$(f).png"))
    end

    dists, max_responses, max_dist, max_dist_feature
end



function make_average_feature_plot(; x, g, pert_node, folder, prefix="", squeeze_f = tanh)
    dists = bellman_ford_shortest_paths(g, pert_node).dists
    # @show maximum(dists)
    sort_idx = sortperm(dists)

    mkpath(folder)

    # cbar_t = LaTeXString("\n\$tanh(\\langle x^f_n(step)\\rangle_f)\$")
    cbar_t = LaTeXString("\n\$ \\textrm{tanh}(\\langle x^f_n(\\textrm{step})\\rangle_f)\$")
    mean_f = mean(x[1:end, sort_idx, :], dims=3)[:, :, 1]
    Plots.heatmap(squeeze_f.(mean_f'),
    xlabel="step",
    ylabel="sorted nodes",
    colorbar_title=cbar_t,
    colorbar_size=100,
    clims=(-1., 1.),
    left_margin = left_margin,
    right_margin = right_margin,
    titlefontsize = 10,
    top_margin = 5Plots.mm,
    bottom_margin = bottom_margin,
    tickfontsize=tickfontsize,
    guidefontsize=guidefontsize,
    legendfontsize=legendfontsize,
    colorbar_titlefontsize=guidefontsize,
    # title="Response to random signal at node $pert_node."
    dpi=600,
    )
    savefig(joinpath(folder, prefix * "spreading_from_node_$(pert_node)_mean_feature.png"))
end



function make_dirichlet_energy(; x, g, T, folder, prefix="")
    L = laplacian_matrix(g)

    @views D = [tr(x[t, :, :]' * L * x[t, :, :])/tr(x[t, :, :]' * x[t, :, :]) for t in 1:T]

    mkpath(folder)

    plot(D,
    xlabel="t in steps",
    ylabel="Dirichlet Energy")
    savefig(joinpath(folder, prefix * "Dirichlet_energy.png"))
end
