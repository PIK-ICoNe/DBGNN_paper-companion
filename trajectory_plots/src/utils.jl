function prepare_data(s; dim, g, Δ_init, gf, l, oscillatory=false, make_folder=true)

    Random.seed!(s)

    B = incidence_matrix(g; oriented=true)
    T = 2500
    pert_node = 1
    W = Δ_init .* randn(dim, dim)
    if oscillatory
        W_p = - W'
        @assert dim ÷ 2 * 2 == dim
        β_upper = 0.1 .* Δ_init .* randn(dim ÷ 2, dim ÷ 2)
        β = [zeros(dim ÷ 2, dim ÷ 2)  β_upper ;
             -β_upper' zeros(dim ÷ 2, dim ÷ 2)]
        of = "_oscillatory"
    else
        W_p = Δ_init .* randn(dim, dim)
        β = 0.1 .* Δ_init .* randn(dim, dim)
        of = ""
    end
    if make_folder == true
        folder = joinpath("plots", gf, "l-$(l)_Δ_init-$(Δ_init)_dim-$(dim)"*of)
    else
        folder = false
    end
    B, β, T, pert_node, W, W_p, folder 
end

function distance_excitation_one_grid(g, T, W, W_p, β, B, l, dim, step)
    nv = size(g.fadjlist,1)
    max_distances = Array{Float64,1}(undef, nv)
    for pert_node in 1:nv
        x,e = make_trajectory(; g, T, W, W_p, β, B, l, dim, pert_node, step = step)
        max_distances[pert_node] = get_max_distance_excitation(g, x, pert_node)
    end
    return max_distances
end


function get_max_distance_excitation(g, x, pert_node)
    max_responses = [findmax(x[end, :, i])[2] for i in 1:dim]
    dists = bellman_ford_shortest_paths(g, pert_node).dists
    max_dist, max_dist_feature = findmax(dists[max_responses])
    return max_dist
end