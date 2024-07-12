Random.seed!(42)

gs = (grid = (DiGraph(Graphs.grid([5, 20])), "grid"),
path_graph = (DiGraph(Graphs.path_graph(71)), "path_graph"),
ladder_graph = (DiGraph(Graphs.ladder_graph(50)), "ladder_graph"),
random_regular = (DiGraph(random_regular_graph(100, 4)), "random_regular"),
watts_strogatz = (DiGraph(watts_strogatz(200, 6, 0.04)), "watts_strogatz"),
cycle_graph = (DiGraph(Graphs.cycle_graph(71)), "cycle_graph"),
)

