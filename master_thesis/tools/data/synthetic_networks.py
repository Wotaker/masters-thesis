from typing import Dict

import random
import networkx as nx


def scale_free_network(network_config: Dict, seed: int = 42) -> nx.DiGraph:

    def shuffle_digraph(g: nx.DiGraph) -> nx.DiGraph:

        # Collect edges
        new_edges = []
        for u, v in g.edges():
            new_edges.append((u, v))

        # Shuffle edges
        random.Random(seed).shuffle(new_edges)

        # Create a new graph and return it
        return nx.DiGraph(new_edges)

    # Create initial graph
    type_initial = network_config.pop("initial_network")
    n_initial = network_config.pop("initial_network_size")
    if type_initial == "clique":
        g_init = nx.MultiDiGraph(nx.complete_graph(9))
    elif type_initial == "ring":
        g_init = nx.MultiDiGraph([(i, (i+1)%n_initial) for i in range(n_initial)])
    else:
        print("Unsuported initial type, choose from [clique, ring]")
        exit(1)
    
    # Create a scale-free directed graph
    g = nx.scale_free_graph(**network_config, initial_graph=g_init, seed=seed)

    # Convert multigraph to a simple graph
    g = nx.DiGraph(g)

    # Remove self-loops
    g.remove_edges_from(nx.selfloop_edges(g))

    # Reorder nodes
    g = shuffle_digraph(g)

    # Return
    return g
