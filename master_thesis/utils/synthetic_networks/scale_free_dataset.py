from argparse import ArgumentParser
from typing import Dict, List
from copy import copy

import os
import json
import random
import numpy as np
import networkx as nx


def normalize_distribution(x: List, n: int):
    return np.array(x) / sum(x) if x else np.ones(n) / n


def draw_network(g: nx.DiGraph, with_labels=False):
    nx.draw(g, pos=nx.circular_layout(g), node_size=5, width=0.1, with_labels=with_labels)


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
    type_initial = network_config.pop("type_initial")
    n_initial = network_config.pop("n_initial")
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


def generate_class_representatives(size: int, network_config: Dict, initial_seed: int):
    
    # Get class label
    label = network_config.pop("label")

    # Iterate over the number of samples
    seed = initial_seed
    for i in range(size):
        seed += 1

        # Generate network
        network = scale_free_network(copy(network_config), seed=seed)

        # Save network
        np.save(
            os.path.join(out_dir, f"{label}_{i}.npy"),
            nx.to_numpy_array(network)
        )


if __name__ == "__main__":

    # Load dataset configuration from config file
    parser = ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=1000)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-c", "--class_configs", type=str, required=True, nargs="*")
    parser.add_argument("-d", "--class_distribution", type=float, nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Extract arguments
    seed = args.seed
    verbose = args.verbose
    dataset_size = args.size
    out_dir = args.output_dir
    class_configs = args.class_configs
    class_distribution = normalize_distribution(args.class_distribution, len(args.class_configs))

    # Print configuration
    if verbose:
        print(f"Config files:       {class_configs}")
        print(f"Class distribution: {class_distribution}")

    # Generate dataset
    for ratio, config in zip(class_distribution, class_configs):
        network_config = json.loads(open(config, "r").read())
        generate_class_representatives(int(ratio * dataset_size), network_config, seed)
