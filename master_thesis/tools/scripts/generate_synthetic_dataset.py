"""
Script to generate a synthetic dataset of directed networks.
"""

from argparse import ArgumentParser
from typing import Dict, List
from copy import copy

import os
import json
import numpy as np
import networkx as nx
from tqdm import tqdm

from master_thesis.tools.data import scale_free_network


NETWORK_GENERATORS = {
    "scale_free": scale_free_network
}


def normalize_distribution(x: List, n: int):
    return np.array(x) / sum(x) if x else np.ones(n) / n


def generate_class_representatives(size: int, network_config: Dict, initial_seed: int):
    
    # Get class label and network type
    label = network_config.pop("label")
    network_type = network_config.pop("network_type")

    # Iterate over the number of samples
    seed = initial_seed
    for i in tqdm(range(size)):
        seed += 1

        # Generate network
        network = NETWORK_GENERATORS[network_type](copy(network_config), seed=seed)

        # Save network
        np.save(
            os.path.join(out_dir, f"sub-{label}{i}.npy"),
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
    for ratio, config in tqdm(zip(class_distribution, class_configs), total=len(class_configs)):
        network_config = json.loads(open(config, "r").read())
        generate_class_representatives(int(ratio * dataset_size), network_config, seed)
