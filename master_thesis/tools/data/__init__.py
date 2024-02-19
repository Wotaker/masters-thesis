from typing import Optional, Tuple, List

import os
import numpy as np
import networkx as nx

from master_thesis.tools.data.preprocessing import Preprocessing
from master_thesis.tools.data.synthetic_networks import scale_free_network
from master_thesis.tools.data.connectivity_dataset import ConnectivityDataset


def load_np_data(networks_dir: str, channel: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:

    # Get filenames
    filenames = [f for f in os.listdir(networks_dir) if f.endswith(".npy")]

    # Create paths
    paths = [os.path.join(networks_dir, f) for f in filenames]

    # Load networks to numpy arrays
    np_networks = [np.load(p) for p in paths]
    np_networks = [net[channel] if channel is not None else net for net in np_networks]

    # Extract labels
    labels = [int(x.split("-")[1][:3] == "PAT") for x in filenames]

    return np_networks, labels

