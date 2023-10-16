import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def load_networks(networks_dir: str, causal_coeff_strength: float = 1.5, verbose: bool = False):
    """
    Load networks from a directory, binarize them and convert them to networkx graphs.
    The binarization is done by thresholding the score of the networks, which is calculated
    as the mean of all networks plus a multiple of the standard deviation.

    Parameters
    ----------
    networks_dir : str
        _description_
    causal_coeff_strength : float, optional
        _description_, by default 1.5
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    dataset : list[networkx.DiGraph]
        _description_
    labels : numpy.ndarray
        _description_
    (n_control, n_patological) : tuple(int, int)
        _description_
    """

    # List of all networks
    np_files = os.listdir(networks_dir)

    # Get labels from file names
    labels = np.array([np_file[4:7] == 'PAT' for np_file in np_files])

    # Load all networks, and calculate mean score and std
    np_matrices = [np.load(os.path.join(networks_dir, np_file))[1] for np_file in np_files]
    score_mean, score_std = np.array(np_matrices).mean(), np.array(np_matrices).std()

    # Binarize networks
    np_networks = np_matrices > score_mean + causal_coeff_strength * score_std

    # Convert to networkx graphs
    dataset = [nx.from_numpy_array(np_network, create_using=nx.DiGraph) for np_network in np_networks]

    # Balance of control and patological samples
    n_control = len(labels) - labels.sum()
    n_patological = labels.sum()

    if verbose:
        print(f"No of control samples: {n_control}")
        print(f"No of patological samples: {n_patological}")
    
    return dataset, labels, (n_control, n_patological)