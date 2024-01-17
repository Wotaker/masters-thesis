from typing import Optional, Tuple, Union

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import LocalDegreeProfile


class ConnectivityDataset(Dataset):
    """
    Loads the effective connectivity networks to a PyTorch Geometric Dataset.

    Parameters
    ----------
    networks_dir : str
        Path to the directory containing the networks.
    causal_coeff_strength : Optional[float]
        Regulates the strength of the causal coefficients to be considered as edges.
    causal_coeff_threshold : Optional[float]
        Explicit threshold for the causal coefficients to be considered as edges.
    device : str
        Device type to which the data is loaded. Choose from 'cpu', 'cuda' and 'mps', by default 'cpu'.

    Returns
    -------
    ConnectivityDataset : Dataset
        PyTorch Geometric Dataset containing the networks.
    """

    @staticmethod
    def nx2geometric(nx_graph: nx.DiGraph, label: int, device: str, ldp: bool = False) -> Data:
        """
        Converts a networkx graph to a PyTorch Geometric Data object.
        """

        data = Data(
            x= None if ldp else torch.ones(len(nx_graph.nodes), 1),
            edge_index=torch.tensor(list(nx_graph.edges)).t().contiguous(),
            edge_attr=torch.tensor(list(nx.get_edge_attributes(nx_graph, 'weight').values()), dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long)
        ).to(torch.device(device))

        assert data.edge_index.shape == (2, len(nx_graph.edges)), f"Edge index has wrong shape ({data.edge_index.shape})"

        return LocalDegreeProfile()(data) if ldp else data

    def __init__(
            self,
            networks_dir: str,
            causal_coeff_strength: Optional[float] = None,
            causal_coeff_threshold: Optional[float] = None,
            ldp: bool = False,
            device: str = 'cpu'
        ):

        assert causal_coeff_strength is not None or causal_coeff_threshold is not None, \
            "Either causal_coeff_strength or causal_coeff_threshold must be set"

        self.networks_dir = networks_dir
        self.causal_coeff_strength = causal_coeff_strength
        self.causal_coeff_threshold = causal_coeff_threshold
       
       # List of all networks
        np_files = os.listdir(self.networks_dir)

        # Get labels from file names
        self.labels = np.array([np_file[4:7] == 'PAT' for np_file in np_files])

        # Load all networks to numpy networks
        np_networks = [np.load(os.path.join(networks_dir, np_file))[1] for np_file in np_files]

        # Preprocess networks
        if causal_coeff_strength is not None:
            score_mean, score_std = np.array(np_networks).mean(), np.array(np_networks).std()
            np_networks = np_networks > score_mean + self.causal_coeff_strength * score_std
        else:
            np_networks = np.where(np.array(np_networks) < self.causal_coeff_threshold, 0, np_networks)
        
        # Convert to networkx graphs
        nx_networks = [nx.from_numpy_array(np_network, create_using=nx.DiGraph) for np_network in np_networks]
        self.nx_dataset = [self.nx2geometric(g, l, device, ldp) for g, l in zip(nx_networks, self.labels)]

        # Balance of control and patological samples
        self.n_control = len(self.labels) - self.labels.sum()
        self.n_patological = self.labels.sum()

        # Shuffle the dataset (to avoid pattern in labels)
        self.shuffle()
    
    def subjects_info(self):
        return {"Control": self.n_control, "Patological": self.n_patological}
    
    def weight_histogram(self, idx):
        graph = self.get(idx)
        sns.histplot(graph.edge_attr.numpy(), kde=True, bins=100)
        idx_label = "control" if self.labels[idx] == 0 else "patological"
        plt.title(f"Weights distribution ({idx_label} sample)")
        plt.show()
    
    def shuffle(self, perm: Optional[Union[Tensor, np.ndarray]] = None) -> None:
        if perm is None:
            perm = torch.randperm(len(self.nx_dataset))
        self.nx_dataset = [self.nx_dataset[idx] for idx in perm]
        self.labels = self.labels[perm]
    
    def len(self) -> int:
        return len(self.nx_dataset)
    
    def get(self, idx) -> Data:
        return self.nx_dataset[idx]
    
    def __getitem__(self, idx) -> Data:
        return self.nx_dataset[idx]
    
    def __len__(self) -> int:
        return self.len()


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


def balance_split_dataset(X, y, half_test_size, verbose: bool = False, seed: int = 42):

    # Set seed for reproducibility
    np.random.seed(seed)

    # Split the dataset into control and patological
    X_con, X_pat = X[y == 0], X[y == 1]

    # Holdout the test set, keep both classes balanced by taking N of each
    X_test = np.concatenate((X_con[-half_test_size:], X_pat[-half_test_size:]))
    y_test = np.concatenate((np.zeros(half_test_size), np.ones(half_test_size)))

    # Shuffle the test set (to avoid pattern in labels)
    permutation = np.random.permutation(2 * half_test_size)
    X_test, y_test = X_test[permutation], y_test[permutation]

    # The rest is the training set, keep both classes balanced by resampling
    half_train = max(len(X_con) - half_test_size, len(X_pat) - half_test_size)
    X_train = np.concatenate((
        X_con[np.random.choice(X_con[:-half_test_size].shape[0], size=half_train, replace=True)],
        X_pat[np.random.choice(X_pat[:-half_test_size].shape[0], size=half_train, replace=True)]
    ))
    y_train = np.concatenate((np.zeros(half_train), np.ones(half_train)))

    # Shuffle the train set (to avoid pattern in labels)
    permutation = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[permutation], y_train[permutation]

    # Print info about the dataset
    if verbose:
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Ratio of positive samples in train: {y_train.mean()}")
        print(f"Ratio of positive samples in test: {y_test.mean()}")
    
    return X_train, y_train, X_test, y_test
