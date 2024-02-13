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


def load_networks(networks_dir: str, causal_coeff_strength: float = 1.5, verbose: bool = False):
    """
    Load effective connectivity networks from a directory, binarize them and convert them to
    networkx graphs. The binarization is done by thresholding the score of the networks,
    which is calculated as the mean of all networks plus a multiple of the standard deviation.

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


def get_balanced_train_test_masks(labels: np.ndarray, half_test_size, seed: int = 42):

    # Set seed for reproducibility
    np.random.seed(seed)

    # Get indices of control and pathology subjects
    indices = np.arange(len(labels))
    indices_con, indices_pat = indices[labels == 0], indices[labels == 1]

    # Get balanced test mask as holdout set
    mask_test = np.concatenate((
        np.random.choice(indices_con, size=half_test_size, replace=False),
        np.random.choice(indices_pat, size=half_test_size, replace=False)
    ))

    # Remove test indices from train set
    indices_con_train = np.setdiff1d(indices_con, mask_test)
    indices_pat_train = np.setdiff1d(indices_pat, mask_test)

    # Get balanced train mask by resampling from remaining indices
    half_train_size = max(len(indices_con) - half_test_size, len(indices_pat) - half_test_size)
    mask_train = np.concatenate((
        np.random.choice(indices_con_train, size=half_train_size, replace=(len(indices_con) < len(indices_pat))),
        np.random.choice(indices_pat_train, size=half_train_size, replace=(len(indices_pat) < len(indices_con)))
    ))

    return mask_train, mask_test



def balance_split_dataset(
        X, y, half_test_size, verbose: bool = False, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into train and test sets, keeping both classes balanced. The balance in test set is
    achieved by holding out the same number of examples from each class (`half_test_size`). Whereas the
    balance in train set is achieved by resampling with replacement from the minority class.

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    half_test_size : _type_
        _description_
    verbose : bool, optional
        _description_, by default False
    seed : int, optional
        _description_, by default 42

    Returns
    -------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_test : _type_
        _description_
    y_test : _type_
        _description_
    """

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
