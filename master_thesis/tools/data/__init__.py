from typing import Optional, Tuple, List

import os
import logging
import numpy as np
import networkx as nx

from master_thesis.tools.data.preprocessing import Preprocessing
from master_thesis.tools.data.synthetic_networks import scale_free_network
from master_thesis.tools.data.connectivity_dataset import ConnectivityDataset


def load_np_data(
        networks_dir: str,
        channel: Optional[int] = None,
        hem_connections: Optional[str] = None,
        with_filenames: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

    # TODO Extract a mock function that returns the same class with random labels

    # Get filenames
    filenames = [f for f in os.listdir(networks_dir) if f.endswith(".npy")]

    # Create paths
    paths = [os.path.join(networks_dir, f) for f in filenames]

    # Load networks to numpy arrays
    np_networks = [np.load(p) for p in paths]
    np_networks = [net[channel] if channel is not None else net for net in np_networks]
    for i, net in enumerate(np_networks):
        if net.sum() == 0:
            logging.warning(f"Network {filenames[i]} is empty, removing it from the dataset.")
            del np_networks[i]
            del filenames[i]
    np_networks = np.array(np_networks)

    # Filter hemispheric connections
    if hem_connections is not None:
        if hem_connections.upper() == "LL":
            np_networks = np_networks[:, :50, :50]
        elif hem_connections.upper() == "LR":
            np_networks = np_networks[:, :50, 50:]
        elif hem_connections.upper() == "RL":
            np_networks = np_networks[:, 50:, :50]
        elif hem_connections.upper() == "RR":
            np_networks = np_networks[:, 50:, 50:]
        else:
            raise ValueError(f"Invalid value for `hem_connections`: {hem_connections}, choose from 'LL', 'LR', 'RL' or 'RR'")


    # Extract labels
    labels = [int(x.split("-")[1][:3] == "PAT") for x in filenames]
    labels = np.array(labels)

    # Return dataset
    if with_filenames:
        return np_networks, labels, filenames
    return np_networks, labels


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

