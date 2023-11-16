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
