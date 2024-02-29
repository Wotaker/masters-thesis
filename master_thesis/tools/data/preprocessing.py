from typing import Optional, Tuple, List, Callable

import numpy as np
import networkx as nx

class Preprocessing():
    """
    Preprocesses the effective connectivity networks. Define the parameters in the constructor and call
    the object with the numpy networks and labels to perform the preprocessing and convert the networks to
    networkx format.

    Parameters
    ----------
    connection_weight_threshold : Optional[float]
        Weights below the threshold are set to zero.
    connection_significance_threshold : Optional[float]
        Only weights that varies more than the mean plus the significance threshold times the standard deviation
        are considered as edges.
    undirected : bool
        If True, the networks are converted to undirected graphs.
    
    Examples
    --------
    Define the preprocessing object and call it with the networks and labels:

    >>> np_networks = ...
    >>> labels = ...
    >>> preprocessing = Preprocessing(connection_weight_threshold=0.5)
    >>> nx_networks, labels = preprocessing(np_networks, labels)

    """

    def __init__(
            self,
            connection_weight_threshold: Optional[float] = None,
            connection_significance_threshold: Optional[float] = None,
            undirected: bool = False,
            shuffle: bool = True,
            seed: Optional[int] = None
        ) -> None:

        assert connection_weight_threshold is None or connection_significance_threshold is None, \
            "Either connection_weight_threshold or connection_significance_threshold can be set, but not both."
        
        self.connection_weight_threshold = connection_weight_threshold
        self.connection_significance_threshold = connection_significance_threshold
        self.undirected = undirected
        self.shuffle = shuffle
        self.seed = seed

    def __call__(self, np_networks: List[np.ndarray], labels: List[int]) -> Tuple[List[nx.DiGraph], List[int]]:
        
        # Cast to numpy array
        np_networks = np.stack(np_networks)
        self.score_mean, self.score_std = np_networks.mean(), np_networks.std()
        
        # Preprocess networks
        if self.connection_weight_threshold is not None:
            np_networks = np.where(np_networks < self.connection_weight_threshold, 0, np_networks)
        elif self.connection_significance_threshold is not None:
            np_networks = np.where(np_networks < self.score_mean + self.connection_significance_threshold * self.score_std, 0, np_networks)
            # np_networks = np_networks > self.score_mean + self.connection_significance_threshold * self.score_std

        # Convert to networkx
        nx_networks = [nx.from_numpy_array(net, create_using=nx.DiGraph) for net in np_networks]

        # Convert to undirected
        if self.undirected:
            nx_networks = [net.to_undirected() for net in nx_networks]
        
        # Shuffle data
        if self.seed is not None:
                np.random.seed(self.seed)
        if self.shuffle:
            idx = np.arange(len(nx_networks))
            np.random.shuffle(idx)
            nx_networks = [nx_networks[i] for i in idx]
            labels = [labels[i] for i in idx]

        # Return networkx networks and labels
        return nx_networks, labels