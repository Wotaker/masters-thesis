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
        ) -> None:

        assert connection_weight_threshold is None or connection_significance_threshold is None, \
            "Either connection_weight_threshold or connection_significance_threshold can be set, but not both."
        
        self.connection_weight_threshold = connection_weight_threshold
        self.connection_significance_threshold = connection_significance_threshold

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

        # Return networkx networks and labels
        return nx_networks, labels