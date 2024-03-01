"""
Local Topology Profile (LTP) model for graph classification. Based on https://github.com/j-adamczyk/LTP,
which is a code provided by the authors of the paper "Strengthening structural baselines for graph
classification using Local Topological Profile", which can be cited as:

@article{adamczyk2023strengthening,
  title={Strengthening structural baselines for graph classification using Local Topological Profile},
  author={Adamczyk, Jakub and Czech, Wojciech},
  journal={arXiv preprint arXiv:2305.00724},
  year={2023}
}
"""

from typing import Dict, List

import torch
import numpy as np
from networkx import DiGraph
from networkit.nxadapter import nx2nk
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_networkx
from torch_scatter import (
    scatter_min,
    scatter_max,
    scatter_mean,
    scatter_std,
    scatter_sum,
)

from master_thesis.classification_models.base_model import BaseModel, CLASSIC_CLASSIFIERS_MAP
from master_thesis.classification_models.utils import (
    calculate_shortest_paths,
    calculate_edge_betweenness,
    calculate_jaccard_index,
    calculate_local_degree_score,
)


def _extract_single_graph_features(
    data: Data,
    degree_sum: bool,
    shortest_paths: bool,
    edge_betweenness: bool,
    jaccard_index: bool,
    local_degree_score: bool,
) -> np.array:
    # adapted from PyTorch Geometric
    row, col = data.edge_index
    N = data.num_nodes

    # calculate out-degree features
    deg_out = degree(row, N, dtype=torch.float)
    deg_out_col = deg_out[col]

    deg_out_min, _ = scatter_min(deg_out_col, row, dim_size=N)
    deg_out_min[deg_out_min > 10000] = 0
    deg_out_max, _ = scatter_max(deg_out_col, row, dim_size=N)
    deg_out_max[deg_out_max < -10000] = 0
    deg_out_mean = scatter_mean(deg_out_col, row, dim_size=N)
    deg_out_stddev = scatter_std(deg_out_col, row, dim_size=N)

    # calculate in-degree features
    deg_in = degree(col, N, dtype=torch.float)
    deg_in_row = deg_in[row]

    deg_in_min, _ = scatter_min(deg_in_row, col, dim_size=N)
    deg_in_min[deg_in_min > 10000] = 0
    deg_in_max, _ = scatter_max(deg_in_row, col, dim_size=N)
    deg_in_max[deg_in_max < -10000] = 0
    deg_in_mean = scatter_mean(deg_in_row, col, dim_size=N)
    deg_in_stddev = scatter_std(deg_in_row, col, dim_size=N)

    ldp_features = {
        "deg_out": deg_out.numpy(),
        "deg_out_min": deg_out_min.numpy(),
        "deg_out_max": deg_out_max.numpy(),
        "deg_out_mean": deg_out_mean.numpy(),
        "deg_out_std": deg_out_stddev.numpy(),
        "deg_in": deg_in.numpy(),
        "deg_in_min": deg_in_min.numpy(),
        "deg_in_max": deg_in_max.numpy(),
        "deg_in_mean": deg_in_mean.numpy(),
        "deg_in_std": deg_in_stddev.numpy(),
    }

    if degree_sum:
        # calculate sum of out-degrees
        deg_out_sum = scatter_sum(deg_out_col, row, dim_size=N)
        deg_out_sum = deg_out_sum.numpy()
        ldp_features["deg_out_sum"] = deg_out_sum

        # calculate sum of in-degrees
        deg_in_sum = scatter_sum(deg_in_row, col, dim_size=N)
        deg_in_sum = deg_in_sum.numpy()
        ldp_features["deg_in_sum"] = deg_in_sum

    if any(
        [
            shortest_paths,
            edge_betweenness,
            jaccard_index,
            local_degree_score,
        ]
    ):
        graph = to_networkx(data, to_undirected=True)
        digraph = to_networkx(data, to_undirected=False)

        graph = nx2nk(graph)
        digraph = nx2nk(digraph)

        graph.indexEdges()
        digraph.indexEdges()

    if shortest_paths:
        sp_lengths = calculate_shortest_paths(digraph)
        ldp_features["shortest_paths"] = sp_lengths

    if edge_betweenness:
        eb = calculate_edge_betweenness(digraph)
        ldp_features["edge_betweenness"] = eb

    if jaccard_index:
        ji = calculate_jaccard_index(graph)
        ldp_features["jaccard_index"] = ji

    if local_degree_score:
        lds = calculate_local_degree_score(digraph)
        lds[lds > 1] = 1    # Range [0, 1], where smaller values denote more important edges
        ldp_features["local_degree_score"] = lds

    # make sure that all features have the same dtype
    ldp_features = {k: v.astype(np.float32) for k, v in ldp_features.items()}

    return ldp_features


def _aggregate_features(
    features_dict: Dict[str, np.ndarray],
    n_bins: int,
    log_degree: bool = False,
):
    x = np.empty(len(features_dict.keys()) * n_bins, dtype=np.float32)

    # features that use logarithm of values if log_degree is True
    log_features = [
        "deg_out",
        "deg_out_min",
        "deg_out_max",
        "deg_out_mean",
        "deg_out_sum",
        "deg_in",
        "deg_in_min",
        "deg_in_max",
        "deg_in_mean",
        "deg_in_sum",
    ]

    col_start = 0
    col_end = n_bins

    for feature_name, feature_values in features_dict.items():

        if log_degree is True and feature_name in log_features:
            # add small value to avoid problems with degree 0
            feature_values = np.log(feature_values + 1e-3)
        
        feature_values, _ = np.histogram(feature_values, bins=n_bins, density=False)

        x[col_start:col_end] = feature_values
        col_start += n_bins
        col_end += n_bins

    return x


class LTPModel(BaseModel):

    def __init__(
            self,
            device: str = 'cpu',
            degree_sum: bool = False,
            shortest_paths: bool = False,
            edge_betweenness: bool = True,
            jaccard_index: bool = True,
            local_degree_score: bool = True,
            log_degree: bool = False,
            n_bins: int = 10,
            classifier_type: str = 'random_forest',
            classifier_kwargs: Dict = {}
        ):
        self.device = device
        self.degree_sum = degree_sum
        self.shortest_paths = shortest_paths
        self.edge_betweenness = edge_betweenness
        self.jaccard_index = jaccard_index
        self.local_degree_score = local_degree_score
        self.log_degree = log_degree
        self.n_bins = n_bins
        self.classifier = CLASSIC_CLASSIFIERS_MAP[classifier_type](**classifier_kwargs)

    def _ltp_transform(self, X: List[Data]) -> List[Data]:
        X = [_aggregate_features(_extract_single_graph_features(
            data=x,
            degree_sum=self.degree_sum,
            shortest_paths=self.shortest_paths,
            edge_betweenness=self,
            jaccard_index=self.jaccard_index,
            local_degree_score=self.local_degree_score,
        ), n_bins=self.n_bins, log_degree=self.log_degree) for x in X]
        return X
    
    def fit(self, X: List[DiGraph], y: List[int]):
        X = [self.nx2geometric(self.device, x, x_attr=None, label=label) for x, label in zip(X, y)]
        X = self._ltp_transform(X)
        self.classifier.fit(X, y)

    def predict(self, X: List[DiGraph]):
        X = [self.nx2geometric(self.device, x, x_attr=None) for x in X]
        X = self._ltp_transform(X)
        y_hat = self.classifier.predict(X)
        return y_hat
