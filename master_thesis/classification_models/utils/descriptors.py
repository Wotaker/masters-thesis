import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from networkit.centrality import Betweenness
from networkit.distance import APSP
from networkit.graph import Graph
from networkit.linkprediction import JaccardIndex
from networkit.sparsification import LocalDegreeScore


@functional_transform('add_ones')
class AddOnes(BaseTransform):
    def __init__(self, dim: int = 1):
        self.dim = dim

    def forward(self, data: Data) -> Data:
        data.x = torch.ones((data.num_nodes, self.dim), dtype=torch.float)
        return data


@functional_transform('agile_local_degree_profile')
class LocalDegreeProfile(BaseTransform):

    def __init__(self):
        from torch_geometric.nn.aggr.fused import FusedAggregation
        self.aggr = FusedAggregation(['min', 'max', 'mean', 'std'])

    def forward(self, data: Data) -> Data:
        row, col = data.edge_index
        N = data.num_nodes

        deg_out = degree(row, N, dtype=torch.float).view(-1, 1)
        xs = [deg_out] + self.aggr(deg_out[col], row, dim_size=N)

        if data.is_directed():
            deg_in = degree(col, N, dtype=torch.float).view(-1, 1)
            xs += [deg_in] + self.aggr(deg_in[row], col, dim_size=N)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x] + xs, dim=-1)
        else:
            data.x = torch.cat(xs, dim=-1)

        return data


def calculate_shortest_paths(graph: Graph) -> np.array:
    # Networkit is faster than NetworkX for large graphs
    apsp = APSP(graph)
    apsp.run()
    path_lengths = apsp.getDistances(asarray=True)

    path_lengths = path_lengths.ravel()

    # filter out 0 length "paths" from node to itself
    path_lengths = path_lengths[np.nonzero(path_lengths)]

    # Networkit assigns extremely high values (~1e308) to mark infinite
    # distances for disconnected components, so we simply drop them
    path_lengths = path_lengths[path_lengths < 1e100]

    return path_lengths


def calculate_edge_betweenness(graph: Graph) -> np.ndarray:
    betweeness = Betweenness(graph, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    scores = np.array(scores, dtype=np.float32)
    return scores


def calculate_jaccard_index(graph: Graph) -> np.ndarray:
    jaccard_index = JaccardIndex(graph)
    scores = [jaccard_index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    return scores


def calculate_local_degree_score(graph: Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, dtype=np.float32)
