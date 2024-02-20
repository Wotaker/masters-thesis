from typing import Optional, List, Tuple

import torch
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from master_thesis.classification_models.base_model import BaseModel, plot_confusion_matrix


def draw_network(
        g: nx.DiGraph,
        axis: Optional[plt.Axes]=None,
        node_size: int=5, 
        width: float=0.1, 
        with_labels: bool=False
    ):
    nx.draw(
        g, pos=nx.circular_layout(g), ax=axis,
        node_size=node_size, width=width, with_labels=with_labels
    )

def plot_weight_histogram(network: nx.DiGraph, label: int, bins: int=100):
    weights = torch.tensor(list(nx.get_edge_attributes(network, 'weight').values()), dtype=torch.float)
    datapoint = BaseModel.nx2geometric("cpu", network, x_attr=weights)
    sns.histplot(datapoint.x.numpy(), kde=True, bins=bins)
    plt.title(f"Weights distribution ({'CON' if label == 0 else 'PAT'} sample)")
    plt.show()


def plot_sample_networks(
    networks: List[nx.DiGraph],
    labels: List[int],
    rows: int=4,
    figsize: Optional[Tuple[int]]=None,
    save_path: Optional[str]=None
):

    # Get indices of each class
    labels = np.array(labels)
    labels_pat = np.where(labels == True)[0]
    labels_con = np.where(labels == False)[0]
    indices_pat = np.random.choice(len(labels_pat), rows, replace=False)
    indices_con = np.random.choice(len(labels_con), rows, replace=False)

    # Define plot
    fig, axes = plt.subplots(rows, 2, figsize=figsize if figsize else (6, 3 * rows))
    for row in range(rows):

        # Get axes
        if rows == 1:
            ax_pat, ax_con = axes[0], axes[1]
        else:
            ax_pat, ax_con = axes[row, 0], axes[row, 1]

        # Draw pathological sample
        draw_network(networks[indices_pat[row]], axis=ax_pat)
        ax_pat.set_title("Pathological")

        # Draw control sample
        draw_network(networks[indices_con[row]], axis=ax_con)
        ax_con.set_title("Control")
    
    # Save or show plot
    plt.tight_layout()
    plt.savefig(save_path) if save_path else plt.show()


def plot_2d_embeddings(embeddings, labels, save_path=None):
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(embeddings[labels == 0, 0], embeddings[labels == 0, 1], c='tab:blue', s=5, alpha=0.5, label='control')
    ax.scatter(embeddings[labels == 1, 0], embeddings[labels == 1, 1], c='tab:orange', s=5, alpha=0.5, label='patological')
    ax.set_title('Graph2Vec embeddings')
    plt.legend()
    plt.savefig(save_path) if save_path else plt.show()