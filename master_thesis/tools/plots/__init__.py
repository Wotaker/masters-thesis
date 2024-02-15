from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import confusion_matrix


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


def plot_confusion_matrix(y_true, y_pred, save_path=None):

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.savefig(save_path) if save_path else plt.show()
    plt.clf()


def plot_2d_embeddings(embeddings, labels, save_path=None):
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(embeddings[labels == 0, 0], embeddings[labels == 0, 1], c='tab:blue', s=5, alpha=0.5, label='control')
    ax.scatter(embeddings[labels == 1, 0], embeddings[labels == 1, 1], c='tab:orange', s=5, alpha=0.5, label='patological')
    ax.set_title('Graph2Vec embeddings')
    plt.legend()
    plt.savefig(save_path) if save_path else plt.show()