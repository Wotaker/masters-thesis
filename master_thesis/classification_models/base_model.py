from abc import abstractmethod
from typing import Optional

import torch
import networkx as nx
from sklearn.metrics import *
from torch_geometric.data import Data

from master_thesis.tools.plots import plot_confusion_matrix


def evaluate(y_train, y_train_hat, y_test, y_test_hat):

    print(f"Train accuracy: {accuracy_score(y_train, y_train_hat):.2f},\t\tTest accuracy: {accuracy_score(y_train, y_train_hat):.2f}")
    print(f"Train recall: {recall_score(y_train, y_train_hat):.2f}, Test recall: {recall_score(y_test, y_test_hat):.2f}")
    print(f"Train precision: {precision_score(y_train, y_train_hat):.2f}, Test precision: {precision_score(y_test, y_test_hat):.2f}")
    print(f"Train f1: {f1_score(y_train, y_train_hat):.2f}, Test f1: {f1_score(y_test, y_test_hat):.2f}")
    print(f"Train AUC: {roc_auc_score(y_train, y_train_hat):.2f}, Test AUC: {roc_auc_score(y_test, y_test_hat):.2f}")
    plot_confusion_matrix(y_test, y_test_hat)


class BaseModel():

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : Array[networkx.DiGraph]
            Array of directed connectivity networks.
        y : Array[int]
            Array of labels.

        Returns
        -------
        self : BaseModel
            Returns the instance of the fitted model.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Returns the model predictions for the given data.

        Parameters
        ----------
        X : Array[networkx.DiGraph]
            Array of directed connectivity networks.
        
        Returns
        -------
        Array[int]
            Array of predicted labels.
        """
        pass

    @staticmethod
    def nx2geometric(
        device: str,
        nx_graph: nx.DiGraph,
        label: Optional[int] = None,
        x_attr: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Converts a networkx graph to a PyTorch Geometric Data object.
        """

        return Data(
            edge_index=torch.tensor(list(nx_graph.edges)).t().contiguous(),
            x=x_attr,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=len(nx_graph.nodes),
            num_edges=len(nx_graph.edges)
        ).to(torch.device(device))

    def evaluate(self, X_train, y_train, X_test, y_test):
        y_train_hat = self.predict(X_train)
        y_test_hat = self.predict(X_test)
        evaluate(y_train, y_train_hat, y_test, y_test_hat)



