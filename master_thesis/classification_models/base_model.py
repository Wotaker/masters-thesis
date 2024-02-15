from abc import abstractmethod
from typing import Optional

import torch
import networkx as nx
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from torch_geometric.data import Data

from master_thesis.tools.plots import plot_confusion_matrix


CLASSIC_CLASSIFIERS_MAP = {
    "svc": SVC,
    "mlp": MLPClassifier,
    "ridge": RidgeClassifier,
    "random_forest": RandomForestClassifier,
}


def evaluate(y_gold, y_hat):

    print(f"Train accuracy: {accuracy_score(y_gold, y_hat):.2f}")
    print(f"Train recall: {recall_score(y_gold, y_hat):.2f}")
    print(f"Train precision: {precision_score(y_gold, y_hat):.2f}")
    print(f"Train f1: {f1_score(y_gold, y_hat):.2f}")
    print(f"Train AUC: {roc_auc_score(y_gold, y_hat):.2f}")
    plot_confusion_matrix(y_gold, y_hat)


def evaluate_train_test(y_train, y_train_hat, y_test, y_test_hat):

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
        X : List[networkx.DiGraph]
            Array of directed connectivity networks.
        y : List[int]
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
        X : List[networkx.DiGraph]
            Array of directed connectivity networks.
        
        Returns
        -------
        List[int]
            Array of predicted labels.
        """
        pass

    @staticmethod
    def nx2geometric(
        device: str,
        nx_graph: nx.DiGraph,
        x_attr: Optional[torch.Tensor] = None,
        label: Optional[int] = None,
    ) -> Data:
        """
        Converts a networkx graph to a PyTorch Geometric Data object.
        """

        return Data(
            edge_index=torch.tensor(list(nx_graph.edges)).t().contiguous(),
            x=x_attr,
            y=torch.tensor([label], dtype=torch.long) if label is not None else None,
            num_nodes=len(nx_graph.nodes),
            num_edges=len(nx_graph.edges)
        ).to(torch.device(device))
    
    @staticmethod
    def evaluate(y_gold, y_hat):
        evaluate(y_gold, y_hat)

    def predict_and_evaluate(self, X_train, y_train, X_test, y_test):
        y_train_hat = self.predict(X_train)
        y_test_hat = self.predict(X_test)
        evaluate_train_test(y_train, y_train_hat, y_test, y_test_hat)



