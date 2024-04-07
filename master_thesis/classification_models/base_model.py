from abc import abstractmethod
from typing import Optional
from dataclasses import dataclass

import torch
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import ndarray as Array
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx


CLASSIC_CLASSIFIERS_MAP = {
    "svc": SVC,
    "mlp": MLPClassifier,
    "ridge": RidgeClassifier,
    "random_forest": RandomForestClassifier,
}


@dataclass
class EvaluationScores:
    accuracy: float
    recall: float
    precision: float
    f1: float
    auc: float

    def __str__(self):
        representation = ""
        representation += f"Accuracy:  {self.accuracy:.2f}\n"
        representation += f"Recall:    {self.recall:.2f}\n"
        representation += f"Precision: {self.precision:.2f}\n"
        representation += f"f1 score:  {self.f1:.2f}\n"
        representation += f"AUC score: {self.auc:.2f}\n"
        return representation


def plot_confusion_matrix(y_true: Array, y_pred: Array, save_path: Optional[str] = None):

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.savefig(save_path) if save_path else None
    plt.clf();


def evaluate(y_gold: Array, y_hat: Array, save_path: Optional[str]) -> EvaluationScores:

    evaluation_scores = EvaluationScores(
        accuracy=accuracy_score(y_gold, y_hat),
        recall=recall_score(y_gold, y_hat),
        precision=precision_score(y_gold, y_hat),
        f1=f1_score(y_gold, y_hat),
        auc=roc_auc_score(y_gold, y_hat),
    )
    plot_confusion_matrix(y_gold, y_hat, save_path)
    
    return evaluation_scores


def evaluate_train_test(y_train, y_train_hat, y_test, y_test_hat):

    print(f"Train accuracy: {accuracy_score(y_train, y_train_hat):.2f},\t\tTest accuracy: {accuracy_score(y_train, y_train_hat):.2f}")
    print(f"Train recall: {recall_score(y_train, y_train_hat):.2f}, Test recall: {recall_score(y_test, y_test_hat):.2f}")
    print(f"Train precision: {precision_score(y_train, y_train_hat):.2f}, Test precision: {precision_score(y_test, y_test_hat):.2f}")
    print(f"Train f1: {f1_score(y_train, y_train_hat):.2f}, Test f1: {f1_score(y_test, y_test_hat):.2f}")
    print(f"Train AUC: {roc_auc_score(y_train, y_train_hat):.2f}, Test AUC: {roc_auc_score(y_test, y_test_hat):.2f}")
    plot_confusion_matrix(y_test, y_test_hat)


class BaseModel():

    @abstractmethod
    def fit(self, X, y, dataset_config):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : List[networkx.DiGraph]
            Array of directed connectivity networks.
        y : List[int]
            Array of labels.
        dataset_config : Optional[Dict]
            Optional dictionary containing the dataset configuration, when the preprocessing is needed. Defaults to None.

        Returns
        -------
        self : BaseModel
            Returns the instance of the fitted model.
        """
        pass

    @abstractmethod
    def predict(self, X, dataset_config):
        """
        Returns the model predictions for the given data.

        Parameters
        ----------
        X : List[networkx.DiGraph]
            Array of directed connectivity networks.
        dataset_config : Optional[Dict]
            Optional dictionary containing the dataset configuration, when the preprocessing is needed. Defaults to None.
        
        Returns
        -------
        List[int]
            Array of predicted labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X, dataset_config):
        """
        Returns the model predictions for the given data.

        Parameters
        ----------
        X : Union[np.ndarray, List[Graph]
            Array of directed connectivity networks.
        dataset_config : Optional[Dict]
            Optional dictionary containing the dataset configuration, when the preprocessing is needed. Defaults to None.
        
        Returns
        -------
        np.ndarray
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

        data = from_networkx(nx_graph)
        data.x = x_attr
        data.y = torch.tensor([label], dtype=torch.long) if label is not None else None
        data.num_nodes = len(nx_graph.nodes)
        data.num_edges = len(nx_graph.edges)

        return data.to(torch.device(device))
    
    @staticmethod
    def evaluate(y_gold: Array, y_hat: Array, save_path: Optional[str] = None) -> EvaluationScores:
        return evaluate(y_gold, y_hat, save_path)

    def predict_and_evaluate(self, X_train, y_train, X_test, y_test):
        y_train_hat = self.predict(X_train)
        y_test_hat = self.predict(X_test)
        evaluate_train_test(y_train, y_train_hat, y_test, y_test_hat)



