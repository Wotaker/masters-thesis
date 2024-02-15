from typing import Dict, List

from networkx import DiGraph
import numpy as np
from numpy import ndarray as Array
from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile

from master_thesis.classification_models.base_model import BaseModel, CLASSIC_CLASSIFIERS_MAP

class VectorModel(BaseModel):

    def __init__(
            self,
            device: str = 'cpu',
            classifier_type: str = 'random_forest',
            classifier_kwargs: Dict = {}
        ):
        self.device = device
        self.classifier = CLASSIC_CLASSIFIERS_MAP[classifier_type](**classifier_kwargs)
    
    def fit(self, X: Array, y: List[int]):
        self.classifier.fit(X, y)

    def predict(self, X: Array):
        y_hat = self.classifier.predict(X)
        return y_hat
