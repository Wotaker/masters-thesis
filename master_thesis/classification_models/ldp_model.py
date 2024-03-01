from typing import Dict, List

import torch
import numpy as np
from networkx import DiGraph
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

from master_thesis.classification_models.base_model import BaseModel, CLASSIC_CLASSIFIERS_MAP
from master_thesis.classification_models.utils import LocalDegreeProfile


class LDPModel(BaseModel):

    def __init__(
            self,
            device: str = 'cpu',
            n_bins: int = 10,
            classifier_type: str = 'random_forest',
            classifier_kwargs: Dict = {}
        ):
        self.device = device
        self.n_bins = n_bins
        self.classifier = CLASSIC_CLASSIFIERS_MAP[classifier_type](**classifier_kwargs)
    
    def _aggregate_ldp(self, data: Data) -> np.ndarray:
        n_features = data.x.shape[1]
        x = [np.histogram(data.x[:, i], bins=self.n_bins, density=True)[0] for i in range(n_features)]
        x = np.concatenate(x, axis=0)
        return x

    def _ldp_transform(self, X: List[Data]) -> List[Data]:
        X = [LocalDegreeProfile()(x) for x in X]
        X = [self._aggregate_ldp(x) for x in X]
        X = np.array(X)
        return X
    
    def fit(self, X: List[DiGraph], y: List[int]):
        X = [self.nx2geometric(self.device, x, x_attr=None, label=label) for x, label in zip(X, y)]
        X = self._ldp_transform(X)
        self.classifier.fit(X, y)

    def predict(self, X: List[DiGraph]) -> np.ndarray:
        X = [self.nx2geometric(self.device, x, x_attr=None) for x in X]
        X = self._ldp_transform(X)
        y_hat = self.classifier.predict(X)
        return y_hat
