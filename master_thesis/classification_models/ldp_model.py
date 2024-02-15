from typing import Dict, List

import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile

from master_thesis.classification_models import *


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
    
    def fit(self, X, y):
        X = [self.nx2geometric(x, self.device, label, x_attr=None) for x, label in zip(X, y)]
        X = self._ldp_transform(X)
        self.classifier.fit(X, y)

    def predict(self, X):
        X = [self.nx2geometric(x, self.device, x_attr=None) for x in X]
        X = self._ldp_transform(X)
        y_hat = self.classifier.predict(X)
        return y_hat
