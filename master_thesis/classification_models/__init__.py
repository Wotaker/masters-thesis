from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier

from master_thesis.classification_models.base_model import BaseModel
from master_thesis.classification_models.ldp_model import LDPModel


CLASSIC_CLASSIFIERS_MAP = {
    "svc": SVC,
    "mlp": MLPClassifier,
    "ridge": RidgeClassifier,
    "random_forest": RandomForestClassifier,
}