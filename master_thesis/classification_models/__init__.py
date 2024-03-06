from typing import Dict

from master_thesis.classification_models.base_model import BaseModel
from master_thesis.classification_models.ldp_model import LDPModel
from master_thesis.classification_models.ltp_model import LTPModel
from master_thesis.classification_models.vector_model import VectorModel
from master_thesis.classification_models.gcn_model import GCNModel

MODELS_MAP: Dict[str, BaseModel] = {
    "GCN": GCNModel,
    "LDP": LDPModel,
    "LTP": LTPModel,
    "VECTOR": VectorModel,
    "GRAPH2VEC": VectorModel
}
