import torch

from master_thesis.classification_models.utils.descriptors import *

LOSS_FUNCTIONS = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "nll": torch.nn.NLLLoss,
}
