from typing import Tuple, Optional

import os
import glob
import logging
import torch

from master_thesis.classification_models.utils.descriptors import *

LOSS_FUNCTIONS = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "nll": torch.nn.NLLLoss,
}

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_metric: float,
        checkpoint_dir: str,
        is_best: bool = False
    ) -> None:
    """Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
        validation_metric (float): The validation metric value (e.g., accuracy).
        checkpoint_dir (str): Directory to save the checkpoint.
        is_best (bool, optional): Whether this is the best model so far based on the validation metric. Defaults to False.
    """

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_filename = f"gcn_checkpoint_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_metric': validation_metric
    }

    torch.save(state, checkpoint_path)

    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "gcn_best_model.pt")

        # Create a symbolic link (remove the old one if it exists)
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        os.symlink(checkpoint_filename, best_model_path)

def load_checkpoint(
        checkpoint_dir: str,
        device: str,
        model: torch.nn.Module
    ) -> Optional[Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]]:
    """Loads the model checkpoint from the specified directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        device (torch.device): Device to load the model on (CPU or GPU).

    Returns:
        tuple: A tuple containing the loaded model, optimizer, epoch number, and validation metric.
    """

    best_checkpoint_path = os.path.join(checkpoint_dir, "gcn_best_model.pt")

    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=torch.device(device))

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        validation_metric = checkpoint['validation_metric']

        return model, optimizer, epoch, validation_metric
    else:
        logging.warning(f"No checkpoints found in {checkpoint_dir}")
        return None

