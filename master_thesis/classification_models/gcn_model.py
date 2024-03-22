from typing import List, Tuple, Optional, Dict, Any, Union

import datetime
from networkx import Graph
from numpy import ndarray as Array
import logging
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from master_thesis.classification_models.base_model import BaseModel, EvaluationScores
from master_thesis.classification_models.utils import *
from master_thesis.tools.data import Preprocessing

DEFAULT_CHECKPOINT_DIR = "master_thesis/classification_models/checkpoints/gcn"


class GCN(torch.nn.Module):
    def __init__(
            self,
            num_node_features: int,
            num_classes: int,
            hidden_channels: int,
            dropout: float
        ):
        super(GCN, self).__init__()
        self.convIn = GCNConv(num_node_features, hidden_channels)
        self.convHidden = GCNConv(hidden_channels, hidden_channels)
        self.linHidden = Linear(hidden_channels, hidden_channels)
        self.linFinal = Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.convIn(x, edge_index)
        x = x.relu()
        x = self.convHidden(x, edge_index)
        x = x.relu()
        x = self.convHidden(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linHidden(x)
        x = x.relu()
        x = self.linFinal(x)
        
        return x


class GCNModel(BaseModel):

    def __init__(
            self,
            device: str = 'cpu',
            ldp_features: bool = False,
            batch_size: int = 64,
            validation_size: float = 0.2,
            hidden_channels: int = 16,
            learning_rate: float = 0.01,
            loss: str = 'cross_entropy',
            dropout: float = 0.5,
            epochs: int = 100,
            print_every: int = 10,
            checkpoint_dir: Optional[str] = None,
            seed: int = 42
        ):
        if checkpoint_dir is None:
            checkpoint_dir = DEFAULT_CHECKPOINT_DIR
            checkpoint_dir = os.path.join(
                checkpoint_dir,
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )

        self.device = device
        self.ldp_features = ldp_features
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.loss = LOSS_FUNCTIONS[loss]
        self.dropout = dropout
        self.epochs = epochs
        self.print_every = print_every
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.train_loss_weights = None
        self.val_loss_weights = None
        torch.manual_seed(seed)
    
    def _train(self):
        self.model.train()

        for data in self.train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.loss(weight=self.train_loss_weights)(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
    
    def _test(self, dataloader: DataLoader, loss_weights: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()

        preds_accum = []
        golds_accum = []
        loss_accum = 0
        for data in dataloader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss(weight=loss_weights)(out, data.y)  # Compute the loss.
            loss_accum += loss.item()  
            preds_accum.append(out.argmax(dim=1))
            golds_accum.append(data.y)
        preds_accum = torch.cat(preds_accum, dim=0)
        golds_accum = torch.cat(golds_accum, dim=0)
        
        return loss_accum / len(dataloader.dataset), preds_accum, golds_accum
    
    def _predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()

        y_hat = []
        for data in dataloader:
            out = self.model(data.x, data.edge_index, data.batch)
            y_hat.append(out.argmax(dim=1))
        y_hat = torch.cat(y_hat, dim=0)
        
        return y_hat
    
    def _log(
            self,
            epoch: int,
            train_loss: float,
            val_loss: float,
            train_scores: EvaluationScores,
            val_scores: EvaluationScores
        ):
        if logging.getLogger().level <= logging.INFO:
            print(f"======= Epoch: {epoch:03d} =======")
            print(f"    Train loss: {train_loss:.4f}")
            print(train_scores)
            print(f"    Validation loss: {val_loss:.4f}")
            print(val_scores, end='\n\n')
    
    def _calc_loss_weights(self, y: List[int]) -> torch.Tensor:
        class_counts = torch.bincount(torch.tensor(y))
        return 1 / class_counts.float()
        
    
    def fit(self, X: Union[np.ndarray, List[Graph]], y: np.ndarray, dataset_config: Optional[Dict] = None):

        # Split data into training and validation sets
        validation_size = int(len(X) * self.validation_size)
        X_train, X_val = X[:-validation_size], X[-validation_size:]
        y_train, y_val = y[:-validation_size], y[-validation_size:]

        # Preprocessing and control mean subtraction
        if dataset_config is not None:
            if dataset_config["subtract_mean"]:
                total_control_mean = X[y == 0].mean(axis=0)
                train_control_mean = X_train[y_train == 0].mean(axis=0)
                X_val = X_val - total_control_mean
                X_train = X_train - train_control_mean
            X_train, y_train = Preprocessing(**dataset_config["preprocessing"])(X_train, y_train)
            X_val, y_val = Preprocessing(**dataset_config["preprocessing"])(X_val, y_val)
        
        # Convert to PyTorch Geometric
        X_train = [self.nx2geometric(
                device=self.device,
                nx_graph=x,
                x_attr=None,
                label=label
            ) for x, label in zip(X_train, y_train)]
        X_val = [self.nx2geometric(
                device=self.device,
                nx_graph=x,
                x_attr=None,
                label=label
            ) for x, label in zip(X_val, y_val)]

        # Add LDP features
        X_train = [LocalDegreeProfile()(x) for x in X_train] if self.ldp_features else [AddOnes()(x) for x in X_train]
        X_val = [LocalDegreeProfile()(x) for x in X_val] if self.ldp_features else [AddOnes()(x) for x in X_val]
        
        # Define dataloaders
        self.train_loader = DataLoader(X_train, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(X_val, batch_size=self.batch_size, shuffle=True)
        self.train_loss_weights = self._calc_loss_weights(y_train)
        self.val_loss_weights = self._calc_loss_weights(y_val)

        # Define model and optimizer
        self.model = GCN(X_train[0].x.shape[1], 2, self.hidden_channels, self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train model
        best_auc = 0.
        for epoch in range(1, self.epochs + 1):
            self._train()
            train_loss, train_preds, train_golds = self._test(self.train_loader, self.train_loss_weights)
            val_loss, val_preds, val_golds = self._test(self.validation_loader, self.val_loss_weights)
            train_evaluation_scores = self.evaluate(train_golds.cpu().numpy(), train_preds.cpu().numpy())
            val_evaluation_scores = self.evaluate(val_golds.cpu().numpy(), val_preds.cpu().numpy())

            # Save checkpoint
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_evaluation_scores.auc,
                self.checkpoint_dir,
                val_evaluation_scores.auc > best_auc
            )
            best_auc = max(best_auc, val_evaluation_scores.auc)

            # Print info
            if epoch % self.print_every == 0:
                self._log(epoch, train_loss, val_loss, train_evaluation_scores, val_evaluation_scores)
            

    def predict(self, X: Union[np.ndarray, List[Graph]], dataset_config: Optional[Dict] = None) -> np.ndarray:
        
        # Load best model
        loaded = load_checkpoint(
            self.checkpoint_dir,
            self.device,
            self.model
        )
        if loaded is not None:
            self.model, _, epoch, validation_metric = loaded
            logging.info(f"Loaded model from epoch {epoch} with validation metric {validation_metric:.4f}")
        
        # Preprocessing
        if dataset_config is not None:
            X, _ = Preprocessing(**dataset_config["preprocessing"], shuffle=False)(X, None)

        # Convert to PyTorch Geometric
        X = [self.nx2geometric(self.device, x) for x in X]

        # Add LDP features
        X = [LocalDegreeProfile()(x) for x in X] if self.ldp_features else [AddOnes()(x) for x in X]
        
        # Define dataloaders
        self.test_loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)

        # Make predictions
        return self._predict(self.test_loader).cpu().numpy()
