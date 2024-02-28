from typing import List

from networkx import Graph
from numpy import ndarray as Array
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from master_thesis.classification_models.base_model import BaseModel
from master_thesis.classification_models.utils import LOSS_FUNCTIONS


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, hidden_channels: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


class UndirectedGCNModel(BaseModel):

    def __init__(
            self,
            device: str = 'cpu',
            batch_size: int = 64,
            validation_size: float = 0.2,
            hidden_channels: int = 16,
            learning_rate: float = 0.01,
            loss: str = 'cross_entropy',
            epochs: int = 100,
            print_every: int = 10,
            seed: int = 42
        ):
        self.device = device
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.loss = LOSS_FUNCTIONS[loss]()
        self.epochs = epochs
        self.print_every = print_every
        self.model = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        torch.manual_seed(seed)
    
    def _train(self):
        self.model.train()

        for data in self.train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.loss(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
    
    def _test(self, dataloader: DataLoader) -> float:
        self.model.eval()

        loss_accum = 0
        for data in dataloader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss(out, data.y)  # Compute the loss.
            loss_accum += loss.item()  
        
        return loss_accum / len(dataloader.dataset) # Normalize the loss.
    
    def _predict(self, dataloader: DataLoader) -> Array:
        self.model.eval()

        y_hat = []
        for data in dataloader:
            out = self.model(data.x, data.edge_index, data.batch)
            y_hat.append(out.argmax(dim=1))
        
        return torch.cat(y_hat, dim=0).cpu().numpy()
    
    def fit(self, X: List[Graph], y: List[int]):

        # Define model and optimizer
        self.model = GCN(1, 2, self.hidden_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Prepare data and dataloaders
        X = [self.nx2geometric(
                self.device,
                x, 
                x_attr=torch.ones((len(x), 1), dtype=torch.float),
                label=label
            ) for x, label in zip(X, y)]
        validation_size = int(len(X) * self.validation_size)
        self.train_loader = DataLoader(X[:-validation_size], batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(X[-validation_size:], batch_size=self.batch_size, shuffle=False)

        # Train model
        for epoch in range(1, self.epochs + 1):
            self._train()
            train_loss = self._test(self.train_loader)
            val_loss = self._test(self.validation_loader)

            # Print info
            if epoch % self.print_every == 0:
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def predict(self, X: List[Graph]) -> Array:

        # Define model and optimizer
        self.model = GCN(1, 2, self.hidden_channels).to(self.device)

        # Prepare data and dataloaders
        X = [self.nx2geometric(
                self.device,
                x, 
                x_attr=torch.ones((len(x), 1), dtype=torch.float),
            ) for x in X]
        self.test_loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)

        # Make predictions
        return self._predict(self.test_loader)
