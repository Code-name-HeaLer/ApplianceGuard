import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional import for PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None

class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network Encoder.
    Takes node features (average signals) and graph structure.
    Outputs static enriched state embeddings. Run offline.
    """
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 8, dropout: float = 0.1, activation=F.gelu): 
        super().__init__()
        
        if GCNConv is None: 
            raise ImportError("PyTorch Geometric (GCNConv) required for GCNEncoder.")
        
        if num_layers < 1: 
            raise ValueError("GCN layers must be >= 1.")

        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation = activation
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        else: # Only 1 layer
            self.convs[0] = GCNConv(node_feature_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        current_edge_index = edge_index
        current_edge_weight = edge_weight

        for i in range(self.num_layers):
            x = self.convs[i](x, current_edge_index, current_edge_weight)
            if i < self.num_layers - 1: # Apply activation and dropout to all but the output layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # NO final activation applied to GCN output (returns logits/features)
        return x