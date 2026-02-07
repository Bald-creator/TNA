"""
GNN Layer Implementations
Various graph neural network layers for brain network analysis
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils


GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
]

EDGE_GNN_TYPES = [
    'gine', 'gcn',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4'
]


def get_simple_gnn_layer(gnn_type, embed_dim, out_dim=None, **kwargs):
    """
    Get GNN layer by type
    
    Args:
        gnn_type: Type of GNN ('gcn', 'gin', 'gine', 'pna', etc.)
        embed_dim: Input embedding dimension
        out_dim: Output dimension (defaults to embed_dim)
        **kwargs: Additional arguments (edge_dim, deg, etc.)
        
    Returns:
        nn.Module: GNN layer
    """
    if out_dim is None:
        out_dim = embed_dim
    
    edge_dim = kwargs.get('edge_dim', None)
    
    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, out_dim)
    
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, out_dim)
    
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, out_dim)
        else:
            if embed_dim != out_dim:
                raise NotImplementedError("Custom GCNConv only supports in_dim == out_dim")
            return GCNConv(embed_dim, edge_dim)
    
    elif gnn_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, out_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    
    elif gnn_type == "gine":
        mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, out_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    
    elif gnn_type == "pna":
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        deg = kwargs.get('deg', None)
        return gnn.PNAConv(
            embed_dim, out_dim,
            aggregators=aggregators, scalers=scalers,
            deg=deg, towers=4, pre_layers=1, post_layers=1,
            divide_input=True, edge_dim=edge_dim
        )
    
    elif gnn_type == "pna2":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        return gnn.PNAConv(
            embed_dim, out_dim,
            aggregators=aggregators, scalers=scalers,
            deg=deg, towers=4, pre_layers=1, post_layers=1,
            divide_input=True, edge_dim=edge_dim
        )
    
    elif gnn_type == "pna3":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        return gnn.PNAConv(
            embed_dim, out_dim,
            aggregators=aggregators, scalers=scalers,
            deg=deg, towers=1, pre_layers=1, post_layers=1,
            divide_input=False, edge_dim=edge_dim
        )
    
    elif gnn_type == "pna4":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        return gnn.PNAConv(
            embed_dim, out_dim,
            aggregators=aggregators, scalers=scalers,
            deg=deg, towers=8, pre_layers=1, post_layers=1,
            divide_input=True, edge_dim=edge_dim
        )
    
    elif gnn_type == "mpnn":
        aggregators = ['sum']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        return gnn.PNAConv(
            embed_dim, out_dim,
            aggregators=aggregators, scalers=scalers,
            deg=deg, towers=4, pre_layers=1, post_layers=1,
            divide_input=True, edge_dim=edge_dim
        )
    
    else:
        raise ValueError(f"GNN type '{gnn_type}' not implemented!")


class GCNConv(gnn.MessagePassing):
    """
    Custom GCN layer with edge features
    """
    
    def __init__(self, embed_dim, edge_dim):
        """
        Initialize GCN layer
        
        Args:
            embed_dim: Node embedding dimension
            edge_dim: Edge feature dimension
        """
        super(GCNConv, self).__init__(aggr='add')
        
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            
        Returns:
            torch.Tensor: Updated node features
        """
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)
        
        row, col = edge_index
        
        deg = utils.degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)
    
    def message(self, x_j, edge_attr, norm):
        """Compute messages"""
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        """Update node features"""
        return aggr_out

