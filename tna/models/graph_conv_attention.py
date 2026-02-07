"""
Graph Convolutional Attention (ESC-inspired)
Replaces full attention with efficient graph convolutions
"""
import torch
from torch import nn
import torch.nn.functional as F
from .gnn_layers import get_simple_gnn_layer


class GraphConvolutionalAttention(nn.Module):
    """
    Graph convolutional attention mechanism (ESC-inspired)
    - Static K-hop convolutions: Capture long-range dependencies
    - Dynamic 1-hop convolutions: Instance-dependent weighting
    """
    
    def __init__(self, embed_dim, pdim=None, k_hops=3, **kwargs):
        """
        Initialize graph convolutional attention
        
        Args:
            embed_dim: Embedding dimension
            pdim: Dimension for graph convolutions (default: embed_dim // 4)
            k_hops: Number of hops for static convolutions
            **kwargs: Additional arguments for GNN layers
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pdim = pdim if pdim else embed_dim // 4
        self.k_hops = k_hops
        
        gnn_type = kwargs.pop('gnn_type', 'gine')
        
        # Static K-hop graph convolutions
        self.static_convs = nn.ModuleList([
            get_simple_gnn_layer(gnn_type, self.pdim, self.pdim, **kwargs)
            for _ in range(k_hops)
        ])
        
        # Dynamic edge weight generator
        self.dynamic_weight_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.pdim)
        )
        nn.init.zeros_(self.dynamic_weight_gen[-1].weight)
        nn.init.zeros_(self.dynamic_weight_gen[-1].bias)
        
        # Aggregation layer
        self.aggr = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features [batch*N, embed_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes (optional)
            
        Returns:
            torch.Tensor: Updated node features
        """
        # Split features: first pdim channels for convolution, rest skip
        x_conv = x[:, :self.pdim]
        x_skip = x[:, self.pdim:]
        
        # 1. Static K-hop graph convolutions (large receptive field)
        x_static = x_conv
        for conv_layer in self.static_convs:
            if edge_attr is not None:
                x_static = F.relu(conv_layer(x_static, edge_index, edge_attr))
            else:
                x_static = F.relu(conv_layer(x_static, edge_index))
            x_static = F.dropout(x_static, p=0.1, training=self.training)
        
        # 2. Dynamic 1-hop graph convolution (instance-dependent)
        edge_weights = self.dynamic_weight_gen(x)
        edge_weights_src = edge_weights[edge_index[0]]
        x_dynamic = x_conv[edge_index[1]] * edge_weights_src
        
        # Aggregate to target nodes
        x_dynamic_agg = torch.zeros_like(x_conv)
        x_dynamic_agg.index_add_(0, edge_index[0], x_dynamic)
        
        # 3. Combine static and dynamic
        x_out = x_static + x_dynamic_agg
        
        # 4. Concatenate with skip connection
        x_full = torch.cat([x_out, x_skip], dim=-1)
        x_full = self.aggr(x_full)
        
        return x_full

