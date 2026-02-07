"""
Transformer Encoder Layer with Hybrid Attention
"""
import torch
from torch import nn
from .attention import SpatialAttention
from .graph_conv_attention import GraphConvolutionalAttention


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom Transformer encoder layer with hybrid attention
    Supports both full spatial attention and graph convolutional attention
    """
    
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 use_graph_conv_attn=False, k_hops=3, pdim=None,
                 **kwargs):
        """
        Initialize transformer layer
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function
            batch_norm: Use batch normalization
            pre_norm: Pre-normalization
            use_graph_conv_attn: Use graph conv attention (Layer 2-4)
            k_hops: K-hop for graph conv attention
            pdim: Dimension for graph conv
            **kwargs: Additional arguments
        """
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        
        self.use_graph_conv_attn = use_graph_conv_attn
        if use_graph_conv_attn:
            # ESC-inspired graph convolutional attention (Layer 2-4)
            self.self_attn = GraphConvolutionalAttention(
                d_model, pdim=pdim, k_hops=k_hops, **kwargs
            )
        else:
            # Full spatial attention with distance and community weights (Layer 1)
            self.self_attn = SpatialAttention(
                d_model, nhead, dropout=dropout, bias=False, **kwargs
            )
        
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
    
    def forward(self, x, edge_index, subgraph_edge_index=None,
                subgraph_edge_attr=None, edge_attr=None, ptr=None,
                batch=None, return_attn=False, dm=None, coord=None,
                comm_edge_index=None, comm_edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            subgraph_edge_index: Subgraph edge indices
            subgraph_edge_attr: Subgraph edge attributes
            edge_attr: Edge attributes
            ptr: Batch pointers
            batch: Batch indices
            return_attn: Return attention weights
            dm: Distance matrix
            coord: Node coordinates
            comm_edge_index: Community edge indices
            comm_edge_attr: Community edge attributes
            
        Returns:
            torch.Tensor: Updated node features
        """
        if self.pre_norm:
            x = self.norm1(x)
        
        # Attention mechanism selection
        if self.use_graph_conv_attn:
            # Graph convolutional attention (uses community subgraph)
            x2 = self.self_attn(x, subgraph_edge_index, subgraph_edge_attr)
            attn = None
        else:
            # Full spatial attention (with all priors)
            x2, attn = self.self_attn(
                x, edge_index,
                edge_attr=edge_attr,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                batch=batch,
                return_attn=return_attn,
                dm=dm,
                coord=coord,
                comm_edge_index=comm_edge_index,
                comm_edge_attr=comm_edge_attr
            )
        
        x = x + self.dropout1(x2)
        
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        
        if not self.pre_norm:
            x = self.norm2(x)
        
        return x

