"""
Spatial Feature Extractor with Community-Aware GNN
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from einops import rearrange
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES


class SpatialExtractor(nn.Module):
    """
    Spatial feature extractor with community-aware processing
    Three-stage: Local GNN -> Community aggregation -> Fusion GNN
    """
    
    def __init__(self, embed_dim, gnn_type="gcn", batch_norm=True,
                 commgnn=True, comm_boundaries=None, **kwargs):
        """
        Initialize spatial extractor
        
        Args:
            embed_dim: Embedding dimension
            gnn_type: Type of GNN layer
            batch_norm: Use batch normalization
            commgnn: Use community-aware GNN
            comm_boundaries: Community boundaries
            **kwargs: Additional GNN arguments
        """
        super().__init__()
        self.commgnn = commgnn
        self.gnn_type = gnn_type
        self.batch_norm = batch_norm
        
        self.comm_conv = get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)
        self.local_conv = get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)
        self.glob_conv = get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)
        
        # Fusion GNN: concatenate local and community features, then fuse
        self.fuse_conv = get_simple_gnn_layer(gnn_type, 2 * embed_dim, embed_dim, **kwargs)
        
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim)
        
        if comm_boundaries is not None:
            self.comm_index = comm_boundaries
        else:
            self.comm_index = [0, 48, 71, 91, 113, 129, 152, 200]
        
        self.num_communities = len(self.comm_index) - 1
    
    def forward(self, x, edge_index, subgraph_edge_index, edge_attr=None,
                subgraph_edge_attr=None, comm_edge_index=None,
                comm_edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Global edge indices
            subgraph_edge_index: Community subgraph edge indices
            edge_attr: Global edge attributes
            subgraph_edge_attr: Subgraph edge attributes
            comm_edge_index: Community-level edge indices
            comm_edge_attr: Community-level edge attributes
            batch: Batch indices
            
        Returns:
            torch.Tensor: Extracted spatial features
        """
        if self.gnn_type in EDGE_GNN_TYPES:
            if edge_attr is None:
                x_local = F.relu(self.local_conv(x, subgraph_edge_index))
                x_lg = F.relu(self.glob_conv(x_local, edge_index))
            else:
                # Local GNN
                x_local = F.relu(self.local_conv(x, subgraph_edge_index, subgraph_edge_attr))
                x_local = F.dropout(x_local, p=0.2, training=self.training)
                
                # Community aggregation
                x_local_dense, mask = to_dense_batch(x_local, batch)
                com_mean = []
                com_max = []
                for start, end in zip(self.comm_index, self.comm_index[1:]):
                    com_mean.append(x_local_dense[:, start:end, :].mean(dim=1, keepdim=True))
                    com_max.append(x_local_dense[:, start:end, :].max(dim=1, keepdim=True)[0])
                
                com_fea_mean = torch.cat(com_mean, dim=1)
                com_fea_max = torch.cat(com_max, dim=1)
                com_fea_mean = com_fea_mean + com_fea_max
                com_fea_mean = rearrange(com_fea_mean, 'b n d -> (b n) d')
                
                # Community GNN (only if comm_edge_index is provided)
                if comm_edge_index is not None:
                    com_fea_mean = F.relu(self.comm_conv(com_fea_mean, comm_edge_index, comm_edge_attr))
                    com_fea_mean = F.dropout(com_fea_mean, p=0.2, training=self.training)
                com_fea_mean = rearrange(com_fea_mean, '(b n) d -> b n d', n=self.num_communities)
                
                # Broadcast community features back to nodes
                comm = []
                for start, end, i in zip(self.comm_index, self.comm_index[1:], range(len(self.comm_index[1:]))):
                    comm.append(com_fea_mean[:, i, :].unsqueeze(1).repeat(1, end - start, 1))
                x_comm = torch.cat(comm, dim=1)[mask]
                
                # Fusion: concatenate local and community, then apply GNN
                x_fused = torch.cat([x_local, x_comm], dim=-1)
                x_j = F.relu(self.fuse_conv(x_fused, subgraph_edge_index, subgraph_edge_attr))
                x_j = F.dropout(x_j, p=0.3, training=self.training)
                x_lg = x + x_j
        else:
            x_local = F.relu(self.local_conv(x, subgraph_edge_index))
            x_lg = F.relu(self.glob_conv(x_local, edge_index))
        
        x_struct = x_lg
        
        if self.batch_norm:
            x_struct = self.bn(x_struct)
        
        return x_struct

