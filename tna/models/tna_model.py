"""
TNA: TNA: Template-aware Network-collaboration Model with Neuromodulated Attention
Supports single-atlas and dual-atlas architectures
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, dense_to_sparse

from .spatial_extractor import SpatialExtractor
from .transformer_layer import TransformerEncoderLayer
from .clustering import AdaptiveBrainCluster


class TNA(nn.Module):
    """
    TNA model for single atlas
    ESC-inspired hybrid attention: Layer 1 uses full attention, Layer 2-4 use graph convolutions
    """
    
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, pe=False, pe_dim=0,
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=2,
                 num_nodes=200, comm_boundaries=None,
                 use_gnn=True, use_attention=True, use_hierarchical_graph=True,
                 **kwargs):
        """
        Initialize TNA
        
        Args:
            in_size: Input feature size (num_nodes)
            num_class: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_layers: Number of transformer layers
            batch_norm: Use batch normalization
            pe: Use positional encoding
            pe_dim: Positional encoding dimension
            gnn_type: GNN type
            se: Spatial extractor type
            use_edge_attr: Use edge attributes
            num_edge_features: Number of edge features
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        if comm_boundaries is None:
            # Default: CC200 with Yeo-7
            comm_boundaries = [0, 48, 71, 91, 113, 129, 152, 200]
        self.comm_boundaries = comm_boundaries
        
        # Ablation settings
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.use_hierarchical_graph = use_hierarchical_graph
        
        self.pe = pe
        self.pe_dim = pe_dim
        if pe and pe_dim > 0:
            self.embedding_pe = nn.Linear(pe_dim, d_model)
        
        self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=True)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 8)
            self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=edge_dim, bias=False)
            self.embedding_comm_edge = nn.Linear(in_features=1, out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None
        
        self.gnn_type = gnn_type
        self.se = se
        
        # ESC-inspired hybrid architecture
        pdim = d_model // 4
        k_hops = 3
        
        layer_kwargs = kwargs.copy()
        layer_kwargs['gnn_type'] = gnn_type
        layer_kwargs['num_nodes'] = num_nodes
        layer_kwargs['comm_boundaries'] = comm_boundaries
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                # Layer 1: Full spatial attention
                layer = TransformerEncoderLayer(
                    d_model, num_heads, dim_feedforward, dropout,
                    batch_norm=batch_norm, use_graph_conv_attn=False, **layer_kwargs
                )
            else:
                # Layer 2-4: Graph convolutional attention
                layer = TransformerEncoderLayer(
                    d_model, num_heads, dim_feedforward, dropout,
                    batch_norm=batch_norm, use_graph_conv_attn=True,
                    k_hops=k_hops, pdim=pdim, **layer_kwargs
                )
            layers.append(layer)
        
        # Spatial feature extractor (GNN module) - conditional
        if self.use_gnn:
            self.ds_spatial_feature_extractor = SpatialExtractor(
                d_model, gnn_type=gnn_type, batch_norm=batch_norm, 
                comm_boundaries=comm_boundaries, **kwargs
            )
        else:
            self.ds_spatial_feature_extractor = None
        
        # Transformer encoder (Attention module) - conditional
        if self.use_attention:
            class HybridEncoder(nn.Module):
                def __init__(self, layers):
                    super().__init__()
                    self.layers = nn.ModuleList(layers)
                    self.num_layers = len(layers)
                    self.norm = None
                
                def forward(self, x, edge_index, **kwargs):
                    output = x
                    for mod in self.layers:
                        output = mod(output, edge_index, **kwargs)
                    if self.norm is not None:
                        output = self.norm(output)
                    return output
            
            self.gt_encoder = HybridEncoder(layers)
        else:
            # No attention: use identity
            self.gt_encoder = nn.Identity()
        
        # Adaptive Brain Clustering (替代原DEC)
        self.encoder = nn.Sequential(
            nn.Linear(d_model * num_nodes, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, d_model * num_nodes),
        )
        self.dec = AdaptiveBrainCluster(
            num_clusters=100,
            hidden_dimension=d_model,
            encoder=self.encoder,
            alpha=1.0,
            temperature=1.0,
            use_orthogonal=True,
            freeze_centers=False,
            assignment_method='projection'
        )
        self.dim_reduction = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.LeakyReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(100 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_class)
        )
        
        self.node_rearranged_len = comm_boundaries
    
    def forward(self, data, return_attn=False):
        """
        Forward pass
        
        Args:
            data: Graph data object
            return_attn: Return attention weights
            
        Returns:
            torch.Tensor: Class logits
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        pe = data.pe
        coord = data.coord
        coord = coord / torch.norm(coord, dim=1, keepdim=True)
        comm_graph = data.comm_graph
        
        # Distance matrix
        diff = coord[:self.num_nodes, :].unsqueeze(1) - coord[:self.num_nodes, :].unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=-1))
        
        # Hierarchical graph structure (global + subgraph + community)
        if self.se == "commgnn" and self.use_hierarchical_graph:
            # Use hierarchical graph: subgraph for local, community graph for inter-community
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_edge_attr = data.subgraph_edge_attr
            comm_edge_index, comm_edge_attr = dense_to_sparse(comm_graph)
            comm_edge_attr = comm_edge_attr.unsqueeze(1).float()
        elif self.se == "commgnn" and not self.use_hierarchical_graph:
            # Use global graph only: subgraph = global, no community-level graph
            subgraph_edge_index = edge_index  # Use global graph for local conv
            subgraph_edge_attr = edge_attr
            comm_edge_index = None  # Disable community-level GNN
            comm_edge_attr = None
        else:
            # No community GNN
            subgraph_edge_index = None
            subgraph_edge_attr = None
            comm_edge_index = None
            comm_edge_attr = None
        
        output = self.embedding(x)
        
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
            if comm_edge_attr is not None:
                comm_edge_attr = self.embedding_comm_edge(comm_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None
            comm_edge_attr = None
        
        # Conditional GNN module
        if self.use_gnn:
            output = self.ds_spatial_feature_extractor(
                x=output, edge_index=edge_index, edge_attr=edge_attr,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                comm_edge_index=comm_edge_index,
                comm_edge_attr=comm_edge_attr,
                batch=batch
            )
        
        if self.pe and pe is not None:
            pe = torch.cat([pe, coord], dim=-1)
            pe = self.embedding_pe(pe)
            output = output + pe
        
        # Conditional Attention module
        if self.use_attention:
            output = self.gt_encoder(
                output, edge_index,
                edge_attr=edge_attr,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=data.ptr,
                batch=batch,
                return_attn=return_attn,
                dm=dist_matrix,
                comm_edge_index=comm_edge_index,
                comm_edge_attr=comm_edge_attr,
                coord=coord
            )
        
        # DEC clustering
        x_dense, mask = to_dense_batch(output, batch)
        x, assignment = self.dec(x_dense)
        #print(f"x.shape: {x.shape}")
        x = self.dim_reduction(x)
        #print(f"x.shape: {x.shape}")
        x = x.reshape((x.shape[0], -1))
        #print(f"x.shape: {x.shape}")
        
        return self.fc(x)


class DualAtlasTNA(nn.Module):
    """
    Dual-atlas TNA with hybrid fusion after DEC clustering
    - CC200 branch: 200 nodes
    - AAL116 branch: 116 nodes
    - Fusion: concatenate clustering features from both branches
    """
    
    def __init__(self, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, pe=False, pe_dim=0,
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=2,
                 use_gnn=True, use_attention=True, use_hierarchical_graph=True,
                 **kwargs):
        """
        Initialize DualAtlasTNA
        
        Args:
            num_class: Number of output classes
            d_model: Model dimension
            (other args same as TNA)
        """
        super().__init__()
        
        # CC200 branch (200 nodes)
        self.branch_cc200 = self._create_single_branch(
            in_size=200, d_model=d_model, num_heads=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, num_layers=num_layers,
            batch_norm=batch_norm, pe=pe, pe_dim=pe_dim,
            gnn_type=gnn_type, se=se, use_edge_attr=use_edge_attr,
            num_edge_features=num_edge_features,
            num_nodes=200,
            comm_boundaries=[0, 48, 71, 91, 113, 129, 152, 200],
            use_gnn=use_gnn, use_attention=use_attention,
            use_hierarchical_graph=use_hierarchical_graph,
            **kwargs
        )
        
        # AAL116 branch (116 nodes)
        self.branch_aal116 = self._create_single_branch(
            in_size=116, d_model=d_model, num_heads=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, num_layers=num_layers,
            batch_norm=batch_norm, pe=pe, pe_dim=pe_dim,
            gnn_type=gnn_type, se=se, use_edge_attr=use_edge_attr,
            num_edge_features=num_edge_features,
            num_nodes=116,
            comm_boundaries=[0, 40, 52, 57, 69, 82, 91, 116],
            use_gnn=use_gnn, use_attention=use_attention,
            use_hierarchical_graph=use_hierarchical_graph,
            **kwargs
        )
        
        # Fusion classifier
        self.fc = nn.Sequential(
            nn.Linear(100 * 8 + 100 * 8, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_class)
        )
    
    def _create_single_branch(self, in_size, d_model, num_heads, dim_feedforward,
                              dropout, num_layers, batch_norm, pe, pe_dim,
                              gnn_type, se, use_edge_attr, num_edge_features,
                              num_nodes, comm_boundaries, use_gnn, use_attention, 
                              use_hierarchical_graph, **kwargs):
        """Create a single-atlas branch"""
        branch = nn.Module()
        
        # Store ablation settings
        branch.use_gnn = use_gnn
        branch.use_attention = use_attention
        branch.use_hierarchical_graph = use_hierarchical_graph
        
        branch.pe = pe
        branch.pe_dim = pe_dim
        if pe and pe_dim > 0:
            branch.embedding_pe = nn.Linear(pe_dim, d_model)
        
        branch.embedding = nn.Linear(in_size, d_model, bias=True)
        
        branch.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 8)
            branch.embedding_edge = nn.Linear(num_edge_features, edge_dim, bias=False)
            branch.embedding_comm_edge = nn.Linear(1, edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None
        
        branch.gnn_type = gnn_type
        branch.se = se
        
        # Spatial feature extractor (GNN module) - conditional
        if use_gnn:
            branch.ds_spatial_feature_extractor = SpatialExtractor(
                d_model, gnn_type=gnn_type, batch_norm=batch_norm, 
                comm_boundaries=comm_boundaries, **kwargs
            )
        else:
            branch.ds_spatial_feature_extractor = None
        
        # Transformer encoder
        pdim = d_model // 4
        k_hops = 3
        layer_kwargs = kwargs.copy()
        layer_kwargs['gnn_type'] = gnn_type
        layer_kwargs['num_nodes'] = num_nodes
        layer_kwargs['comm_boundaries'] = comm_boundaries
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = TransformerEncoderLayer(
                    d_model, num_heads, dim_feedforward, dropout,
                    batch_norm=batch_norm, use_graph_conv_attn=False, **layer_kwargs
                )
            else:
                layer = TransformerEncoderLayer(
                    d_model, num_heads, dim_feedforward, dropout,
                    batch_norm=batch_norm, use_graph_conv_attn=True,
                    k_hops=k_hops, pdim=pdim, **layer_kwargs
                )
            layers.append(layer)
        
        # Transformer encoder (Attention module) - conditional
        if use_attention:
            class HybridEncoder(nn.Module):
                def __init__(self, layers):
                    super().__init__()
                    self.layers = nn.ModuleList(layers)
                    self.num_layers = len(layers)
                    self.norm = None
                
                def forward(self, x, edge_index, **kwargs):
                    output = x
                    for mod in self.layers:
                        output = mod(output, edge_index, **kwargs)
                    if self.norm is not None:
                        output = self.norm(output)
                    return output
            
            branch.gt_encoder = HybridEncoder(layers)
        else:
            # No attention: use identity
            branch.gt_encoder = nn.Identity()
        
        # Adaptive Brain Clustering (替代原DEC)
        branch.encoder = nn.Sequential(
            nn.Linear(d_model * num_nodes, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, d_model * num_nodes),
        )
        branch.dec = AdaptiveBrainCluster(
            num_clusters=100,
            hidden_dimension=d_model,
            encoder=branch.encoder,
            alpha=1.0,
            temperature=1.0,
            use_orthogonal=True,
            freeze_centers=False,
            assignment_method='projection'
        )
        branch.dim_reduction = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.LeakyReLU()
        )
        
        branch.node_rearranged_len = comm_boundaries
        branch.num_nodes = num_nodes
        
        return branch
    
    def _forward_single_branch(self, branch, data, return_attn=False):
        """Forward pass for a single branch"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        pe = data.pe
        coord = data.coord
        coord = coord / torch.norm(coord, dim=1, keepdim=True)
        comm_graph = data.comm_graph
        
        num_nodes = branch.num_nodes
        diff = coord[:num_nodes, :].unsqueeze(1) - coord[:num_nodes, :].unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=-1))
        
        # Hierarchical graph structure (global + subgraph + community)
        if branch.se == "commgnn" and branch.use_hierarchical_graph:
            # Use hierarchical graph: subgraph for local, community graph for inter-community
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_edge_attr = data.subgraph_edge_attr
            comm_edge_index, comm_edge_attr = dense_to_sparse(comm_graph)
            comm_edge_attr = comm_edge_attr.unsqueeze(1).float()
        elif branch.se == "commgnn" and not branch.use_hierarchical_graph:
            # Use global graph only: subgraph = global, no community-level graph
            subgraph_edge_index = edge_index  # Use global graph for local conv
            subgraph_edge_attr = edge_attr
            comm_edge_index = None  # Disable community-level GNN
            comm_edge_attr = None
        else:
            # No community GNN
            subgraph_edge_index = None
            subgraph_edge_attr = None
            comm_edge_index = None
            comm_edge_attr = None
        
        output = branch.embedding(x)
        
        if branch.use_edge_attr and edge_attr is not None:
            edge_attr = branch.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = branch.embedding_edge(subgraph_edge_attr)
            if comm_edge_attr is not None:
                comm_edge_attr = branch.embedding_comm_edge(comm_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None
            comm_edge_attr = None
        
        # Conditional GNN module
        if branch.use_gnn:
            output = branch.ds_spatial_feature_extractor(
                x=output, edge_index=edge_index, edge_attr=edge_attr,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                comm_edge_index=comm_edge_index,
                comm_edge_attr=comm_edge_attr,
                batch=batch
            )
        
        if branch.pe and pe is not None:
            pe = torch.cat([pe, coord], dim=-1)
            pe = branch.embedding_pe(pe)
            output = output + pe
        
        # Conditional Attention module
        if branch.use_attention:
            output = branch.gt_encoder(
                output, edge_index,
                edge_attr=edge_attr,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=data.ptr,
                batch=batch,
                return_attn=return_attn,
                dm=dist_matrix,
                comm_edge_index=comm_edge_index,
                comm_edge_attr=comm_edge_attr,
                coord=coord
            )
        
        #print(f"output.shape: {output.shape}")
        x_dense, mask = to_dense_batch(output, batch)
        x, assignment = branch.dec(x_dense)
        #print(f"x.shape: {x.shape}")
        x = branch.dim_reduction(x)
        #print(f"x.shape: {x.shape}")
        x = x.reshape((x.shape[0], -1))
        #print(f"x.shape: {x.shape}")
        
        return x
    
    def forward(self, data_cc200, data_aal116, return_attn=False):
        """
        Forward pass for dual-atlas model
        
        Args:
            data_cc200: Data object for CC200 atlas
            data_aal116: Data object for AAL116 atlas
            return_attn: Return attention weights (default: False to avoid huge logs)
            
        Returns:
            torch.Tensor: Class logits
        """
        feat_cc200 = self._forward_single_branch(self.branch_cc200, data_cc200, return_attn)
        #print(f"feat_cc200.shape: {feat_cc200.shape}")
        feat_aal116 = self._forward_single_branch(self.branch_aal116, data_aal116, return_attn)
        #print(f"feat_aal116.shape: {feat_aal116.shape}")
        feat_fused = torch.cat([feat_cc200, feat_aal116], dim=-1)
        #print(f"feat_fused.shape: {feat_fused.shape}")

        return self.fc(feat_fused)

