"""
Model Configuration Module
Centralized configuration for TNA model hyperparameters
"""


class TNAConfig:
    """Configuration class for TNA model"""
    
    def __init__(self):
        # Model architecture
        self.num_heads = 8
        self.num_layers = 2
        self.dim_hidden = 256
        self.dim_feedforward = 512  # 2 * dim_hidden
        self.dropout = 0.2
        self.batch_norm = True
        
        # Position encoding
        self.pe = 'rw+mni'  # 'rw', 'lap', or 'rw+mni'
        self.pe_dim = 33  # 30 for RW + 3 for MNI coordinates
        
        # GNN settings
        self.gnn_type = 'gine'  # 'gcn', 'gin', 'gine', 'pna', etc.
        self.se = 'commgnn'  # Spatial extractor type
        self.use_edge_attr = True
        self.edge_dim = 4
        
        # Attention settings
        self.use_dw = True  # Use distance weight
        self.use_cw = True  # Use community weight
        
        # Training hyperparameters
        self.lr = 0.0001
        self.weight_decay = 1e-6
        self.batch_size = 64
        self.epochs = 70
        
        # Data settings
        self.dataset = 'REST-MDD'  # Dataset name
        self.atlas = 'cc200'  # Atlas name ('cc200' or 'aal116')
        self.dual_atlas = False  # Use dual-atlas model
        self.Kfold = 10  # Number of folds for cross-validation
        self.topk_edge = 30  # Number of edges to keep per node
        self.FBNC = 'PCC'  # Functional brain network construction method
        
        # Logging
        self.save_logs = False
        self.use_tensorboard = True
        self.use_csv = True
        
        # Other
        self.warmup = None  # Warmup iterations (None to disable)
        self.layer_norm = False  # Use layer normalization
        self.global_pool = 'mean'  # Global pooling method
        
        # Ablation study settings
        self.use_gnn = True  # Use GNN (SpatialExtractor) module
        self.use_attention = True  # Use Attention (Transformer) module
        self.use_hierarchical_graph = True  # Use hierarchical graph structure (global + subgraph + community)
    
    def update(self, **kwargs):
        """
        Update configuration from keyword arguments
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def __repr__(self):
        items = ', '.join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"TNAConfig({items})"

