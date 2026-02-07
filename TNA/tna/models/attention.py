"""
Spatial Attention with Distance and Community Weights
"""
import torch
from torch import nn
import torch_geometric.nn as gnn
from einops import rearrange


class SpatialAttention(gnn.MessagePassing):
    """
    Spatial attention mechanism with distance and community priors
    Uses learnable distance weights and community weights
    """
    
    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
                 symmetric=False, num_nodes=200, comm_boundaries=None, **kwargs):
        """
        Initialize spatial attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Use bias in projections
            symmetric: Use symmetric Q/K projections
            num_nodes: Number of nodes in graph
            comm_boundaries: Community boundaries
        """
        super().__init__(node_dim=0, aggr='add')
        
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.num_nodes = num_nodes
        
        if comm_boundaries is not None:
            self.comm_boundaries = comm_boundaries
        else:
            self.comm_boundaries = [0, 48, 71, 91, 113, 129, 152, 200]
        
        self.attend = nn.Softmax(dim=-1)
        
        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()
        
        # Learnable parameters for distance and community weights
        self.alpha = nn.Parameter(torch.ones(num_heads), requires_grad=True)
        self.theta1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.ones(1), requires_grad=True)
        num_communities = len(self.comm_boundaries) - 1
        self.epsilon = nn.Parameter(torch.ones(self.num_heads, num_communities), requires_grad=True)
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)
    
    def forward(self, x, edge_index, subgraph_edge_index=None,
                subgraph_edge_attr=None, edge_attr=None, ptr=None,
                batch=None, return_attn=False, dm=None, coord=None,
                comm_edge_index=None, comm_edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge indices
            dm: Distance matrix
            coord: Node coordinates
            return_attn: Whether to return attention weights
            
        Returns:
            tuple: (output, attention_weights)
        """
        v = self.to_v(x)
        
        if self.symmetric:
            qk = self.to_qk(x)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x).chunk(2, dim=-1)
        
        out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn, dm=dm, coord=coord)
        return self.out_proj(out), attn
    
    def self_attn(self, qk, v, ptr, return_attn=False, dm=None, coord=None):
        """Compute self-attention with spatial priors"""
        k, q = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d',
                                       n=self.num_nodes, h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Distance weights
        distance_weights = [
            self.distance_weight_matrix(dm, self.alpha[i])
            for i in range(self.num_heads)
        ]
        distance_weights = torch.stack(distance_weights, dim=0)
        
        # Community weights
        comm_weights = [
            self.community_weight_matrix(self.epsilon[i])
            for i in range(self.num_heads)
        ]
        comm_weights = torch.stack(comm_weights, dim=0)
        
        # Normalize weights
        min_dis = distance_weights.min()
        max_dis = distance_weights.max()
        distance_weights = (distance_weights - min_dis) / (max_dis - min_dis)
        
        min_comm = comm_weights.min()
        max_comm = comm_weights.max()
        comm_weights = (comm_weights - min_comm) / (max_comm - min_comm)
        
        # Add spatial priors to attention
        dots = dots + 0.2 * distance_weights * self.theta1 + 0.2 * comm_weights * self.theta2
        
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        
        v = rearrange(v, '(b n) (h d) -> b h n d', n=self.num_nodes, h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> (b n) (h d)')
        
        if return_attn:
            # Optional: save attention weights when TNA_ATTN_LOG_DIR is set (e.g. for debugging)
            import os
            attn_log_dir = os.environ.get('TNA_ATTN_LOG_DIR')
            if attn_log_dir:
                import datetime
                os.makedirs(attn_log_dir, exist_ok=True)
                log_path = os.path.join(attn_log_dir, 'attention_weights.txt')
                reduced_weights = dots.mean(dim=0).mean(dim=0)
                np_array = reduced_weights.detach().cpu().numpy()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_path, 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Timestamp: {timestamp} | Matrix shape: {np_array.shape} ({self.num_nodes} nodes)\n")
                    f.write(f"{'='*80}\n")
                    for row in np_array:
                        line = ' '.join([f"{x:.6f}" for x in row])
                        f.write(line + '\n')
                    f.write(f"{'='*80}\n\n")
            return out, dots
        return out, None
    
    def distance_weight_matrix(self, distances, alpha):
        """Compute distance-based weights using Gaussian kernel"""
        weight_matrix = torch.exp(-distances ** 2 / (2 * alpha ** 2))
        weight_matrix = weight_matrix.clone()
        weight_matrix.fill_diagonal_(0)
        return weight_matrix
    
    def community_weight_matrix(self, epsilon):
        """Compute community-based weights"""
        comm_index = self.comm_boundaries
        
        comm_weight = torch.zeros(comm_index[-1], comm_index[-1]).to(
            next(self.parameters()).device
        )
        for start, end, eps in zip(comm_index[:-1], comm_index[1:], epsilon):
            comm_weight[start:end, start:end].fill_(eps)
        
        comm_weight = comm_weight.clone()
        comm_weight.fill_diagonal_(0)
        return comm_weight

