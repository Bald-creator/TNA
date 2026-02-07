"""
Community Utilities for Brain Atlas
Functions for community detection, distance matrix computation, and graph processing
"""
import os
import pickle
import numpy as np
import torch
import torch_geometric.utils as utils
from torch_scatter import scatter_add
from pyreadr import pyreadr

from ..configs.atlas_config import get_atlas_config


def get_comm_index(atlas_name='cc200', metadata_dir=None):
    """
    Get community rearrangement indices for specified atlas
    
    Args:
        atlas_name: Name of the atlas
        metadata_dir: Directory containing atlas metadata files
        
    Returns:
        np.ndarray: Rearranged node indices
    """
    config = get_atlas_config(atlas_name)
    
    if metadata_dir is None:
        metadata_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_dir = os.path.join(metadata_dir, '..', '..', 'data', 'atlas_metadata')
    
    rearrange_file = os.path.join(metadata_dir, config['rearrange_file'])
    with open(rearrange_file, 'rb') as handle:
        rearrange_data = pickle.load(handle)
    return rearrange_data['rearranged_indices']


def get_dist_matrix(atlas_name='cc200', metadata_dir=None):
    """
    Get distance matrix and MNI coordinates for atlas
    
    Args:
        atlas_name: Name of the atlas
        metadata_dir: Directory containing atlas metadata files
        
    Returns:
        tuple: (distance_matrix, coordinates) as numpy arrays
    """
    config = get_atlas_config(atlas_name)
    
    if metadata_dir is None:
        metadata_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_dir = os.path.join(metadata_dir, '..', '..', 'data', 'atlas_metadata')
    
    rda_path = os.path.join(metadata_dir, config['rda_file'])
    
    result = pyreadr.read_r(rda_path)
    df = result[config['rda_key']]
    coordinate = df.loc[:, "x.mni":"z.mni"]
    x = df["x.mni"]
    y = df["y.mni"]
    z = df["z.mni"]
    
    num_nodes = len(x)
    dist_matrix = np.ones((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            X = np.array([x[i], y[i], z[i]])
            Y = np.array([x[j], y[j], z[j]])
            dist_matrix[i, j] = np.sqrt(sum(np.power((X - Y), 2)))
    
    return dist_matrix, coordinate


def get_community_graph(brain_graph, comm_index):
    """
    Construct community-level graph from node-level brain graph
    
    Args:
        brain_graph: NetworkX graph of brain network
        comm_index: Community boundaries (list of indices)
        
    Returns:
        np.ndarray: Community-level adjacency matrix
    """
    num_communities = len(comm_index) - 1
    community_graph = np.zeros((num_communities, num_communities))
    
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            inter_community_edges = 0
            for node_i in range(comm_index[i], comm_index[i+1]):
                for node_j in range(comm_index[j], comm_index[j+1]):
                    if brain_graph.has_edge(node_i, node_j):
                        inter_community_edges += 1
            
            if inter_community_edges > 0:
                community_graph[i, j] = inter_community_edges
                community_graph[j, i] = inter_community_edges
    
    return community_graph


def rearrange_nodes(node_feature, rearranged_indices):
    """
    Rearrange nodes according to community structure
    
    Args:
        node_feature: Node feature matrix
        rearranged_indices: New node order
        
    Returns:
        np.ndarray: Rearranged node feature matrix
    """
    node_feature_rearranged = node_feature[rearranged_indices, :]
    node_feature_rearranged = node_feature_rearranged[:, rearranged_indices]
    return node_feature_rearranged


def compute_pe(edge_index, num_nodes, pos_enc_dim):
    """
    Compute random walk positional encoding
    
    Args:
        edge_index: Graph edge index
        num_nodes: Number of nodes
        pos_enc_dim: Dimension of positional encoding
        
    Returns:
        torch.Tensor: Positional encoding matrix
    """
    W0 = normalize_adj(edge_index, num_nodes=num_nodes).tocsc()
    W = W0
    vector = torch.zeros((num_nodes, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
    
    return vector.float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    """
    Normalize adjacency matrix with node degree
    
    Args:
        edge_index: Graph edge index
        edge_weight: Edge weights (optional)
        num_nodes: Number of nodes (optional)
        
    Returns:
        scipy.sparse matrix: Normalized adjacency matrix
    """
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


def normalize(matrix):
    """
    Min-max normalization
    
    Args:
        matrix: Input matrix
        
    Returns:
        np.ndarray: Normalized matrix
    """
    min_val = matrix.min()
    max_val = matrix.max()
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def keep_top_k_symmetric(matrix, k):
    """
    Keep only top-k largest values per row (symmetric)
    
    Args:
        matrix: Input adjacency matrix
        k: Number of edges to keep per node
        
    Returns:
        np.ndarray: Sparse adjacency matrix
    """
    for i in range(matrix.shape[0]):
        top_k_indices = np.argsort(matrix[i])[:-k - 1:-1]
        
        for j in range(matrix.shape[1]):
            if j not in top_k_indices and matrix[i][j] != 0:
                matrix[i][j] = 0
                matrix[j][i] = 0
    
    return matrix


def get_comm_edge_index(comm_index, edge_index, edge_attr, num_nodes, pos_enc_dim):
    """
    Extract community subgraphs and compute their properties
    
    Args:
        comm_index: Community boundaries
        edge_index: Full graph edge index
        edge_attr: Edge attributes
        num_nodes: Total number of nodes
        pos_enc_dim: Positional encoding dimension
        
    Returns:
        tuple: (edge_indices, edge_attributes, pe, degrees)
    """
    edge_indices = []
    edge_attributes = []
    pe = []
    deg = []
    
    for start, end in zip(comm_index, comm_index[1:]):
        node_set = torch.IntTensor(list(range(start, end)))
        sub_edge_index, sub_edge_attr = utils.subgraph(
            node_set, edge_index, edge_attr,
            num_nodes=num_nodes,
            return_edge_mask=False
        )
        edge_indices.append(sub_edge_index)
        edge_attributes.append(sub_edge_attr)
        
        if sub_edge_index.numel() > 0:
            sub_edge_index = sub_edge_index - start
        
        sub_pe = compute_pe(sub_edge_index, len(node_set), pos_enc_dim)
        pe.append(sub_pe)
        
        sub_deg = utils.degree(sub_edge_index[0], len(node_set))
        deg.append(sub_deg)
    
    return (
        torch.cat(edge_indices, dim=1),
        torch.cat(edge_attributes, dim=0),
        torch.cat(pe, dim=0),
        torch.cat(deg, dim=0)
    )

