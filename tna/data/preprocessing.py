"""
Data preprocessing utilities for brain network data
"""
import os.path as osp
import h5py
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as utils
from networkx.convert_matrix import from_numpy_array
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

from ..configs.atlas_config import get_atlas_config
from .atlas_utils import (
    rearrange_nodes,
    keep_top_k_symmetric,
    compute_pe,
    get_comm_edge_index,
    get_community_graph,
    normalize,
)


def read_brain_data(data_dir, filename, dist_matrix, coordinate, rearranged_indices,
                   edge_num=30, pe_dim=30, atlas_name='cc200'):
    """
    Read and preprocess brain network data from H5 file
    
    Args:
        data_dir: Directory containing data files
        filename: Name of H5 data file
        dist_matrix: Distance matrix between brain regions
        coordinate: MNI coordinates of brain regions
        rearranged_indices: Node rearrangement indices (for community structure)
        edge_num: Number of top edges to keep per node
        pe_dim: Dimension of positional encoding
        atlas_name: Name of atlas ('cc200' or 'aal116')
        
    Returns:
        torch_geometric.data.Data: Preprocessed brain network data
    """
    with h5py.File(osp.join(data_dir, filename), 'r') as f:
        correlation_matrix = f['corr'][:]
        label = f['label'][()]
    
    correlation_matrix[np.isnan(correlation_matrix)] = 0
    num_nodes = correlation_matrix.shape[0]
    
    rearranged_corr = rearrange_nodes(correlation_matrix, rearranged_indices)
    rearranged_dist = rearrange_nodes(dist_matrix, rearranged_indices)
    
    adjacency = np.abs(rearranged_corr)
    row, col = np.diag_indices_from(adjacency)
    adjacency[row, col] = 0
    
    sparse_adj = keep_top_k_symmetric(adjacency, edge_num)
    
    G = from_numpy_array(sparse_adj)
    A = nx.to_scipy_sparse_array(G)
    sparse_adj = A.tocoo()
    
    edge_attr = np.zeros((len(sparse_adj.row), 2))
    distance_weights = np.exp(-rearranged_dist)
    
    for i in range(len(sparse_adj.row)):
        edge_attr[i, 0] = adjacency[sparse_adj.row[i], sparse_adj.col[i]]
        edge_attr[i, 1] = distance_weights[sparse_adj.row[i], sparse_adj.col[i]]
    
    edge_index = np.stack([sparse_adj.row, sparse_adj.col])
    edge_index, edge_attr = remove_self_loops(
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_attr)
    )
    edge_index = edge_index.long()
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
    
    x = torch.from_numpy(rearranged_corr).float()
    x[x == float('inf')] = 0
    y = torch.from_numpy(np.array(label)).long()
    edge_attr = edge_attr.float()
    
    pe = compute_pe(edge_index, num_nodes, pos_enc_dim=pe_dim)
    
    config = get_atlas_config(atlas_name)
    comm_boundaries = config['comm_boundaries']
    
    subgraph_edge_index, subgraph_edge_attr, subgraph_pe, subgraph_degree = get_comm_edge_index(
        comm_boundaries, edge_index, edge_attr, num_nodes, pos_enc_dim=pe_dim
    )
    
    community_graph = get_community_graph(G, comm_boundaries)
    community_graph = normalize(community_graph)
    comm_graph = torch.from_numpy(community_graph)
    
    degree = utils.degree(edge_index[0], num_nodes)
    
    coord = torch.from_numpy(np.array(coordinate)[rearranged_indices, :]).float()
    rearranged_dist = torch.from_numpy(rearranged_dist).float()
    
    data = Data(
        x=x,
        edge_index=edge_index.long(),
        y=y,
        edge_attr=edge_attr,
        subgraph_edge_index=subgraph_edge_index,
        subgraph_edge_attr=subgraph_edge_attr,
        pe=pe,
        deg=degree,
        subgraph_pe=subgraph_pe,
        subgraph_deg=subgraph_degree,
        coord=coord,
        dist_matrix=rearranged_dist,
        comm_graph=comm_graph
    )
    
    return data

