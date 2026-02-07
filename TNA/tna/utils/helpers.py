"""
Helper utilities for TNA model
"""
import torch
from torch_scatter import scatter


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def pad_batch(x, ptr, return_mask=False):
    """
    Pad variable-length batch to fixed size
    
    Args:
        x: Input tensor or list of tensors
        ptr: Pointer tensor indicating batch boundaries
        return_mask: Whether to return padding mask
        
    Returns:
        Padded tensor(s) and optionally padding mask
    """
    bsz = len(ptr) - 1
    max_num_nodes = torch.diff(ptr).max().item()
    
    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()
    
    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    
    if return_mask:
        return new_x, padding_mask
    return new_x


def unpad_batch(x, ptr):
    """
    Unpad batch back to variable length
    
    Args:
        x: Padded tensor (batch_size, max_nodes, dim)
        ptr: Pointer tensor indicating batch boundaries
        
    Returns:
        torch.Tensor: Unpacked tensor
    """
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    
    return new_x


def extract_node_feature(data, reduce='add'):
    """
    Extract node features from edge attributes
    
    Args:
        data: Graph data object
        reduce: Reduction method ('add', 'mean', 'max')
        
    Returns:
        data: Updated graph data with node features
    """
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(
            data.edge_attr,
            data.edge_index[0],
            dim=0,
            dim_size=data.num_nodes,
            reduce=reduce
        )
    else:
        raise ValueError(f'Unknown aggregation type: {reduce}')
    return data

