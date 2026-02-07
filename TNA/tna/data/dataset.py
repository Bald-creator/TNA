"""
TNA Dataset Classes
Generic dataset loaders for brain network analysis (supports multiple datasets and atlases)
"""
import torch
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Dataset
from os.path import isfile
from os import listdir
import numpy as np
import os.path as osp
from tqdm import tqdm

from .preprocessing import read_brain_data
from .atlas_utils import get_dist_matrix, get_comm_index


class TNADataset(InMemoryDataset):
    """
    Generic brain network dataset loader
    Supports multiple datasets (ABIDE, MDD, etc.) and atlases (CC200, AAL116, etc.)
    """
    
    def __init__(self, root, dataset_name='rest-mdd', atlas_name='cc200',
                 transform=None, pre_transform=None, edge_num=30, pe_dim=30):
        """
        Initialize brain network dataset
        
        Args:
            root: Root directory containing data
            dataset_name: Name of dataset ('abide', 'rest-mdd', etc.)
            atlas_name: Name of atlas ('cc200', 'aal116')
            transform: Data transformation function
            pre_transform: Pre-transformation function
            edge_num: Number of top edges to keep per node
            pe_dim: Positional encoding dimension
        """
        self.root = root
        self.dataset_name = dataset_name
        self.atlas_name = atlas_name
        self.edge_num = edge_num
        self.pe_dim = pe_dim
        
        super(TNADataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_dir(self) -> str:
        """Directory containing raw data files (dataset and atlas-specific)"""
        return osp.join(self.root, 'raw', self.dataset_name, self.atlas_name)
    
    @property
    def processed_dir(self) -> str:
        """Directory for processed data files"""
        return osp.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        """List of raw data filenames"""
        data_dir = self.raw_dir
        if not osp.exists(data_dir):
            return []
        onlyfiles = [f for f in listdir(data_dir) if isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    
    @property
    def processed_file_names(self):
        """Processed data filename (dataset and atlas-specific)"""
        return f'data_{self.dataset_name}_{self.atlas_name}.pt'
    
    def download(self):
        """Download data (not implemented, assumes data is already present)"""
        pass
    
    def process(self):
        """Process raw data into PyTorch Geometric format"""
        data_list = []
        
        metadata_dir = osp.join(self.root, 'atlas_metadata')
        dist_matrix, coordinates = get_dist_matrix(self.atlas_name, metadata_dir)
        rearranged_indices = get_comm_index(self.atlas_name, metadata_dir)
        
        for i, filename in enumerate(tqdm(self.raw_file_names,
                                         desc=f"Processing {self.atlas_name} data")):
            data = read_brain_data(
                self.raw_dir,
                filename,
                dist_matrix,
                coordinates,
                rearranged_indices,
                edge_num=self.edge_num,
                pe_dim=self.pe_dim,
                atlas_name=self.atlas_name
            )
            if data is not None:
                data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
    
    def __repr__(self):
        return f'{self.__class__.__name__}(dataset={self.dataset_name}, atlas={self.atlas_name}, n={len(self)})'


class DualAtlasTNADataset(Dataset):
    """
    Dual-atlas brain network dataset
    Loads both CC200 and AAL116 data for each subject
    """
    
    def __init__(self, root, dataset_name='rest-mdd',
                 atlas_cc200='cc200', atlas_aal116='aal116',
                 edge_num=30, pe_dim=30):
        """
        Initialize dual-atlas dataset
        
        Args:
            root: Root directory containing data
            dataset_name: Name of dataset
            atlas_cc200: Name of first atlas (default 'cc200')
            atlas_aal116: Name of second atlas (default 'aal116')
            edge_num: Number of top edges to keep per node
            pe_dim: Positional encoding dimension
        """
        self.root = root
        self.dataset_name = dataset_name
        
        self.dataset_cc200 = TNADataset(
            root, dataset_name, atlas_cc200, edge_num=edge_num, pe_dim=pe_dim
        )
        
        self.dataset_aal116 = TNADataset(
            root, dataset_name, atlas_aal116, edge_num=edge_num, pe_dim=pe_dim
        )
        
        assert len(self.dataset_cc200) == len(self.dataset_aal116), \
            f"Dataset size mismatch: {atlas_cc200}={len(self.dataset_cc200)}, " \
            f"{atlas_aal116}={len(self.dataset_aal116)}"
    
    def __len__(self):
        return len(self.dataset_cc200)
    
    def __getitem__(self, idx):
        """
        Get data for both atlases
        
        Args:
            idx: Index (int, list, slice, or numpy array)
            
        Returns:
            For single index: tuple of (data_cc200, data_aal116)
            For multiple indices: DualAtlasTNASubset
        """
        if isinstance(idx, (list, slice, np.ndarray)):
            if isinstance(idx, slice):
                indices = list(range(*idx.indices(len(self))))
            elif isinstance(idx, np.ndarray):
                indices = idx.tolist()
            else:
                indices = idx
            return DualAtlasTNASubset(self, indices)
        
        if isinstance(idx, (int, np.integer)):
            idx = int(idx)
            data_cc200 = self.dataset_cc200[idx]
            data_aal116 = self.dataset_aal116[idx]
            
            label_cc200 = data_cc200.y.item() if data_cc200.y.numel() == 1 else data_cc200.y
            label_aal116 = data_aal116.y.item() if data_aal116.y.numel() == 1 else data_aal116.y
            
            if isinstance(label_cc200, int) and isinstance(label_aal116, int):
                assert label_cc200 == label_aal116, \
                    f"Label mismatch at index {idx}: CC200={label_cc200}, AAL116={label_aal116}"
            
            return data_cc200, data_aal116
        
        raise TypeError(f"Invalid index type: {type(idx)}")
    
    @property
    def num_node_features(self):
        return self.dataset_cc200.num_node_features
    
    @property
    def num_edge_features(self):
        return self.dataset_cc200.num_edge_features
    
    def __repr__(self):
        return (f'DualAtlasTNADataset(dataset={self.dataset_name}, '
                f'CC200={len(self.dataset_cc200)}, AAL116={len(self.dataset_aal116)})')


class DualAtlasTNASubset(Dataset):
    """
    Subset of DualAtlasTNADataset for handling indexed access
    """
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]
    
    def __repr__(self):
        return f'DualAtlasTNASubset(size={len(self)})'

