"""
Path Configuration Module
Centralized path management for data, logs, and model checkpoints
"""
import os
from pathlib import Path


class PathConfig:
    """Configuration class for file paths"""
    
    def __init__(self, base_dir=None):
        """
        Initialize path configuration
        
        Args:
            base_dir: Base directory of the project.
                     If None, uses current file's parent directory.
        """
        if base_dir is None:
            # Auto-detect project root (3 levels up: configs -> tna -> project root)
            current_file = Path(__file__).resolve()
            self.base_dir = str(current_file.parent.parent.parent)
        else:
            self.base_dir = str(base_dir)
        
        # Data paths
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed')
        self.atlas_metadata_dir = os.path.join(self.data_dir, 'atlas_metadata')
        
        # Log paths
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.output_dir = os.path.join(self.logs_dir, 'output')
        self.tensorboard_log_dir = os.path.join(self.logs_dir, 'tensorboard')
        
        # Model checkpoint paths (deprecated, use output_dir)
        self.checkpoint_dir = self.output_dir
    
    def get_raw_data_path(self, atlas_name):
        """Get path to raw data for specific atlas"""
        return os.path.join(self.raw_data_dir, atlas_name)

    def get_processed_data_path(self, atlas_name):
        """Get path to processed data file for specific atlas"""
        return os.path.join(self.processed_data_dir, f'data_{atlas_name}.pt')

    def get_atlas_metadata_path(self, filename):
        """Get path to atlas metadata file"""
        return os.path.join(self.atlas_metadata_dir, filename)

    def get_log_dir(self, experiment_name='default'):
        """Get log directory for experiment"""
        return os.path.join(self.logs_dir, experiment_name)

    def create_directories(self):
        """Create all necessary directories"""
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.atlas_metadata_dir,
            self.logs_dir,
            self.output_dir,
            self.tensorboard_log_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

    def to_dict(self):
        """Convert paths to dictionary"""
        return {
            'base_dir': self.base_dir,
            'data_dir': self.data_dir,
            'raw_data_dir': self.raw_data_dir,
            'processed_data_dir': self.processed_data_dir,
            'atlas_metadata_dir': self.atlas_metadata_dir,
            'logs_dir': self.logs_dir,
            'output_dir': self.output_dir,
            'tensorboard_log_dir': self.tensorboard_log_dir,
            'checkpoint_dir': self.checkpoint_dir,
        }

