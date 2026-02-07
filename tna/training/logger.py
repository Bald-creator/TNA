"""
Logging utilities for experiment tracking
"""
import os
import csv
from pathlib import Path
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Unified logger for experiments
    Supports TensorBoard and CSV logging
    """
    
    def __init__(self, log_dir, use_tensorboard=True, use_csv=True):
        """
        Initialize experiment logger
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard logging
            use_csv: Whether to use CSV logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        
        if self.use_tensorboard:
            tensorboard_dir = self.log_dir / 'tensorboard'
            self.tb_writer = SummaryWriter(str(tensorboard_dir))
        else:
            self.tb_writer = None
        
        if self.use_csv:
            self.csv_path = self.log_dir / 'logs.csv'
            self.csv_data = []
        else:
            self.csv_path = None
            self.csv_data = None
    
    def log_epoch(self, epoch, train_metrics, val_metrics=None):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict (optional)
        """
        # TensorBoard logging
        if self.tb_writer is not None:
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'train/{key}', value, epoch)
            
            if val_metrics is not None:
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(f'val/{key}', value, epoch)
        
        # CSV logging
        if self.csv_data is not None:
            row = {'epoch': epoch}
            row.update({f'train_{k}': v for k, v in train_metrics.items()})
            if val_metrics is not None:
                row.update({f'val_{k}': v for k, v in val_metrics.items()})
            self.csv_data.append(row)
    
    def log_scalars(self, tag, scalar_dict, global_step):
        """
        Log multiple scalars under same tag
        
        Args:
            tag: Main tag (e.g., 'loss', 'accuracy')
            scalar_dict: Dictionary of {name: value}
            global_step: Global step (epoch number)
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(tag, scalar_dict, global_step)
    
    def save_csv(self):
        """Save CSV logs to disk"""
        if self.csv_data:
            df = pd.DataFrame(self.csv_data)
            df.to_csv(self.csv_path, index=False)
    
    def save_final_results(self, results_dict):
        """
        Save final experiment results
        
        Args:
            results_dict: Dictionary of final results
        """
        results_path = self.log_dir / 'final_results.csv'
        df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['value'])
        df.index.name = 'metric'
        df.to_csv(results_path)
    
    def save_results(self, results_dict, filename="results.csv"):
        """
        Save results to CSV file
        
        Args:
            results_dict: Dictionary of results
            filename: Name of the output file
        """
        results_path = self.log_dir / filename
        
        # Handle different result formats
        if isinstance(results_dict, dict):
            # Check if it's a simple dict or nested
            if any(isinstance(v, (list, dict)) for v in results_dict.values()):
                # Complex nested structure
                import json
                with open(str(results_path).replace('.csv', '.json'), 'w') as f:
                    json.dump(results_dict, f, indent=2)
            else:
                # Simple dict
                df = pd.DataFrame([results_dict])
                df.to_csv(results_path, index=False)
    
    def save_model_checkpoint(self, model_state_dict, config_dict, filename="model.pth"):
        """
        Save model checkpoint
        
        Args:
            model_state_dict: Model state dictionary
            config_dict: Configuration dictionary
            filename: Checkpoint filename (can be full path or just filename)
        """
        import torch
        
        # Handle both full path and filename
        if os.path.isabs(filename):
            checkpoint_path = filename
        else:
            checkpoint_path = self.log_dir / filename
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'config': config_dict
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to: {checkpoint_path}")
    
    def close(self):
        """Close logger and save all pending data"""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.csv_data is not None:
            self.save_csv()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

