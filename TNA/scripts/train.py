"""
Training script for TNA
Supports single-atlas and dual-atlas modes with K-fold cross-validation
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tna.models import TNA, DualAtlasTNA
from tna.data.dataset import TNADataset, DualAtlasTNADataset, DualAtlasTNASubset
from tna.data.splits import train_test_splitKFold, StratifiedKFold_tr_te_lab
from tna.training.logger import ExperimentLogger
from tna.training.trainer import TNATrainer
from tna.utils.helpers import count_parameters
from tna.configs.model_config import TNAConfig
from tna.configs.path_config import PathConfig


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config, device):
    """Create model based on configuration"""
    model_kwargs = {
        'd_model': config.dim_hidden,
        'num_heads': config.num_heads,
        'dim_feedforward': config.dim_hidden * 4,
        'dropout': config.dropout,
        'num_layers': config.num_layers,
        'batch_norm': True,
        'pe': config.pe is not None and config.pe != "",
        'pe_dim': config.pe_dim,
        'gnn_type': config.gnn_type,
        'se': config.se,
        'use_edge_attr': config.use_edge_attr,
        'num_edge_features': 2,  # Fixed: actual edge feature dimension is 2
        'edge_dim': config.edge_dim,
        'use_gnn': config.use_gnn,  # Ablation: GNN module
        'use_attention': config.use_attention,  # Ablation: Attention module
        'use_hierarchical_graph': config.use_hierarchical_graph,  # Ablation: Hierarchical graph
    }
    
    if config.dual_atlas:
        model = DualAtlasTNA(
            num_class=2,
            **model_kwargs
        )
    else:
        from tna.configs.atlas_config import get_atlas_config
        atlas_cfg = get_atlas_config(config.atlas)
        
        model = TNA(
            in_size=atlas_cfg['num_nodes'],
            num_class=2,
            num_nodes=atlas_cfg['num_nodes'],
            comm_boundaries=atlas_cfg['comm_boundaries'],
            **model_kwargs
        )
    
    model = model.to(device)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    return model


def create_dataloaders(config, path_config, fold_idx=0):
    """Create dataloaders for training and testing"""
    if config.dual_atlas:
        # Dual-atlas mode
        full_dataset = DualAtlasTNADataset(
            root=path_config.data_dir,
            dataset_name=config.dataset.lower().replace('-', '_'),
            atlas_cc200='cc200',
            atlas_aal116='aal116'
        )
        
        # Get K-fold split
        num_samples = len(full_dataset)
        train_idx, test_idx = train_test_splitKFold(
            kfold=config.Kfold,
            random_state=42,
            n_sub=num_samples
        )[fold_idx]
        
        train_dataset = DualAtlasTNASubset(full_dataset, train_idx)
        test_dataset = DualAtlasTNASubset(full_dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    else:
        # Single-atlas mode
        full_dataset = TNADataset(
            root=path_config.data_dir,
            dataset_name=config.dataset.lower().replace('-', '_'),
            atlas_name=config.atlas
        )
        
        num_samples = len(full_dataset)
        train_idx, test_idx = train_test_splitKFold(
            kfold=config.Kfold,
            random_state=42,
            n_sub=num_samples
        )[fold_idx]
        
        train_dataset = full_dataset[train_idx]
        test_dataset = full_dataset[test_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Fold {fold_idx + 1}/{config.Kfold}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def train_kfold(config, path_config, device, run_name):
    """Train model with K-fold cross-validation"""
    all_fold_results = []
    
    # Create run-specific directories
    run_output_dir = os.path.join(path_config.output_dir, run_name)
    run_tensorboard_dir = os.path.join(path_config.tensorboard_log_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(run_tensorboard_dir, exist_ok=True)
    
    print(f"Run directory: {run_name}")
    print(f"Output logs: {run_output_dir}")
    print(f"TensorBoard logs: {run_tensorboard_dir}\n")
    
    # Lists to collect metrics across folds
    acc_list = []
    auc_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for fold_idx in range(config.Kfold):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx + 1}/{config.Kfold}")
        print(f"{'='*60}")
        
        # Create model for this fold
        model = create_model(config, device)
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(config, path_config, fold_idx)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.warmup is not None:
            # Custom warmup + decay scheduler
            lr_steps = (config.lr - 1e-6) / config.warmup
            decay_factor = config.lr * config.warmup ** 0.5
            
            def lr_scheduler_fn(iteration):
                if iteration < config.warmup:
                    lr = 1e-6 + iteration * lr_steps
                else:
                    lr = decay_factor * iteration ** -0.5
                return lr
            
            lr_scheduler = lr_scheduler_fn
        else:
            # ReduceLROnPlateau: reduce LR when test accuracy plateaus
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max',
                factor=0.5,
                patience=10,
                min_lr=1e-05,
                verbose=True
            )
        
        # Create logger for this fold
        fold_log_dir = os.path.join(run_tensorboard_dir, f"fold{fold_idx + 1}")
        logger = ExperimentLogger(
            log_dir=fold_log_dir,
            use_tensorboard=config.save_logs,
            use_csv=True
        )
        
        # Create trainer
        trainer = TNATrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            logger=logger,
            device=device
        )
        
        # Train
        trainer.fit(train_loader, test_loader)
        
        # Collect metrics from this fold
        best_metrics = trainer.best_test_metrics
        acc_list.append(trainer.best_test_acc)
        auc_list.append(best_metrics.get('auc', 0))
        sensitivity_list.append(best_metrics.get('sensitivity', 0))
        specificity_list.append(best_metrics.get('specificity', 0))
        precision_list.append(best_metrics.get('precision', 0))
        recall_list.append(best_metrics.get('recall', 0))
        f1_list.append(best_metrics.get('f1', 0))
        
        # Save fold results
        fold_result = {
            'fold': fold_idx + 1,
            'best_test_acc': trainer.best_test_acc,
            'best_test_auc': best_metrics.get('auc', 0),
            'sensitivity': best_metrics.get('sensitivity', 0),
            'specificity': best_metrics.get('specificity', 0),
            'precision': best_metrics.get('precision', 0),
            'recall': best_metrics.get('recall', 0),
            'f1': best_metrics.get('f1', 0),
        }
        all_fold_results.append(fold_result)
        
        # Save model checkpoint to run-specific output directory
        checkpoint_path = os.path.join(
            run_output_dir,
            f"model_fold{fold_idx + 1}.pth"
        )
        logger.save_model_checkpoint(model.state_dict(), config.to_dict(), checkpoint_path)
        
        logger.close()
        
        print(f"\nFold {fold_idx + 1} completed:")
        print(f"  Best Test Accuracy: {trainer.best_test_acc:.4f}")
        print(f"  Best Test AUC: {best_metrics.get('auc', 0):.4f}")
        print(f"  Sensitivity: {best_metrics.get('sensitivity', 0):.4f}")
        print(f"  Specificity: {best_metrics.get('specificity', 0):.4f}")
    
    # Compute and print average results (same format as original project)
    print(f"\n{'='*60}")
    print("K-Fold Cross-Validation Results")
    print(f"{'='*60}")
    
    # Print individual fold results
    print(acc_list)
    
    # Compute statistics
    avg_acc = np.mean(acc_list) * 100
    std_acc = np.std(acc_list) * 100
    avg_auc = np.mean(auc_list) * 100
    std_auc = np.std(auc_list) * 100
    avg_sen = np.mean(sensitivity_list) * 100
    std_sen = np.std(sensitivity_list) * 100
    avg_spec = np.mean(specificity_list) * 100
    std_spec = np.std(specificity_list) * 100
    avg_recall = np.mean(recall_list) * 100
    std_recall = np.std(recall_list) * 100
    avg_prec = np.mean(precision_list) * 100
    std_prec = np.std(precision_list) * 100
    avg_f1 = np.mean(f1_list) * 100
    std_f1 = np.std(f1_list) * 100
    
    # Print results (same format as original project)
    print(f"test acc mean {avg_acc} std {std_acc}")
    print(f"test auc mean {avg_auc} std {std_auc}")
    print(f"test sensitivity mean {avg_sen} std {std_sen}")
    print(f"test specficity mean {avg_spec} std {std_spec}")
    print(f"test recall mean {avg_recall} std {std_recall}")
    print(f"test precision mean {avg_prec} std {std_prec}")
    print(f"test f1_micro mean {avg_f1} std {std_f1}")
    
    # Save summary to run-specific output directory
    summary_logger = ExperimentLogger(
        log_dir=run_output_dir,
        use_tensorboard=False,
        use_csv=True
    )
    summary_logger.save_results(
        {
            'avg_test_acc': avg_acc / 100,
            'std_test_acc': std_acc / 100,
            'avg_test_auc': avg_auc / 100,
            'std_test_auc': std_auc / 100,
            'avg_sensitivity': avg_sen / 100,
            'std_sensitivity': std_sen / 100,
            'avg_specificity': avg_spec / 100,
            'std_specificity': std_spec / 100,
            'avg_recall': avg_recall / 100,
            'std_recall': std_recall / 100,
            'avg_precision': avg_prec / 100,
            'std_precision': std_prec / 100,
            'avg_f1': avg_f1 / 100,
            'std_f1': std_f1 / 100,
            'folds': all_fold_results
        },
        filename="kfold_summary.json"
    )
    summary_logger.close()
    
    return all_fold_results


def main():
    """Main entry point"""
    # Default config for optional args (no defaults in CLI for reproducibility)
    defaults = TNAConfig()
    parser = argparse.ArgumentParser(description='Train TNA')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory (data and logs under this path)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (e.g., REST-MDD, ABIDE)')
    parser.add_argument('--atlas', type=str, default=None,
                        help='Atlas name (cc200 or aal116)')
    parser.add_argument('--dual_atlas', action='store_true',
                        help='Use dual-atlas mode')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--kfold', type=int, default=None,
                        help='Number of folds for cross-validation')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # Ablation study arguments
    parser.add_argument('--no-gnn', action='store_true',
                        help='Disable GNN (SpatialExtractor) module for ablation study')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable Attention (Transformer) module for ablation study')
    parser.add_argument('--no-hierarchical-graph', action='store_true',
                        help='Disable hierarchical graph structure (use global graph only for ablation study)')
    
    args = parser.parse_args()
    
    # Apply optional args (use config defaults when not provided)
    dataset = args.dataset if args.dataset is not None else defaults.dataset
    atlas = args.atlas if args.atlas is not None else defaults.atlas
    epochs = args.epochs if args.epochs is not None else defaults.epochs
    batch_size = args.batch_size if args.batch_size is not None else defaults.batch_size
    lr = args.lr if args.lr is not None else defaults.lr
    kfold = args.kfold if args.kfold is not None else defaults.Kfold
    gpu = args.gpu if args.gpu is not None else 0
    seed = args.seed if args.seed is not None else 42

    set_seed(seed)

    config = TNAConfig()
    config.update(
        dataset=dataset,
        atlas=atlas,
        dual_atlas=args.dual_atlas,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        Kfold=kfold,
        use_gnn=not args.no_gnn,
        use_attention=not args.no_attention,
        use_hierarchical_graph=not args.no_hierarchical_graph
    )
    
    path_config = PathConfig(base_dir=args.base_dir)
    
    # Setup device
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(path_config.output_dir, exist_ok=True)
    os.makedirs(path_config.tensorboard_log_dir, exist_ok=True)
    
    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    atlas_str = "dual_atlas" if config.dual_atlas else config.atlas
    run_name = f"{config.dataset}_{atlas_str}_{timestamp}"
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Run Name: {run_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Atlas: {config.atlas}")
    print(f"Dual Atlas: {config.dual_atlas}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.lr}")
    print(f"K-Fold: {config.Kfold}")
    print(f"Use GNN: {config.use_gnn}")
    print(f"Use Attention: {config.use_attention}")
    print(f"Use Hierarchical Graph: {config.use_hierarchical_graph}")
    print(f"{'='*60}\n")
    
    # Train with K-fold cross-validation
    train_kfold(config, path_config, device, run_name)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

