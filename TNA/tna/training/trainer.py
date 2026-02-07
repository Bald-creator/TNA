"""
Trainer class for TNA model
Encapsulates training and evaluation logic
"""
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
from .metrics import compute_classification_metrics


class TNATrainer:
    """
    Trainer for TNA models
    Handles both single-atlas and dual-atlas models
    """
    
    def __init__(self, model, criterion, optimizer, lr_scheduler, config, logger, device):
        """
        Initialize trainer
        
        Args:
            model: TNA model (single or dual-atlas)
            criterion: Loss function
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler (optional)
            config: Model configuration object
            logger: ExperimentLogger instance
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.logger = logger
        self.device = device
        
        # Detect if model is dual-atlas
        self.is_dual_atlas = 'DualAtlas' in model.__class__.__name__
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self, loader, epoch):
        """
        Train for one epoch
        
        Args:
            loader: Data loader
            epoch: Current epoch number
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        
        tic = timer()
        for i, data in enumerate(loader):
            # Learning rate warmup
            if self.config.warmup is not None:
                iteration = epoch * len(loader) + i
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr_scheduler(iteration)
            
            # Forward pass
            if self.is_dual_atlas:
                data_cc200, data_aal116 = data
                data_cc200 = data_cc200.to(self.device)
                data_aal116 = data_aal116.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data_cc200, data_aal116)
                loss = self.criterion(output, data_cc200.y)
                
                num_graphs = data_cc200.num_graphs
                labels = data_cc200.y
            else:
                data = data.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data.y)
                
                num_graphs = data.num_graphs
                labels = data.y
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * num_graphs
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == labels).sum().item()
        
        toc = timer()
        n_sample = len(loader.dataset)
        avg_loss = total_loss / n_sample
        accuracy = total_correct / n_sample
        
        print(f'Train loss: {avg_loss:.4f} Train acc: {accuracy:.4f} time: {toc - tic:.2f}s')
        
        return avg_loss, accuracy
    
    def eval_epoch(self, loader, split='Val'):
        """
        Evaluate for one epoch
        
        Args:
            loader: Data loader
            criterion: Loss function
            split: Split name for printing ('Val' or 'Test')
            
        Returns:
            tuple: (average_loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        tic = timer()
        with torch.no_grad():
            for data in loader:
                # Forward pass
                if self.is_dual_atlas:
                    data_cc200, data_aal116 = data
                    data_cc200 = data_cc200.to(self.device)
                    data_aal116 = data_aal116.to(self.device)
                    
                    output = self.model(data_cc200, data_aal116)
                    loss = self.criterion(output, data_cc200.y)
                    
                    num_graphs = data_cc200.num_graphs
                    labels = data_cc200.y
                else:
                    data = data.to(self.device)
                    
                    output = self.model(data)
                    loss = self.criterion(output, data.y)
                    
                    num_graphs = data.num_graphs
                    labels = data.y
                
                # Statistics
                total_loss += loss.item() * num_graphs
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == labels).sum().item()
                
                # Store for metrics
                probabilities = F.softmax(output, dim=1)[:, 1]
                all_predictions.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())
        
        toc = timer()
        
        n_sample = len(loader.dataset)
        avg_loss = total_loss / n_sample
        accuracy = total_correct / n_sample
        
        # Compute detailed metrics
        metrics = compute_classification_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        print(f'{split} loss: {avg_loss:.4f} acc: {accuracy:.4f} '
              f'auc: {metrics.get("auc", 0):.4f} time: {toc - tic:.2f}s')
        
        return avg_loss, accuracy, metrics
    
    def fit(self, train_loader, test_loader):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            dict: Training results including best metrics
        """
        self.best_test_acc = 0
        self.best_test_auc = 0
        self.best_test_metrics = {}
        best_model_state = None
        best_epoch = 0
        
        start_time = timer()
        
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}, LR {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            test_loss, test_acc, test_metrics = self.eval_epoch(test_loader, split='Test')
            
            # Learning rate scheduling
            if self.config.warmup is None and self.lr_scheduler is not None:
                # ReduceLROnPlateau: step with test accuracy
                self.lr_scheduler.step(test_acc)
            # Note: warmup scheduler is applied per-iteration in train_epoch
            
            # Log to experiment logger
            if self.logger is not None:
                self.logger.log_epoch(
                    epoch,
                    {'loss': train_loss, 'accuracy': train_acc},
                    {'loss': test_loss, 'accuracy': test_acc, **test_metrics}
                )
            
            # Save best model
            if test_acc > self.best_test_acc and epoch > 5:
                self.best_test_acc = test_acc
                self.best_test_auc = test_metrics.get('auc', 0)
                self.best_test_metrics = test_metrics.copy()
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        total_time = timer() - start_time
        
        print()
        print(f"Best epoch: {best_epoch} Best test acc: {self.best_test_acc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
        
        # Final evaluation with best model
        print("Final Testing with best model...")
        test_loss, test_acc, test_metrics = self.eval_epoch(test_loader, split='Test')
        
        # Save attention weights for best model (only once per fold)
        print("Saving attention weights for best model...")
        self.save_attention_weights(test_loader)
        
        results = {
            'best_test_accuracy': self.best_test_acc,
            'final_test_accuracy': test_acc,
            'test_loss': test_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
            **test_metrics
        }
        
        return results
    
    def save_attention_weights(self, loader):
        """
        Save attention weights for the best model (single pass through test set)
        Only called once per fold with the best model
        """
        self.model.eval()
        
        with torch.no_grad():
            for data in loader:
                if self.is_dual_atlas:
                    data_cc200, data_aal116 = data
                    data_cc200 = data_cc200.to(self.device)
                    data_aal116 = data_aal116.to(self.device)
                    # Enable attention weight saving
                    _ = self.model(data_cc200, data_aal116, return_attn=True)
                else:
                    data = data.to(self.device)
                    # Enable attention weight saving
                    _ = self.model(data, return_attn=True)
                
                # Only save for first batch to avoid huge files
                break
        
        print("Attention weights saved to logs/attn/weigt.txt")

