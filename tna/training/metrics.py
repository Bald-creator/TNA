"""
Evaluation metrics for brain disease classification
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report
)


def compute_classification_metrics(labels, predictions, probabilities=None):
    """
    Compute comprehensive classification metrics
    
    Args:
        labels: True labels (numpy array or list)
        predictions: Predicted labels (numpy array or list)
        probabilities: Prediction probabilities for positive class (optional)
        
    Returns:
        dict: Dictionary containing various metrics
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = (predictions == labels).mean()
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # AUC (if probabilities provided)
    if probabilities is not None:
        probabilities = np.array(probabilities)
        metrics['auc'] = roc_auc_score(labels, probabilities)
    
    # Per-class metrics
    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    
    # Sensitivity and Specificity for binary classification
    if len(np.unique(labels)) == 2:
        metrics['sensitivity'] = report.get('1.0', report.get('1', {})).get('recall', 0)
        metrics['specificity'] = report.get('0.0', report.get('0', {})).get('recall', 0)
    
    return metrics


def format_metrics(metrics, prefix=''):
    """
    Format metrics dictionary for printing
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string (e.g., 'Train', 'Val', 'Test')
        
    Returns:
        str: Formatted string
    """
    lines = []
    if prefix:
        lines.append(f"{prefix} Metrics:")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return '\n'.join(lines)

