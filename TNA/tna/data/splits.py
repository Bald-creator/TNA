"""
Data splitting utilities for K-fold cross-validation
"""
import random
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def train_test_splitKFold(kfold=5, random_state=42, n_sub=None):
    """
    K-fold train-test split
    
    Args:
        kfold: Number of folds
        random_state: Random seed
        n_sub: Number of subjects
        
    Returns:
        list: List of tuples (train_indices, test_indices)
    """
    indices = list(range(n_sub))
    random.seed(random_state)
    random.shuffle(indices)
    kf = KFold(n_splits=kfold, random_state=random_state, shuffle=True)
    
    splits = []
    for tr, te in kf.split(np.array(indices)):
        splits.append((tr.astype(np.int64), te.astype(np.int64)))
    
    return splits


def StratifiedKFold_tr_te_lab(n_splits=5, random_state=42, n_sub=None, x=None, label=None):
    """
    Stratified K-fold split based on labels
    
    Args:
        n_splits: Number of splits
        random_state: Random seed
        n_sub: Number of subjects (not used if x and label provided)
        x: Feature array (optional)
        label: Label array
        
    Returns:
        list: List of tuples (train_indices, test_indices)
    """
    if label is None:
        raise ValueError("label must be provided for stratified split")
    
    if x is None:
        x = np.arange(len(label))
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    splits = []
    for tr, te in skf.split(x, label):
        splits.append((tr.astype(np.int64), te.astype(np.int64)))
    
    return splits

