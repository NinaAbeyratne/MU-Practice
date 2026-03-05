"""
Membership Inference Attack (MIA) module for measuring privacy leakage.

This module provides black-box membership inference attacks using model confidence scores.
"""

import torch
import numpy as np


def evaluate_mia(model, member_loader, non_member_loader, threshold=None):
    """
    Black-box Membership Inference Attack using model confidence.
    
    Measures privacy leakage by determining if samples were used in training
    based on the model's confidence scores.
    
    Args:
        model: Trained classification model
        member_loader: DataLoader with training data (members -known to the model)
        non_member_loader: DataLoader with non-training data (non-members - unknown to the model)
        threshold: Decision threshold for attack. If None, uses mean confidence of members.
    
    Returns:
        dict with keys:
            - mia_accuracy: Overall attack accuracy (correct predictions / total)
            - tpr: True Positive Rate (members correctly identified)
            - fpr: False Positive Rate (non-members wrongly identified as members)
            - threshold: Decision threshold used
            - member_mean_confidence: Average confidence on member samples
            - non_member_mean_confidence: Average confidence on non-member samples
    
    Note:
        - Works on GPU if model parameters are on CUDA device
        - No training occurs in this function (pure evaluation)
        - Lower MIA accuracy = better privacy
    """
    model.eval()
    device = next(model.parameters()).device
    
    member_confidences = []
    non_member_confidences = []
    
    # Collect confidence scores from members (should be high)
    with torch.no_grad():
        for x, y in member_loader:
            x = x.to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1)
            max_conf = probs.max(dim=1).values
            member_confidences.extend(max_conf.cpu().numpy())
    
    # Collect confidence scores from non-members (should be low after unlearning)
    with torch.no_grad():
        for x, y in non_member_loader:
            x = x.to(device)
            output = model(x)
            probs = torch.softmax(output, dim=1)
            max_conf = probs.max(dim=1).values
            non_member_confidences.extend(max_conf.cpu().numpy())
    
    member_confidences = np.array(member_confidences)
    non_member_confidences = np.array(non_member_confidences)
    
    # Auto-compute threshold if not provided
    if threshold is None:
        threshold = member_confidences.mean()
    
    # Attack predictions: confidence > threshold → predict member (1), else non-member (0)
    member_predictions = (member_confidences > threshold).astype(int)  # Should be mostly 1
    non_member_predictions = (non_member_confidences > threshold).astype(int)  # Should be mostly 0
    
    # Compute metrics
    tp = member_predictions.sum()  # Correctly identified members
    fp = non_member_predictions.sum()  # Incorrectly identified non-members as members
    tn = (1 - non_member_predictions).sum()  # Correctly identified non-members
    fn = (1 - member_predictions).sum()  # Incorrectly identified members as non-members
    
    # True Positive Rate: Of actual members, how many did we identify?
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # False Positive Rate: Of actual non-members, how many did we wrongly identify as members?
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Overall attack accuracy
    total = len(member_confidences) + len(non_member_confidences)
    correct = tp + tn
    mia_accuracy = correct / total if total > 0 else 0.0
    
    return {
        "mia_accuracy": float(mia_accuracy),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "threshold": float(threshold),
        "member_mean_confidence": float(member_confidences.mean()),
        "non_member_mean_confidence": float(non_member_confidences.mean())
    }
