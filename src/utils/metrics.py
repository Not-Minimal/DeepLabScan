"""
Metrics calculation utilities
"""

import numpy as np
from typing import List, Tuple, Dict


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU score
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap(recalls: List[float], precisions: List[float]) -> float:
    """
    Calculate Average Precision (AP)
    
    Args:
        recalls: List of recall values
        precisions: List of precision values
        
    Returns:
        Average Precision score
    """
    # Add sentinel values
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]
    
    # Compute the precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP as area under curve
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    
    return ap


def calculate_confusion_matrix(predictions: List, 
                               ground_truths: List, 
                               num_classes: int,
                               iou_threshold: float = 0.5) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        predictions: List of predicted boxes with class labels
        ground_truths: List of ground truth boxes with class labels
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Confusion matrix as numpy array
    """
    # TODO: Implement confusion matrix calculation
    matrix = np.zeros((num_classes, num_classes))
    return matrix


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score
    
    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        
    Returns:
        (precision, recall, f1_score)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def calculate_map(aps: List[float]) -> float:
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        aps: List of Average Precision for each class
        
    Returns:
        mAP score
    """
    return sum(aps) / len(aps) if len(aps) > 0 else 0
