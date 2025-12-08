"""
Utility functions for data loading and processing
"""

from pathlib import Path
from typing import List, Tuple, Dict
import yaml


def load_yolo_annotations(label_path: str) -> List[List[float]]:
    """
    Load YOLO format annotations from a label file
    
    Args:
        label_path: Path to .txt label file
        
    Returns:
        List of annotations [class_id, x_center, y_center, width, height]
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            annotations.append(values)
    return annotations


def parse_data_config(config_path: str) -> Dict:
    """
    Parse data configuration YAML file
    
    Args:
        config_path: Path to data config YAML
        
    Returns:
        Dictionary with data configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_statistics(data_dir: str) -> Dict:
    """
    Calculate dataset statistics (number of images, classes, etc.)
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    
    stats = {
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'classes': {}
    }
    
    # Count images in each split
    for split in ['train', 'valid', 'test']:
        split_path = data_path / split / 'images'
        if split_path.exists():
            images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))
            count = len(images)
            stats['total_images'] += count
            
            if split == 'train':
                stats['train_images'] = count
            elif split == 'valid':
                stats['val_images'] = count
            else:
                stats['test_images'] = count
    
    return stats


def validate_dataset(data_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate dataset structure and integrity
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    data_path = Path(data_dir)
    
    # Check required directories exist
    required_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            issues.append(f"Missing required directory: {dir_name}")
    
    # Check that each image has a corresponding label
    for split in ['train', 'valid', 'test']:
        img_dir = data_path / split / 'images'
        label_dir = data_path / split / 'labels'
        
        if img_dir.exists() and label_dir.exists():
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            for img_path in images:
                label_path = label_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    issues.append(f"Missing label for image: {img_path.name}")
    
    is_valid = len(issues) == 0
    return is_valid, issues
