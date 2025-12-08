# Source Code Directory

This directory contains all Python scripts for the YOLO project.

## Structure

- **training/**: Scripts for model training
  - `train.py`: Main training script
  - `augmentation.py`: Data augmentation utilities
  
- **inference/**: Scripts for model inference and prediction
  - `predict.py`: Run predictions on images/videos
  - `detect.py`: Real-time detection
  
- **evaluation/**: Scripts for model evaluation
  - `evaluate.py`: Calculate metrics (precision, recall, mAP)
  - `visualize.py`: Visualize predictions and results
  
- **utils/**: Utility functions and helpers
  - `data_loader.py`: Data loading utilities
  - `config.py`: Configuration management
  - `metrics.py`: Metric calculation functions

## Usage

Each subdirectory will contain specific scripts for different stages of the ML pipeline.
