# Configurations Directory

This directory contains configuration files for training, evaluation, and inference.

## Configuration Files

- `training_config.yaml`: Training hyperparameters (epochs, batch size, learning rate, etc.)
- `model_config.yaml`: Model architecture configuration
- `data_config.yaml`: Dataset paths and class definitions
- `inference_config.yaml`: Inference settings and thresholds

## Example Structure

```yaml
# training_config.yaml
epochs: 100
batch_size: 16
img_size: 640
learning_rate: 0.01
optimizer: Adam
```

## Usage

Configuration files are loaded by training and inference scripts to manage parameters centrally.
