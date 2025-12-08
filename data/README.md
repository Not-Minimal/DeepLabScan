# Data Directory

This directory contains all datasets and annotations for the YOLO project.

## Structure

- **raw/**: Original, unprocessed images and videos
- **processed/**: Preprocessed data ready for training (resized, augmented, etc.)
- **annotations/**: Manual annotations and labels
- **roboflow/**: Datasets exported from Roboflow in YOLO format

## Usage

1. Place your raw data in the `raw/` folder
2. Export your labeled dataset from Roboflow to the `roboflow/` folder
3. Processed data will be generated in the `processed/` folder during training preparation

## Data Format

YOLO format expects:
- Images in `.jpg`, `.png` format
- Labels in `.txt` format (one file per image)
- Each line: `class_id x_center y_center width height` (normalized 0-1)
