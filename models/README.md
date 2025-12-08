# Models Directory

This directory stores YOLO model weights and checkpoints.

## Structure

- **pretrained/**: Pre-trained YOLO weights (YOLOv5, YOLOv8, etc.)
- **trained/**: Your trained model weights and checkpoints

## Pre-trained Models

Download pre-trained weights from:
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8: https://github.com/ultralytics/ultralytics

## Trained Models

Your trained models will be saved here with naming convention:
- `best.pt`: Best model based on validation metrics
- `last.pt`: Most recent checkpoint
- `epoch_*.pt`: Checkpoints from specific epochs
