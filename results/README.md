# Results Directory

This directory stores all training results, metrics, and visualizations.

## Structure

- **metrics/**: Training and evaluation metrics
  - Precision, Recall, mAP scores
  - Confusion matrices
  - Performance reports (CSV, JSON)
  
- **visualizations/**: Plots and visual results
  - Training curves (loss, accuracy)
  - Detection results on test images
  - Confusion matrix plots
  
- **logs/**: Training logs and TensorBoard files
  - Console logs
  - TensorBoard events
  - Training history

## Metrics

Standard YOLO metrics include:
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **mAP (mean Average Precision)**: Average precision across all classes
- **mAP@0.5**: mAP at IoU threshold of 0.5
- **mAP@0.5:0.95**: mAP averaged over IoU thresholds from 0.5 to 0.95
