# Task 3.1: SAHI Integration for Small Objects

## Description
Implement Slicing Aided Hyper Inference (SAHI) in `src/inference/sahi_pipeline.py` to improve detection performance on small weapons in high-resolution frames.

## Details
1. **Slicing Logic**: Divide high-resolution input images into overlapping patches (slices).
2. **Inference Pipeline**: Run detection on each slice and the full image simultaneously.
3. **Merging Results**: Use Non-Maximum Suppression (NMS) or Weighted Boxes Fusion (WBF) to merge detections from multiple slices into a final prediction.

## Learning Resources
- [SAHI: Slicing Aided Hyper Inference GitHub](https://github.com/obss/sahi)
- [Small Object Detection: Challenges and Solutions](https://blog.roboflow.com/small-object-detection/)
- [Weighted Boxes Fusion (WBF) Explained](https://arxiv.org/abs/1910.13302)
