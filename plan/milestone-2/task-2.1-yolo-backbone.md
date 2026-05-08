# Task 2.1: YOLO Backbone Implementation

## Description
Implement or customize the YOLO backbone in `models/backbones/yolo_backbone.py` to serve as the primary feature extractor.

## Details
1. **Backbone Selection**: Use YOLOv11/v12 CSP (Cross Stage Partial) connections for efficient feature extraction.
2. **Modular Design**: Ensure the backbone can export features at multiple scales (P3, P4, P5) for the subsequent neck/head components.
3. **Weight Initialization**: Implement logic to load pre-trained COCO weights to speed up convergence.

## Learning Resources
- [YOLOv11 Architecture Breakdown](https://docs.ultralytics.com/models/yolov11/)
- [Deep Learning for Vision: CSPNet Explained](https://arxiv.org/abs/1911.11929)
- [PyTorch Custom Model Implementation](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html)
