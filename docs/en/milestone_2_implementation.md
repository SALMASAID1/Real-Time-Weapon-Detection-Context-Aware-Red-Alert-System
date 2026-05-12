# Milestone 2: Hybrid Model Implementation Details

This document provides a comprehensive, step-by-step breakdown of how Milestone 2 (Hybrid Model Development) was implemented in the **Real-Time Weapon Detection & Context-Aware Red Alert System**.

The goal of this milestone was to architect the core neural network by bridging state-of-the-art CNN feature extraction with global spatial context mapping and robust loss computation.

---

## Step 1: Implementing the YOLO Backbone Feature Extractor
**File:** `models/backbones/yolo_backbone.py`

Instead of building a CNN from scratch, we wrapped the battle-tested `ultralytics.YOLO` model (defaulting to the `yolo11m.pt` variant) to leverage transfer learning from COCO.

**Key Actions:**
1. **Module Wrapper:** Created the `YOLOBackbone` class inheriting from `torch.nn.Module`.
2. **Forward Hooks:** To extract intermediate feature maps without interrupting the YOLO internal graph, PyTorch forward hooks were registered to the `[15, 18, 21]` layers. This successfully captures the **P3** (stride 8), **P4** (stride 16), and **P5** (stride 32) spatial levels.
3. **Training Control:** Implemented `.freeze()` and `.unfreeze()` methods. Freezing the backbone during the initial training epochs ensures that the randomly initialized Neck and Head do not corrupt the pre-trained CNN weights with unstable gradients.

---

## Step 2: Integrating the Swin Transformer Neck
**File:** `models/necks/swin_neck.py`

Standard CNN pyramids aggregate features purely through local convolutions and struggle with long-range relationships (e.g., matching a hand in the corner to a weapon elsewhere). We introduced a Swin Transformer Neck to solve this.

**Key Actions:**
1. **Swin Blocks:** Leveraged the `timm` library to instantiate `SwinTransformerBlock`s. These blocks apply a Shifted Window Self-Attention mechanism with a `window_size=7` on the P4 feature level.
2. **Channel Projections:** Built `nn.Conv2d` layers to project the raw YOLO P4 channel depths into a consistent `512` Swin embedding dimension before processing, and back out afterward.
3. **FPN Fusion:** Implemented Feature Pyramid Network (FPN) lateral connections. The globally enriched P4 context is fused:
   - Upwards to P3 via bilinear interpolation.
   - Downwards to P5 via adaptive max pooling.

---

## Step 3: Designing the Decoupled Detection Head
**File:** `models/heads/detection_head.py`

Weapon detection datasets suffer from massive background class imbalance (easy "confuser" objects). We implemented a decoupled head with Focal Loss to address this.

**Key Actions:**
1. **Decoupled Architecture:** Created three separate convolutional branches (Classification, Box Regression, Objectness) for each feature level (P3, P4, P5), following the YOLOX design pattern.
2. **Focal Loss Integration:** Authored a custom `focal_loss()` function using PyTorch's `binary_cross_entropy_with_logits`.
   - **Gamma (`γ=2.0`):** Acts as a focusing exponent that aggressively down-weights the loss contribution from easily classified background patches, forcing the network to focus on hard samples (e.g., partially occluded weapons).
   - **Alpha (`α=0.25`):** Acts as the balancing factor for class frequencies.

---

## Step 4: Assembling the Hybrid Detector
**File:** `models/hybrid_model.py`

The three independent sub-modules required a unified interface for the inference engine and training loop.

**Key Actions:**
1. **Graph Stitching:** Created the `HybridWeaponDetector` module. The `forward()` pass elegantly pipes data: 
   `Raw Frame -> YOLOBackbone -> SwinNeck -> DetectionHead -> (cls_logits, bbox_offsets, objectness)`
2. **State Management:** Implemented `.save()` and `.load()` to atomically save the combined `state_dict` of all three components into a single `best.pt` file.
3. **Inference Loop:** Set up a stub for `.predict()` to handle incoming numpy arrays or SAHI tiles during active deployment.

---

## Step 5: Developing the Phase 2 Training Pipeline
**File:** `notebooks/Phase2_Training.ipynb`

To train this highly custom architecture, we built a custom PyTorch training pipeline that bridges the hybrid model with standard YOLO data formats.

**Key Actions:**
1. **Real Data Loading:** Integrated `ultralytics.data.dataset.YOLODataset` to handle the 50k image dataset, including mosaic and mixup augmentations.
2. **Target Matching (The "Fix"):** Implemented a spatial matching logic within the `DetectionHead` to map ground truth bounding boxes to the model's multi-scale anchors (P3, P4, P5).
3. **Composite Loss:** Connected the classification Focal Loss and the Objectness loss into a single `.compute_loss()` method.
4. **Training Strategy:**
   - **Phase 1 (Epochs 1–10):** Backbone frozen; only Neck and Head are trained at `lr=1e-4` to stabilise global context.
   - **Phase 2 (Epochs 10+):** Full model fine-tuning at `lr=1e-5` to refine low-level weapon features.
   - **Persistence:** Automatic checkpointing to `models/weights/` every 5 epochs.
