# Milestone 2: Hybrid Model Implementation Details

This document provides a comprehensive, step-by-step breakdown of how Milestone 2 (Hybrid Model Development) was implemented in the **Real-Time Weapon Detection & Context-Aware Red Alert System**.

The goal of this milestone was to architect the core neural network by bridging state-of-the-art CNN feature extraction with global spatial context mapping and robust loss computation.

---

## Step 1: Implementing the YOLO Backbone Feature Extractor
**File:** `models/backbones/yolo_backbone.py`

Instead of building a CNN from scratch, we wrapped the battle-tested `ultralytics.YOLO` model (defaulting to the `yolo11m.pt` variant) to leverage transfer learning from COCO.

**Key Actions:**
1. **Module Wrapper:** Created the `YOLOBackbone` class inheriting from `torch.nn.Module`.
2. **Manual Layer Iteration:** To extract intermediate feature maps in a DataParallel-safe manner, we implemented a manual layer-by-layer forward pass that stops early after extracting features at indices `[4, 6, 10]` (for YOLOv11/v12). This successfully captures the **P3** (stride 8), **P4** (stride 16), and **P5** (stride 32) spatial levels without relying on hooks that can break under multi-GPU setups.
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
3. **BBox Decoding & NMS (The "Fix"):** Implemented the `decode_predictions()` pipeline to convert raw model outputs into localized bounding boxes.
   - **DFL Decoding:** Implemented Distribution Focal Loss (DFL) decoding to transform regression offsets into spatial distances (`ltrb`).
   - **Coordinate Transformation:** Added logic to map stride-relative distances back to normalized `[x1, y1, x2, y2]` coordinates.
   - **NMS:** Integrated `torchvision.ops.nms` to prune overlapping detections and ensure one detection per weapon.

---

## Step 4: Assembling the Hybrid Detector
**File:** `models/hybrid_model.py`

The three independent sub-modules required a unified interface for the inference engine and training loop.

**Key Actions:**
1. **Graph Stitching:** Created the `HybridWeaponDetector` module. The `forward()` pass elegantly pipes data: 
   `Raw Frame -> YOLOBackbone -> SwinNeck -> DetectionHead -> (cls_logits, bbox_offsets, objectness)`
2. **State Management:** Implemented `.save()` and `.load()` to atomically save the combined `state_dict` of all three components into a single `best.pt` file.
3. **Optimized Prediction:** Finalized the `.predict()` method with full preprocessing:
   - **BGR Preservation:** OpenCV frames are kept in BGR order throughout the pipeline — this matches the `YOLODataset` training pipeline that also uses BGR. No color space conversion is applied, ensuring consistency between training and inference.
   - **In-Graph Decoding:** Connected the head's decoding logic to provide ready-to-use detections directly to the inference engine.

---

## Step 5: Developing the Phase 2 Training Pipeline
**File:** `notebooks/Phase2_Training.ipynb`

To train this highly custom architecture, we built a custom PyTorch training pipeline that bridges the hybrid model with standard YOLO data formats.

**Key Actions:**
1. **Real Data Loading:** Integrated `ultralytics.data.dataset.YOLODataset` to handle the 50k image dataset, including mosaic and mixup augmentations.
2. **Target Matching:** Implemented a spatial matching logic within the `DetectionHead` to map ground truth bounding boxes to the model's multi-scale anchors.
3. **CIoU Loss Optimization:** Implemented **Complete IoU (CIoU)** loss for bounding box regression. Corrected the center calculation logic (`cx_new = cx_anchor + (r - l) / 2`) to handle asymmetric object boundaries within grid cells.
4. **Training Strategy (v7 Optimized):**
   - **Phase 1 (Epochs 1–10):** Backbone frozen; only Neck and Head are trained at `lr=1e-4` with a Cosine Annealing scheduler.
   - **Phase 2 (Epochs 10+):** Full model fine-tuning with differential learning rates (`1e-5` for backbone, `5e-5` for head/neck).
   - **Persistence:** Automatic checkpointing to `models/weights/` every epoch during the 50-epoch run.
