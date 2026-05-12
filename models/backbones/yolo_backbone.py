"""
models/backbones/yolo_backbone.py
==================================
YOLOv11/v12 feature extractor wrapper.

Role in the pipeline
--------------------
The backbone is responsible for the FIRST stage of feature extraction.
Given a raw image (or SAHI tile), it produces a set of multi-scale feature
maps at three pyramid levels:

    P3  (stride 8)  — fine-grained spatial detail  → small weapon parts
    P4  (stride 16) — balanced semantics + location → main detection level
    P5  (stride 32) — rich semantics, coarse location → context & person body

These feature maps are NOT used for final predictions here. They are passed
to the Swin Transformer Neck (models/necks/swin_neck.py) for global context
enrichment before reaching the detection head.

Design decisions
----------------
- We wrap Ultralytics YOLO rather than rebuilding it from scratch. This
  preserves the battle-tested CSP/C2f block implementations and allows us to
  load any official pretrained checkpoint as a starting point (transfer
  learning from COCO weights).
- The wrapper interface is deliberately thin so the backbone can be swapped
  (e.g. YOLOv11 → v12) without touching any other module. Only the output
  feature map shapes and names are part of the public contract.
- Weights are frozen for the first N epochs (configurable) to allow the Swin
  neck and detection head to stabilise before fine-tuning the backbone.

Public API
----------
    YOLOBackbone(model_variant, pretrained, freeze_epochs)
        .forward(x) -> dict[str, Tensor]
            Keys: "P3", "P4", "P5"
            Shapes: (B, C, H/s, W/s) where s is the stride

Dependencies
------------
    ultralytics >= 8.2.0
    torch >= 2.2.0
"""

import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOBackbone(nn.Module):
    """
    Thin wrapper around Ultralytics YOLOv11/v12.

    Parameters
    ----------
    model_variant : str
        Ultralytics model descriptor, e.g. "yolo11n.pt", "yolo11m.pt".
        A pretrained COCO checkpoint is used as the base for transfer learning.
    pretrained : bool
        Whether to initialise from an official Ultralytics pretrained weight.
        Always True for production training runs.
    freeze_backbone_epochs : int
        Number of training epochs during which backbone weights are frozen.
        Recommended: 5–10 epochs to allow the neck/head to warm up first.
    intermediate_layers : list[str]
        Names of the internal YOLO layers whose outputs correspond to
        P3, P4, P5. These layer indices are model-variant-specific and must
        be verified against the parsed model graph after loading.
    """
    def __init__(self, model_variant="yolo11m.pt", pretrained=True, freeze_backbone_epochs=5, intermediate_layers=None):
        super().__init__()
        self.freeze_backbone_epochs = freeze_backbone_epochs
        
        # Standard backbone indices for P3, P4, P5
        if intermediate_layers is None:
            if "yolo11" in model_variant or "yolo12" in model_variant:
                self.intermediate_layers = [4, 6, 10]
            else: # Fallback to YOLOv8/v9/v10
                self.intermediate_layers = [4, 6, 9]
        else:
            self.intermediate_layers = intermediate_layers
        
        # Load the base model
        base_model = YOLO(model_variant)
        # Use the DetectionModel wrapper (base_model.model)
        self.model = base_model.model
        
        # Identify which layers we need to save for the manual forward pass
        # We only care about layers up to the last intermediate layer (P5)
        last_target_idx = max(self.intermediate_layers)
        self.save = set(self.intermediate_layers)
        for i, m in enumerate(self.model.model):
            if i > last_target_idx: break # Optimization: Ignore layers past backbone
            if hasattr(m, 'f'):
                if isinstance(m.f, int):
                    if m.f != -1: self.save.add(m.f)
                else:
                    for f in m.f:
                        if f != -1: self.save.add(f)
        
        # Get channel sizes by running a dummy forward pass
        device = next(self.model.parameters()).device
        dummy_input = torch.zeros(1, 3, 640, 640).to(device)
        features = self.forward(dummy_input)
        
        # Safety check: ensure all levels were extracted
        for level in ["P3", "P4", "P5"]:
            if level not in features:
                raise KeyError(f"Failed to extract {level} from YOLO backbone at indices {self.intermediate_layers}")

        self.out_channels = {
            "P3": features["P3"].shape[1],
            "P4": features["P4"].shape[1],
            "P5": features["P5"].shape[1]
        }

    def forward(self, x):
        """
        DataParallel-safe forward pass that manually iterates through YOLO layers.
        Stops early after extracting P5 to save GPU memory.
        """
        y = []
        features = {}
        last_target_idx = max(self.intermediate_layers)

        for i, m in enumerate(self.model.model):
            # Resolve input(s) for current layer
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            
            # Forward current layer
            x = m(x)
            
            # Cache output if needed for later skip connections or feature extraction
            y.append(x if i in self.save else None)
            
            # Extract features at target levels
            if i == self.intermediate_layers[0]:
                features["P3"] = x
            elif i == self.intermediate_layers[1]:
                features["P4"] = x
            elif i == self.intermediate_layers[2]:
                features["P5"] = x
            
            # Optimization: Stop immediately once we have the backbone outputs
            if i == last_target_idx:
                break
                
        return features

    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
