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
    def __init__(self, model_variant="yolo11m.pt", pretrained=True, freeze_backbone_epochs=5, intermediate_layers=[4, 6, 10]):
        super().__init__()
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.intermediate_layers = intermediate_layers
        
        # Load the base model
        base_model = YOLO(model_variant)
        # Use the DetectionModel wrapper which handles internal routing (Concat layers, etc.)
        self.model = base_model.model
        
        # Store features
        self.features = {}
        
        # Register forward hooks
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # YOLOv11/v12 Backbone indices for P3, P4, P5 are typically 4, 6, 10
        for name, layer_idx in zip(["P3", "P4", "P5"], self.intermediate_layers):
            layer = self.model.model[layer_idx]
            layer.register_forward_hook(get_activation(name))
                
        # Get channel sizes by running a dummy forward pass
        device = next(self.model.parameters()).device
        dummy_input = torch.zeros(1, 3, 640, 640).to(device)
        self.model(dummy_input)
        self.out_channels = {
            "P3": self.features["P3"].shape[1],
            "P4": self.features["P4"].shape[1],
            "P5": self.features["P5"].shape[1]
        }

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch, shape (B, 3, H, W).
            For SAHI, this is a single tile of shape (B, 3, 640, 640).

        Returns
        -------
        dict[str, torch.Tensor]
            {
                "P3": Tensor of shape (B, C3, H/8,  W/8),
                "P4": Tensor of shape (B, C4, H/16, W/16),
                "P5": Tensor of shape (B, C5, H/32, W/32),
            }
            C3/C4/C5 are variant-dependent channel counts (e.g. 128/256/512
            for yolo11n, 256/512/1024 for yolo11l).
        """
        self.features = {} # Clear previous features
        _ = self.model(x)
        return self.features

    def freeze(self):
        """Freeze all backbone parameters (called at epoch 0)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze backbone parameters (called at epoch freeze_backbone_epochs)."""
        for param in self.model.parameters():
            param.requires_grad = True
