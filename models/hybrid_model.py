"""
models/hybrid_model.py
========================
Assembly point: YOLOBackbone → SwinNeck → DetectionHead.

This module is the single import point for the full inference engine.
It instantiates and chains the three sub-modules and provides a unified
.predict() interface used by src/inference/engine.py.

Forward pass summary
--------------------
  raw_frame (or SAHI tile)
      │
      ▼
  YOLOBackbone.forward(x)         → {P3, P4, P5} feature maps
      │
      ▼
  SwinNeck.forward(features)      → enriched {P3, P4, P5}
      │
      ▼
  DetectionHead.forward(features) → (cls_logits, bbox_offsets, objectness)
      │
      ▼
  decode_predictions()            → List[Detection(bbox, class_id, confidence)]

The model checkpoint saved to models/weights/best.pt contains the state_dicts
of all three sub-modules jointly, allowing atomic save/load.

Weight loading strategy
-----------------------
1. Load YOLOBackbone from official Ultralytics pretrained checkpoint (COCO).
2. Initialise SwinNeck from timm pretrained Swin weights (ImageNet).
3. Initialise DetectionHead from scratch (random).
4. Fine-tune jointly on the weapon dataset with backbone frozen for the first
   `freeze_backbone_epochs` epochs.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from models.backbones.yolo_backbone import YOLOBackbone
from models.necks.swin_neck import SwinNeck
from models.heads.detection_head import DetectionHead

class HybridWeaponDetector(nn.Module):
    """
    Unified model: Backbone + Neck + Head.

    Parameters
    ----------
    backbone_variant : str   — e.g. "yolo11m.pt"
    pretrained       : bool  — load pretrained weights for backbone and neck
    nc               : int   — number of classes (3, matches dataset.yaml)
    device           : str   — "cuda" or "cpu"
    """
    def __init__(self, backbone_variant="yolo11n.pt", pretrained=True, nc=3, device="cuda"):
        super().__init__()
        self.device = device
        self.nc = nc
        
        # 1. Backbone
        self.backbone = YOLOBackbone(
            model_variant=backbone_variant,
            pretrained=pretrained
        )
        
        # Get channel counts from backbone
        in_channels = self.backbone.out_channels
        
        # 2. Neck
        self.neck = SwinNeck(
            in_channels=in_channels,
            embed_dim=256,
            num_heads=4,
            window_size=7,
            num_blocks=1,  # Reduced from 2 to 1 for Nano speed
            imgsz=640      # Default, but predict() now handles dynamic resize
        )
        
        # 3. Head
        self.head = DetectionHead(
            in_channels=in_channels,
            nc=nc
        )
        
        self.to(device)

    def forward(self, x):
        """
        Forward pass producing raw tensors.
        """
        features = self.backbone(x)
        
        # Robustness check for multi-GPU gathering/extraction
        for key in ["P3", "P4", "P5"]:
            if key not in features:
                raise KeyError(f"Backbone failed to provide {key} features. Check layer indices.")
                
        enriched_features = self.neck(features)
        return self.head(enriched_features)

    def predict(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run a full forward pass and decode results.
        """
        self.eval()
        with torch.inference_mode():
            # Basic preprocessing (expects BGR numpy array; normalize to [0, 1])
            if isinstance(frame, np.ndarray):
                # Use current image dimensions (DetectionHead is now dynamic!)
                h, w = frame.shape[:2]
                # Snap to nearest 32 for YOLO compatibility
                new_h = (h + 16) // 32 * 32
                new_w = (w + 16) // 32 * 32
                
                if (new_h, new_w) != (h, w):
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Keep BGR order — matches YOLODataset training pipeline
                x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                x = x.unsqueeze(0).to(self.device)
            else:
                x = frame.to(self.device)
                
            # If the model is in half precision, input must also be half precision
            if next(self.parameters()).dtype == torch.float16:
                x = x.half()
                
            # Forward pass
            preds = self.forward(x)
            
            # Decode using the head's logic (NMS + BBox transform)
            return self.head.decode_predictions(
                preds, 
                conf_thres=conf_threshold, 
                iou_thres=iou_threshold
            )

    def save(self, path: str):
        """Save full model state_dict to path."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        """Load a saved checkpoint and return a ready-to-use model instance."""
        model = cls(device=device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
