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
    def __init__(self, backbone_variant="yolo11m.pt", pretrained=True, nc=3, device="cuda"):
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
            embed_dim=512,
            num_heads=8,
            window_size=7,
            num_blocks=2
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
        enriched_features = self.neck(features)
        return self.head(enriched_features)

    def predict(self, frame, conf_threshold=0.25):
        """
        Run a full forward pass and decode results.
        """
        self.eval()
        with torch.no.grad():
            # Basic preprocessing (BGR to RGB and Normalisation)
            if isinstance(frame, np.ndarray):
                # Simple Resize if not 640x640
                if frame.shape[:2] != (640, 640):
                    import cv2
                    frame = cv2.resize(frame, (640, 640))
                
                x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                x = x.unsqueeze(0).to(self.device)
            else:
                x = frame.to(self.device)
                
            cls_logits, bbox_offsets, objectness = self.forward(x)
            
            # Simple decoding for Milestone 2/3 health check
            # In production, this would use TAL/Anchor decoding
            probs = torch.sigmoid(cls_logits) * torch.sigmoid(objectness)
            conf, class_ids = torch.max(probs, dim=2)
            
            mask = conf > conf_threshold
            
            # Map batch=0 for simplicity in this prediction method
            detections = []
            valid_conf = conf[0][mask[0]]
            valid_ids = class_ids[0][mask[0]]
            
            # Placeholder for bbox decoding (Milestone 3 Task)
            # For now, we return valid confidence and classes to verify the pipeline
            for c, cid in zip(valid_conf, valid_ids):
                detections.append({
                    "bbox": [0, 0, 50, 50], # Placeholder box
                    "class_id": int(cid),
                    "class_name": ["Weapon", "Person", "Confuser"][int(cid)],
                    "confidence": float(c)
                })
                
            return detections

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
