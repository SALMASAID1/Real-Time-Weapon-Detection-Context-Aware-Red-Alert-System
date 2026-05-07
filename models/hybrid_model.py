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


class HybridWeaponDetector:
    """
    Unified model: Backbone + Neck + Head.

    Parameters
    ----------
    backbone_variant : str   — e.g. "yolo11m.pt"
    pretrained       : bool  — load pretrained weights for backbone and neck
    nc               : int   — number of classes (7, matches dataset.yaml)
    device           : str   — "cuda" or "cpu"
    """

    def predict(self, frame):
        """
        Run a full forward pass on a single frame or SAHI tile.

        Parameters
        ----------
        frame : np.ndarray | torch.Tensor
            BGR image (OpenCV convention) or pre-normalised tensor.

        Returns
        -------
        list[dict]
            Each dict contains:
              'bbox'       : [x1, y1, x2, y2] in pixel coordinates
              'class_id'   : int  (maps to dataset.yaml names)
              'class_name' : str
              'confidence' : float (0.0–1.0)
        """
        ...

    def save(self, path: str):
        """Save full model state_dict to path."""
        ...

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        """Load a saved checkpoint and return a ready-to-use model instance."""
        ...
