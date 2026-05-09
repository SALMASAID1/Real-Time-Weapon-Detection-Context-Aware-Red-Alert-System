"""
models/heads/detection_head.py
================================
Detection head with Focal Loss for class-imbalanced weapon detection.

Why Focal Loss?
---------------
In a weapon detection dataset, every batch contains thousands of background
anchor proposals and only a handful of positive weapon instances. Standard
Binary Cross-Entropy treats every sample equally — the loss is dominated by
easy negatives (background patches that are trivially classified). The model
never learns the hard weapon cases.

Focal Loss addresses this with a modulating factor (1 - p_t)^γ that
down-weights easy examples. As a correctly classified example's confidence p_t
approaches 1, the factor approaches 0 — its contribution to the gradient
vanishes. Hard, ambiguous examples (small, partially occluded weapons) retain
full gradient signal.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

Where:
    γ (gamma)  — focusing parameter. Recommended 2.0. Higher values focus
                  more aggressively on hard examples.
    α (alpha)  — per-class balance weight. Typically set proportionally to
                  the inverse class frequency to handle dataset imbalance.

Architecture
------------
Input:  Enriched multi-scale feature maps from SwinNeck.
Output: Per-anchor predictions for each feature map level:
          - Classification logits  (B, num_anchors * nc)
          - Bounding box offsets   (B, num_anchors * 4)
          - Objectness score       (B, num_anchors * 1)

The head uses a lightweight decoupled design (separate cls and reg branches)
following the YOLOX pattern, which has proven superior to the coupled head for
small-object detection.

Key hyperparameters
-------------------
    nc              : int   — Number of classes (must match dataset.yaml nc: 7)
    gamma           : float — Focal Loss focusing parameter (default 2.0)
    alpha           : float — Focal Loss balance weight (default 0.25 per class)
    reg_max         : int   — DFL (Distribution Focal Loss) regression max (16)

Dependencies
------------
    torch >= 2.2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """
    Decoupled detection head with Focal Loss and DFL regression.

    Parameters
    ----------
    in_channels : dict[str, int]
        Input channel counts from the SwinNeck, keyed by "P3", "P4", "P5".
    nc : int
        Number of detection classes. Must match dataset.yaml `nc`.
    gamma : float
        Focal Loss focusing exponent. Start at 2.0; reduce if mAP plateaus.
    alpha : float
        Per-class Focal Loss weighting factor. Can be set per-class as a list
        of length nc if class imbalance is highly non-uniform.
    """
    def __init__(self, in_channels, nc=3, gamma=2.0, alpha=0.25, reg_max=16):
        super().__init__()
        self.nc = nc
        self.gamma = gamma
        self.alpha = alpha
        self.reg_max = reg_max
        
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        self.obj_heads = nn.ModuleList()
        
        for level in ["P3", "P4", "P5"]:
            ch = in_channels[level]
            # Classification branch
            self.cls_heads.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(),
                nn.Conv2d(ch, nc, 1)
            ))
            # Regression branch (DFL uses reg_max * 4 channels)
            self.reg_heads.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(),
                nn.Conv2d(ch, 4 * reg_max, 1)
            ))
            # Objectness branch
            self.obj_heads.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.SiLU(),
                nn.Conv2d(ch, 1, 1)
            ))

    def forward(self, features: dict):
        """
        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Enriched feature maps from SwinNeck.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (cls_logits, bbox_offsets, objectness)
            Shapes vary by feature map level and anchor count.
            These are decoded by the inference engine into final detections.
        """
        cls_logits = []
        bbox_offsets = []
        objectness = []
        
        for i, level in enumerate(["P3", "P4", "P5"]):
            x = features[level]
            
            # Forward pass through branches
            cls_out = self.cls_heads[i](x)
            reg_out = self.reg_heads[i](x)
            obj_out = self.obj_heads[i](x)
            
            # Flatten predictions for this level
            B, _, H, W = x.shape
            cls_out = cls_out.flatten(2).transpose(1, 2)  # (B, H*W, nc)
            reg_out = reg_out.flatten(2).transpose(1, 2)  # (B, H*W, 4*reg_max)
            obj_out = obj_out.flatten(2).transpose(1, 2)  # (B, H*W, 1)
            
            cls_logits.append(cls_out)
            bbox_offsets.append(reg_out)
            objectness.append(obj_out)
            
        # Concatenate across all levels
        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_offsets = torch.cat(bbox_offsets, dim=1)
        objectness = torch.cat(objectness, dim=1)
        
        return cls_logits, bbox_offsets, objectness

    def focal_loss(self, pred_logits, targets, gamma: float = None, alpha: float = None):
        """
        Compute Focal Loss for a batch of classification predictions.

        Parameters
        ----------
        pred_logits : torch.Tensor — Raw (unactivated) class predictions.
        targets     : torch.Tensor — One-hot ground truth labels.
        gamma       : float        — Focusing parameter.
        alpha       : float        — Balance weight.

        Returns
        -------
        torch.Tensor — Scalar loss value.
        """
        gamma = gamma if gamma is not None else self.gamma
        alpha = alpha if alpha is not None else self.alpha
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of target
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        
        return focal_loss.mean()
