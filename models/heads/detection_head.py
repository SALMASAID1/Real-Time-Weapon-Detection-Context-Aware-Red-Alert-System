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


class DetectionHead:
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
        ...

    def focal_loss(self, pred_logits, targets, gamma: float, alpha: float):
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
        ...
