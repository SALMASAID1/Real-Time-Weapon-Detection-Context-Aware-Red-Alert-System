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
    nc              : int   — Number of classes (must match dataset.yaml nc: 3)
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

        # Cache anchor centers/strides per device to avoid recomputing meshgrids every batch.
        # Keyed by stringified device (e.g. "cpu", "cuda:0").
        self._anchor_cache = {}
        
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
        Returns (cls_logits, bbox_offsets, objectness)
        """
        cls_logits = []
        bbox_offsets = []
        objectness = []
        
        for i, level in enumerate(["P3", "P4", "P5"]):
            x = features[level]
            cls_out = self.cls_heads[i](x)
            reg_out = self.reg_heads[i](x)
            obj_out = self.obj_heads[i](x)
            
            B, _, H, W = x.shape
            cls_out = cls_out.flatten(2).transpose(1, 2)  # (B, H*W, nc)
            reg_out = reg_out.flatten(2).transpose(1, 2)  # (B, H*W, 4*reg_max)
            obj_out = obj_out.flatten(2).transpose(1, 2)  # (B, H*W, 1)
            
            cls_logits.append(cls_out)
            bbox_offsets.append(reg_out)
            objectness.append(obj_out)
            
        return torch.cat(cls_logits, dim=1), torch.cat(bbox_offsets, dim=1), torch.cat(objectness, dim=1)

    def _generate_anchor_centers(self, device):
        """
        Pre-compute normalized (cx, cy) anchor centres for each FPN level.

        For a 640×640 input the grid sizes are:
            P3: 80×80 (stride  8)
            P4: 40×40 (stride 16)
            P5: 20×20 (stride 32)

        Returns
        -------
        anchor_centers : Tensor (total_anchors, 2)
            Each row is (cx, cy) in [0, 1] normalized coordinates.
        anchor_strides : Tensor (total_anchors, 1)
            Stride for each anchor in pixel units.
        """
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # Ensure a stable cache key even when callers pass device="cuda".
            device = torch.device("cuda", torch.cuda.current_device())

        cache_key = str(device)
        cached = self._anchor_cache.get(cache_key)
        if cached is not None:
            return cached

        strides = [8, 16, 32]
        imgsz = 640  # training resolution
        centers_list = []
        strides_list = []

        for s in strides:
            grid_h = imgsz // s
            grid_w = imgsz // s
            shift_y, shift_x = torch.meshgrid(
                torch.arange(grid_h, dtype=torch.float32, device=device),
                torch.arange(grid_w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # Normalize centres to [0, 1]
            cx = (shift_x + 0.5) * s / imgsz
            cy = (shift_y + 0.5) * s / imgsz
            centers_list.append(torch.stack([cx, cy], dim=-1).reshape(-1, 2))
            strides_list.append(torch.full((grid_h * grid_w, 1), s, dtype=torch.float32, device=device))

        anchors = (torch.cat(centers_list, dim=0), torch.cat(strides_list, dim=0))
        self._anchor_cache[cache_key] = anchors
        return anchors

    def build_targets(self, pred_shape, batch, device, topk=13):
        """
        Top-K Anchor Matcher: assigns each GT to the K nearest anchors.

        Using K=13 (following SimOTA from YOLOX) increases positive ratio
        from ~0.1% (single anchor) to ~1.5%, giving the cls/reg heads
        substantially more gradient signal during training.

        pred_shape: (B, total_anchors, nc)
        batch: Dictionary from YOLODataset containing 'cls', 'batch_idx', 'bboxes'.

        Returns
        -------
        target_cls  : (B, num_anchors, nc)  — one-hot class targets
        target_obj  : (B, num_anchors, 1)   — objectness targets
        target_bbox : (B, num_anchors, 4)   — normalized [cx, cy, w, h] box targets
        fg_mask     : (B, num_anchors)       — bool mask for positive anchors
        """
        B, num_anchors, nc = pred_shape
        target_cls  = torch.zeros((B, num_anchors, nc), device=device)
        target_obj  = torch.zeros((B, num_anchors, 1), device=device)
        target_bbox = torch.zeros((B, num_anchors, 4), device=device)
        fg_mask     = torch.zeros((B, num_anchors), dtype=torch.bool, device=device)

        cls      = batch['cls']        # (N_gt,) or (N_gt, 1)
        batch_idx = batch['batch_idx'] # (N_gt,)
        bboxes   = batch.get('bboxes') # (N_gt, 4)  normalized [cx, cy, w, h]

        if cls.shape[0] == 0:
            return target_cls, target_obj, target_bbox, fg_mask

        # Flatten cls to 1-D if (N_gt, 1)
        if cls.ndim > 1:
            cls = cls.squeeze(-1)

        # Anchor centres — (num_anchors, 2)
        anchor_centers, _ = self._generate_anchor_centers(device)

        # Clamp K to the number of available anchors
        k = min(topk, num_anchors)

        for i in range(len(cls)):
            b_idx   = int(batch_idx[i])
            cls_idx = int(cls[i])

            if bboxes is not None:
                gt_cx, gt_cy = bboxes[i, 0].item(), bboxes[i, 1].item()
                # Compute squared distance from GT centre to every anchor centre
                dists = (anchor_centers[:, 0] - gt_cx) ** 2 + (anchor_centers[:, 1] - gt_cy) ** 2
                # Select the K nearest anchors
                _, topk_indices = dists.topk(k, largest=False)

                # Vectorized assignment (avoids Python loops)
                target_cls[b_idx, topk_indices, cls_idx] = 1.0
                target_obj[b_idx, topk_indices, 0] = 1.0
                gt_box = bboxes[i].to(device)
                target_bbox[b_idx, topk_indices] = gt_box.unsqueeze(0).expand(topk_indices.shape[0], -1)
                fg_mask[b_idx, topk_indices] = True
            else:
                # Fallback: no bboxes available — assign all anchors (gradient-flow check)
                target_cls[b_idx, :, cls_idx] = 1.0
                target_obj[b_idx, :, 0]       = 1.0

        return target_cls, target_obj, target_bbox, fg_mask

    # ── CIoU loss ────────────────────────────────────────────────────────────
    @staticmethod
    def _ciou_loss(pred_boxes, target_boxes, eps=1e-7):
        """
        Complete IoU loss between two sets of [cx, cy, w, h] boxes.

        Parameters
        ----------
        pred_boxes   : (N, 4) — predicted boxes in normalized [cx, cy, w, h]
        target_boxes : (N, 4) — ground-truth boxes in normalized [cx, cy, w, h]

        Returns
        -------
        loss : scalar — mean CIoU loss over the N pairs
        """
        import math

        # Convert [cx, cy, w, h] → [x1, y1, x2, y2]
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        gt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        gt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        gt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        gt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area   = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area + eps

        iou = inter_area / union_area

        # Enclosing box
        enc_x1 = torch.min(pred_x1, gt_x1)
        enc_y1 = torch.min(pred_y1, gt_y1)
        enc_x2 = torch.max(pred_x2, gt_x2)
        enc_y2 = torch.max(pred_y2, gt_y2)

        # Centre distance squared
        rho2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
               (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

        # Diagonal of enclosing box squared
        c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

        # Aspect ratio penalty
        v = (4 / (math.pi ** 2)) * (
            torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps)) -
            torch.atan(pred_boxes[:, 2] / (pred_boxes[:, 3] + eps))
        ) ** 2
        with torch.no_grad():
            alpha_ciou = v / (1 - iou + v + eps)

        ciou = iou - rho2 / c2 - alpha_ciou * v
        return (1 - ciou).mean()

    def compute_loss(self, preds, batch, device):
        """
        Comprehensive loss = Focal-cls + BCE-obj + CIoU-box.

        The CIoU component trains bounding-box regression so the model
        learns to localize weapons, not just classify anchor cells.
        """
        cls_logits, reg_offsets, objectness = preds

        # 1. Match targets to anchors
        target_cls, target_obj, target_bbox, fg_mask = self.build_targets(
            cls_logits.shape, batch, device
        )

        # 2. Classification Focal Loss (all anchors)
        loss_cls = self.focal_loss(cls_logits, target_cls)

        # 3. Objectness Loss (all anchors)
        loss_obj = F.binary_cross_entropy_with_logits(objectness, target_obj)

        # 4. CIoU Box Regression Loss (positive anchors only)
        loss_box = torch.tensor(0.0, device=device)
        num_fg = fg_mask.sum().item()

        if num_fg > 0:
            # Decode DFL regression offsets → [cx, cy, w, h] for positive anchors
            anchor_centers, anchor_strides = self._generate_anchor_centers(device)

            # Foreground indices (num_fg, 2): [batch_index, anchor_index]
            fg_idx = fg_mask.nonzero(as_tuple=False)
            fg_b = fg_idx[:, 0]
            fg_a = fg_idx[:, 1]

            # Gather foreground predictions: (num_fg, 4*reg_max)
            fg_reg = reg_offsets[fg_b, fg_a]

            # DFL decode: softmax over reg_max bins → expected value per side
            B_fg = fg_reg.shape[0]
            fg_reg = fg_reg.reshape(B_fg, 4, self.reg_max)      # (num_fg, 4, reg_max)
            fg_reg = F.softmax(fg_reg, dim=-1)                   # (num_fg, 4, reg_max)
            proj = torch.arange(self.reg_max, dtype=torch.float32, device=device)
            fg_dist = (fg_reg * proj).sum(dim=-1)                # (num_fg, 4) — ltrb

            # Anchor centres and strides for positive anchors (vectorized)
            fg_centers = anchor_centers[fg_a]  # (num_fg, 2)
            fg_strides = anchor_strides[fg_a]  # (num_fg, 1)

            # Convert ltrb distances (in stride units) to [cx, cy, w, h] normalized
            imgsz = 640.0
            stride_norm = fg_strides / imgsz  # normalize stride to [0,1]
            
            # Correct BBox center: cx_new = cx_anchor + (r - l) / 2
            # fg_dist: [l, t, r, b]
            pred_w  = (fg_dist[:, 0:1] + fg_dist[:, 2:3]) * stride_norm
            pred_h  = (fg_dist[:, 1:2] + fg_dist[:, 3:4]) * stride_norm
            pred_cx = fg_centers[:, 0:1] + (fg_dist[:, 2:3] - fg_dist[:, 0:1]) / 2 * stride_norm
            pred_cy = fg_centers[:, 1:2] + (fg_dist[:, 3:4] - fg_dist[:, 1:2]) / 2 * stride_norm
            
            pred_boxes = torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=1)  # (num_fg, 4)

            gt_boxes = target_bbox[fg_b, fg_a]  # (num_fg, 4)
            loss_box = self._ciou_loss(pred_boxes, gt_boxes)

        # Loss weights (following YOLO convention: box=7.5, cls=0.5, obj=1.5)
        loss = 7.5 * loss_box + 0.5 * loss_cls + 1.5 * loss_obj
        return loss

    def decode_predictions(self, preds, conf_thres=0.25, iou_thres=0.45):
        """
        Decodes raw head outputs into filtered detections.
        
        Parameters
        ----------
        preds      : tuple (cls_logits, reg_offsets, objectness)
        conf_thres : float — confidence threshold for filtering
        iou_thres  : float — NMS IoU threshold
        
        Returns
        -------
        List of dicts: [{"bbox": [x1, y1, x2, y2], "class_id": int, "confidence": float}]
        """
        from torchvision.ops import nms
        
        cls_logits, reg_offsets, objectness = preds
        device = cls_logits.device
        
        # 1. Compute class probabilities (cls * obj)
        probs = torch.sigmoid(cls_logits) * torch.sigmoid(objectness)
        conf, class_ids = torch.max(probs, dim=2)  # (B, total_anchors)
        
        # 2. Filter by threshold
        mask = conf > conf_thres
        if not mask.any():
            return []
            
        # For simplicity, we process only the first image in the batch (inference context)
        b = 0
        img_conf = conf[b][mask[b]]
        img_class_ids = class_ids[b][mask[b]]
        img_reg_offsets = reg_offsets[b][mask[b]]
        
        # 3. Decode Bounding Boxes
        anchor_centers, anchor_strides = self._generate_anchor_centers(device)
        fg_centers = anchor_centers[mask[b]]
        fg_strides = anchor_strides[mask[b]]
        
        # DFL decode: softmax over reg_max bins -> expected value per side
        num_fg = img_reg_offsets.shape[0]
        fg_reg = img_reg_offsets.reshape(num_fg, 4, self.reg_max)
        fg_reg = F.softmax(fg_reg, dim=-1)
        proj = torch.arange(self.reg_max, dtype=torch.float32, device=device)
        fg_dist = (fg_reg * proj).sum(dim=-1) # (num_fg, 4) - ltrb in stride units
        
        # Convert ltrb to [x1, y1, x2, y2] normalized
        imgsz = 640.0
        stride_norm = fg_strides / imgsz
        
        # pred_ltrb = [l, t, r, b]
        # x1 = cx - l * stride
        # y1 = cy - t * stride
        # x2 = cx + r * stride
        # y2 = cy + b * stride
        x1 = fg_centers[:, 0:1] - fg_dist[:, 0:1] * stride_norm
        y1 = fg_centers[:, 1:2] - fg_dist[:, 1:2] * stride_norm
        x2 = fg_centers[:, 0:1] + fg_dist[:, 2:3] * stride_norm
        y2 = fg_centers[:, 1:2] + fg_dist[:, 3:4] * stride_norm
        
        boxes = torch.cat([x1, y1, x2, y2], dim=1).clamp(0, 1)
        
        # 4. Non-Maximum Suppression (NMS)
        keep = nms(boxes, img_conf, iou_thres)
        
        results = []
        for i in keep:
            cid = int(img_class_ids[i])
            results.append({
                "bbox": boxes[i].tolist(),
                "class_id": cid,
                "class_name": ["Weapon", "Person", "Confuser"][cid],
                "confidence": float(img_conf[i]),
                "is_weapon": cid == 0,  # Only class 0 is a weapon
            })
            
        return results

    def focal_loss(self, pred_logits, targets, gamma: float = None, alpha: float = None):
        gamma = gamma if gamma is not None else self.gamma
        alpha = alpha if alpha is not None else self.alpha
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = alpha * (1 - pt) ** gamma * bce_loss
        
        return f_loss.mean()
