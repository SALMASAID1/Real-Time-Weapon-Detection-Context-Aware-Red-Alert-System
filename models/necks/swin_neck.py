"""
models/necks/swin_neck.py
==========================
Swin Transformer Neck — global context enrichment between backbone and head.

Why a Transformer neck?
-----------------------
Standard CNN feature pyramids (FPN, PANet) aggregate features purely through
local convolutions. They cannot reason about the relationship between spatially
distant objects, e.g. a hand in the bottom-left corner and a weapon in the
top-right. This is a critical failure mode for Hand-Weapon IoU scoring.

Swin Transformer blocks process feature maps with a shifted-window self-
attention mechanism. Each window attends to its neighbours across shift cycles,
achieving global receptive field in O(n) rather than O(n²) complexity.

Architecture
------------
Input:  P3, P4, P5 feature maps from YOLOBackbone.
        P4 (stride 16) is the primary processing level — it balances semantic
        richness and spatial resolution for weapon-sized objects.

Processing:
  1. Project P4 channels to the Swin embedding dimension (e.g. 512d).
  2. Apply N Swin Transformer blocks (window size 7×7, shift size 3).
  3. Project back to the original P4 channel count.
  4. Fuse the enriched P4 with P3 and P5 via bilinear up/downsampling and
     channel concatenation — standard FPN-style lateral connections.

Output: Three enriched feature maps (same shapes as input P3/P4/P5) that now
        encode global spatial context, passed to the detection head.

Key hyperparameters
-------------------
    embed_dim        : int  — Swin internal embedding dimension (default 512)
    num_heads        : int  — Number of attention heads per Swin block (8)
    window_size      : int  — Local attention window size in tokens (7)
    num_swin_blocks  : int  — Depth of Swin processing (2–4 recommended)
    drop_path_rate   : float — Stochastic depth regularisation (0.1–0.2)

Dependencies
------------
    timm >= 0.9.0  (SwinTransformerBlock implementation)
    torch >= 2.2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock

class SwinNeck(nn.Module):
    """
    Wraps Swin Transformer blocks as a FPN-style neck.

    Parameters
    ----------
    in_channels : dict[str, int]
        Channel counts of input feature maps, e.g. {"P3": 128, "P4": 256, "P5": 512}.
        Must match the output of the YOLOBackbone variant being used.
    embed_dim : int
        Internal Swin embedding dimension. P4 is projected to this size before
        the Swin blocks and projected back afterwards.
    num_heads : int
        Number of self-attention heads in each Swin block.
    window_size : int
        Side length of the local attention window (in feature map tokens).
        Must divide evenly into the P4 spatial dimensions at the target input size.
    num_blocks : int
        Number of consecutive Swin Transformer blocks to stack.
        More blocks → richer global context but higher memory and latency.
    """
    def __init__(self, in_channels, embed_dim=512, num_heads=8, window_size=7, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # P4 projection
        self.p4_proj_in = nn.Conv2d(in_channels["P4"], embed_dim, kernel_size=1)
        
        # Swin Blocks
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(640//16, 640//16), # Default for 640x640 at P4
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2
            ) for i in range(num_blocks)
        ])
        
        # P4 projection out
        self.p4_proj_out = nn.Conv2d(embed_dim, in_channels["P4"], kernel_size=1)
        
        # FPN Lateral connections
        self.lat_p4_to_p3 = nn.Conv2d(in_channels["P4"], in_channels["P3"], kernel_size=1)
        self.lat_p4_to_p5 = nn.Conv2d(in_channels["P4"], in_channels["P5"], kernel_size=1)

    def forward(self, features: dict):
        """
        Parameters
        ----------
        features : dict[str, torch.Tensor]
            Output from YOLOBackbone.forward().
            Keys: "P3", "P4", "P5".

        Returns
        -------
        dict[str, torch.Tensor]
            Enriched feature maps with the same keys and shapes as input.
            P4 is globally enriched via Swin; P3 and P5 receive enriched P4
            context via lateral FPN connections.
        """
        p3, p4, p5 = features["P3"], features["P4"], features["P5"]
        
        # 1. Project P4 to embed_dim
        x = self.p4_proj_in(p4)
        B, C, H, W = x.shape
        
        # Reshape for Swin (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. Apply Swin blocks
        for block in self.swin_blocks:
            x = block(x)
            
        # Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # 3. Project back to P4 channels
        enriched_p4 = self.p4_proj_out(x)
        
        # 4. Fuse with P3 and P5
        # Upsample P4 to P3 and add
        p4_up = F.interpolate(self.lat_p4_to_p3(enriched_p4), size=p3.shape[2:], mode="bilinear", align_corners=False)
        enriched_p3 = p3 + p4_up
        
        # Downsample P4 to P5 and add
        p4_down = F.adaptive_max_pool2d(self.lat_p4_to_p5(enriched_p4), output_size=p5.shape[2:])
        enriched_p5 = p5 + p4_down
        
        return {
            "P3": enriched_p3,
            "P4": enriched_p4,
            "P5": enriched_p5
        }
