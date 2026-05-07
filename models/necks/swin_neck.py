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


class SwinNeck:
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
        ...
