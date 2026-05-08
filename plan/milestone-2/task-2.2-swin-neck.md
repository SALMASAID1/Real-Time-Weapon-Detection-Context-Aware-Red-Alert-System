# Task 2.2: Swin Transformer Neck Integration

## Description
Integrate Swin Transformer blocks into the model's neck within `models/necks/swin_neck.py` to capture long-range spatial dependencies.

## Details
1. **Swin Blocks**: Implement Hierarchical Vision Transformer with Shifted Windows.
2. **Feature Fusion**: Use the Swin Transformer to process multi-scale features from the YOLO backbone before passing them to the detection head.
3. **Model Assembly**: Implement the full `HybridWeaponDetector` in `models/hybrid_model.py` to chain the Backbone, Swin Neck, and Detection Head (`models/heads/detection_head.py`).
4. **Context Awareness**: Leverage the self-attention mechanism to better distinguish weapons from background noise in complex scenes.

## Learning Resources
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Integrating Transformers into YOLO](https://github.com/iscas-tc/YOLO-Transformer)
- [Attention Mechanisms in Computer Vision](https://towardsdatascience.com/everything-you-need-to-know-about-attention-mechanisms-in-computer-vision-913a30283c74)
