# Task 3.3: Explainability (XAI) with Grad-CAM

## Description
Implement Grad-CAM (Gradient-weighted Class Activation Mapping) in `src/xai/gradcam.py` to ensure model transparency and validate that the model is focusing on relevant weapon features (e.g., barrel, trigger).

## Details
1. **Heatmap Generation**: Generate visual heatmaps indicating which pixels influenced the model's classification.
2. **Bias Detection**: Use XAI to verify that the model is not relying on background context or biased elements.
3. **Integration**: Enable the Grad-CAM visualization for sample detections in the final dashboard.

## Learning Resources
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Implementing Grad-CAM in PyTorch](https://github.com/jacobgil/pytorch-grad-cam)
- [XAI for Object Detection](https://towardsdatascience.com/explainable-ai-for-object-detection-653a921d7b)
