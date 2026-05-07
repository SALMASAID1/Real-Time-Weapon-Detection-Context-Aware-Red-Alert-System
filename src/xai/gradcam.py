"""
src/xai/gradcam.py
====================
Grad-CAM heatmap generation for weapon detection explainability.

What Grad-CAM produces
-----------------------
Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question:
"Which pixels caused the model to predict class C with this confidence?"

It computes the gradient of the target class score with respect to the
activations of the FINAL convolutional feature map (last layer before the
detection head). The gradient tells us which channels are most influential
for the predicted class. These channel importances are used to take a
weighted sum of the feature map — producing a low-resolution heatmap that
is then bilinearly upsampled to the input image size.

    α_c^k  = (1/Z) ΣΣ (∂y^c / ∂A^k_{ij})
    L^c_GradCAM = ReLU(Σ_k α_c^k · A^k)

Where A^k is the k-th feature map and y^c is the score for class c.

The ReLU is critical: it removes features that DECREASE the class score —
we only want to visualise activating evidence, not suppressive evidence.

Hook strategy for YOLO+Swin
-----------------------------
Registering the gradient hook on the correct layer is model-architecture-
specific. For our HybridWeaponDetector, we hook the OUTPUT of the SwinNeck
(after global context enrichment, before the detection head). This ensures
the heatmap reflects BOTH local backbone features AND global Swin context.

Hook registration: `nn.Module.register_forward_hook` to capture activations,
`nn.Module.register_full_backward_hook` to capture gradients.

Bias detection protocol
-----------------------
For each training checkpoint, run Grad-CAM on a validation set sample of
~100 confirmed weapon detections. Manually inspect:
  - Does the heatmap concentrate on the weapon body (barrel, blade, trigger)?
  - Or does it fire on background elements (red walls, dark clothing, hands)?

If background activations dominate → dataset bias — add more hard negatives
(Phase 1 Task 1.2) or apply stronger data augmentation.

Public API
----------
    GradCAMGenerator(model, target_layer_name)
        .generate(frame, detection) -> np.ndarray  (heatmap, shape H×W, float32 0–1)
"""


class GradCAMGenerator:
    """
    Parameters
    ----------
    model             : HybridWeaponDetector
    target_layer_name : str  — Name of the layer to hook. Recommended:
                                "swin_neck.output_proj" (last SwinNeck projection).
                                Verify against model.named_modules() output.
    """

    def generate(self, frame, detection: dict):
        """
        Generate a Grad-CAM heatmap for a single detection event.

        Parameters
        ----------
        frame     : np.ndarray  — BGR image (same as input to inference engine)
        detection : dict        — A single detection from HybridWeaponDetector
                                  (must include 'class_id', 'bbox', 'confidence')

        Returns
        -------
        np.ndarray
            Float32 heatmap of shape (H, W) matching input frame dimensions.
            Values in [0, 1]. 1.0 = maximum class activation.
        """
        ...

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""
        ...

    def _remove_hooks(self):
        """Detach hooks after generation to prevent memory leaks."""
        ...
