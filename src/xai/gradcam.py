import cv2
import numpy as np
import torch
import logging
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

logger = logging.getLogger(__name__)


class WeaponClassTarget:
    """
    Custom target for Grad-CAM that works with our detection head output.

    The HybridWeaponDetector.forward() returns a tuple:
        (cls_logits, bbox_offsets, objectness)
    where cls_logits has shape (B, num_anchors, num_classes).

    This target extracts the mean class activation for the specified class
    across all spatial locations, producing the scalar value that Grad-CAM
    needs for gradient computation.
    """
    def __init__(self, class_id=0):
        self.class_id = class_id

    def __call__(self, model_output):
        # model_output is the tuple (cls, bbox, obj) from forward()
        # cls_logits shape: (B, num_anchors, num_classes)
        if isinstance(model_output, (tuple, list)):
            cls_logits = model_output[0]
        else:
            cls_logits = model_output

        # Extract the target class activations and take the mean
        # This gives us a scalar that Grad-CAM can backpropagate through
        return cls_logits[..., self.class_id].mean()


class GradCAMGenerator:
    """
    Grad-CAM heatmap generation for weapon detection explainability.

    Target layer strategy
    ---------------------
    The primary target layer is `neck.p4_proj_out` — the 1×1 Conv2d that
    projects the Swin-enriched P4 features back to the original channel
    count. This layer captures the globally-enriched spatial context that
    drives the detection head's decisions.

    If the hybrid model structure changes, a fallback walks the neck's
    children to find the last Conv2d layer automatically.
    """

    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.cam = None

    def _get_target_layer(self):
        """
        Resolve the target layer for Grad-CAM from the HybridWeaponDetector.

        Priority:
          1. model.neck.p4_proj_out  (Swin Neck output projection)
          2. Last Conv2d in the neck (auto-discover fallback)
          3. Last module in the backbone (emergency fallback)
        """
        # 1. Primary: Swin Neck output projection
        try:
            layer = self.model.neck.p4_proj_out
            if layer is not None:
                return [layer]
        except AttributeError:
            pass

        # 2. Fallback: walk neck children for the last Conv2d
        try:
            last_conv = None
            for module in self.model.neck.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                logger.warning("Grad-CAM: Using fallback — last Conv2d in neck")
                return [last_conv]
        except AttributeError:
            pass

        # 3. Emergency: backbone last layer
        try:
            layer = list(self.model.backbone.model.model.children())[-1]
            logger.warning("Grad-CAM: Using emergency fallback — backbone last layer")
            return [layer]
        except (AttributeError, IndexError):
            raise RuntimeError(
                "Grad-CAM: Cannot resolve any target layer from the model. "
                "Check HybridWeaponDetector structure."
            )

    def generate(self, frame, detection: dict):
        """
        Generates a Grad-CAM heatmap overlaid on the original frame.

        Parameters
        ----------
        frame     : np.ndarray — BGR frame from OpenCV
        detection : dict       — Must contain 'class_id' key

        Returns
        -------
        bytes — JPEG-encoded composite image (BGR heatmap overlay).
                Returns None if generation fails.
        """
        try:
            # 1. Prepare image for torch — model expects BGR (matches training pipeline)
            img_resized = frame
            if frame.shape[:2] != (640, 640):
                img_resized = cv2.resize(frame, (640, 640))

            img_float_bgr = np.float32(img_resized) / 255
            input_tensor = torch.from_numpy(img_float_bgr).permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.to(next(self.model.parameters()).device)

            # 2. Setup Grad-CAM with target layers
            target_layers = self._get_target_layer()

            # Use our custom target that handles the tuple output
            class_id = detection.get('class_id', 0)
            targets = [WeaponClassTarget(class_id=class_id)]

            cam = GradCAM(model=self.model, target_layers=target_layers)

            # 3. Generate grayscale heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            # Clean up
            cam.activations_and_grads.release()

            # 4. Create color overlay (show_cam_on_image needs RGB float [0,1])
            img_float_rgb = np.float32(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)) / 255
            cam_image = show_cam_on_image(img_float_rgb, grayscale_cam, use_rgb=True)

            # Convert back to BGR for OpenCV/Storage
            cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            # 5. Encode as JPEG bytes (matches InferenceEngine contract)
            _, buffer = cv2.imencode('.jpg', cam_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return None
