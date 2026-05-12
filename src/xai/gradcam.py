import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAMGenerator:
    """
    Grad-CAM heatmap generation for weapon detection explainability.
    """

    def __init__(self, model, target_layer_name=None):
        self.model = model
        # Default target layer: the output of the Swin Transformer Neck
        self.target_layer_name = target_layer_name or "neck.output_layers[-1]"
        self.cam = None

    def _get_target_layer(self):
        """
        Finds the target layer object by name within the model.
        """
        # This is architecture-dependent. We'll need to verify this
        # when the full HybridModel class is implemented.
        try:
            # Assuming the model has a 'neck' attribute with submodules
            return [self.model.neck.output_proj] # Example target
        except AttributeError:
            # Fallback for standard YOLO backbones if hybrid isn't ready
            return [self.model.model.model[-1]] 

    def generate(self, frame, detection: dict):
        """
        Generates a Grad-CAM heatmap overlaid on the original frame.
        """
        # 1. Prepare image for torch — model expects BGR (matches training pipeline)
        img_float_bgr = np.float32(frame) / 255
        input_tensor = torch.from_numpy(img_float_bgr).permute(2, 0, 1).unsqueeze(0)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # 2. Setup Grad-CAM with target layers
        target_layers = self._get_target_layer()
        self.cam = GradCAM(model=self.model, target_layers=target_layers)

        # 3. Specify target class (Weapon is class 0)
        targets = [ClassifierOutputTarget(detection['class_id'])]

        # 4. Generate grayscale heatmap
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # 5. Create color overlay (show_cam_on_image needs RGB float [0,1])
        img_float_rgb = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) / 255
        cam_image = show_cam_on_image(img_float_rgb, grayscale_cam, use_rgb=True)
        
        # Convert back to BGR for OpenCV/Storage
        return cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    def __del__(self):
        if self.cam:
            self.cam.activations_and_grads.release()
