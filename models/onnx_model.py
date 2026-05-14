import cv2
import torch
import numpy as np
import onnxruntime as ort
import logging
from models.heads.detection_head import DetectionHead

logger = logging.getLogger(__name__)

class ONNXWeaponDetector:
    """
    ONNX Runtime wrapper that provides the exact same .predict() interface
    as the HybridWeaponDetector, but runs inference via ONNX for speed.
    """
    def __init__(self, onnx_path: str, device: str = "cuda"):
        self.device = device
        
        # Set execution providers based on device
        providers = []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            logger.info("ONNX: Using CUDAExecutionProvider")
        else:
            providers.append("CPUExecutionProvider")
            logger.info("ONNX: Using CPUExecutionProvider")
            
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Instantiate a DetectionHead purely to reuse its decode_predictions method
        # The in_channels don't matter because we won't run its .forward() method.
        dummy_in_channels = {"P3": 1, "P4": 1, "P5": 1}
        self.head = DetectionHead(in_channels=dummy_in_channels, nc=3)
        self.head.to(device)

    def parameters(self):
        """
        Mock parameters() to satisfy checks in engine.py
        """
        class MockParam:
            @property
            def device(self):
                return torch.device(self.device_str)
            @property
            def dtype(self):
                return torch.float32
        
        mock = MockParam()
        mock.device_str = self.device
        return [mock]

    def half(self):
        """Mock method for API compatibility."""
        pass

    def eval(self):
        """Mock method for API compatibility."""
        pass

    def predict(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        Runs the ONNX graph and decodes the results.
        """
        # Save original dimensions for scaling boxes back later
        orig_shape = None
        if isinstance(frame, np.ndarray):
            orig_shape = frame.shape[:2]
            
        # 1. Preprocess
        if isinstance(frame, np.ndarray):
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            # HWC -> CHW, BGR is kept
            x = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
            x = np.expand_dims(x, axis=0)
        else:
            # If tensor, convert to numpy
            x = frame.cpu().numpy()
            if x.dtype == np.float16:
                x = x.astype(np.float32)

        # 2. Run ONNX Session
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: x})
        cls_logits, reg_offsets, objectness = outputs

        # 3. Decode Predictions using PyTorch head
        preds = (
            torch.from_numpy(cls_logits).to(self.device),
            torch.from_numpy(reg_offsets).to(self.device),
            torch.from_numpy(objectness).to(self.device)
        )
        
        results = self.head.decode_predictions(
            preds, 
            conf_thres=conf_threshold, 
            iou_thres=iou_threshold
        )
        
        # 4. Scale bounding boxes back to original frame size
        if orig_shape and orig_shape != (640, 640):
            h_orig, w_orig = orig_shape
            scale_x = w_orig / 640.0
            scale_y = h_orig / 640.0
            for det in results:
                b = det['bbox']
                det['bbox'] = [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                
        return results
