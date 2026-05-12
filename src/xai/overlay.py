"""
src/xai/overlay.py
====================
Composites Grad-CAM heatmap onto the raw video frame for WebSocket delivery.

Compositing pipeline
--------------------
1. Receive raw heatmap (float32 H×W, values 0–1) from GradCAMGenerator.
2. Apply a perceptually-linearised colormap (COLORMAP_JET or COLORMAP_INFERNO).
   INFERNO is preferred: it is perceptually uniform and accessible to
   red-green colour-blind operators.
3. Resize the colourised heatmap to match the display frame dimensions
   (may differ from the model input tile size after coordinate remapping).
4. Blend with the original BGR frame using alpha compositing:
       composite = (1 - alpha) * frame + alpha * heatmap_rgb
   Default alpha = 0.45 — visible but not obscuring the original footage.
5. Draw the original detection bounding box on the composite.
6. Encode the composite as JPEG (quality=85) and return as bytes.
   The FastAPI WebSocket router base64-encodes this for JSON transport.

Why JPEG and not PNG
--------------------
A 1080p PNG overlay is ~2–6 MB. Over a WebSocket at 30 FPS this saturates
a 1Gbps connection. JPEG at quality=85 produces ~80–200 KB/frame — a 10–30×
reduction with visually negligible artefacts for surveillance analysis.

Public API
----------
    OverlayRenderer(colormap, alpha, jpeg_quality)
        .render(frame, heatmap, detection) -> bytes  (JPEG)
"""

import cv2
import numpy as np


# Threat level colors (BGR)
_LEVEL_COLORS = {
    "HIGH":   (0, 0, 255),     # Red
    "LOW":    (0, 165, 255),   # Orange
    "NONE":   (0, 255, 0),     # Green
}

# Class label colors (BGR)
_CLASS_COLORS = {
    0: (0, 0, 255),    # Weapon → Red
    1: (255, 200, 0),  # Person → Cyan-blue
    2: (0, 200, 255),  # Confuser → Orange-yellow
}


class OverlayRenderer:
    """
    Composites Grad-CAM heatmaps and detection bounding boxes onto video frames.

    Parameters
    ----------
    colormap     : int   — OpenCV colormap constant (cv2.COLORMAP_INFERNO)
    alpha        : float — Heatmap opacity for alpha blend (default 0.45)
    jpeg_quality : int   — JPEG compression quality 0–100 (default 85)
    """

    def __init__(self, colormap=cv2.COLORMAP_INFERNO, alpha=0.45, jpeg_quality=85):
        self.colormap = colormap
        self.alpha = alpha
        self.jpeg_quality = jpeg_quality

    def render(self, frame, heatmap, detection: dict) -> bytes:
        """
        Composite heatmap onto frame and encode as JPEG.

        Parameters
        ----------
        frame     : np.ndarray  — Original BGR frame, shape (H, W, 3)
        heatmap   : np.ndarray  — Float32 activation map, shape (Hm, Wm), values 0–1
        detection : dict        — Must contain 'bbox' key with [x1, y1, x2, y2].
                                  Optional: 'class_id', 'class_name', 'confidence',
                                  'threat_level'.

        Returns
        -------
        bytes  — JPEG-encoded composite image.
                 Caller (InferenceEngine) base64-encodes this for WebSocket JSON.
        """
        h, w = frame.shape[:2]
        composite = frame.copy()

        # 1. Colorise and resize heatmap
        if heatmap is not None and heatmap.size > 0:
            # Normalize to 0–255 uint8
            heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))

            # Apply perceptually-uniform colormap
            heatmap_color = cv2.applyColorMap(heatmap_uint8, self.colormap)

            # Resize to match frame dimensions
            if heatmap_color.shape[:2] != (h, w):
                heatmap_color = cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_LINEAR)

            # 2. Alpha blend: composite = (1 - α)·frame + α·heatmap
            composite = cv2.addWeighted(frame, 1 - self.alpha, heatmap_color, self.alpha, 0)

        # 3. Draw bounding box annotation
        composite = self._draw_bbox(composite, detection)

        # 4. Encode as JPEG
        _, buffer = cv2.imencode(
            '.jpg', composite,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return buffer.tobytes()

    def render_detections(self, frame, detections: list) -> bytes:
        """
        Draw all detection bounding boxes on a frame (no heatmap).
        Used for standard frame annotation without Grad-CAM.

        Parameters
        ----------
        frame      : np.ndarray  — Original BGR frame
        detections : list[dict]  — List of detection dicts

        Returns
        -------
        bytes  — JPEG-encoded annotated frame.
        """
        composite = frame.copy()

        for det in detections:
            composite = self._draw_bbox(composite, det)

        _, buffer = cv2.imencode(
            '.jpg', composite,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return buffer.tobytes()

    @staticmethod
    def _draw_bbox(frame, detection: dict):
        """Draw a single bounding box with label on the frame."""
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return frame

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp coordinates to frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Color by class or threat level
        class_id = detection.get("class_id", -1)
        color = _CLASS_COLORS.get(class_id, (255, 255, 255))

        threat_level = detection.get("threat_level")
        if threat_level and threat_level in _LEVEL_COLORS:
            color = _LEVEL_COLORS[threat_level]

        # Draw rectangle
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        class_name = detection.get("class_name", f"cls_{class_id}")
        confidence = detection.get("confidence", 0.0)
        label = f"{class_name} {confidence:.2f}"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)

        # Draw label text (white on coloured background)
        cv2.putText(
            frame, label,
            (x1 + 2, label_y2 - baseline - 2),
            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
        )

        return frame
