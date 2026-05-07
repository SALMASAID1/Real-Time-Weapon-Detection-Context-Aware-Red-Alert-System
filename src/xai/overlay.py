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


class OverlayRenderer:
    """
    Parameters
    ----------
    colormap     : int   — OpenCV colormap constant (cv2.COLORMAP_INFERNO)
    alpha        : float — Heatmap opacity for alpha blend (default 0.45)
    jpeg_quality : int   — JPEG compression quality 0–100 (default 85)
    """

    def render(self, frame, heatmap, detection: dict) -> bytes:
        """
        Composite heatmap onto frame and encode as JPEG.

        Parameters
        ----------
        frame     : np.ndarray  — Original BGR frame, shape (H, W, 3)
        heatmap   : np.ndarray  — Float32 activation map, shape (H, W)
        detection : dict        — Used to draw bbox annotation on the composite

        Returns
        -------
        bytes  — JPEG-encoded composite image.
                 Caller (InferenceEngine) base64-encodes this for WebSocket JSON.
        """
        ...
