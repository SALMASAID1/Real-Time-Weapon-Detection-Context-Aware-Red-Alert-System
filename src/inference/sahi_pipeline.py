"""
src/inference/sahi_pipeline.py
================================
Slicing Aided Hyper Inference (SAHI) — tiled inference for 4K footage.

Why SAHI is required
--------------------
YOLOv11 processes images resized to 640×640. A 4K frame (3840×2160) downscaled
to 640×640 compresses pixel information by a factor of ~36x. A weapon that
occupies 100×100 pixels in the original frame becomes 17×17 pixels — below
the detection threshold of most CNN backbones.

SAHI preserves resolution by partitioning the frame into overlapping tiles,
running inference on each at full resolution, then stitching results back.

Tiling strategy
---------------
  tile_size    : 640  — matches model input (no rescaling needed per tile)
  overlap_ratio: 0.2  — 20% overlap on each edge (128px for 640 tiles)
  stride       : 512  — effective step between tile origins

For a 4K frame: ceil(3840/512) × ceil(2160/512) = 8×5 = 40 tiles per frame.
With GPU batching (batch_size=8), this requires 5 forward passes per frame.

Overlap rationale
-----------------
Weapons straddling tile boundaries appear partially in two adjacent tiles.
With 20% overlap, any object larger than 128px will be fully visible in at
least one tile. For sub-128px weapons, overlap is increased to 0.3.

Post-merge NMS
--------------
After stitching tile detections back to full-frame coordinates, predictions
from overlapping tiles produce duplicate bounding boxes for the same object.
A final NMS pass with iou_threshold=0.5 suppresses duplicates.

Critical: this NMS threshold is DIFFERENT from the Hand-Weapon IoU threshold
in threat_logic. Do not conflate them.

Public API
----------
    SAHIPipeline(model, tile_size, overlap_ratio, batch_size, nms_iou_threshold)
        .run(frame: np.ndarray) -> list[dict]
            Returns merged, NMS-filtered detections in full-frame coordinates.
"""


class SAHIPipeline:
    """
    Tiled inference pipeline wrapping HybridWeaponDetector.

    Parameters
    ----------
    model            : HybridWeaponDetector
    tile_size        : int   — pixel side length of each square tile (640)
    overlap_ratio    : float — fractional overlap between adjacent tiles (0.2)
    batch_size       : int   — number of tiles per GPU forward pass (8)
    nms_iou_threshold: float — IoU threshold for post-merge duplicate suppression (0.5)
    """

    def run(self, frame):
        """
        Parameters
        ----------
        frame : np.ndarray  — Full-resolution BGR frame (e.g. 3840×2160).

        Returns
        -------
        list[dict]
            Detections in full-frame pixel coordinates.
            Same schema as HybridWeaponDetector.predict() output.
        """
        ...

    def _tile_frame(self, frame):
        """
        Partition frame into overlapping tiles.

        Returns
        -------
        list[tuple[np.ndarray, tuple[int,int]]]
            Each element: (tile_image, (tile_origin_x, tile_origin_y))
        """
        ...

    def _to_full_frame_coords(self, detections, tile_origin):
        """
        Translate tile-relative bounding boxes to full-frame coordinates.

        Parameters
        ----------
        detections  : list[dict]  — Detections from HybridWeaponDetector.
        tile_origin : tuple[int, int]  — (x_offset, y_offset) of this tile.

        Returns
        -------
        list[dict]  — Same detections with translated bbox coordinates.
        """
        ...

    def _nms(self, detections, iou_threshold):
        """
        Non-Maximum Suppression across all stitched tile detections.

        Parameters
        ----------
        detections    : list[dict]  — All merged full-frame detections.
        iou_threshold : float       — Overlap threshold for suppression.

        Returns
        -------
        list[dict]  — Filtered, duplicate-free detection list.
        """
        ...
