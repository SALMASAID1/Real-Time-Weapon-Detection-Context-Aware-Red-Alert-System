import numpy as np
import torch
from typing import List, Dict

class SAHIPipeline:
    """
    Tiled inference pipeline wrapping HybridWeaponDetector.
    """

    def __init__(self, model, tile_size=640, overlap_ratio=0.2, batch_size=8, nms_iou_threshold=0.5, conf_threshold=0.25, iou_threshold=0.45):
        self.model = model
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.nms_iou_threshold = nms_iou_threshold
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.stride = int(tile_size * (1 - overlap_ratio))

    def run(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs tiled inference on a full frame and merges results.
        """
        h, w = frame.shape[:2]
        
        # 1. Generate tiles and their origins
        tiles_data = self._tile_frame(frame)
        all_detections = []

        # Track if frame was resized (small-image path)
        was_resized = (h <= self.tile_size and w <= self.tile_size
                       and (h != self.tile_size or w != self.tile_size))
        scale_x = w / self.tile_size if was_resized else 1.0
        scale_y = h / self.tile_size if was_resized else 1.0

        # 2. Process tiles in batches for efficiency
        for i in range(0, len(tiles_data), self.batch_size):
            batch = tiles_data[i : i + self.batch_size]
            batch_imgs = [t[0] for t in batch]
            batch_origins = [t[1] for t in batch]

            # In a real scenario, we'd batch these into a single tensor
            # For now, we process them sequentially or via model.predict(batch)
            for img, origin in zip(batch_imgs, batch_origins):
                tile_dets = self.model.predict(img, conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)
                
                # 3. Translate tile coordinates to full frame coordinates
                translated_dets = self._to_full_frame_coords(tile_dets, origin)
                all_detections.extend(translated_dets)

        # 4. Final NMS to remove duplicates at tile boundaries
        if not all_detections:
            return []

        merged = self._nms(all_detections, self.nms_iou_threshold)

        # 5. If frame was resized, scale bboxes back to original frame dimensions
        if was_resized:
            for det in merged:
                b = det['bbox']
                det['bbox'] = [b[0] * scale_x, b[1] * scale_y,
                               b[2] * scale_x, b[3] * scale_y]

        return merged

    def _tile_frame(self, frame):
        """
        Partitions the frame into overlapping tiles.
        Falls back to a single tile for frames smaller than tile_size.
        """
        h, w = frame.shape[:2]

        # Guard: if frame is smaller than tile_size, return as single tile
        if h <= self.tile_size and w <= self.tile_size:
            # Pad or resize to tile_size for consistent model input
            import cv2
            tile = cv2.resize(frame, (self.tile_size, self.tile_size))
            return [(tile, (0, 0))]

        tiles = []

        for y in range(0, h - self.tile_size + self.stride, self.stride):
            for x in range(0, w - self.tile_size + self.stride, self.stride):
                # Ensure we don't go out of bounds for the last tiles
                curr_x = min(x, w - self.tile_size)
                curr_y = min(y, h - self.tile_size)
                
                tile = frame[curr_y : curr_y + self.tile_size, curr_x : curr_x + self.tile_size]
                tiles.append((tile, (curr_x, curr_y)))
                
                if x + self.tile_size >= w: break
            if y + self.tile_size >= h: break

        return tiles

    def _to_full_frame_coords(self, detections, tile_origin):
        """
        Translates bbox coordinates from tile-space to full-frame space.
        """
        off_x, off_y = tile_origin
        for det in detections:
            bbox = det['bbox']
            # Convert from normalized [0, 1] to tile pixel coords if they are normalized
            if all(0 <= v <= 1.0 for v in bbox):
                x1 = bbox[0] * self.tile_size
                y1 = bbox[1] * self.tile_size
                x2 = bbox[2] * self.tile_size
                y2 = bbox[3] * self.tile_size
            else:
                x1, y1, x2, y2 = bbox
            det['bbox'] = [x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y]
        return detections

    def _nms(self, detections, iou_threshold):
        """
        Simple CPU-based NMS for merging detections from multiple tiles.
        """
        if not detections:
            return []

        # Convert to tensor for faster processing if available
        bboxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d['confidence'] for d in detections], dtype=torch.float32)
        
        # Standard torchvision NMS
        from torchvision.ops import nms
        keep_indices = nms(bboxes, scores, iou_threshold)
        
        return [detections[i] for i in keep_indices]
