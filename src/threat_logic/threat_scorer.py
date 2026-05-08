"""
src/threat_logic/threat_scorer.py
====================================
Composite threat scoring and Red Alert escalation logic.

Score composition
-----------------
A single detection confidence is insufficient to trigger a Red Alert.
High-confidence detections of weapons in a gun shop, museum display, or
movie prop are genuine detections but not genuine threats. Context matters.

The ThreatScorer combines three orthogonal signals into a composite score:

    S = w1 · Conf + w2 · Proximity + w3 · Persistence

Where:
  Conf        : float  — Model confidence for the weapon detection (0–1)
  Proximity   : float  — Max GIoU between weapon and any detected hand (0–1)
                         from IoUCalculator.max_iou_per_weapon()
  Persistence : float  — Normalised count of consecutive frames this
                         (weapon_class, approximate_location) pair has
                         been detected. Clamped to 1.0 at PERSIST_MAX frames.
  w1, w2, w3  : float  — Tunable weights (default 0.3, 0.5, 0.2)

Threat level thresholds
-----------------------
  S < 0.40              → NONE   (detection logged, no alert)
  0.40 ≤ S < 0.70       → LOW    (low-priority notification)
  S ≥ 0.70              → HIGH   → Red Alert triggered

Temporal persistence tracking
------------------------------
Persistence prevents single-frame false positives from firing Red Alerts.
A weapon appearing in frame 1 only (possibly a misidentified phone) will
have Persistence ≈ 0 and a low composite score even if Conf is high.

Tracking uses a lightweight dict keyed by (class_id, grid_cell) where
grid_cell is a coarse spatial bucket (e.g. 100×100px grid over the full frame).
This avoids requiring a full object tracker (SORT, ByteTrack) which would add
significant latency.

Public API
----------
    ThreatScorer(iou_calculator, weights, thresholds, persist_max)
        .score(detections, frame_id) -> list[ScoredDetection]
        .reset()  — clear persistence state (e.g. on camera switch)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoredDetection:
    """
    Output of ThreatScorer.score() for a single detection.

    Fields
    ------
    detection      : dict        — Original detection dict from InferenceEngine
    confidence     : float       — Model confidence
    proximity_iou  : float       — Max GIoU with nearest hand (0 if no hands)
    persistence    : float       — Normalised frame-count score (0–1)
    composite_score: float       — Weighted composite S
    threat_level   : str         — "NONE" | "LOW" | "HIGH"
    paired_hand_idx: Optional[int] — Index of the hand with max IoU, if any
    """
    detection:       dict
    confidence:      float
    proximity_iou:   float
    persistence:     float
    composite_score: float
    threat_level:    str
    paired_hand_idx: Optional[int] = None


class ThreatScorer:
    """
    Parameters
    ----------
    iou_calculator : IoUCalculator
    weights        : dict  — {"conf": 0.3, "proximity": 0.5, "persistence": 0.2}
    thresholds     : dict  — {"low": 0.40, "high": 0.70}
    persist_max    : int   — Frame count at which persistence score saturates to 1.0 (default 10)
    """

    def score(self, detections: list, frame_id: int) -> list:
        """
        Score all detections in a frame.

        Parameters
        ----------
        detections : list[dict]
            Merged detection list from the two-stream pipeline:
            - "Weapon" & "Confuser" from the Custom Hybrid Model.
            - "Hand" from the Pre-trained YOLO Hand Model.
        frame_id   : int  — Used for persistence tracking.

        Returns
        -------
        list[ScoredDetection]
            Only weapon-class detections are scored.
            Hand detections are used to calculate proximity IoU.
        """
        ...

    def reset(self):
        """Clear all persistence tracking state."""
        ...
