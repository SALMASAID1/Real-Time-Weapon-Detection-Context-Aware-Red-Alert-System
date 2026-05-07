"""
src/threat_logic/iou_calculator.py
=====================================
Pairwise IoU between Hand bounding boxes and Weapon bounding boxes.

Why IoU over centroid distance
------------------------------
Centroid distance measures the gap between box centres, but it is scale-
dependent and ignores box dimensions. A large hand box and a small weapon box
can have identical centroid distance whether the weapon is inside the hand or
far away.

IoU (Intersection over Union) directly measures the fractional overlap of the
two bounding boxes. An IoU > 0 means the boxes physically overlap in pixel
space — a strong geometric proxy for "the person is gripping the weapon".

    IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)

For our use case, a weapon is considered "held" when IoU(hand_box, weapon_box)
exceeds the configured threshold (default 0.45, tunable via Settings API).

Generalised IoU (GIoU) extension
---------------------------------
For near-miss detections (hand adjacent to weapon but not overlapping), pure
IoU = 0 and provides no gradient for threat scoring. GIoU adds a penalty term
for the gap between non-overlapping boxes, producing a smooth score even when
IoU is 0:

    GIoU = IoU - (Area(C) - Area(A ∪ B)) / Area(C)

where C is the smallest enclosing box. This allows the ThreatScorer to assign
non-zero proximity scores to near-miss scenarios — useful for early warning.

Public API
----------
    IoUCalculator(mode="iou")        # mode: "iou" or "giou"
        .calculate(box_a, box_b) -> float
        .pairwise(hands, weapons)   -> np.ndarray  (shape: n_hands × n_weapons)
        .max_iou_per_weapon(hands, weapons) -> list[tuple[float, int]]
"""


class IoUCalculator:
    """
    Parameters
    ----------
    mode : str  — "iou" (standard) or "giou" (generalised, recommended for scoring)
    """

    def calculate(self, box_a: list, box_b: list) -> float:
        """
        Compute IoU or GIoU between two bounding boxes.

        Parameters
        ----------
        box_a, box_b : list[float]
            Bounding boxes in [x1, y1, x2, y2] format (pixel coordinates).

        Returns
        -------
        float — IoU in [0, 1] or GIoU in [-1, 1].
        """
        ...

    def pairwise(self, hands: list, weapons: list):
        """
        Compute full pairwise matrix: every hand vs. every weapon.

        Parameters
        ----------
        hands   : list[dict]  — Detections with class_id == class_id("hand")
        weapons : list[dict]  — Detections with class_id in weapon class IDs

        Returns
        -------
        np.ndarray  — Shape (n_hands, n_weapons). Entry [i, j] is the
                       IoU/GIoU between hand i and weapon j.
        """
        ...

    def max_iou_per_weapon(self, hands: list, weapons: list):
        """
        For each weapon, find the maximum IoU across all detected hands.
        This is the primary input to the ThreatScorer.

        Returns
        -------
        list[tuple[float, int]]
            Each element: (max_iou, hand_index_or_-1_if_no_hands)
        """
        ...
