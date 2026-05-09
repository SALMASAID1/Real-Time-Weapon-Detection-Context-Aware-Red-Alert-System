import numpy as np

class IoUCalculator:
    """
    Pairwise IoU/GIoU between Hand bounding boxes and Weapon bounding boxes.
    """

    def __init__(self, mode="giou"):
        self.mode = mode

    def calculate(self, box_a, box_b):
        """
        Compute IoU or GIoU between two [x1, y1, x2, y2] boxes.
        """
        # 1. Intersection area
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        # 2. Union area
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        if self.mode == "iou":
            return iou
            
        # 3. GIoU extension: Smallest enclosing box
        cx1 = min(box_a[0], box_b[0])
        cy1 = min(box_a[1], box_b[1])
        cx2 = max(box_a[2], box_b[2])
        cy2 = max(box_a[3], box_b[3])
        
        c_area = (cx2 - cx1) * (cy2 - cy1)
        
        giou = iou - (c_area - union_area) / (c_area + 1e-6)
        return giou

    def pairwise(self, hands, weapons):
        """
        Compute full pairwise matrix: hands (rows) x weapons (cols).
        """
        matrix = np.zeros((len(hands), len(weapons)))
        for i, hand in enumerate(hands):
            for j, weapon in enumerate(weapons):
                matrix[i, j] = self.calculate(hand['bbox'], weapon['bbox'])
        return matrix

    def max_iou_per_weapon(self, hands, weapons):
        """
        Finds the highest proximity score for each weapon.
        """
        if not hands or not weapons:
            return [(0.0, -1)] * len(weapons)
            
        matrix = self.pairwise(hands, weapons)
        results = []
        for j in range(len(weapons)):
            max_idx = np.argmax(matrix[:, j])
            max_val = matrix[max_idx, j]
            results.append((max_val, max_idx))
            
        return results
