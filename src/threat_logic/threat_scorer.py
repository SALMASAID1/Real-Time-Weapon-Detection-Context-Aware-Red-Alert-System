from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class ScoredDetection:
    detection:       dict
    confidence:      float
    proximity_iou:   float
    persistence:     float
    composite_score: float
    threat_level:    str
    paired_hand_idx: Optional[int] = None

class ThreatScorer:
    def __init__(self, iou_calculator, weights=None, thresholds=None, persist_max=10):
        self.iou_calc = iou_calculator
        self.weights = weights or {"conf": 0.3, "proximity": 0.5, "persistence": 0.2}
        self.thresholds = thresholds or {"low": 0.40, "high": 0.70}
        self.persist_max = persist_max
        self.persistence_registry = {} # (class_id, grid_x, grid_y) -> count

    def score(self, detections: List[Dict], frame_id: int) -> List[ScoredDetection]:
        # 1. Separate Hands and Weapons
        # Hands  → identified by class_name == "hand" (from the COCO person stream)
        # Weapons → identified by is_weapon == True (from the HybridWeaponDetector)
        #
        # IMPORTANT: Do NOT filter weapons by class_id == 0 alone!
        # COCO person class is also id 0, which caused false RED ALERTs
        # on every person detection.  We now use the explicit is_weapon flag
        # that the engine sets only on detections from the weapon model.
        hands = [d for d in detections if d.get('class_name') == 'hand']
        weapons = [d for d in detections if d.get('is_weapon', False)]
        
        # 2. Get Proximity Scores
        proximity_data = self.iou_calc.max_iou_per_weapon(hands, weapons)
        
        scored_results = []
        
        for idx, (weapon, (iou, hand_idx)) in enumerate(zip(weapons, proximity_data)):
            # 3. Calculate Persistence
            # We use a coarse grid (100x100) to track objects without a complex tracker
            x1, y1, x2, y2 = weapon['bbox']
            gx, gy = int((x1+x2)/200), int((y1+y2)/200) # Grid cell
            key = (weapon['class_id'], gx, gy)
            
            count = self.persistence_registry.get(key, 0) + 1
            self.persistence_registry[key] = count
            persistence_score = min(count / self.persist_max, 1.0)
            
            # 4. Composite Score Calculation
            conf = weapon['confidence']
            # Proximity is normalized from [-1, 1] to [0, 1] for GIoU
            norm_proximity = max(0, iou) 
            
            s = (self.weights['conf'] * conf + 
                 self.weights['proximity'] * norm_proximity + 
                 self.weights['persistence'] * persistence_score)
            
            # 5. Determine Threat Level
            level = "NONE"
            if s >= self.thresholds['high']:
                level = "HIGH"
            elif s >= self.thresholds['low']:
                level = "LOW"
            
            scored_results.append(ScoredDetection(
                detection=weapon,
                confidence=conf,
                proximity_iou=norm_proximity,
                persistence=persistence_score,
                composite_score=s,
                threat_level=level,
                paired_hand_idx=hand_idx if hand_idx != -1 else None
            ))
            
        # 6. Cleanup old persistence (very simple cleanup logic)
        if len(self.persistence_registry) > 100:
             self.persistence_registry = {k: v for k, v in self.persistence_registry.items() if v > 1}

        return scored_results

    def reset(self):
        self.persistence_registry = {}
