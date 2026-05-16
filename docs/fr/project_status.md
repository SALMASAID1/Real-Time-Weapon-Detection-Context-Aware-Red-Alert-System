# 📊 État du Projet : Système de Détection d'Armes
**Auteurs :** SAID Salma & TAMZIRT Mohamed
**Dernière mise à jour :** 16 mai 2026 (Achèvement final du projet)

## 📝 Résumé Exécutif
Le projet a complété toutes les phases de la feuille de route de développement. L'ensemble du pipeline système — de l'architecture du modèle au tableau de bord en temps réel — est structurellement complet, intégré et validé. Le modèle final `best.pt` est actif, et l'environnement de déploiement local est entièrement configuré avec des scripts de démarrage multiplateformes.

---

## ✅ Jalons Terminés

### Jalon 1 : Ingénierie des Données ✅
*   Unification de 50 967 images réparties en 3 classes : `[Weapon, Person, Confuser]`.
*   L'analyse EDA a révélé un goulot d'étranglement de 36,2 % sur les petits objets, ce qui a guidé l'architecture du Cou Swin.
*   Augmentation synthétique (pluie, brouillard, faible luminosité) pour une robustesse 24h/24.

### Jalon 2 : Architecture du Modèle Hybride ✅
*   **YOLOBackbone** (`yolo11n.pt`) : Passage avant couche par couche aux indices `[4, 6, 10]` pour une extraction P3/P4/P5 compatible DataParallel.
*   **SwinNeck** : SwinTransformerBlock timm (window_size=7, 1 bloc) avec connexions latérales FPN. Remplacement de `adaptive_max_pool2d` par `max_pool2d` pour la compatibilité ONNX.
*   **DetectionHead** : Branches cls/reg/obj découplées, Focal Loss (γ=2.0, α=0.25), régression DFL (reg_max=16), perte de boîte CIoU.
*   **Pipeline d'Entraînement** : 7 itérations, v7 finale avec LR différentiel et Cosine Annealing.

### Jalon 3 : Intelligence d'Inférence et XAI ✅
*   **Pipeline SAHI** : Inférence par tuiles (640×640, chevauchement 20 %) avec fusion NMS inter-tuiles.
*   **Score de Menace** : Formule composite `S = 0.3×conf + 0.5×proximité + 0.2×persistance` avec seuils HIGH/LOW.
*   **Calculateur IoU** : Mode GIoU pour la détection de proximité, matrice pairée mains-armes.
*   **Grad-CAM** : Cible la couche `neck.p4_proj_out`, retourne des octets JPEG pour la livraison WebSocket.
*   **Rendu de Superposition** : Compositage avec colormap INFERNO, annotation bbox, encodage JPEG.

### Jalon 4 : Alertes Multimodales et Tableau de Bord ✅
*   **AlertDispatcher** : Telegram (asynchrone), audio Pygame (lecture en 3 boucles), journalisation JSONL, déduplication avec refroidissement de 30s.
*   **InferenceEngine** : Double cadence (SAHI tous les N + léger), double flux (Armes + Mains), diffusion par file asyncio.Queue. Correction du bug des paramètres simulés ONNX.
*   **Backend FastAPI** : WebSocket `/ws/stream/{camera_id}`, REST `/api/threats` (CRUD), `/api/settings` (application en direct).
*   **Tableau de Bord React** : LiveMonitor (VideoCanvas + panneau latéral), ThreatHistory (paginé + filtré), Settings (ThresholdPanel).

### Jalon 5 : Optimisation et Déploiement ✅
*   **Export de Modèle** : Export réussi de `best.pt` vers `best.onnx` (opset 14) via `export_onnx.py`.
*   **Environnement** : Environnement virtuel Python et dépendances Node.js installés.
*   **Scripts de Démarrage** : Création de `start_system.sh` (Linux/macOS) et `start_system.bat` (Windows) avec logique d'arrêt en douceur.
*   **Entraînement Final** : Exécution terminée des 50 époques d'entraînement, atteignant les objectifs de mAP et de précision de validation.

---

## 🏁 État Final du Projet
Le système est prêt pour le déploiement en production. Tous les jalons ont été atteints et les performances dépassent les objectifs initiaux.

---
*Rapport d'état généré par l'Assistant IA Antigravity.*
