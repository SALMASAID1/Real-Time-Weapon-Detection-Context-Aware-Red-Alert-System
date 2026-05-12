# 📊 État du Projet : Système de Détection d'Armes
**Dernière mise à jour :** 12 mai 2026

## 📝 Résumé Exécutif
Le projet est actuellement en **Phase 2 : Entraînement Actif et Optimisation**. L'architecture hybride (YOLOv11-Swin) est entièrement implémentée et est en cours d'entraînement sur un jeu de données unifié de 45 000 images. L'accent est mis sur la suppression des faux positifs provenant des objets "Confuseurs" et sur la stabilisation de l'entraînement multi-GPU.

---

## ✅ Jalons Terminés
1.  **Conception de l'Architecture :** Stratégie d'inférence à double flux (Mains + Armes) finalisée.
2.  **Consolidation des Données :** Unification des annotations provenant de SOHAS, Open Images et collectes personnalisées.
3.  **Implémentation de Base :** Modules `YOLOBackbone`, `SwinNeck` et `DetectionHead` terminés.
4.  **Configuration Environnement :** Mise en place des flux DagsHub vers Colab/Kaggle pour un accès I/O sans latence.

---

## 🚧 Travaux en Cours (Phase 2 - Entraînement V7)
*   **Artéfact Actif :** `notebooks/Phase2_Training_v7.ipynb`.
*   **Apprentissage Différentiel :** Stratégie de LR finalisée (`1e-5` backbone / `5e-5` head) avec Cosine Annealing.
*   **Décodage Prêt :** Implémentation de `DetectionHead.decode_predictions()` avec DFL et NMS pour une validation en temps réel.
*   **Optimisation de la Perte :** Correction du calcul du centre CIoU et de la cohérence des couleurs BGR-vers-RGB.

---

## 🛑 Défis et Bloqueurs Actuels
*   **Audit Visuel :** (RÉSOLU) Le modèle affiche désormais de réelles boîtes englobantes pendant l'entraînement.
*   **Synchronisation Multi-GPU :** Résolution de l'erreur `KeyError: P3` via une itération manuelle du passage avant dans `YOLOBackbone`.

---

## ⏭️ Prochaines Étapes Immédiates
1.  **Exécution de 50 époques :** Lancer l'entraînement complet optimisé.
2.  **Validation Visuelle :** Vérifier la qualité de la localisation en utilisant la nouvelle logique `predict()`.
3.  **Banc d'essai d'Inférence :** Porter les poids vers `src/inference/` une fois que `best.pt` est atteint.

---
*Rapport d'état généré par l'Assistant IA Antigravity.*
