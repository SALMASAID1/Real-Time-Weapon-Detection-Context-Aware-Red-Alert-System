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
*   **Apprentissage Différentiel :** Implémentation d'un taux d'apprentissage plus bas pour le backbone (`1e-5`) par rapport au neck/head (`5e-5`) pour préserver les caractéristiques COCO pré-entraînées.
*   **Réglage de la Focal Loss :** Ajustement de $\alpha$ et $\gamma$ pour pénaliser les négatifs faciles de la classe "Confuseur".
*   **Assainissement des Étiquettes :** Suppression des boîtes englobantes en double et nettoyage des masques de segmentation bruyants.

---

## 🛑 Défis et Bloqueurs Actuels
*   **Déséquilibre des Classes :** Les "Confuseurs" sous-performent actuellement par rapport aux "Armes".
*   **Synchronisation Multi-GPU :** Résolution de l'erreur `KeyError: P3` lors de l'utilisation de `nn.DataParallel` sur Kaggle Dual T4 (Solution : passée avant manuelle implémentée dans `YOLOBackbone`).

---

## ⏭️ Prochaines Étapes Immédiates
1.  **Exécution de 50 époques :** Lancer un entraînement complet avec le planificateur Cosine Annealing.
2.  **Audit Visuel :** Utiliser FiftyOne pour inspecter les 100 images les moins performantes de l'ensemble de validation.
3.  **Banc d'essai d'Inférence :** Porter les poids vers `src/inference/` pour mesurer les FPS en conditions réelles.

---
*Rapport d'état généré par l'Assistant IA Antigravity.*
