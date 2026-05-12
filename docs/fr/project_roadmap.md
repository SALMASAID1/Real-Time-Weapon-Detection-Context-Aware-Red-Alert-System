# 🚀 Feuille de Route du Projet : Détection d'Armes en Temps Réel et Système d'Alerte Rouge

## 🌟 Vision et Objectif
Le **Système de Détection d'Armes en Temps Réel et d'Alerte Rouge Sensible au Contexte** est une plateforme d'intelligence de surveillance de nouvelle génération. Il dépasse la simple classification pour identifier les menaces actives en analysant la **relation spatiale** entre les humains et les armes, distinguant ainsi une arme rangée d'une menace imminente.

---

## 🏗️ Architecture Technique (L'Inférence à "Double Flux")
Notre système utilise un pipeline **d'Inférence à Double Flux** à la pointe de la technologie :
1.  **Flux Armes (YOLO-Swin Hybride) :** Un modèle personnalisé combinant la vitesse de YOLOv11 avec le contexte global du Swin Transformer pour détecter les Armes et les "Confuseurs" (téléphones, outils).
2.  **Flux Mains (YOLO-Hand) :** Un flux spécialisé pour la détection robuste des mains afin de faciliter l'analyse de proximité.
3.  **Logique de Menace :** Un moteur d'IoU Géométrique qui valide si une arme est physiquement manipulée.

---

## 📊 Métriques de Succès
| Métrique | Cible | Description |
| :--- | :--- | :--- |
| **mAP@0.5:0.95** | ≥ 0,72 | Précision moyenne pour la localisation des armes. |
| **TFP (Confuseurs)** | < 2,0 % | Taux de Faux Positifs sur les objets non dangereux. |
| **Vitesse d'Inférence** | ≥ 40 FPS | Performance en temps réel sur matériel Edge (Jetson). |
| **Latence de bout en bout** | < 500ms | Temps entre la détection et la notification. |

---

## 🗺️ Feuille de Route du Projet

### Phase 1 : Ingénierie des Données et Fondations (Terminée ✅)
*   **Construction du Jeu de Données Unifié :** Agrégation de 45k+ images (Open Images, SOHAS, Synthétique).
*   **EDA :** Visualisation des déséquilibres de classes et de la géométrie des petits objets avec FiftyOne.
*   **Assainissement :** Standardisation des annotations vers le schéma `[Weapon, Person, Confuser]`.

### Phase 2 : Intelligence du Modèle Hybride (Active ⚡)
*   **Assemblage de l'Architecture :** Intégration des backbones YOLOv11 avec les Necks Swin-Transformer.
*   **Perte Personnalisée :** Implémentation de la Focal Loss pour gérer la sous-représentation de la classe "Confuseur".
*   **Entraînement Sensible aux Phases :** Utilisation de phases de "warmup" à backbone gelé suivies d'un réglage fin complet.

### Phase 3 : Intelligence d'Inférence et XAI (À venir 🚀)
*   **Intégration SAHI :** Inférence par tuiles pour le traitement des flux de surveillance 4K.
*   **Proximité Mains-Armes :** Logique de validation des menaces basée sur la proximité.
*   **IA Explicable (XAI) :** Cartes de chaleur Grad-CAM pour visualiser la prise de décision du modèle.

### Phase 4 : Alertes Multimodales et UI (Futur 🔮)
*   **Moteur de Notification :** Intégration du Bot Telegram et des alertes audio locales (Pygame).
*   **Tableau de Bord React :** Interface de surveillance en temps réel avec historique des menaces.
*   **Optimisation Edge :** Quantification TensorRT pour un déploiement à haut débit (FPS).

---

## 📚 Ressources d'Apprentissage

### Apprentissage Profond et Vision par Ordinateur
*   [Fast.ai Practical Deep Learning](https://course.fast.ai/) : Meilleure base PyTorch disponible (en anglais).
*   [DeepLearning.AI - CNNs](https://www.coursera.org/learn/convolutional-neural-networks) : Théorie de base des architectures YOLO.

### Recherche Avancée
*   [Papier Swin Transformer (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030) : Théorie derrière notre "Neck" à contexte global.
*   [Focal Loss (arXiv:1708.02002)](https://arxiv.org/abs/1708.02002) : Solution mathématique pour le déséquilibre des classes.

### Outils et Frameworks
*   [Ultralytics YOLOv11](https://docs.ultralytics.com/) : Notre moteur de détection principal.
*   [Documentation FiftyOne](https://docs.voxel51.com/) : Curation de données et QA visuelle.
*   [Bibliothèque Timm](https://huggingface.co/docs/timm/index) : Source pour les implémentations de blocs transformer.

---
*Créé par l'Assistant IA Antigravity.*
