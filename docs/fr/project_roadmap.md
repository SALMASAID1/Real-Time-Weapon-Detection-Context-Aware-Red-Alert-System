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
*   **Construction du Jeu de Données Unifié :** Agrégation de 50k+ images (Open Images, SOHAS, Synthétique).
*   **EDA :** Visualisation des déséquilibres de classes et de la géométrie des petits objets avec FiftyOne.
*   **Assainissement :** Standardisation des annotations vers le schéma `[Weapon, Person, Confuser]`.   

### Phase 2 : Intelligence du Modèle Hybride (Terminée ✅)
*   **Assemblage de l'Architecture :** Intégration des backbones YOLOv11 avec les Necks Swin-Transformer.
*   **Perte Personnalisée :** Implémentation de la Focal Loss + CIoU pour le déséquilibre de classes et la localisation.
*   **Entraînement Sensible aux Phases :** Warmup à backbone gelé suivi d'un réglage fin à taux d'apprentissage différentiels.
*   **Pipeline d'Entraînement :** 7 itérations (v1→v7) avec décodage DFL, NMS et validation visuelle.

### Phase 3 : Intelligence d'Inférence et XAI (Terminée ✅)
*   **Intégration SAHI :** Inférence par tuiles pour le traitement des flux de surveillance 4K.
*   **Proximité Mains-Armes :** Score de proximité GIoU avec persistance temporelle.
*   **IA Explicable (XAI) :** Cartes de chaleur Grad-CAM ciblant le Cou Swin pour la visualisation des décisions du modèle.
*   **Rendu de Superposition :** Compositage avec colormap INFERNO et encodage JPEG pour livraison WebSocket.

### Phase 4 : Alertes Multimodales et Tableau de Bord (Terminée ✅)
*   **Moteur de Notification :** Bot Telegram (asynchrone, fire-and-forget) et alertes audio locales (Pygame).
*   **Moteur d'Inférence :** Boucle à double cadence (SAHI + léger), double flux (Armes + Mains), diffusion par file d'attente WebSocket.
*   **Tableau de Bord React :** Surveillance en direct (VideoCanvas), historique des menaces (REST paginé), paramètres en temps réel (ThresholdPanel).
*   **Déduplication des Alertes :** Période de refroidissement par clé spatiale pour éviter le spam de notifications.

### Phase 5 : Optimisation et Déploiement (Terminée ✅)
*   **Entraînement Complet du Modèle :** Exécution de 50 époques terminée avec des taux d'apprentissage différentiels. Précision cible atteinte.
*   **Optimisation Edge :** Intégration d'ONNX Runtime pour une inférence haute performance.
*   **Validation Système :** Vérification du mAP, audit TFP et benchmarks de latence de bout en bout.

### Phase 6 : Maintenance Post-Projet (Active 📋)
*   **Surveillance Continue :** Suivi des performances dans des environnements réels.
*   **Expansion du Jeu de Données :** Mises à jour périodiques avec de nouveaux cas limites ou objets confuseurs.
*   **FiftyOne Brain :** Calcul des scores d'unicité pour élaguer les images d'armes redondantes.
*   **Rééquilibrage des Classes :** Réduction du ratio Arme:Confuseur pour des frontières de décision plus nettes.
*   **QA Visuelle :** Inspection manuelle des négatifs difficiles et des instances mal étiquetées.

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
