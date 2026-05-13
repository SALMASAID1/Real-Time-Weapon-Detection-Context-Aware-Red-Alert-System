# Rapport de Présentation : Système de Détection d'Armes en Temps Réel

## 1. Introduction
### Problématique
Dans le contexte sécuritaire actuel, la détection rapide et automatique d'armes à feu dans les flux de vidéosurveillance est devenue un enjeu majeur pour la sécurité publique et privée. Les systèmes de surveillance traditionnels reposent souvent sur une surveillance humaine constante, sujette à la fatigue et aux erreurs d'inattention.

### Importance du sujet
Une détection proactive permet de réduire drastiquement le temps d'intervention des forces de l'ordre. L'automatisation par l'IA offre une vigilance 24h/24, capable d'analyser simultanément plusieurs flux avec une précision constante.

### Nécessité d'une solution innovante
Les modèles standards ont des difficultés dans des environnements complexes. Notre solution hybride combine la rapidité des CNN et la capacité d'analyse contextuelle des Transformers.

---

## 2. Solution Proposée
Notre solution **SALMAS** intègre :
- **Modèle Hybride YOLO+Swin** : Fusion de YOLOv11 et Swin Transformer.
- **Inférence Intelligente (SAHI)** : Pour la détection de petits objets.
- **Analyse Contextuelle** : Calcul de proximité mains-armes (GIoU).
- **Système d'Alerte Multimodal** : Telegram et alertes sonores.
- **Tableau de Bord React** : Visualisation temps réel et explicabilité (Grad-CAM).

---

## 3. Dataset (Base de Données)
### Composition
- **50 967 images** unifiées.
- Format YOLO avec annotations normalisées.

### Classes
1. **Weapon** (Arme)
2. **Person** (Personne)
3. **Confuser** (Objets du quotidien pour réduire les faux positifs)

### Analyse
36,2 % des objets sont de petite taille. Augmentations synthétiques (pluie, brouillard) appliquées pour la robustesse.

---

## 4. Architecture et Justification
### Architecture Adoptée
1. **Backbone (YOLOv11)** : Extraction locale rapide (P3, P4, P5).
2. **Neck (Swin Transformer)** : Attention globale au niveau P4 pour capturer les relations à longue distance.
3. **Head (Decoupled Detection Head)** : Séparation classification/régression et Focal Loss.

### Justification
Besoin de concilier **vitesse** et **précision contextuelle**. Le Swin Transformer compense les faiblesses des CNN sur la compréhension globale.

---

## 5. Évaluation
### Métriques
- **mAP@.5** et **mAP@.5:.95**
- Précision / Rappel
- Score de Persistance Temporelle

### État Actuel (Placeholder)
*L'entraînement de 50 époques est en cours.* Objectif : **mAP@0.72**. Les premiers tests montrent une convergence stable.

---

## 6. Démonstration
1. **Flux Live** : Détection en temps réel.
2. **XAI** : Cartes Grad-CAM pour la transparence.
3. **Alertes** : Déclenchement "ROUGE" et notification Telegram.
