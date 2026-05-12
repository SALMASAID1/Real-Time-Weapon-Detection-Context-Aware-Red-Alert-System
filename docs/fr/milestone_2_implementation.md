# Jalon 2 : Détails de l'Implémentation du Modèle Hybride

Ce document fournit une analyse complète, étape par étape, de la manière dont le Jalon 2 (Développement du Modèle Hybride) a été implémenté dans le **Système de Détection d'Armes en Temps Réel et d'Alerte Rouge Contextuelle**.

L'objectif de ce jalon était de concevoir le réseau de neurones central en reliant l'extraction de caractéristiques CNN de pointe à une cartographie du contexte spatial global et à un calcul de perte robuste.

---

## Étape 1 : Implémentation de l'Extracteur de Caractéristiques YOLO Backbone
**Fichier :** `models/backbones/yolo_backbone.py`

Au lieu de construire un CNN à partir de zéro, nous avons enveloppé le modèle `ultralytics.YOLO` éprouvé (en utilisant par défaut la variante `yolo11m.pt`) pour tirer parti de l'apprentissage par transfert à partir de COCO.

**Actions Clés :**
1. **Wrapper de Module** : Création de la classe `YOLOBackbone` héritant de `torch.nn.Module`.
2. **Hooks de Passage Avant (Forward Hooks)** : Pour extraire des cartes de caractéristiques intermédiaires sans interrompre le graphe interne de YOLO, des hooks de passage avant PyTorch ont été enregistrés sur les couches `[15, 18, 21]`. Cela permet de capturer avec succès les niveaux spatiaux **P3** (foulée 8), **P4** (foulée 16) et **P5** (foulée 32).
3. **Contrôle de l'Entraînement** : Implémentation des méthodes `.freeze()` (geler) et `.unfreeze()` (dégeler). Le gel du backbone pendant les premières époques d'entraînement garantit que le Cou et la Tête initialisés de manière aléatoire ne corrompent pas les poids CNN pré-entraînés avec des gradients instables.

---

## Étape 2 : Intégration du Cou Swin Transformer (Swin Transformer Neck)
**Fichier :** `models/necks/swin_neck.py`

Les pyramides CNN standard agrègent les caractéristiques uniquement par des convolutions locales et ont des difficultés avec les relations à longue distance (par exemple, faire correspondre une main dans un coin à une arme ailleurs). Nous avons introduit un Cou Swin Transformer pour résoudre ce problème.

**Actions Clés :**
1. **Blocs Swin** : Utilisation de la bibliothèque `timm` pour instancier des `SwinTransformerBlock`. Ces blocs appliquent un mécanisme d'auto-attention à fenêtre glissante avec une `window_size=7` sur le niveau de caractéristique P4.
2. **Projections de Canaux** : Construction de couches `nn.Conv2d` pour projeter les profondeurs de canaux YOLO P4 brutes dans une dimension d'intégration Swin cohérente de `512` avant le traitement, et inversement après.
3. **Fusion FPN** : Implémentation des connexions latérales du Feature Pyramid Network (FPN). Le contexte P4 enrichi globalement est fusionné :
   - Vers le haut vers P3 via une interpolation bilinéaire.
   - Vers le bas vers P5 via un pooling adaptatif maximum.

---

## Étape 3 : Conception de la Tête de Détection Découplée (Decoupled Detection Head)
**Fichier :** `models/heads/detection_head.py`

Les ensembles de données de détection d'armes souffrent d'un déséquilibre massif des classes de fond (objets "confuseurs" faciles). Nous avons implémenté une tête découplée avec la Focal Loss pour remédier à cela.

**Actions Clés :**
1. **Architecture Découplée** : Création de trois branches convolutionnelles distinctes (Classification, Régression de Boîte, Objectivité) pour chaque niveau de caractéristique (P3, P4, P5), en suivant le modèle de conception de YOLOX.
2. **Intégration de la Focal Loss** : Écriture d'une fonction personnalisée `focal_loss()` utilisant `binary_cross_entropy_with_logits` de PyTorch.
   - **Gamma (`γ=2,0`)** : Agit comme un exposant de focalisation qui réduit agressivement la pondération de la perte provenant des zones de fond faciles à classifier, forçant le réseau à se concentrer sur les échantillons difficiles (par exemple, les armes partiellement masquées).
   - **Alpha (`α=0,25`)** : Agit comme facteur d'équilibrage pour les fréquences de classes.

---

## Étape 4 : Assemblage du Détecteur Hybride
**Fichier :** `models/hybrid_model.py`

Les trois sous-modules indépendants nécessitaient une interface unifiée pour le moteur d'inférence et la boucle d'entraînement.

**Actions Clés :**
1. **Assemblage du Graphe** : Création du module `HybridWeaponDetector`. Le passage `forward()` achemine élégamment les données :
   `Image Brute -> YOLOBackbone -> SwinNeck -> DetectionHead -> (cls_logits, bbox_offsets, objectness)`
2. **Gestion de l'État** : Implémentation de `.save()` et `.load()` pour sauvegarder de manière atomique le `state_dict` combiné des trois composants dans un seul fichier `best.pt`.
3. **Boucle d'Inférence** : Mise en place d'un bouchon (stub) pour `.predict()` afin de gérer les tableaux numpy entrants ou les tuiles SAHI lors du déploiement actif.

---

## Étape 5 : Développement du Pipeline d'Entraînement de la Phase 2
**Fichier :** `notebooks/Phase2_Training.ipynb`

Pour entraîner cette architecture hautement personnalisée, nous avons construit un pipeline d'entraînement PyTorch sur mesure qui relie le modèle hybride aux formats de données YOLO standard.

**Actions Clés :**
1. **Chargement des Données Réelles** : Intégration de `ultralytics.data.dataset.YOLODataset` pour gérer l'ensemble de données de 50 000 images, incluant les augmentations mosaic et mixup.
2. **Correspondance des Cibles (Le "Fix")** : Implémentation d'une logique de correspondance spatiale au sein de la `DetectionHead` pour mapper les boîtes englobantes de vérité terrain (ground truth) aux ancres multi-échelles du modèle (P3, P4, P5).
3. **Perte Composite** : Connexion de la Focal Loss de classification et de la perte d'Objectivité dans une seule méthode `.compute_loss()`.
4. **Stratégie d'Entraînement** :
   - **Phase 1 (Époques 1–10)** : Backbone gelé ; seuls le Cou et la Tête sont entraînés à `lr=1e-4` pour stabiliser le contexte global.
   - **Phase 2 (Époques 10+)** : Réglage fin du modèle complet à `lr=1e-5` pour affiner les caractéristiques des armes à bas niveau.
   - **Persistance** : Points de contrôle automatiques dans `models/weights/` toutes les 5 époques.
