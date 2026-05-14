# Jalon 2 : Détails de l'Implémentation du Modèle Hybride

Ce document fournit une analyse complète, étape par étape, de la manière dont le Jalon 2 (Développement du Modèle Hybride) a été implémenté dans le **Système de Détection d'Armes en Temps Réel et d'Alerte Rouge Contextuelle**.

L'objectif de ce jalon était de concevoir le réseau de neurones central en reliant l'extraction de caractéristiques CNN de pointe à une cartographie du contexte spatial global et à un calcul de perte robuste.

---

## Étape 1 : Implémentation de l'Extracteur de Caractéristiques YOLO Backbone
**Fichier :** `models/backbones/yolo_backbone.py`

Au lieu de construire un CNN à partir de zéro, nous avons enveloppé le modèle `ultralytics.YOLO` éprouvé (en utilisant par défaut la variante `yolo11m.pt`) pour tirer parti de l'apprentissage par transfert à partir de COCO.

**Actions Clés :**
1. **Wrapper de Module** : Création de la classe `YOLOBackbone` héritant de `torch.nn.Module`.
2. **Itération Manuelle des Couches** : Pour extraire des cartes de caractéristiques intermédiaires de manière compatible avec DataParallel, nous avons implémenté un passage avant couche par couche qui s'arrête prématurément après l'extraction des caractéristiques aux indices `[4, 6, 10]` (pour YOLOv11/v12). Cela permet de capturer avec succès les niveaux spatiaux **P3** (foulée 8), **P4** (foulée 16) et **P5** (foulée 32) sans s'appuyer sur des hooks qui peuvent échouer en configuration multi-GPU.
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
3. **Décodage BBox & NMS (Le "Fix")** : Implémentation du pipeline `decode_predictions()` pour convertir les sorties brutes du modèle en boîtes englobantes localisées.
   - **Décodage DFL** : Implémentation du décodage de la Distribution Focal Loss (DFL) pour transformer les décalages de régression en distances spatiales (`ltrb`).
   - **Transformation de Coordonnées** : Ajout d'une logique pour mapper les distances relatives à la foulée vers des coordonnées normalisées `[x1, y1, x2, y2]`.
   - **NMS** : Intégration de `torchvision.ops.nms` pour supprimer les détections redondantes et garantir une seule détection par arme.

---

## Étape 4 : Assemblage du Détecteur Hybride
**Fichier :** `models/hybrid_model.py`

Les trois sous-modules indépendants nécessitaient une interface unifiée pour le moteur d'inférence et la boucle d'entraînement.

**Actions Clés :**
1. **Assemblage du Graphe** : Création du module `HybridWeaponDetector`. Le passage `forward()` achemine élégamment les données :
   `Image Brute -> YOLOBackbone -> SwinNeck -> DetectionHead -> (cls_logits, bbox_offsets, objectness)`
2. **Gestion de l'État** : Implémentation de `.save()` et `.load()` pour sauvegarder de manière atomique le `state_dict` combiné des trois composants dans un seul fichier `best.pt`.
3. **Prédiction Optimisée** : Finalisation de la méthode `.predict()` avec un prétraitement complet :
   - **Préservation BGR** : Les trames OpenCV sont conservées en ordre BGR tout au long du pipeline — cela correspond au pipeline d'entraînement `YOLODataset` qui utilise également BGR. Aucune conversion d'espace colorimétrique n'est appliquée, garantissant la cohérence entre l'entraînement et l'inférence.
   - **Décodage Intégré au Graphe** : Connexion de la logique de décodage de la tête pour fournir des détections prêtes à l'emploi directement au moteur d'inférence.

---

## Étape 5 : Développement du Pipeline d'Entraînement de la Phase 2
**Fichier :** `notebooks/Phase2_Training.ipynb`

Pour entraîner cette architecture hautement personnalisée, nous avons construit un pipeline d'entraînement PyTorch sur mesure qui relie le modèle hybride aux formats de données YOLO standard.

**Actions Clés :**
1. **Chargement des Données Réelles** : Intégration de `ultralytics.data.dataset.YOLODataset` pour gérer l'ensemble de données de 50 000 images, incluant les augmentations mosaic et mixup.
2. **Correspondance des Cibles** : Implémentation d'une logique de correspondance spatiale au sein de la `DetectionHead` pour mapper les boîtes englobantes de vérité terrain aux ancres multi-échelles.
3. **Optimisation de la Perte CIoU** : Implémentation de la perte **Complete IoU (CIoU)** pour la régression des boîtes. Correction de la logique de calcul du centre (`cx_new = cx_anchor + (r - l) / 2`) pour gérer les limites asymétriques des objets dans les cellules de la grille.
4. **Stratégie d'Entraînement (v7 Optimisée)** :
   - **Phase 1 (Époques 1–10)** : Backbone gelé ; seuls le Cou et la Tête sont entraînés à `lr=1e-4` avec un planificateur Cosine Annealing.
   - **Phase 2 (Époques 10+)** : Réglage fin du modèle complet avec des taux d'apprentissage différentiels (`1e-5` pour le backbone, `5e-5` pour la tête/le cou).
   - **Persistance** : Points de contrôle automatiques dans `models/weights/` à chaque époque pendant l'exécution de 50 époques.
