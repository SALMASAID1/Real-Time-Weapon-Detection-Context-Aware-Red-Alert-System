# Guide d'Entraînement sur Kaggle avec Double T4 (Données Pré-traitées)

Ce guide explique comment entraîner votre modèle sur Kaggle en utilisant votre **ensemble de données déjà traité**.

> [!IMPORTANT]
> **N'exécutez pas `build_unified_dataset.py` sur Kaggle.** Ce guide suppose que vous avez déjà traité vos 50 000 images localement et que vous les téléchargez sous la forme d'un ensemble de données prêt à l'emploi.

## 1. Téléchargement de vos Données Traitées
1. **Préparer le Dossier** : Localisez votre dossier local `data/processed/yolo_dataset`.
2. **Créer l'Ensemble de Données Kaggle** :
   - Allez sur Kaggle -> **Datasets** -> **+ New Dataset**.
   - Nommez-le exactement : `WD-data`.
   - Téléchargez l'**intégralité du contenu** de votre dossier `yolo_dataset` (incluant `train/`, `val/`, `test/` et `data.yaml`).
   - Créez l'ensemble de données.

## 2. Téléchargement de votre Code Source
1. **Compresser le Code** : Compressez ces dossiers/fichiers depuis la racine de votre projet :
   - `models/` (Requis : contient l'architecture)
   - `scripts/` (Contient la configuration)
   - `src/` (Contient la logique d'assistance)
   - `requirements.txt`
2. **Télécharger** : Vous pouvez le télécharger en tant qu'un autre ensemble de données Kaggle (par exemple, `Weapon-Detection-Source`) afin de pouvoir l'attacher facilement à votre notebook.

---

## 3. Configuration du Kernel
1. **Accélérateur** : Sélectionnez **GPU T4 x2**.
2. **Internet** : Activez l'option **ON** (nécessaire pour le téléchargement initial des poids de YOLOv11).

## 4. Exécution (Le Notebook v3)
Utilisez **`Phase2_Training_v3.ipynb`**. Il inclut un **Correcteur de Chemin (Path Fixer)** qui aligne automatiquement vos chemins `data.yaml` locaux avec la structure `/kaggle/input/WD-data/` de Kaggle.

- **Taille du Batch (Batch Size)** : Définie sur `32`.
- **Nombre de Workers (Num Workers)** : Défini sur `2`.
- **Pondération des Classes** : `Confuser` (Index 2) est automatiquement défini sur un **poids de 2,5x** pour réduire les Faux Positifs.

## 5. Récupération des Artefacts
Après l'entraînement (ou lorsque la limite de 12h approche) :
- Vos modèles seront dans `/kaggle/working/models/weights/`.
- Téléchargez `best.pt` pour le déploiement.
- Si vous avez besoin de reprendre plus tard, téléchargez `last.pt` et téléchargez-le à nouveau dans le même dossier lors de votre prochaine session.
