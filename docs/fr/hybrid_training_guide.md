# Guide d'Entraînement du Modèle Hybride

Ce document fournit un flux de travail étape par étape pour l'entraînement du système de détection d'armes en temps réel dans différents environnements (Local, Lightning AI Studio et Kaggle).

## 1. Préparation de l'Environnement

### Local / Lightning Studio
1. **Cloner le Projet** :
   ```bash
   git clone <votre-url-de-depot>
   cd Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System
   ```
2. **Installer les Dépendances** :
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

### Kaggle
1. **Télécharger les Sources** : Compressez les dossiers `models/`, `scripts/` et `utils/` en un fichier ZIP et téléchargez-les en tant qu'ensemble de données Kaggle ou script utilitaire.
2. **Paramètres GPU** : Allez dans **Settings -> Accelerator** et sélectionnez **GPU T4 x2**.
3. **Internet** : Assurez-vous que l'option **Internet on** est activée dans la barre latérale.

---

## 2. Préparation des Données
Avant l'entraînement, vous devez avoir l'ensemble de données unifié prêt.

1. **Construire l'Ensemble de Données** :
   Exécutez la commande suivante pour fusionner les données multi-sources (OI v7, COCO, Simuletic) dans le format YOLO propre à 3 classes :
   ```bash
   python scripts/data/build_unified_dataset.py
   ```
2. **Vérifier les Chemins** :
   Assurez-vous que `data/processed/yolo_dataset/data.yaml` existe et pointe vers des chemins d'image absolus.

---

## 3. Flux de Travail de l'Entraînement

### Choisir le Bon Notebook
| Environnement | Notebook à Utiliser | Fonctionnalités |
| :--- | :--- | :--- |
| **Local / Studio** | `Phase2_Training_v2.ipynb` | GPU unique, Reprise automatique, Sauvegarde périodique. |
| **Kaggle** | `Phase2_Training_v3.ipynb` | **Double GPU T4**, AMP (Précision Mixte), Pondération des classes. |

### Étapes d'Exécution
1. **Initialiser le Modèle** : Le script téléchargera automatiquement le backbone `yolo11m.pt`.
2. **Phase d'Échauffement (Époques 1-10)** :
   *   Le backbone YOLO est **gelé**.
   *   Seuls le cou Swin Transformer et la tête focale s'entraînent.
   *   Taux d'Apprentissage (Learning Rate) : `1e-4`.
3. **Phase de Réglage Fin (Époques 11-50)** :
   *   Le backbone est **dégelé**.
   *   L'ensemble de l'architecture de bout en bout est optimisé.
   *   Taux d'Apprentissage : `1e-5` (plus bas pour préserver les caractéristiques pré-entraînées).

---

## 4. Surveillance et Artefacts

- **Courbes de Perte** : Surveillez la perte d'entraînement/validation dans la sortie du notebook.
- **Poids** :
  - `best.pt` : Sauvegardé automatiquement lorsque la perte de validation s'améliore.
  - `last.pt` : Sauvegardé toutes les 2 à 5 époques pour la reprise automatique.
- **Équilibre des Classes** : Dans la v3, nous appliquons une **pénalité de 2,5x** aux erreurs "Confuseur" pour garantir que le système "Alerte Rouge" ait un taux de faux positifs très bas.

---

## 5. Dépannage

### "CUDA Out of Memory"
- Réduisez la `batch_size` dans la fonction `get_dataloaders` (par exemple, de 32 à 16).
- Assurez-vous qu' `autocast()` est actif dans la boucle d'entraînement.

### "ModuleNotFoundError: models"
- Assurez-vous d'avoir exécuté `pip install -e .` ou que la racine du projet est ajoutée à `sys.path`.

### "FileNotFoundError: data.yaml"
- Vérifiez la variable de chemin `DATA_YAML` dans la cellule "Exécution" du notebook. Sur Kaggle, cela doit être `/kaggle/input/WD-data/data.yaml`.
