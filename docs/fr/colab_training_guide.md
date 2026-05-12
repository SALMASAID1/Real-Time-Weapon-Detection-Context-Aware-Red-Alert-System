# Entraînement sur Google Colab : Un Guide Étape par Étape

Ce guide détaille comment transférer le **Système de Détection d'Armes en Temps Réel et d'Alerte Rouge Contextuelle** vers Google Colab pour un entraînement accéléré utilisant les GPU du cloud.

> [!TIP]
> **Nouveau** : Nous recommandons désormais d'utiliser **DagsHub Storage** pour un meilleur versionnage des données. Consultez le [Guide de Migration DagsHub](dagshub_colab_setup.md) pour le dernier flux de travail.

## 1. Prérequis
- Un compte Google.
- Les fichiers du projet téléchargés sur Google Drive ou un dépôt GitHub.
- Recommandé : Google Colab Pro pour des sessions plus longues (facultatif).

## 2. Configuration de l'Environnement

### 2.1. Ouvrir Google Colab
Allez sur [colab.research.google.com](https://colab.research.google.com) et créez un nouveau notebook.

### 2.2. Sélectionner l'Exécution GPU
1. Cliquez sur **Exécution** dans le menu supérieur.
2. Sélectionnez **Modifier le type d'exécution**.
3. Choisissez **T4 GPU** (ou A100 si disponible) dans le menu déroulant de l'accélérateur matériel.

### 2.3. Monter Google Drive (Recommandé)
Cela vous permet de conserver votre ensemble de données et les poids du modèle.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.4. Cloner le Dépôt ou Télécharger les Fichiers
Si votre code est sur GitHub :
```bash
!git clone https://github.com/votre-utilisateur/Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System.git
%cd Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System
```
Sinon, téléchargez le dossier de votre projet sur Drive et accédez-y :
```python
%cd /content/drive/MyDrive/Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System
```

## 3. Installation

Installez toutes les dépendances requises.
```bash
!pip install -r requirements.txt
```

## 4. Préparation des Données

### 4.1. Configurer les Répertoires de Données
Assurez-vous que la structure des répertoires est prête :
```bash
!python scripts/data/build_unified_dataset.py
```
*(Remarque : Si vous n'avez pas encore téléchargé les données brutes, exécutez d'abord le script de récupération)*
```bash
!python scripts/data/fetch_external_data.py
```

### 4.2. Vérifier l'Ensemble de Données
Vérifiez si `data/processed/yolo_dataset/data.yaml` existe et pointe vers les bons chemins relatifs.

## 5. Exécution de l'Entraînement

Vous pouvez lancer l'entraînement directement à l'aide du notebook fourni ou via une commande CLI si vous convertissez le notebook en script.

### 5.1. Exécution du Notebook
Ouvrez `notebooks/Phase2_Training.ipynb` dans Colab. Vous devrez peut-être ajuster les chemins :
- Mettez à jour le chemin `data_yaml` dans la cellule d'exécution.
- Assurez-vous que le répertoire `models` existe pour sauvegarder les poids.

### 5.2. Ajustement du Script d'Entraînement
Assurez-vous que votre `HybridTrainer` est configuré pour Colab :
```python
# Dans Phase2_Training.ipynb
trainer = HybridTrainer(model, train_loader, val_loader, device="cuda")
trainer.run(epochs=50)
```

## 6. Surveillance

### 6.1. TensorBoard
Vous pouvez surveiller la progression de l'entraînement en temps réel dans Colab :
```python
%load_ext tensorboard
%tensorboard --logdir runs/train
```

### 6.2. Weights & Biases (Facultatif)
Si vous avez un compte WandB, il suivra automatiquement l'entraînement du backbone Ultralytics si vous vous connectez :
```bash
!pip install wandb
import wandb
wandb.login()
```

## 7. Sauvegarde et Exportation
Après l'entraînement, vos meilleurs poids seront dans `models/weights/best.pt`.
- **Téléchargement** : Vous pouvez le télécharger directement depuis l'explorateur de fichiers Colab.
- **Drive** : Si vous avez monté Drive, il y est déjà sauvegardé !

---
**Conseil** : Si vous rencontrez des erreurs "Out of Memory" (OOM), réduisez la `batch_size` dans l'appel de la fonction `get_dataloaders` (par exemple, passez de 16 à 8 ou 4).
