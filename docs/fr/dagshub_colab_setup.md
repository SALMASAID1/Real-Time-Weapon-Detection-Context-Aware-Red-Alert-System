# Entraînement sur Google Colab avec DagsHub

Ce guide fournit la configuration complète de l'environnement et le flux de travail pour l'entraînement du **Système de Détection d'Armes en Temps Réel et d'Alerte Rouge Contextuelle** en utilisant Google Colab et **DagsHub Storage**.

---

## 1. Architecture à deux dépôts

Ce projet utilise **deux dépôts DagsHub distincts** avec des objectifs différents :

| Dépôt | Objectif | Contenu |
|-------|----------|---------|
| `dagshub-drive` | **Stockage de l'ensemble de données** | Plus de 41 000 images stockées dans DagsHub S3 à l'adresse `s3:/dagshub-drive/data/processed/yolo_dataset` |
| Code source | **Code du modèle** | Conservé sur Google Drive (chargé dans Colab au moment de l'exécution) |

> **Clé** : Le `DagsHubFilesystem` doit pointer vers `dagshub-drive` — le dépôt de stockage — et NON vers le nom du dépôt de code.
> Ceci est confirmé par l'URL de l'ensemble de données : `dagshub.com/TAMZIRT-MOHAMED/dagshub-drive/src/main/s3:/dagshub-drive/data/processed/yolo_dataset`

- L'ensemble de données doit se trouver à l'emplacement `data/processed/yolo_dataset/` à l'intérieur de `dagshub-drive`.
- Assurez-vous que `data.yaml` utilise des chemins relatifs (par exemple, `train: train/images`).

---

## 2. Configuration initiale de Google Colab

1.  **Ouvrir le Notebook** : Téléchargez et ouvrez `notebooks/Phase2_Training_v5.ipynb` dans [Google Colab](https://colab.research.google.com).
2.  **Accélérateur matériel** :
    *   Allez dans **Exécution** > **Modifier le type d'exécution**.
    *   Sélectionnez **T4 GPU** (Standard sur le niveau gratuit).
3.  **Préparation de l'environnement** :
    *   Exécutez la cellule de l'**Étape 1** du notebook.
    *   Cela installe `dagshub`, `ultralytics` et corrige l'**incompatibilité Numpy 2.0**.
    *   **CRITIQUE** : Si la sortie vous avertit à propos de Numpy 2.x, cliquez sur **"REDÉMARRER LA SESSION"** dans la fenêtre contextuelle et passez à l'étape 2.

---

## 3. Connexion au stockage DagsHub

À l'**Étape 2** du notebook, vous allez vous authentifier et monter les deux systèmes de stockage.

```python
# ── MODÈLE CORRECT (v5) ──────────────────────────────────────────────────
from dagshub.streaming import DagsHubFilesystem
import dagshub
import dagshub.colab

# 1. S'authentifier (ouvre l'OAuth du navigateur une fois ; le jeton est mis en cache)
dagshub.colab.login()

# 2. Instancier DagsHubFilesystem avec une URL HTTPS explicite du dépôt
#    ⚠️  Ne PAS utiliser : dagshub.streaming.install_hooks(repo_url=...) ← défectueux
#    Cette forme essaie de lire .git/HEAD qui n'existe pas dans Colab.
fs = DagsHubFilesystem(
    repo_url="https://dagshub.com/PROPRIÉTAIRE/NOM_DU_DÉPÔT",
    branch="main",
    project_root="/content/dagshub_streaming",
)
fs.install_hooks()   # patche open(), os.listdir(), pathlib, etc.

# 3. Monter Google Drive (code source + poids)
drive.mount('/content/drive')
```

**Ce que cela fait** :
- **Code source** : Chargé depuis `/content/drive/MyDrive/...`. Vos derniers changements sont instantanément disponibles.
- **Ensemble de données** : Diffusé depuis DagsHub. Cela évite les heures de synchronisation que Google Drive requiert habituellement pour plus de 50 000 images.
- **Points de contrôle (Checkpoints)** : Automatiquement sauvegardés dans votre dossier GDrive `models/weights/` pour la persistance.

---

## 4. Vérification de l'environnement

Une fois monté, le notebook vérifiera la structure :
- **CWD (Répertoire de travail)** : Automatiquement déplacé vers la racine du projet.
- **Chemin Python** : La racine du projet est ajoutée à `sys.path` pour que `from models...` fonctionne.
- **Vérification des données** : Utilisez la cellule de diagnostic (si fournie) pour vous assurer que les images sont visibles.

---

## 5. Logique d'entraînement (Config Alerte Rouge)

L'environnement est pré-configuré pour nos objectifs du **Jalon 2** :
- **Optimiseur** : AdamW avec précision mixte (`torch.cuda.amp.autocast`).
- **Matériel** : `batch_size=16` et `num_workers=2`.
- **Intelligence** : Les poids des classes `[1.0, 1.0, 2.5]` sont appliqués à la tête pour donner la priorité à la réduction des faux positifs (classe Confuseur).
- **Stratégie** : Backbone gelé pendant 10 époques suivi d'un réglage fin complet (fine-tuning).

---

## 6. Dépannage des problèmes courants

### ModuleNotFoundError: No module named 'models'
- **Cause** : L'environnement d'exécution a été redémarré et l'étape 2 (Alignement) n'a pas encore été exécutée.
- **Solution** : Ré-exécutez la cellule de l'étape 2 pour rajouter le chemin du projet à la mémoire de Python.

### PytorchStreamReader failed reading zip archive
- **Cause** : Un fichier de poids `.pt` corrompu (souvent `yolo11m.pt`).
- **Solution** : Supprimez le fichier corrompu de votre stockage DagsHub et laissez le code le télécharger à nouveau.

### FileNotFoundError: `/content/dagshub_streaming/.git/HEAD` + UnsupportedProtocol
- **Cause profonde** : L'ancien code appelait `dagshub.streaming.install_hooks(repo_url=repo_url)` où `repo_url` était la valeur de retour de `dagshub.colab.login()`. La fonction de niveau module `install_hooks()` n'accepte **pas** `repo_url` comme argument — elle l'ignore et essaie à la place de détecter automatiquement le dépôt en lisant `.git/HEAD` dans le système de fichiers, qui n'existe pas dans un nouvel environnement Colab. L'URL interne résultante est alors malformée → `UnsupportedProtocol`.
- **Correction** (appliquée dans la v5) : Utilisez toujours `DagsHubFilesystem(repo_url="https://dagshub.com/UTILISATEUR/DÉPÔT", ...).install_hooks()` — instanciez d'abord la classe avec une URL HTTPS explicite, puis appelez `.install_hooks()` sur l'**instance**.

### No images found in .../images
- **Cause** : Les fichiers `labels.cache` provenant d'une autre machine interfèrent.
- **Solution** : Supprimez tous les fichiers `labels.cache` dans les sous-dossiers de `data/processed/yolo_dataset/` pour forcer une nouvelle analyse.

---

## 7. Points de contrôle persistants
Tous les poids sont sauvegardés dans `models/weights/` sur le lecteur DagsHub monté.
- **last.pt** : Sauvegardé à chaque époque (contient l'état de l'optimiseur pour la reprise automatique).
- **best.pt** : Sauvegardé chaque fois que la perte de validation s'améliore.
- **epoch_N.pt** : Points de contrôle permanents sauvegardés toutes les 5 époques.
