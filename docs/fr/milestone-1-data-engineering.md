# Jalon 1 : Ingénierie des Données et Augmentation Synthétique

## Aperçu
L'objectif du Jalon 1 était de construire un ensemble de données robuste, équilibré et de haute qualité d'environ 50 000 images adaptées à la détection d'armes en temps réel dans diverses conditions de surveillance.

## Composants Clés

### 1. Agrégation de Données et Mappage des Classes
Nous avons consolidé des images provenant de six sources disparates dans un ensemble de données unifié au format YOLO.
*   **Ontologie Finale des Classes** :
    *   `0: Weapon` (Arme) : Pistolets, Fusils, Couteaux.
    *   `1: Person` (Personne) : Échantillons humains de haute qualité provenant de COCO.
    *   `2: Confuser` (Confondeur) : Objets souvent confondus avec des armes (Parapluies, Perceuses, Téléphones).
*   **Équilibre** : Nous avons atteint un ratio proche de 1:1 entre les Armes (~40k au total) et les Personnes (~37k au total) pour garantir une détection sensible au contexte.

### 2. Recherche de Négatifs Difficiles (Hard Negative Mining)
Pour minimiser les taux de faux positifs (FPR), nous avons intégré plus de 16 000 objets "Confuseurs". Ceux-ci apprennent au modèle à distinguer une arme létale d'un objet courant de la vie quotidienne comme un smartphone ou un outil électrique.

### 3. Injection Synthétique (Météo et Faible Luminosité)
En utilisant la bibliothèque `albumentations`, nous avons "injecté" des conditions défavorables simulées dans une partie de l'ensemble de données :
*   **Pluie et Brouillard** : Simulations basées sur la physique pour apprendre au modèle à voir à travers le bruit environnemental.
*   **Faible Luminosité/Nuit** : Simulation du bruit des capteurs de vidéosurveillance et des chutes de luminosité pour une fiabilité 24h/24 et 7j/7.
*   **Nommage** : Toutes les images synthétiques sont préfixées par `syn_` pour un suivi facile.

## Résumé des Statistiques
| Répartition | Total d'Images | Boîtes d'Armes | Boîtes de Personnes | Boîtes de Confuseurs |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 41 444 | 32 737 | 30 061 | 13 359 |
| **Test** | 4 764 | 3 726 | 3 471 | 1 691 |
| **Val** | 4 759 | 3 684 | 3 372 | 1 564 |
| **Total** | **50 967** | **40 150** | **36 904** | **16 614** |

## Scripts du Pipeline
*   `scripts/data/fetch_external_data.py` : Récupère les données de FiftyOne Zoo.
*   `scripts/data/build_unified_dataset.py` : Normalise et fusionne toutes les sources.
*   `scripts/data/augment_synthetic.py` : Génère des échantillons météo/faible luminosité.
*   `scripts/setup_data.sh` : Exécution en un clic de l'ensemble du pipeline.
