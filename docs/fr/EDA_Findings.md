# Résultats de l'Analyse Exploratoire des Données (EDA) : Une Étude Approfondie de la Géométrie de Détection des Armes

Ce document résume les conclusions techniques de l'analyse exploratoire des données (EDA) effectuée sur l'ensemble de données unifié de 50 000 détections d'armes. Ces informations orientent directement les décisions architecturales du **Système d'Alerte Rouge Contextuel et de Détection d'Armes en Temps Réel**.

---

## 1. Aperçu de l'Ensemble de Données
- **Total d'Images** : 4 764 (Échantillon de test)
- **Classes** : `Weapon` (Arme) (3 727), `Person` (Personne) (3 471), `Confuser` (Confondeur) (1 691)
- **Objectif** : Identifier les armes létales tout en minimisant les faux positifs provenant des "Confuseurs" (téléphones portables, portefeuilles, outils).

---

## 2. Conclusion Clé : Le Goulot d'Étranglement de la Détection de Petits Objets
### Les Données
Notre analyse a révélé que **36,2 %** de toutes les boîtes englobantes occupent moins de **1 %** de la surface totale de l'image. Ces "micro-objets" représentent des armes tenues à distance ou partiellement masquées.

### Le Défi
Les réseaux de neurones convolutionnels (CNN) standard ont souvent des difficultés avec les petits objets car l'échantillonnage descendant répété (pooling/stride) réduit considérablement la résolution spatiale des caractéristiques fines. Au moment où la carte des caractéristiques atteint la tête de détection, un pistolet de 20x20 pixels pourrait être réduit à un seul pixel.

### La Solution Architecturale : Cou Swin Transformer
Pour remédier à cela, nous utilisons un **cou basé sur Swin Transformer** (remplaçant ou augmentant le YOLO FPN/PAN standard).
- **Représentation Hiérarchique** : Contrairement aux Transformers standard, Swin construit des cartes de caractéristiques hiérarchiques, permettant au modèle de traiter l'information à différentes échelles.
- **Attention par Fenêtre Glissante (Shifted Window Attention)** : Ce mécanisme permet la modélisation des dépendances à longue portée tout en maintenant l'efficacité computationnelle (complexité $O(hw)$), garantissant que même les indices spatiaux infimes (par exemple, le pontet d'un pistolet) sont capturés à travers l'image.

---

## 3. Conclusion Clé : Déséquilibre des Classes et le Défi du "Confuseur"
### Les Données
La classe `Confuser` est considérablement sous-représentée, apparaissant à seulement **~50 %** de la fréquence des armes ou des personnes.

### Le Défi
Dans la détection d'objets, le ratio entre le "fond" (négatifs faciles) et les "objets" (échantillons positifs) est extrême. Les négatifs faciles dominent la fonction de perte pendant l'entraînement, rendant le modèle "paresseux" et l'empêchant d'apprendre les différences subtiles entre un smartphone noir et une arme de poing noire.

### La Solution : Focal Loss
Nous implémentons la **Focal Loss** pour remédier à ce déséquilibre.
- **Facteur de Modulation** : La Focal Loss ajoute un facteur $(1 - p_t)^\gamma$ à l'entropie croisée standard.
- **Exploitation des Négatifs Difficiles (Hard Negative Mining)** : Cela "sous-pondère" mathématiquement la perte provenant des échantillons de fond faciles à classifier et force le modèle à se concentrer sur les exemples "difficiles" — spécifiquement la classe `Confuser` qui nécessite une discrimination de haute précision.

---

## 4. Conclusion Clé : Distribution Spatiale et Contexte de Surveillance
### Les Données
L'analyse par carte de chaleur montre un fort **biais central** dans la localisation des armes. La plupart des images sources (extraites d'Internet) présentent l'arme comme sujet principal au centre du cadre.

### L'Implication pour la Surveillance
Les images de vidéosurveillance réelles placent rarement la menace en plein centre. Les menaces apparaissent généralement à la périphérie, dans les coins, ou se déplacent à travers le cadre.

### La Stratégie : Augmentation Mosaic et Translation
Pour combler l'écart entre le biais de l'ensemble de données et le déploiement en conditions réelles, nous utilisons :
- **Augmentation Mosaic** : Combinaison de 4 images d'entraînement en une seule, forçant le modèle à détecter des objets à différentes échelles et dans différents quadrants du cadre.
- **Translation Aléatoire** : Déplacement de l'arme du centre vers les bords pour briser le biais spatial.

---

## 5. Références Techniques et Sources

### Architecture et Transformers
1.  **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Liu et al., 2021). [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
2.  **Documentation YOLOv8/v11** : Informations sur les structures SPPF (Spatial Pyramid Pooling - Fast) et CSP (Cross Stage Partial) pour l'extraction de caractéristiques.

### Fonctions de Perte et Déséquilibre
3.  **Focal Loss for Dense Object Detection** (Lin et al., 2017). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) - L'article fondateur pour la gestion du déséquilibre des classes dans les détecteurs à une seule étape.
4.  **Imbalance Problems in Object Detection: A Review** (Oksuz et al., 2020). [IEEE Xplore](https://ieeexplore.ieee.org/document/8972412)

### Ensemble de Données et Méthodologie
5.  **Documentation FiftyOne** : Meilleures pratiques pour la visualisation d'ensembles de données et l'analyse de la géométrie des BBox. [Voxel51 Docs](https://docs.voxel51.com/)
6.  **Ensemble de données Open Images V7** : Source principale pour les classes `Weapon` et `Person`.
