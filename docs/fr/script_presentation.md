# Script de Présentation : Évaluation et Système d'Alerte (SALMAS)

Ce script est conçu pour accompagner les diapositives du fichier `presentation_evaluation.tex`. Les durées sont indicatives pour une présentation de 5 à 7 minutes.

---

## Diapositive 1 : Introduction (30s)

*"Bonjour à tous. Aujourd'hui, je vais vous présenter la stratégie d'évaluation et le fonctionnement du système de 'Red Alert' de notre projet SALMAS. Nous allons voir comment nous passons d'un modèle mathématique brut à un système d'alerte intelligent capable de comprendre le contexte d'une menace."*

---

## Diapositive 2 : Pourquoi ces métriques ? (1m)

*"Avant d'entrer dans les chiffres, il est crucial de justifier nos choix. Dans un système de sécurité, une simple précision globale ne suffit pas. 
D'abord, le contexte : une arme posée sur une table n'est pas la même menace qu'une arme dans une main. 
Ensuite, la taille : nos cibles sont souvent de petits objets (AP_small). 
Enfin, le coût de l'erreur : une fausse alerte peut causer une panique inutile. C'est pourquoi nous avons structuré notre évaluation en trois niveaux, de l'entraînement pur jusqu'à l'intelligence en temps réel."*

---

## Diapositive 3 : Niveau 1 — Convergence (Focal Loss) (1m)

*"Le premier niveau concerne la convergence pendant l'entraînement. Ici, nous surveillons la perte (Loss). 
Le point clé est l'utilisation de la **Focal Loss**. Pourquoi ? Parce que notre jeu de données est déséquilibré : il y a énormément de zones vides par rapport au nombre d'armes. 
La Focal Loss force le modèle à ignorer les exemples trop faciles et à se concentrer sur les cas ambigus. Cela nous permet d'obtenir un 'best checkpoint' beaucoup plus robuste dès la phase de sélection."*

---

## Diapositive 4 : Niveau 2 — Qualité de Détection Offline (1m)

*"Une fois le modèle entraîné, nous passons à l'évaluation 'offline'. Nous utilisons le **mAP** (mean Average Precision) pour évaluer l'équilibre entre précision et rappel sur plusieurs niveaux de chevauchement IoU.
Le **F1-Score** nous aide à trouver le seuil de confiance idéal. 
Mais surtout, nous surveillons l'**AP_small**, car dans la vidéosurveillance, les armes sont souvent distantes et ne représentent que quelques dizaines de pixels."*

---

## Diapositive 5 : Niveau 3 — Système Temps Réel (Pipeline) (1m)
*"Le niveau 3 est le plus innovant : c'est l'intégration en temps réel. 
Notre système n'est pas juste un détecteur d'objets. Il utilise **SAHI** (Slicing Aided Hyper Inference) pour ne rater aucune arme minuscule. 
Parallèlement, nous faisons tourner un modèle léger pour détecter les **mains**. L'idée est simple : une arme devient une alerte critique uniquement si elle est en interaction spatiale avec une personne."*

---

## Diapositive 6 : Le Score de Menace Composite (1m30s)
*"Voici le cœur de notre logique d'alerte : le score composite $S$. Il repose sur trois piliers :
1. **La Confiance (30%)** : C'est la certitude du modèle que l'objet est bien une arme.
2. **Le GIoU (50%)** : C'est le poids le plus lourd. Il mesure la proximité entre l'arme et une main détectée. Si l'arme est brandie, ce score explose.
3. **La Persistance (20%)** : $N$ représente le nombre d'images consécutives où l'objet est vu. Cela nous permet de filtrer les bruits de détection passagers qui durent moins d'une fraction de seconde."*

---

## Diapositive 7 : Conclusion et Seuils (30s)
*"Pour conclure, nous avons défini deux seuils :
- Au-dessus de **0.40**, nous avons une alerte 'Basse' (notification).
- Au-dessus de **0.70**, c'est la **RED ALERT**. Le système déclenche automatiquement une explication Grad-CAM pour montrer à l'opérateur exactement ce qui a causé l'alerte. 
Grâce à cette hiérarchie de métriques, SALMAS est un système non seulement précis, mais surtout intelligent et contextuel. Merci pour votre attention."*
