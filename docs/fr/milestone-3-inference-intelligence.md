# Jalon 3 : Intelligence d'Inférence et Logique de Menace

## Aperçu
Le Jalon 3 s'est concentré sur la couche "Intelligence" qui se situe au-dessus des détections brutes du modèle. Elle garantit que le système peut gérer des séquences 4K et distinguer une simple détection d'une véritable menace "Alerte Rouge".

## Composants Clés

### 1. Pipeline SAHI (Slicing Aided Hyper Inference)
Les modèles YOLO standard échouent sur les petits objets dans les images 4K car les objets rétrécissent trop lors du redimensionnement de l'image.
*   **La Logique** : Nous partitionnons l'image 4K en tuiles de 640x640 se chevauchant.
*   **Avantages** : Préserve les détails en haute résolution, permettant au modèle de détecter de petits couteaux ou pistolets même lorsque le sujet est loin de la caméra.
*   **Fusion** : Utilise le NMS (Non-Maximum Suppression) pour unifier les détections à travers les tuiles qui se chevauchent.

### 2. Score de Menace Sensible au Contexte
Nous utilisons une formule de score composite ($S$) pour déterminer le niveau de risque d'une détection :
$$S = (0,3 \cdot \text{Confiance}) + (0,5 \cdot \text{Proximité}) + (0,2 \cdot \text{Persistance})$$

#### **A. Proximité (GIoU)**
Nous calculons l'**Intersection over Union Généralisée (GIoU)** entre les boîtes d'Armes et les boîtes de Mains.
*   **GIoU > 0** : L'arme est physiquement dans une main (Menace Élevée).
*   **Proximité Immédiate** : Le GIoU fournit un score même si les boîtes sont très proches mais ne se touchent pas encore, permettant une alerte précoce.

#### **B. Persistance Temporelle**
Une détection doit persister sur plusieurs images (10 par défaut) pour atteindre un niveau de menace "ÉLEVÉ". Cela filtre efficacement les faux positifs "clignotants" du flux d'alerte final.

### 3. XAI (IA Explicable avec Grad-CAM)
Lorsqu'une menace "ÉLEVÉE" est déclenchée, le système génère une carte de chaleur Grad-CAM.
*   **Objectif** : Montre quelles caractéristiques visuelles (par exemple, la détente, le canon) ont provoqué le déclenchement du modèle.
*   **Avantage** : Renforce la confiance des opérateurs humains et permet une vérification manuelle des alertes.

## Scripts du Pipeline
*   `src/inference/sahi_pipeline.py` : La logique de découpage en tuiles et de fusion.
*   `src/threat_logic/iou_calculator.py` : Le calcul mathématique de proximité GIoU.
*   `src/threat_logic/threat_scorer.py` : Le moteur de prise de décision finale.
*   `src/xai/gradcam.py` : Génération de cartes de chaleur.
