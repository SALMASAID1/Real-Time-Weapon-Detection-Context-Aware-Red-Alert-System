# Task 6.1: Dataset Optimization and Imbalance Handling

## Description
This phase focuses on optimizing the dataset after initial synthesis to ensure the model focuses on discriminative features rather than just learning from the dominant class.

## Justification
Équilibrage des classes via FiftyOne : Afin de limiter les faux positifs causés par le déséquilibre entre la classe Arme (23k+) et la classe Confuser (Umbrellas/Phones), nous utilisons FiftyOne pour effectuer un élagage intelligent (Pruning). En calculant l'unicité des échantillons (Uniqueness), nous réduisons la redondance des images d'armes pour forcer le modèle à se concentrer sur les frontières de décision fines entre un couteau et un parapluie.

## Workflow
1. **Load Dataset**: Load the unified `data/processed/yolo_dataset/` into FiftyOne.
2. **Compute Uniqueness**: Use `fiftyone.brain.compute_uniqueness` to identify heavily redundant weapon images.
3. **Pruning**: Remove highly redundant images to balance the ratio between Weapons and Confusers.
4. **Visual QA**: Compute visualization embeddings to manually inspect hard negatives (Confusers) and fix mislabeled instances.
