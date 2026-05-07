# Task 4.1: Hand-Weapon Logic & Proximity Analysis

## Description
Develop a logic layer that understands the relationship between people and weapons, specifically focusing on Hand-Weapon interaction.

## Details
1. **Parallel Detection**: Ensure the model detects "Hands" and "Weapons" as separate classes (or use a secondary model for hands).
2. **Intersection over Union (IoU)**: Calculate the IoU between detected Hand boxes and Weapon boxes.
3. **Contextual Alerting**: Trigger "High Priority" alerts only when the IoU exceeds a specific threshold (indicating a weapon is being held).

## Learning Resources
- [Hand-Object Interaction Detection](https://arxiv.org/abs/2004.03684)
- [Calculating IoU between multiple classes](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [Building Contextual Intelligence in CV](https://towardsdatascience.com/contextual-intelligence-in-computer-vision-2f7d3e0b2e7a)
