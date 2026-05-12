# Milestone 1: Data Engineering & Synthetic Augmentation

## Overview
The goal of Milestone 1 was to build a robust, balanced, and high-quality dataset of ~50,000 images tailored for real-time weapon detection in varied surveillance conditions.

## Key Components

### 1. Data Aggregation & Class Mapping
We consolidated images from six disparate sources into a unified YOLO-format dataset.
*   **Final Class Ontology**: 
    *   `0: Weapon`: Handguns, Rifles, Knives.
    *   `1: Person`: High-quality human samples from COCO.
    *   `2: Confuser`: Objects often mistaken for weapons (Umbrellas, Drills, Phones).
*   **Balance**: We achieved a near 1:1 ratio between Weapons (~40k total) and Persons (~37k total) to ensure context-aware detection.

### 2. Hard Negative Mining
To minimize False Positive Rates (FPR), we integrated 16,000+ "Confuser" objects. These teach the model to distinguish between a lethal weapon and a common everyday object like a smartphone or a power tool.

### 3. Synthetic Injection (Weather & Low-Light)
Using the `albumentations` library, we "injected" simulated adverse conditions into a portion of the dataset:
*   **Rain & Fog**: Physics-based simulations to teach the model to see through environmental noise.
*   **Low-Light/Night**: Simulating CCTV sensor noise and brightness drops for 24/7 reliability.
*   **Naming**: All synthetic images are prefixed with `syn_` for easy tracking.

## Statistics Summary
| Split | Total Images | Weapon Boxes | Person Boxes | Confuser Boxes |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 41,444 | 32,737 | 30,061 | 13,359 |
| **Test** | 4,764 | 3,726 | 3,471 | 1,691 |
| **Val** | 4,759 | 3,684 | 3,372 | 1,564 |
| **Total** | **50,967** | **40,150** | **36,904** | **16,614** |

## Pipeline Scripts
*   `scripts/data/fetch_external_data.py`: Fetches data from FiftyOne Zoo.
*   `scripts/data/build_unified_dataset.py`: Normalizes and merges all sources.
*   `scripts/data/augment_synthetic.py`: Generates weather/low-light samples.
*   `scripts/setup_data.sh`: One-click execution of the entire pipeline.
