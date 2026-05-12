# 🚀 Project Roadmap: Real-Time Weapon Detection & Red Alert System

## 🌟 Vision & Objective
The **Real-Time Weapon Detection & Context-Aware Red Alert System** is a next-generation surveillance intelligence platform. It moves beyond simple classification to identify active threats by analyzing the **spatial relationship** between humans and weapons, effectively distinguishing a holstered weapon from an imminent threat.

---

## 🏗️ Technical Architecture (The "Hybrid" Edge)
Our system utilizes a state-of-the-art **Two-Stream Inference Pipeline**:
1.  **Weapon Stream (Hybrid YOLO-Swin):** A custom model combining YOLOv11 speed with Swin Transformer global context to detect Weapons and "Confusers" (cellphones, tools).
2.  **Hand Stream (YOLO-Hand):** A specialized stream for robust hand detection to facilitate proximity analysis.
3.  **Threat Logic:** A Geometric IoU engine that validates if a weapon is physically being handled.

---

## 📊 Success Metrics
| Metric | Target | Description |
| :--- | :--- | :--- |
| **mAP@0.5:0.95** | ≥ 0.72 | Mean Average Precision for weapon localization. |
| **FPR (Confusers)** | < 2.0% | False Positive Rate on non-threat objects. |
| **Inference Speed** | ≥ 40 FPS | Real-time performance on edge hardware (Jetson). |
| **End-to-End Latency** | < 500ms | Detection-to-Notification time. |

---

## 🗺️ Project Roadmap

### Phase 1: Data Engineering & Foundation (Completed ✅)
*   **Unified Dataset Build:** Aggregated 50k+ images (Open Images, SOHAS, Synthetic).
*   **EDA:** Visualized class imbalances and small-object geometry using FiftyOne.
*   **Sanitization:** Standardized multi-source annotations to `[Weapon, Person, Confuser]`.

### Phase 2: Hybrid Model Intelligence (Completed ✅)
*   **Architecture Stitching:** Integration of YOLOv11 backbones with Swin-Transformer Necks.
*   **Custom Loss:** Implementation of Focal Loss + CIoU for class imbalance and localization.
*   **Phase-Aware Training:** Frozen-backbone warmups followed by differential learning rate fine-tuning.
*   **Training Pipeline:** 7 iterations (v1→v7) with DFL decoding, NMS, and visual validation.

### Phase 3: Inference Intelligence & XAI (Completed ✅)
*   **SAHI Integration:** Tiled inference for 4K surveillance stream processing.
*   **Hand-Weapon Proximity:** GIoU-based proximity scoring with temporal persistence.
*   **Explainable AI:** Grad-CAM heatmaps targeting the Swin Neck for model decision visualization.
*   **Overlay Renderer:** INFERNO colormap compositing with JPEG encoding for WebSocket delivery.

### Phase 4: Multimodal Alerting & Dashboard (Completed ✅)
*   **Notification Engine:** Telegram Bot (async, fire-and-forget) and local audio alerts (Pygame).
*   **InferenceEngine:** Dual-cadence loop (SAHI + lightweight), two-stream (Weapon + Hand), queue-based WebSocket broadcasting.
*   **React Dashboard:** Live monitoring (VideoCanvas), threat history (paginated REST), runtime settings (ThresholdPanel).
*   **Alert Deduplication:** Spatial-key cooldown to prevent notification spam.

### Phase 5: Optimization & Deployment (Active ⚡)
*   **Model Training:** Full 50-epoch run with differential LR (`1e-5` backbone / `5e-5` head).
*   **Edge Optimization:** TensorRT/OpenVINO quantization for ≥40 FPS inference.
*   **System Validation:** mAP evaluation, FPR audit, end-to-end latency benchmarking.

### Phase 6: Dataset Optimization (Planned 📋)
*   **FiftyOne Brain:** Compute uniqueness scores to prune redundant weapon images.
*   **Class Rebalancing:** Reduce Weapon:Confuser ratio for sharper decision boundaries.
*   **Visual QA:** Manual inspection of hard negatives and mislabeled instances.

---

## 📚 Learning Resources

### Deep Learning & Computer Vision
*   [Fast.ai Practical Deep Learning](https://course.fast.ai/): Best-in-class PyTorch foundation.
*   [DeepLearning.AI - CNNs](https://www.coursera.org/learn/convolutional-neural-networks): Core theory of YOLO architectures.

### Advanced Research
*   [Swin Transformer (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030): Theory behind our Global Context Neck.
*   [Focal Loss (arXiv:1708.02002)](https://arxiv.org/abs/1708.02002): Mathematical solution for class imbalance.

### Tools & Frameworks
*   [Ultralytics YOLOv11](https://docs.ultralytics.com/): Our primary detection engine.
*   [FiftyOne Docs](https://docs.voxel51.com/): Dataset curation and Visual QA.
*   [Timm Library](https://huggingface.co/docs/timm/index): Source for transformer block implementations.

---
*Created by Antigravity AI Assistant.*
