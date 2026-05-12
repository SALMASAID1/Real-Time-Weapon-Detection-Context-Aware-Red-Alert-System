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
*   **Unified Dataset Build:** Aggregated 45k+ images (Open Images, SOHAS, Synthetic).
*   **EDA:** Visualized class imbalances and small-object geometry using FiftyOne.
*   **Sanitization:** Standardized multi-source annotations to `[Weapon, Person, Confuser]`.

### Phase 2: Hybrid Model Intelligence (Active ⚡)
*   **Architecture Stitching:** Integration of YOLOv11 backbones with Swin-Transformer Necks.
*   **Custom Loss:** Implementation of Focal Loss to address "Confuser" class under-representation.
*   **Phase-Aware Training:** Utilizing frozen-backbone warmups followed by full model fine-tuning.

### Phase 3: Inference Intelligence & XAI (Upcoming 🚀)
*   **SAHI Integration:** Tiled inference for 4K surveillance stream processing.
*   **Hand-Weapon Proximity:** Proximity-based threat validation logic.
*   **Explainable AI:** Grad-CAM heatmaps to visualize model decision-making.

### Phase 4: Multimodal Alerting & UI (Future 🔮)
*   **Notification Engine:** Telegram Bot and local audio (Pygame) integration.
*   **React Dashboard:** Real-time monitoring UI with live threat history.
*   **Edge Optimization:** TensorRT quantization for high-FPS deployment.

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
