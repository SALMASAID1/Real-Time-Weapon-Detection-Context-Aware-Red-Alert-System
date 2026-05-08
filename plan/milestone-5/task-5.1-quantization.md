# Task 5.1: Model Quantization & Performance Optimization

## Description
Optimize the Hybrid YOLO-Swin model and the Hand model for real-time edge inference using quantization techniques.

## Details
1. **Precision Reduction**: Convert PyTorch models (`.pt`) to Half-Precision (FP16) or 8-bit Integer (INT8) to reduce latency.
2. **Framework Optimization**:
   - Export models to **TensorRT** for NVIDIA hardware.
   - Export models to **OpenVINO** for Intel hardware (CPU/iGPU).
3. **Benchmarking**: Measure inference latency (ms/frame) and FPS across different precisions to ensure the system maintains ≥ 30 FPS.

## Learning Resources
- [Ultralytics: Model Export and Optimization](https://docs.ultralytics.com/modes/export/)
- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [OpenVINO Toolkit for Edge AI](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- [Quantization in Deep Learning: A Tutorial](https://arxiv.org/abs/2103.13630)
