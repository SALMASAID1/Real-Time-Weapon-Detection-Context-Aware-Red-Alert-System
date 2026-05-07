# Task 1.3: Hard Negative Mining & Synthetic Injection

## Description
Integrate "confuser" objects and synthetic data to minimize False Positive Rates (FPR) and handle adverse weather conditions.

## Details
1. **Hard Negative Mining**: Integrate 5,000+ images of objects often mistaken for weapons (umbrellas, drills, wallets, smartphones).
2. **Synthetic Injection**: Use GANs or Simuletic data to simulate:
   - Adverse weather (rain, fog).
   - Low-light and night-time surveillance conditions.
3. **Validation**: Ensure the model is trained to explicitly ignore these confuser objects.

## Learning Resources
- [What is Hard Negative Mining?](https://towardsdatascience.com/hard-negative-mining-for-object-detection-7d4d7a46979a)
- [Synthetic Data for Computer Vision](https://blog.roboflow.com/synthetic-data-computer-vision/)
- [Handling Adverse Weather in Object Detection](https://arxiv.org/abs/2103.02414)
