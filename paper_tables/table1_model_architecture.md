# Table 1: Architectural Parameters

| Component / Module | Specialization & Responsibility | Parameters (M) |
| :--- | :--- | :---: |
| **M1 (Primary YOLOv8s)** | Primary General Detector (Real-Time Anchor-Free Head) | 11.17M |
| **M2 (Occlusion Analyzer)** | Pixel-Level Occlusion & Visibility Estimation (U-Net Lite) | 0.17M |
| **M3 (Small Object Enhancer)** | Super-Resolution Feature Sub-Pixel Upsampling | 0.15M |
| **M4 (Depth Estimator)** | Monocular Pseudo-Depth Transformer Ordering | 0.07M |
| **M5 (Context GNN)** | Spatial Graph Reasoning Across Inter-Object Relations | 0.12M |
| **M6 (Scale Equivariance)** | Multi-Scale Scale-Equivariant Invariance Layer | 0.67M |
| **M7 (Confidence Calibrator)** | Bayesian Residual Confidence Recalibration | 0.04M |
| **Shared Backbone (ResNet-50)** | Multi-Scale Convolutional Feature Extractor | 23.51M |
| **Feature Pyramid (FPN)** | Top-Down Lateral Semantic Feature Aggregation | 1.20M |
| **Adaptive Router** | Lightweight Dynamic Gating Network ($<0.5\text{ ms}$) | 0.06M |
| **Total Parameter Suite** | Complete Multi-Agent Ensemble Capacity | **21.30M** |
