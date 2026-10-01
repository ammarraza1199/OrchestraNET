# Table 5: Leave-One-Out Component Ablation on COCO val2017 (5,000 Images)

| Configuration | mAP@50:95 | mAP@50 | AP_occ | FPS | Δ mAP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full OrchestraNet** | **48.83** | **65.00** | **30.6** | **32.4** | — |
| **−M2 (Occlusion Analyzer)** | 48.56 | 64.74 | 24.1 | 35.8 | −0.27 |
| **−M6 (Amodal Completer)** | 48.64 | 64.83 | 25.8 | 44.7 | −0.19 |
| **−M4 (Depth Estimator)** | 48.71 | 64.91 | 26.4 | 34.2 | −0.12 |
| **−M3 (Small Enhancer)** | 48.73 | 64.89 | 29.8 | 35.1 | −0.10 |
| **−M7 (Calibrator)** | 48.76 | 64.94 | 28.9 | 32.6 | −0.07 |
| **−M5 (Scene Context)** | 48.79 | 64.96 | 29.5 | 33.0 | −0.05 |
| **M1 only (Baseline)** | 48.42 | 65.08 | 22.3 | 138.5 | −0.41 |

> **Ablation Findings:** The methodology depends most heavily on M2 (−0.27 mAP@50:95 and −6.5 AP on severely occluded objects when removed), emphasising explicit occlusion analysis as the major driver of robustness. M6 is ranked second (−0.19 mAP@50:95, −4.8 AP on severely occluded objects) with amodal shape completion, followed by M4 (−0.12 mAP@50:95, −4.2 AP). Compared with the M1-only baseline (a standard single-head detector on the same backbone), the full system gains +0.41 mAP@50:95 overall but +8.3 AP on severely occluded objects (30.6 vs. 22.3), at the cost of throughput (32.4 vs. 138.5 FPS).
