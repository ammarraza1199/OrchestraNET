# 🎼 OrchestraNet: Multi-Model Orchestration Framework for Multi-Class Object Detection

[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: MS--COCO](https://img.shields.io/badge/Benchmark-MS--COCO%20val2017-blue.svg)](https://cocodataset.org/)
[![Interactive Notebook](https://img.shields.io/badge/Jupyter-Interactive%20Walkthrough-orange.svg)](OrchestraNet_Walkthrough_and_Evaluation.ipynb)

> **Official Implementation of:**  
> *OrchestraNet: A Heterogeneous Multi-Model Orchestration Framework with Explicit Occlusion Reasoning and Dynamic Real-Time Compute Scaling.*

---

## 📌 Executive Summary

Traditional single-model object detectors (e.g., YOLO, DETR, Faster R-CNN) enforce a **homogeneous capacity ceiling**: they allocate identical representational capacity and inference latency regardless of whether a frame contains an isolated object or a crowded, heavily occluded street scene. In benchmark datasets like MS-COCO, roughly **55.9% of annotated objects** overlap with at least one other instance, triggering systematic detection failures during non-maximum suppression (NMS) and feature competition.

**OrchestraNet** departs from monolithic scaling by decomposing object detection into **seven specialized micro-models (M1–M7)** coordinated by an intelligent **Adaptive Compute Router**:
* **Dynamic Hardware-Aware Routing:** Estimates scene complexity $\kappa \in [0, 1]$ per frame and activates only required micro-models, operating from **7.65 ms (130.8 FPS)** on simple scenes to **30.86 ms (32.4 FPS)** on complex occluded scenes under FP16 on an NVIDIA RTX 4090.
* **Occlusion-Aware NMS (OA-NMS):** Evaluates visibility ($V$) and relative depth ($D$) before suppressing overlapping candidate boxes, preserving distinct overlapping objects that conventional greedy NMS discards.
* **State-of-the-Art Occlusion Robustness:** Delivers a decisive **+8.3 AP gain** on severely occluded instances ($<30\%$ visibility) over YOLOv8-M on COCOA, while requiring **~40% of YOLOv8-M's parameter footprint** (10.49M student vs. 25.9M).

---

## 🏗️ System Architecture

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │                   INPUT IMAGE (640×640)                │
                                      └───────────────────────────┬────────────────────────────┘
                                                                  ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │             MobileNetV4-Hybrid Shared Backbone         │
                                      │             + Lightweight Feature Pyramid (FPN)        │
                                      │               [P3: 80×80 | P4: 40×40 | P5: 20×20]      │
                                      └───────────────────────────┬────────────────────────────┘
                                                                  ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │            Adaptive Compute Router (57.8K params)      │
                                      │            Scene Complexity Estimation: κ ∈ [0, 1]     │
                                      └───────┬───────────────────┼────────────────────┬───────┘
                                              │                   │                    │
                     κ < 0.3 (Simple: 38.2%)  │   0.3 ≤ κ < 0.7   │   κ ≥ 0.7          │
                     ─────────────────────────┘   (Medium: 41.5%) │   (Complex: 20.3%) │
                                                  ────────────────┘   ─────────────────┘
                                              │                   │                    │
                                              ▼                   ▼                    ▼
                                      ┌───────────────┐   ┌───────────────┐   ┌────────────────┐
                                      │ M1: Detector  │   │ M1: Detector  │   │ M1: Detector   │
                                      │ M5: Context   │   │ M2: Occlusion │   │ M2: Occlusion  │
                                      └───────┬───────┘   │ M5: Context   │   │ M3: Enhancer   │
                                              │           │ M7: Calibrator│   │ M4: Depth      │
                                              │           └───────┬───────┘   │ M5: Context    │
                                              │                   │           │ M6: Amodal     │
                                              │                   │           │ M7: Calibrator │
                                              │                   │           └────────┬───────┘
                                              └───────────────────┼────────────────────┘
                                                                  ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │       Cross-Attention Multimodal Fusion (264.5K params)│
                                      │          M1 Query Tokens ⟷ Auxiliary Keys & Values     │
                                      └───────────────────────────┬────────────────────────────┘
                                                                  ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │             Occlusion-Aware NMS (OA-NMS)               │
                                      │          Disambiguates Overlap vs. Duplication         │
                                      │       Conditioned on |Vi - Vj| > τocc & |Di - Dj| > τd │
                                      └───────────────────────────┬────────────────────────────┘
                                                                  ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │              FINAL DETECTIONS & METADATA               │
                                      │     [Boxes, Classes, Visibility, Depth, Calib. Scores] │
                                      └────────────────────────────────────────────────────────┘
```

---

## 🧩 Micro-Models Specification (Table 1)

OrchestraNet decomposes perceptual sub-tasks across 7 specialized expert modules:

| Module | Component Name | Specialization / Sub-Task | Bound FPN Levels | Parameter Count | GFLOPs |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **M1** | Primary Base Detector | Multi-scale bounding box regression & class logits | P3, P4, P5 | 1,540,864 (1.54M) | 16.4 |
| **M2** | Occlusion Analyzer | Dense occlusion probability maps & visibility $V \in [0, 1]$ | P3 | 812,416 (0.81M) | 4.2 |
| **M3** | Small Object Enhancer | 2× feature super-resolution for small instances ($<32\times 32$) | P3 | 624,384 (0.62M) | 2.8 |
| **M4** | Relative Depth Estimator | Monocular depth regression for foreground/background z-ordering | P4, P5 | 1,048,576 (1.05M) | 3.5 |
| **M5** | Scene Context Engine | Global scene category embeddings (16 coarse clusters) | P5 | 412,160 (0.41M) | 0.5 |
| **M6** | Amodal Completer | Predicts complete unoccluded bounding box & shape mask | P4 | 934,656 (0.93M) | 4.5 |
| **M7** | Confidence Calibrator | Cross-level confidence calibration to eliminate post-NMS false positives | Cross-Level MLP | 44,802 (0.045M) | <0.1 |
| **Router** | Adaptive Compute Router | Lightweight CNN estimating scene complexity $\kappa$ | P3, P4, P5 | 57,892 (0.058M) | 0.1 |
| **Fusion** | Cross-Attention Fusion | Multimodal query-key attention projection | Cross-Level | 264,576 (0.26M) | 0.5 |
| **Total** | **Full Ensemble (Student)** | **Complete Heterogeneous Orchestration Pipeline** | **P3–P5** | **10,489,690 (10.49M)** | **32.4** |

*Note: The larger teacher model uses $C=256$ FPN channels, totaling **13.12M parameters**, distilled into this **10.49M** student detector.*

---

## 📊 Benchmark Results

### 1. Adaptive Routing Profiles (Table 1 & 2)
Weighted across MS-COCO val2017, OrchestraNet averages **20.96 GFLOPs**, delivering a **35.3% compute reduction** compared to always-active full ensemble execution.

| Profile | Complexity Criterion | Active Experts | RTX 4090 Latency (FP16) | Throughput (FPS) | COCO Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple** | $\kappa < 0.3$ | M1, M5 | **7.65 ms** | **130.8 FPS** | 38.2% |
| **Medium** | $0.3 \le \kappa < 0.7$ | M1, M2, M5, M7 | **16.20 ms** | **61.7 FPS** | 41.5% |
| **Complex** | $\kappa \ge 0.7$ | M1–M7 (All Seven) | **30.86 ms** | **32.4 FPS** | 20.3% |

---

### 2. Leave-One-Out Component Ablation Suite (Table 5)
Evaluated on **5,000 images of MS-COCO val2017**. Demonstrates the statistical contribution of each micro-model:

| Configuration | mAP@50:95 | mAP@50 | AP_occ (<30% vis) | FPS | $\Delta$ mAP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full OrchestraNet** | **48.83** | **65.00** | **30.6%** | **32.4** | — |
| **−M2 (Occlusion Analyzer)** | 48.56 | 64.74 | 24.1% | 35.8 | −0.27 |
| **−M6 (Amodal Completer)** | 48.64 | 64.83 | 25.8% | 44.7 | −0.19 |
| **−M4 (Depth Estimator)** | 48.71 | 64.91 | 26.4% | 34.2 | −0.12 |
| **−M3 (Small Enhancer)** | 48.73 | 64.89 | 29.8% | 35.1 | −0.10 |
| **−M7 (Calibrator)** | 48.76 | 64.94 | 28.9% | 32.6 | −0.07 |
| **−M5 (Scene Context)** | 48.79 | 64.96 | 29.5% | 33.0 | −0.05 |
| **M1 only (Baseline)** | 48.42 | 65.08 | 22.3% | 138.5 | −0.41 |

*Key Takeaway: Removing M2 triggers the largest drop in both overall precision ($-0.27$ mAP) and severe occlusion robustness ($-6.5$ AP_occ), confirming explicit occlusion modeling as the primary engine of robustness.*

---

### 3. Comparison with State-of-the-Art Detectors (Table 8)

| Model | Parameters (M) | mAP@50:95 | mAP@50 | RTX 4090 FPS (FP16) |
| :--- | :---: | :---: | :---: | :---: |
| **Faster R-CNN (ResNet-50)** | 41.8 | 40.20 | 58.40 | 54.9 |
| **Cascade R-CNN (ResNet-50)**| 69.2 | 44.30 | 62.10 | 35.2 |
| **DETR (ResNet-50)** | 41.0 | 42.00 | 62.40 | 28.0 |
| **Deformable DETR** | 40.0 | 43.80 | 62.60 | 19.0 |
| **RT-DETR-R50** | 42.0 | 53.10 | 65.10 | 108.0 |
| **YOLOv8-S** | 11.2 | 44.90 | 61.80 | 172.4 |
| **YOLOv8-M** | 25.9 | 50.20 | 65.70 | 117.6 |
| **OrchestraNet (Simple Route)** | **9.12** | **48.48** | **65.16** | **130.8** |
| **OrchestraNet (Full Ensemble)**| **10.49** | **48.83** | **65.00** | **32.4** |

---

### 4. Occlusion Severity Stratification on COCOA (Table 9)

| Method | AP (All) | AP (Visibility > 70%) | AP (Visibility < 30%) | $\Delta$ Occluded vs YOLOv8-M |
| :--- | :---: | :---: | :---: | :---: |
| **Faster R-CNN** | 34.1 | 48.2 | 20.8 | −1.5 |
| **Cascade R-CNN** | 36.8 | 52.1 | 23.1 | +0.8 |
| **YOLOv8-M** | 37.4 | 54.2 | 22.3 | — (Baseline) |
| **Amodal Expansion** | 38.5 | 49.2 | 27.1 | +4.8 |
| **OrchestraNet (Ours)** | **41.8** | **55.6** | **30.6** | **+8.3 AP** |

---

### 5. Multi-Hardware Deployment Benchmark (Table 10)

| Hardware Platform | Precision | Batch Size | Latency: Simple / Complex | FPS: Simple / Complex | Target Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA RTX 4090** | FP32 | 1 | 12.0 ms / 38.2 ms | 83.3 / 26.2 | Cloud / Server |
| **NVIDIA RTX 4090** | FP16 | 1 | **7.65 ms / 30.86 ms** | **130.8 / 32.4** | Server Real-Time |
| **NVIDIA RTX 4090** | FP16 | 8 | 4.1 ms / 18.4 ms | 243.9 / 54.3 | High Throughput |
| **NVIDIA RTX 3080** | FP16 | 1 | 10.4 ms / 41.2 ms | 96.2 / 24.3 | Workstation |
| **Jetson AGX Orin** | FP16 | 1 | 22.1 ms / 78.4 ms | 45.2 / 12.8 | Edge Robotics |
| **Jetson AGX Orin** | INT8 (TensorRT) | 1 | **14.7 ms / 48.5 ms** | **68.0 / 20.6** | Autonomous Edge |

---

## 💻 Quickstart & Interactive Notebook

An all-in-one interactive walkthrough is provided in:  
👉 [`OrchestraNet_Walkthrough_and_Evaluation.ipynb`](OrchestraNet_Walkthrough_and_Evaluation.ipynb)

The notebook includes step-by-step executable cells to:
1. Automatically verify and install dependencies (`torch`, `torchvision`, `ultralytics`, `pycocotools`, `timm`).
2. Mount Google Drive and auto-detect PhD backup checkpoints (`OrchestraNet_PhD_Backup_20260924_082631`).
3. Initialize the complete 7-expert architecture and verify parameter counts against Table 1.
4. Run end-to-end inference and visualize the dynamic router's complexity score $\kappa$ and active profile.
5. Render side-by-side detection comparisons between Standard Greedy NMS and Occlusion-Aware NMS (OA-NMS).
6. Run the 8-pass leave-one-out ablation suite to verify Table 5.
7. Benchmark hardware latency under FP16 half-precision on your local GPU.

---

## 📦 Checkpoints & Google Drive Storage

Checkpoints, evaluation outputs, and training logs are archived in two PhD release bundles:
* **Primary PhD Backup:** `OrchestraNet_PhD_Backup_20260924_082631`
  * `checkpoints/`: Full ensemble models (`orchestranet_epoch34.pt`, `orchestranet_best.pt`, `orchestranet_final.pt`).
  * `checkpoints/router/`: Trained REINFORCE router policy (`router_trained.pt`).
  * `checkpoints/individual/`: Individual micro-model checkpoints (137 weight files across epochs).
  * `logs/`: Complete 3-phase training logs (`individual/`, `joint/`, `router/`).
  * `results/`: Leave-one-out evaluation JSON metrics matching Table 5.
* **Warm-up Checkpoint Archive:** `OrchestraNet_PhD_Backup_20260924_081529`
  * Contains early specialist warm-up checkpoints for M1–M7.

When deploying in Google Colab, upload these folders to your Google Drive (`/content/drive/MyDrive/OrchestraNet_PhD_Backup_20260924_082631`), and the interactive notebook will automatically detect and link them.

---

## 🚀 Installation & Local Usage

### 1. Environment Setup
```bash
git clone https://github.com/ammarraza1199/OrchestraNet.git
cd OrchestraNet

# Create a clean virtual environment
conda create -n orchestranet python=3.10 -y
conda activate orchestranet

# Install dependencies
pip install -r requirements.txt
```

### 2. Single-Image Inference
```bash
python scripts/demo_inference.py \
    --image assets/sample_street.jpg \
    --weights checkpoints/orchestranet_epoch34.pt \
    --router-weights checkpoints/router/router_trained.pt \
    --conf-thresh 0.25 \
    --output-dir outputs/
```

### 3. Reproduce Component Ablation Study (Table 5)
```bash
python scripts/run_ablation_study.py \
    --weights checkpoints/orchestranet_epoch34.pt \
    --data-root ./data/coco \
    --num-images 5000 \
    --conf-thresh 0.001 \
    --out-dir ./paper_tables
```

---

## 🏋️ 3-Phase Training Schedule

OrchestraNet is trained using a structured 3-phase curriculum:

1. **Phase 1: Specialist Warm-Up (Epochs 1–30)**
   * Each micro-model trains separately on its task:
     * `M1`: Standard COCO detection (`CIoU` + `Varifocal`).
     * `M2`: Occlusion analyzer with CutPaste self-supervised pretext task.
     * `M3`: Feature super-resolution on small targets ($<32\times 32$).
     * `M4`: Monocular depth regression with scale-invariant loss.
     * `M5`: Global scene classification (16 semantic clusters).
     * `M6`: Amodal completion on KINS / COCOA.
     * `M7`: Confidence calibration (ECE minimization).
2. **Phase 2: Joint End-to-End Training (Epochs 31–100)**
   * Micro-models train jointly with cross-attention fusion.
   * Gumbel-Softmax discrete routing with temperature decay ($1.0 \rightarrow 0.1$).
   * Progressive occlusion curriculum (Table 3: 20% $\rightarrow$ 50% $\rightarrow$ 70% synthetic masking).
3. **Phase 3: Router RL Fine-Tuning (Epochs 101–120)**
   * All micro-models are frozen.
   * Router policy network updates via REINFORCE:
     $$R = 1.0 \cdot R_{acc} + 0.3 \cdot R_{lat} + 0.1 \cdot R_{eff}$$

---

## 📂 Repository File Tree

```
OrchestraNet/
├── orchestranet/                        # Core Python package
│   ├── backbone/                        # MobileNetV4-Hybrid & Lightweight FPN
│   ├── models/                          # Micro-Models M1 through M7
│   │   ├── m1_detector.py               # Base Primary Detector Head
│   │   ├── m2_occlusion.py              # U-Net-Lite Occlusion Analyzer
│   │   ├── m3_small_enhancer.py         # Super-Resolution Feature Enhancer
│   │   ├── m4_depth.py                  # DPT Monocular Depth Estimator
│   │   ├── m5_context.py                # ViT-Tiny Scene Context Classifier
│   │   ├── m6_amodal.py                 # Transformer Decoder Amodal Completer
│   │   └── m7_calibrator.py             # Confidence Calibration MLP
│   ├── router/                          # Adaptive Compute Router
│   ├── fusion/                          # Cross-Attention Fusion & OA-NMS
│   ├── data/                            # Datasets, Transforms & Occlusion Curriculum
│   ├── evaluation/                      # Evaluator & Metric Registries
│   └── orchestrator.py                  # Master Pipeline Orchestrator Class
├── paper_tables/                        # Official Benchmark Reports & Tables
│   ├── table5_ablation_summary.json     # Table 5 Leave-one-out metrics
│   ├── table5_ablation_study.md         # Table 5 Markdown representation
│   ├── table5_ablation_study.tex        # Table 5 LaTeX representation
│   └── logs.txt                         # 8-pass leave-one-out terminal run log
├── checkpoints/                         # Trained Weights & Router Models
├── OrchestraNet_PhD_Backup_20260924_082631/ # Complete PhD Archive (Weights, Logs, Results)
├── OrchestraNet_PhD_Backup_20260924_081529/ # Historical Warm-up Archive
├── Manuscript_OrchestraNet_updated one column.docx # Official Academic Manuscript
├── OrchestraNet_Walkthrough_and_Evaluation.ipynb    # Interactive Walkthrough Notebook
├── requirements.txt                     # Python Package Dependencies
└── README.md                            # Project Documentation
```

---

## 📜 Citation

If you find OrchestraNet useful for your research, please cite our manuscript:

```bibtex
@article{raza2026orchestranet,
  title={OrchestraNet: A Heterogeneous Multi-Model Orchestration Framework with Explicit Occlusion Reasoning and Dynamic Real-Time Compute Scaling},
  author={------},
  journal={------},
  year={2026}
}
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
