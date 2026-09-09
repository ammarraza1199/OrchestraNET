# OrchestraNet Project Documentation

---

# OrchestraNet Benchmarks & Performance Goals

This document outlines the performance benchmarks extracted from the official `OrchestraNet_IJIES_formatted PG.11.docx` manuscript, as well as the overarching accuracy targets that must be achieved.

> [!IMPORTANT]
> **Primary Goal:** You have promised a model accuracy of **65-70%**. 
> *Note: In standard object detection on MS-COCO, the current OrchestraNet achieves `64.3%` on the **mAP@50** metric (Table 5). Pushing this to the 65-70% range is our critical target.*

---

## 1. Overall Detection Accuracy (MS-COCO val2017)
To surpass the current paper's baseline, the model must exceed these metrics (extracted from Table 5 & Table 8):

| Model | mAP@50:95 | mAP@50 | AP_occ | FPS | Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OrchestraNet (Current)** | **42.7** | **64.3** | **30.6** | **83** | **~5M** |
| YOLOv8-M | 50.2 | — | 22.3 | ~120 | ~25M |
| RT-DETR-R50 | 53.1 | — | — | ~108 | ~42M |
| Faster R-CNN | 43.1 | — | 20.8 | ~15 | ~42M |
| DETR | 42.0 | 62.5 | — | ~28 | ~41M |

*To achieve your promised 65-70% accuracy, we will focus on pushing the `mAP@50` from 64.3 to at least 65.0+.*

## 2. Occlusion-Specific Performance (COCOA)
The primary selling point of OrchestraNet is its performance on heavily occluded objects (Visibility < 30%). We must maintain or surpass these benchmarks from Table 9:

| Method | AP (All) | AP (Vis > 70%) | AP (Vis < 30%) |
| :--- | :--- | :--- | :--- |
| YOLOv8-M | 50.2 | 58.4 | 22.3 |
| Amodal Expansion | 44.8 | 49.2 | 27.1 |
| **OrchestraNet (Current)** | **42.7** | **51.6** | **30.6** |

**Goal:** Maintain the `+8.3 AP` gap over YOLOv8-M on heavily occluded objects.

## 3. Post-Processing: Occlusion-Aware NMS (OA-NMS)
The OA-NMS module must maintain its ability to preserve valid occlusion pairs without spiking the False Positive (FP) rate (from Table 6):

| NMS Method | AP (Vis < 50%) | Preserved Pairs | FP Rate |
| :--- | :--- | :--- | :--- |
| Standard NMS | 24.2 | 0% | 3.2% |
| Soft-NMS | 26.8 | 31% | 4.1% |
| **OA-NMS (Current)**| **30.6** | **73%** | **3.8%** |

## 4. Latency and Hardware Benchmarks
The adaptive compute router aims to keep inference fast despite having 7 micro-models. We must respect these speed limits (from Table 10):

*   **RTX 4090 (FP32, Batch 1):** ~12.0 ms (83 FPS)
*   **RTX 4090 (FP16, Batch 1):** ~7.2 ms (139 FPS)
*   **Jetson Orin (FP16, Batch 1):** ~22.1 ms (45 FPS)
*   **Jetson Orin (INT8 TRT, Batch 1):** ~14.7 ms (68 FPS)

---

## Next Steps for Achieving 65-70%
To push the `mAP@50` from 64.3% to the 65-70% range without destroying the 83 FPS speed limit, we have a few options:
1.  **Enhance M1 (Primary Detector):** Upgrade the M1 backbone or head slightly to capture better baseline features.
2.  **Dataset Enrichment:** Training with the complete KITTI Depth and KINS amodal datasets (as discussed earlier) will likely provide the exact boost needed for M4 and M6 to push the overall accuracy into the 65-70% range.
3.  **Refine the RL Router:** Tune the REINFORCE reward function to slightly favor accuracy (`α`) over latency (`β`), forcing the router to activate more micro-models on borderline scenes.

---

# Dataset Preparation TODO List

Here is the checklist for getting your datasets completely set up to train OrchestraNet (M1-M7) end-to-end, including the required links and storage estimates.

> [!IMPORTANT]
> Make sure you have at least **~25 GB to 30 GB** of free storage space available before beginning these downloads.

## 1. Setup the Amodal Annotations (For M6)
*Target: ~1-2 GB Storage*

- [ ] **Clone the KINS Dataset Repository**
  - **Link:** [KINS Dataset GitHub](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset)
  - Follow the instructions in their README to download the JSON annotation files.
- [ ] **Clone the Amodal API**
  - **Link:** [amodalAPI GitHub](https://github.com/Wakeupbuddy/amodalAPI)
  - This is a fork of the COCO API specifically designed to parse the amodal masks you just downloaded. Install this in your Python environment.

## 2. Setup the KITTI Depth Maps (For M4)
*Target: ~21 GB Storage*

> [!NOTE]
> The KITTI dataset is too large to be hosted directly on GitHub. You must download it from the official academic portal.

- [ ] **Create an account / Login to CVLIBS**
  - **Link:** [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [ ] **Download the Annotated Ground-Truth Data**
  - **Size:** ~14 GB
- [ ] **Download the Projected Raw LiDAR Scans**
  - **Size:** ~5 GB
- [ ] **Download the Validation and Test Sets**
  - **Size:** ~2 GB
- [ ] **Extract the Archives**
  - Unzip all three of the depth archives into the *same base directory* so that their folder structures merge correctly.

## 3. Integration
- [ ] **Align the Datasets**
  - Since KINS is built on top of KITTI, ensure that the image filenames from the KINS annotations correctly map to the depth map filenames in the KITTI directory.
- [ ] **Update `datasets.py`**
  - Modify `orchestranet/data/datasets.py` to utilize the `amodalAPI` (to load `amodal_masks` and `amodal_boxes`) and load the aligned `depth_gt` arrays.

---

# OrchestraNet Codebase Analysis

## 1. High-Level Overview
**OrchestraNet** is a novel, multi-model orchestration framework designed to tackle **occluded object detection** while maintaining **real-time performance**. 

Traditional object detectors rely on a single, monolithic network to process everything. OrchestraNet takes a micro-services-like approach: it replaces the monolithic detector with an ensemble of **7 specialized micro-models**. To maintain high speed, it utilizes an **Adaptive Compute Router** that dynamically selects which micro-models to run on a per-frame basis, depending on the complexity of the scene.

## 2. Core Architecture Pipeline
The system processes an input image through a well-defined pipeline (found in [`orchestranet/orchestrator.py`](file:///c:/Users/DELL/Downloads/OrchestraNet-master/OrchestraNet-master/orchestranet/orchestrator.py)):

1. **Shared Feature Extraction:**
   - **Backbone:** The image passes through a lightweight backbone (`MobileNetV4Backbone`).
   - **FPN:** A `LightweightFPN` (Feature Pyramid Network) unifies the extracted features into a standard channel dimension (default 128 channels).

2. **Adaptive Routing:**
   - The FPN features are fed into the **Adaptive Router** (`AdaptiveRouter` in `orchestranet/router/adaptive_router.py`), which scores the scene complexity.
   - Based on this score, the router categorizes the scene as `simple`, `medium`, or `complex`, and decides which of the 7 micro-models to activate.

3. **Micro-Model Execution:**
   - **Sequential Dependencies:** `M1` (Primary Detector) runs first. `M2` (Occlusion Analyzer) runs next (if active) to provide occlusion context.
   - **Parallel Execution:** The remaining selected models (`M3` to `M7`) run in parallel, utilizing the FPN features and context provided by `M2`.

4. **Fusion and OA-NMS:**
   - The outputs of the active models are fused. If `M7` is active, it recalibrates the detection confidence scores.
   - Finally, the detections go through **Occlusion-Aware Non-Maximum Suppression (OA-NMS)**. Unlike standard NMS that indiscriminately removes highly overlapping bounding boxes (assuming they are duplicates), OA-NMS uses depth and occlusion data to determine if two overlapping boxes are actually two distinct objects (one in front of the other).

## 3. The 7 Micro-Models
The 7 micro-models are defined in `orchestranet/models/`. Each acts as an expert for a specific perceptual task:

| ID | Name | Role |
|----|------|------|
| **M1** | Primary Detector | Generates the initial bounding box proposals and class logits. It is the only model that runs 100% of the time. |
| **M2** | Occlusion Analyzer | Uses a U-Net-Lite architecture with Attention Gates to predict per-pixel occlusion maps and visibility scores. Its outputs provide context for M6, M7, and the OA-NMS algorithm. |
| **M3** | Small Object Enhancer | Focuses on super-resolution and detection refinement for small objects. |
| **M4** | Depth Estimator | Generates a monocular pseudo-depth map, used primarily by OA-NMS to figure out the depth ordering of overlapping boxes. |
| **M5** | Semantic Context | Extracts scene-level contextual priors to aid in detection. |
| **M6** | Amodal Completer | Predicts the complete shape of an object, even the parts hidden behind an occluder. |
| **M7** | Confidence Calibrator| Recalibrates confidence scores based on occlusion metrics, preventing the system from overly penalizing heavily occluded objects. |

## 4. Key Innovations

### A. Adaptive Compute Router (`adaptive_router.py`)
This is the brain behind the real-time capability. By estimating complexity, it routes compute power only where needed:
- **Simple Scenes:** Only runs `M1` (~2ms latency).
- **Medium Scenes:** Runs `M1, M2, M7` (~5ms latency).
- **Complex Scenes:** Runs all models `M1-M7` (~12ms latency).
During training, it uses **Gumbel-Softmax** to allow differentiable routing, which gradually decays into hard-threshold routing for inference.

### B. Occlusion-Aware NMS (`oa_nms.py`)
Standard NMS destroys overlapping boxes based solely on Intersection-over-Union (IoU). OA-NMS fixes this by checking if the overlapping pair has a significant difference in visibility scores (from M2) or depth values (from M4). If they do, the system recognizes an occluder-occludee relationship and keeps both boxes, rather than treating one as a duplicate.

### C. Self-Supervised Occlusion Pretext Task
Found in `M2` (`m2_occlusion.py`), this task allows the model to learn occlusion representations without human annotations by predicting where artificial occlusions were injected into the image. (Likely relies on `synthetic_occlusion.py`).

## 5. Training and Data Pipeline
The `training/` directory contains various scripts, highlighting a sophisticated, multi-stage training regime:
- **`train_individual.py`**: For pre-training micro-models in isolation.
- **`train_router.py`**: Dedicated to training the `AdaptiveRouter`'s routing logic.
- **`train_joint.py`**: For end-to-end joint fine-tuning of the entire orchestrated network.
- **`distill.py`**: Indicates the usage of knowledge distillation, likely to compress the knowledge of all models into the lightweight ensemble or to train the router.

The `data/` directory handles dataset loading and augmentations, explicitly featuring `synthetic_occlusion.py` to generate the artificial occlusions needed for M2's self-supervised learning task. 

## 6. Tech Stack & Engineering
- **Framework:** PyTorch (`torch`, `torchvision`).
- **Models:** Uses `timm` for backbones and `einops` for tensor reshaping.
- **Deployment:** Has provisions for ONNX (`onnx`, `onnxruntime`) and TensorRT export for blazingly fast edge inference.
- **Project Structure:** Managed by `pyproject.toml` with `setuptools`, featuring modern tooling like `ruff`, `black`, and `pytest`.

## 7. Dataset Requirements & Compatibility
While OrchestraNet provides an end-to-end training script (`train_joint.py`) configured for the standard COCO dataset, not all micro-models can be effectively trained on COCO alone due to its lack of specialized annotations.

### Models Fully Supported by COCO:
- **M1 (Primary Detector):** Fully trained using standard COCO bounding boxes.
- **M2 (Occlusion Analyzer):** Fully trained using a self-supervised pretext task. The data loader dynamically injects synthetic occlusions (via `SyntheticOcclusionGenerator`) onto COCO images, eliminating the need for human-annotated occlusion masks.
- **M3 (Small Object Enhancer):** Fully trained using COCO's existing bounding box data for small objects.
- **M5 (Semantic Context):** Fully trained using standard COCO class and scene distributions.
- **M7 (Confidence Calibrator):** Fully trained using the dynamic occlusion scores predicted by M2.

### Models Requiring Specialized Datasets:
- **M4 (Depth Estimator):** Standard COCO lacks depth maps (`depth_gt`). When trained on COCO, M4 falls back to a weak "self-supervised smoothness" loss, which fails to learn accurate depth ordering. **Required Dataset:** A dataset with monocular depth maps, such as **NYU Depth v2** or **KITTI**.
- **M6 (Amodal Completer):** M6 relies on `amodal_boxes` and `amodal_masks` to learn the hidden shapes of occluded objects. Without these targets, its loss evaluates to `0.0`, meaning its weights will not update during training. **Required Dataset:** An amodally-annotated dataset, such as **COCOA** (COCO Amodal) or **KINS** (KITTI INstance Segmentation).

## Summary
OrchestraNet is a highly advanced, modular framework aimed at pushing the state-of-the-art in occlusion handling for object detection. It brilliantly balances the heavy compute requirements of occlusion reasoning by dynamically allocating specialized models only when the scene complexity demands it.

---

# OrchestraNet Training Plan

## Goal Description
Start the end-to-end joint training of OrchestraNet and increase the epoch count to push the model towards the 65-70% mAP benchmark you requested.

> [!WARNING]
> **User Review Required: Missing Datasets & Model Limitations**
> You mentioned that "everything is completed", but I just checked the `data/` directory and it **only contains the standard COCO dataset**. The KINS (amodal masks) and KITTI (depth maps) datasets are still missing from the filesystem.
> 
> Here is what will happen if we start training right now:
> *   **M1, M2, M3, M5, M7:** Will train perfectly on the COCO data.
> *   **M4 (Depth Estimator):** Will fall back to a weak "unsupervised smoothness loss" because it lacks the KITTI `depth_gt` labels.
> *   **M6 (Amodal Completer):** Will output a loss of exactly `0.0` and **will not train at all** because it lacks the KINS `amodal_masks`.
> 
> **Can we hit 65-70% without them?**
> We can certainly try to push the baseline mAP higher by training for longer (e.g., 200 epochs instead of 100), but missing M4 and M6 completely will make it nearly impossible to hit your occlusion benchmarks (the +8.3 AP gap on heavily occluded objects).

## Proposed Changes

If you want to proceed with training *as-is* (without KINS/KITTI), here is the plan:

### 1. Adjust Training Schedule
We need to increase the epochs in `training/train_joint.py` to give the model more time to learn the primary tasks and push past the baseline 64.3% mAP@50.

#### [MODIFY] `train_joint.py`
- Change the default `--epochs` argument from `100` to `200` (or `250`).
- Ensure the occlusion curriculum (`epoch 31-100` decay) is stretched proportionally to match the new 200 epoch schedule.

### 2. Execute Training
We will run the training script in the background. 
> [!IMPORTANT]
> **Hardware Warning:** The paper states that full joint training takes **4 days on 8x A100 GPUs**. Running this locally on a single GPU will take a very long time.

## Open Questions for You
1. **Did you download KINS/KITTI to a different folder?** If so, I need to update `datasets.py` to point to them before we start!
2. **Do you want to proceed with training right now anyway?** I can start the 200-epoch training on just the COCO dataset, but M6 will remain untrained. 

Let me know how you want to proceed!
