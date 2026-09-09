# 🎼 OrchestraNet

**A Novel Multi-Model Orchestration Framework for Multi-Class Object Detection with Explicit Occlusion Reasoning and Real-Time Optimization**

---

## Overview

OrchestraNet replaces the traditional monolithic object detector with an **orchestrated ensemble of 7 specialized micro-models**, each expert at one perceptual task. An intelligent **Adaptive Compute Router** dynamically selects which models to activate per-frame based on scene complexity, achieving superior accuracy on occluded and small objects while maintaining real-time speed.

## Architecture

```
Input → Shared Backbone → Adaptive Router → [M1..M7 Micro-Models] → Fusion → OA-NMS → Output
```

### Micro-Models
| Model | Task | Parameters |
|-------|------|-----------|
| **M1** Primary Detector | Bounding box detection | ~1.2M |
| **M2** Occlusion Analyzer | Occlusion maps & visibility scores | ~800K |
| **M3** Small Object Enhancer | Super-resolution + detection refinement | ~600K |
| **M4** Depth Estimator | Monocular pseudo-depth for occlusion ordering | ~1M |
| **M5** Semantic Context Engine | Scene-level contextual priors | ~400K |
| **M6** Amodal Completer | Complete shape prediction for occluded objects | ~900K |
| **M7** Confidence Calibrator | Occlusion-aware confidence recalibration | ~100K |

### Key Innovations
- **Adaptive Compute Routing**: 2ms for simple scenes, 12ms for complex
- **Occlusion-Aware NMS (OA-NMS)**: Preserves overlapping-but-real detections
- **Self-Supervised Occlusion Pretext**: Learns occlusion understanding without annotations
- **Shared Latent Tensor**: Rich inter-model feature communication

## Quick Start

```bash
# Install
pip install -e .

# Train individual model
python training/train_individual.py --model m1 --config configs/models/m1_detector.yaml

# Joint training
python training/train_joint.py --config configs/training/joint_finetune.yaml

# Real-time inference
python inference/realtime_demo.py --weights checkpoints/orchestranet.pt --source webcam
```

## Project Structure
```
orchestranet/          # Core package
├── backbone/          # Shared feature extractor
├── models/            # 7 specialized micro-models
├── router/            # Adaptive compute router
├── fusion/            # Cross-attention fusion + OA-NMS
├── losses/            # Custom loss functions
├── data/              # Datasets & augmentations
└── utils/             # Metrics, visualization, export
```

## Citation
```bibtex
@article{orchestranet2026,
  title={OrchestraNet: Multi-Model Orchestration for Occluded Object Detection},
  year={2026}
}
```

## License
MIT License
