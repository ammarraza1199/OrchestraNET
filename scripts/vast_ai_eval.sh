#!/usr/bin/env bash
# ==============================================================================
# OrchestraNet — Vast.ai 1-Click Complete Validation & Ablation Runner
# ==============================================================================
set -euo pipefail

echo "========================================================================"
echo "🎻 OrchestraNet — Vast.ai Empirical Benchmark & Ablation Suite"
echo "========================================================================"
date

# 1. Hardware & Environment Inspection
echo -e "\n[1/5] Inspecting GPU Hardware & Environment..."
nvidia-smi || { echo "❌ ERROR: No NVIDIA GPU detected!"; exit 1; }
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

# 2. Dependencies
echo -e "\n[2/5] Ensuring required Python dependencies..."
pip install -q --upgrade pip
pip install -q pycocotools ultralytics torchvision tqdm numpy scipy

# 3. Dataset Verification / Fast Download
echo -e "\n[3/5] Verifying MS COCO 2017 Validation Split..."
DATA_DIR="data/coco"
mkdir -p "$DATA_DIR/annotations"

ANN_FILE="$DATA_DIR/annotations/instances_val2017.json"
VAL_DIR="$DATA_DIR/val2017"

if [ ! -f "$ANN_FILE" ]; then
    echo "Downloading instances_val2017.json (~20MB)..."
    wget -c -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations_trainval2017.zip
    unzip -q -j annotations_trainval2017.zip "annotations/instances_val2017.json" -d "$DATA_DIR/annotations/"
    rm -f annotations_trainval2017.zip
fi

if [ ! -d "$VAL_DIR" ] || [ "$(ls -1 "$VAL_DIR" | wc -l)" -lt 4900 ]; then
    echo "Downloading val2017.zip (~1GB, fast on server connection)..."
    wget -c -q --show-progress http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
    unzip -q val2017.zip -d "$DATA_DIR/"
    rm -f val2017.zip
fi

NUM_VAL_IMAGES=$(ls -1 "$VAL_DIR" | wc -l)
echo "✅ MS COCO val2017 verified: $NUM_VAL_IMAGES images found at $VAL_DIR"

# 4. Checkpoints Verification
echo -e "\n[4/5] Checking model checkpoints..."
mkdir -p checkpoints
if [ ! -f "checkpoints/orchestranet_epoch34.pt" ]; then
    echo "⚠️ Warning: checkpoints/orchestranet_epoch34.pt not found. M1 pretrained detector will run as base."
fi

# 5. Run Full 5,000-Image Empirical Ablation Suite
echo -e "\n[5/5] Running Complete 5,000-Image Leave-One-Out Ablation Suite..."
mkdir -p paper_tables

python3 scripts/run_ablation_study.py \
    --weights checkpoints/orchestranet_epoch34.pt \
    --pretrained-m1 yolov8m \
    --data-root "$DATA_DIR" \
    --num-images 5000 \
    --batch-size 1 \
    --device cuda \
    --out-dir paper_tables 2>&1 | tee paper_tables/eval_val2017_raw.log

# 6. Save Hardware & Reproducibility Spec
nvidia-smi > paper_tables/hardware_nvidia_smi.txt
pip list > paper_tables/python_environment_pip_list.txt
git log -1 > paper_tables/git_commit_hash.txt || true

echo "========================================================================"
echo "🎉 Empirical Validation Complete! All genuine benchmark files saved to:"
echo "   - paper_tables/eval_val2017_raw.log"
echo "   - paper_tables/table5_ablation_study.md"
echo "   - paper_tables/table5_ablation_study.tex"
echo "   - paper_tables/table5_ablation_summary.json"
echo "   - paper_tables/hardware_nvidia_smi.txt"
echo "========================================================================"
