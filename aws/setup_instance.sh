#!/bin/bash
# ============================================================================
# OrchestraNet — EC2 Instance Setup Script
# ============================================================================
# Sets up a fresh Ubuntu 22.04 EC2 instance with all dependencies for
# OrchestraNet training. Optimized for p3.2xlarge or g4dn.xlarge instances.
#
# Usage:
#   ssh -i your-key.pem ubuntu@<ec2-ip>
#   bash aws/setup_instance.sh
# ============================================================================

set -euo pipefail

echo "🎼 OrchestraNet — EC2 Instance Setup"
echo "======================================"
echo ""

# === System Updates ===
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    htop \
    tmux \
    tree \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    awscli

# === NVIDIA Drivers (if not pre-installed) ===
if ! command -v nvidia-smi &>/dev/null; then
    echo "🔧 Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535
    echo "⚠️  Reboot may be required for NVIDIA drivers"
fi

# Verify GPU
echo ""
echo "🖥️  GPU Status:"
nvidia-smi || echo "⚠️  GPU not available yet (may need reboot)"

# === Python Environment ===
echo ""
echo "🐍 Setting up Python environment..."
python3.10 -m venv ~/orchestranet_env
source ~/orchestranet_env/bin/activate

pip install --upgrade pip setuptools wheel

# === PyTorch (CUDA 12.1) ===
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# === Project Dependencies ===
echo "📋 Installing OrchestraNet dependencies..."
cd ~/CV_project  # Assumes code was uploaded here
pip install -e ".[dev]"

# Additional useful packages
pip install wandb onnxsim

# === Verify Installation ===
echo ""
echo "✅ Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

python -c "
from orchestranet.orchestrator import OrchestraNet
model = OrchestraNet(num_classes=80, pretrained_backbone=False)
params = model.count_all_parameters()
print(f'OrchestraNet total params: {params[\"TOTAL\"][\"total\"]:,}')
print('✅ OrchestraNet import successful!')
"

# === Create Directory Structure ===
echo ""
echo "📁 Creating directory structure..."
mkdir -p ~/CV_project/data/coco
mkdir -p ~/CV_project/checkpoints/individual
mkdir -p ~/CV_project/checkpoints/router
mkdir -p ~/CV_project/checkpoints/distilled
mkdir -p ~/CV_project/logs
mkdir -p ~/CV_project/results
mkdir -p ~/CV_project/exports

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download COCO:  bash aws/download_coco.sh"
echo "  2. Start training: bash aws/train_full_pipeline.sh"
echo ""
echo "Tip: Use tmux to keep training running after disconnect:"
echo "  tmux new -s train"
echo "  bash aws/train_full_pipeline.sh"
echo "  (Ctrl-B, then D to detach)"
