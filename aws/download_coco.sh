#!/bin/bash
# ============================================================================
# OrchestraNet — COCO 2017 Dataset Download Script
# ============================================================================
# Downloads COCO2017 train/val images and annotations.
# Total size: ~25GB (images) + ~0.8GB (annotations)
#
# Usage:
#   bash aws/download_coco.sh [/path/to/data/dir]
# ============================================================================

set -euo pipefail

DATA_DIR="${1:-./data/coco}"

echo "🎼 OrchestraNet — COCO 2017 Download"
echo "======================================"
echo "   Target directory: ${DATA_DIR}"
echo ""

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# === Download Annotations ===
echo "📥 Downloading annotations..."
if [ ! -d "annotations" ]; then
    wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "📦 Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "   ✅ Annotations ready"
else
    echo "   ✅ Annotations already exist, skipping"
fi

# === Download Training Images ===
echo ""
echo "📥 Downloading train2017 images (~18GB)..."
if [ ! -d "train2017" ]; then
    wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
    echo "📦 Extracting train2017 (this may take a while)..."
    unzip -q train2017.zip
    rm train2017.zip
    echo "   ✅ Train images ready: $(ls train2017 | wc -l) images"
else
    echo "   ✅ train2017 already exists: $(ls train2017 | wc -l) images"
fi

# === Download Validation Images ===
echo ""
echo "📥 Downloading val2017 images (~1GB)..."
if [ ! -d "val2017" ]; then
    wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
    echo "📦 Extracting val2017..."
    unzip -q val2017.zip
    rm val2017.zip
    echo "   ✅ Val images ready: $(ls val2017 | wc -l) images"
else
    echo "   ✅ val2017 already exists: $(ls val2017 | wc -l) images"
fi

# === Verify ===
echo ""
echo "📊 Dataset Summary:"
echo "   Annotations: $(ls annotations/*.json 2>/dev/null | wc -l) files"
echo "   Train images: $(ls train2017 2>/dev/null | wc -l)"
echo "   Val images:   $(ls val2017 2>/dev/null | wc -l)"
echo ""

# Validate expected counts
TRAIN_COUNT=$(ls train2017 2>/dev/null | wc -l)
VAL_COUNT=$(ls val2017 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -ge 118000 ] && [ "$VAL_COUNT" -ge 5000 ]; then
    echo "✅ COCO 2017 dataset download complete and verified!"
else
    echo "⚠️  Image counts seem low. Expected ~118K train, ~5K val."
    echo "    Got ${TRAIN_COUNT} train, ${VAL_COUNT} val."
    echo "    The download may have been interrupted."
fi

echo ""
echo "Total disk usage: $(du -sh . | cut -f1)"
