#!/bin/bash
# ============================================================================
# OrchestraNet — Automated Google Drive Sync
# ============================================================================
# Automatically synchronises all checkpoints (.pt, .pth), training logs (.log, .jsonl),
# TensorBoard runs, and evaluation results directly to Google Drive.
#
# Works in two environments:
#   1. Google Colab: Syncs directly to mounted Google Drive (/content/drive/MyDrive/...)
#   2. Cloud GPU (Vast.ai / RunPod / AWS): Syncs via rclone remote (gdrive:)
#
# Usage:
#   # One-time sync
#   bash aws/sync_gdrive.sh
#
#   # Continuous background watch (runs every 5 minutes during training)
#   bash aws/sync_gdrive.sh --watch &
#
#   # Custom Drive destination folder
#   GDRIVE_DIR="/path/to/drive" bash aws/sync_gdrive.sh
# ============================================================================

set -euo pipefail

# Default Google Drive destination folder
GDRIVE_DEFAULT="/content/drive/MyDrive/ANVRiksh Project/OrchestraNET"
GDRIVE_DEST="${GDRIVE_DIR:-$GDRIVE_DEFAULT}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:ANVRiksh Project/OrchestraNET}"

echo "🔄 ═══════════════════════════════════════════════════════════"
echo "🔄   OrchestraNet — Google Drive Sync"
echo "🔄 ═══════════════════════════════════════════════════════════"

# Detect whether we have a local Drive mount or need rclone
MODE="unknown"
if [ -d "${GDRIVE_DEST}" ] || [ -d "/content/drive" ]; then
    MODE="colab_mount"
    echo "   Mode: Local Google Drive mount detected at: ${GDRIVE_DEST}"
elif command -v rclone &> /dev/null; then
    MODE="rclone"
    echo "   Mode: rclone detected (Remote: ${RCLONE_REMOTE})"
else
    MODE="python_fallback"
    echo "   Mode: Standard filesystem / Python fallback"
fi
echo ""

sync_checkpoints_and_logs() {
    local timestamp
    timestamp=$(date '+%H:%M:%S')
    echo "[${timestamp}] ☁️ Syncing checkpoints, logs, and results to Google Drive..."

    if [ "${MODE}" = "colab_mount" ]; then
        mkdir -p "${GDRIVE_DEST}/checkpoints" \
                 "${GDRIVE_DEST}/logs" \
                 "${GDRIVE_DEST}/results" \
                 "${GDRIVE_DEST}/exports" 2>/dev/null || true

        # Sync checkpoints (.pt, .pth)
        if [ -d "./checkpoints" ]; then
            cp -r -u ./checkpoints/* "${GDRIVE_DEST}/checkpoints/" 2>/dev/null || true
        fi

        # Sync logs (.log, .jsonl, TensorBoard)
        if [ -d "./logs" ]; then
            cp -r -u ./logs/* "${GDRIVE_DEST}/logs/" 2>/dev/null || true
        fi

        # Sync results (eval tables, plots)
        if [ -d "./results" ]; then
            cp -r -u ./results/* "${GDRIVE_DEST}/results/" 2>/dev/null || true
        fi

        # Sync exports (ONNX)
        if [ -d "./exports" ]; then
            cp -r -u ./exports/* "${GDRIVE_DEST}/exports/" 2>/dev/null || true
        fi

    elif [ "${MODE}" = "rclone" ]; then
        # Checkpoints
        if [ -d "./checkpoints" ]; then
            rclone copy ./checkpoints/ "${RCLONE_REMOTE}/checkpoints/" \
                --include "*.pt" --include "*.pth" \
                --transfers 4 --checkers 8 --quiet 2>/dev/null || true
        fi

        # Logs
        if [ -d "./logs" ]; then
            rclone copy ./logs/ "${RCLONE_REMOTE}/logs/" \
                --include "*.log" --include "*.jsonl" --include "events.out.tfevents.*" \
                --transfers 4 --quiet 2>/dev/null || true
        fi

        # Results & Exports
        if [ -d "./results" ]; then
            rclone copy ./results/ "${RCLONE_REMOTE}/results/" --quiet 2>/dev/null || true
        fi
        if [ -d "./exports" ]; then
            rclone copy ./exports/ "${RCLONE_REMOTE}/exports/" --quiet 2>/dev/null || true
        fi

    else
        echo "   ℹ️ Note: Mount Google Drive or install rclone ('curl https://rclone.org/install.sh | sudo bash') for cloud VM sync."
    fi

    echo "[$(date '+%H:%M:%S')] ✅ Drive sync complete."
}

if [ "${1:-}" = "--watch" ]; then
    echo "👁️  Watch mode active: Syncing every 5 minutes in background..."
    echo "   Press Ctrl+C to stop."
    echo ""
    while true; do
        sync_checkpoints_and_logs
        sleep 300
    done
else
    sync_checkpoints_and_logs
fi
