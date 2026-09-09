#!/bin/bash
# ============================================================================
# OrchestraNet — S3 Checkpoint Sync
# ============================================================================
# Syncs checkpoints, logs, and results to an S3 bucket.
# Can be run as a background cron job during training.
#
# Usage:
#   # One-time sync
#   bash aws/sync_checkpoints.sh
#
#   # Continuous sync (every 5 minutes in background)
#   bash aws/sync_checkpoints.sh --watch
#
#   # Custom bucket
#   S3_BUCKET=s3://my-bucket bash aws/sync_checkpoints.sh
# ============================================================================

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-s3://orchestranet-training}"
PROJECT_NAME="orchestranet"
TIMESTAMP=$(date +%Y%m%d)

S3_PREFIX="${S3_BUCKET}/${PROJECT_NAME}/${TIMESTAMP}"

echo "🔄 OrchestraNet — S3 Sync"
echo "   Bucket: ${S3_PREFIX}"
echo ""

sync_once() {
    echo "[$(date '+%H:%M:%S')] Syncing..."

    # Sync checkpoints (excluding optimizer state for smaller uploads)
    aws s3 sync ./checkpoints/ "${S3_PREFIX}/checkpoints/" \
        --exclude "*.tmp" \
        --quiet

    # Sync logs (TensorBoard + JSON)
    aws s3 sync ./logs/ "${S3_PREFIX}/logs/" \
        --exclude "__pycache__/*" \
        --quiet

    # Sync results
    aws s3 sync ./results/ "${S3_PREFIX}/results/" --quiet

    # Sync exports
    aws s3 sync ./exports/ "${S3_PREFIX}/exports/" --quiet

    echo "[$(date '+%H:%M:%S')] Sync complete"
}

if [ "${1:-}" = "--watch" ]; then
    echo "👁️  Watching mode: syncing every 5 minutes..."
    echo "   Press Ctrl+C to stop"
    echo ""

    while true; do
        sync_once
        sleep 300
    done
else
    sync_once
fi

echo ""
echo "📥 To download checkpoints locally:"
echo "   aws s3 sync ${S3_PREFIX}/checkpoints/ ./checkpoints/"
echo ""
echo "📊 To download TensorBoard logs:"
echo "   aws s3 sync ${S3_PREFIX}/logs/ ./logs/"
echo "   tensorboard --logdir ./logs/"
