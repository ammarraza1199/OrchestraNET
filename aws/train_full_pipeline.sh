#!/bin/bash
# ============================================================================
# OrchestraNet — Full Training Pipeline
# ============================================================================
# Master orchestration script that runs the complete training pipeline:
#   Phase 1: Individual model pre-training (M1-M6)
#   Phase 2: Joint end-to-end training
#   Phase 3: Router RL fine-tuning
#   Phase 4: Evaluation on val2017
#   Phase 5: ONNX export
#   Phase 6: (Optional) Knowledge distillation
#
# Usage:
#   bash aws/train_full_pipeline.sh
#
# Resume from a specific phase:
#   bash aws/train_full_pipeline.sh --start-phase 2
#
# Configuration:
#   Edit the variables below to customize training.
# ============================================================================

set -euo pipefail

# === Configuration ===
DATA_ROOT="./data/coco"
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"
DEVICE="cuda"
BATCH_SIZE=16
NUM_WORKERS=4
START_PHASE="${1:---start-phase}"
START_PHASE_NUM=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-phase)
            START_PHASE_NUM="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Activate environment
source ~/orchestranet_env/bin/activate 2>/dev/null || true

echo "🎼 ═══════════════════════════════════════════════════════════"
echo "🎼   OrchestraNet — Full Training Pipeline"
echo "🎼 ═══════════════════════════════════════════════════════════"
echo ""
echo "   Data:       ${DATA_ROOT}"
echo "   Device:     ${DEVICE}"
echo "   Batch size: ${BATCH_SIZE}"
echo "   Start from: Phase ${START_PHASE_NUM}"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "${PIPELINE_LOG}"
}

# ============================================================================
# Phase 1: Individual Model Pre-training
# ============================================================================
if [ "${START_PHASE_NUM}" -le 1 ]; then
    log "═══ Phase 1: Individual Model Pre-training ═══"

    # Pre-training epochs per model
    declare -A MODEL_EPOCHS=(
        ["m1"]=50 ["m2"]=30 ["m3"]=30
        ["m4"]=30 ["m5"]=20 ["m6"]=30
    )

    for model_id in m1 m2 m3 m4 m5 m6; do
        epochs=${MODEL_EPOCHS[$model_id]}
        log "▶ Training ${model_id} for ${epochs} epochs..."

        python training/train_individual.py \
            --model "${model_id}" \
            --data-root "${DATA_ROOT}" \
            --epochs "${epochs}" \
            --batch-size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --save-dir "${SAVE_DIR}/individual" \
            --log-dir "${LOG_DIR}/individual" \
            --num-workers "${NUM_WORKERS}" \
            --use-ema \
            2>&1 | tee -a "${PIPELINE_LOG}"

        log "   ✅ ${model_id} pre-training complete"
        echo ""
    done

    log "✅ Phase 1 complete: All individual models pre-trained"
    echo ""
fi

# ============================================================================
# Phase 2: Joint End-to-End Training
# ============================================================================
if [ "${START_PHASE_NUM}" -le 2 ]; then
    log "═══ Phase 2: Joint Training ═══"

    python training/train_joint.py \
        --data-root "${DATA_ROOT}" \
        --epochs 100 \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --save-dir "${SAVE_DIR}" \
        --log-dir "${LOG_DIR}/joint" \
        --num-workers "${NUM_WORKERS}" \
        --load-individual "${SAVE_DIR}/individual" \
        --use-ema \
        --warmup-epochs 5 \
        --save-freq 5 \
        2>&1 | tee -a "${PIPELINE_LOG}"

    log "✅ Phase 2 complete: Joint training done"
    echo ""
fi

# ============================================================================
# Phase 3: Router RL Fine-tuning
# ============================================================================
if [ "${START_PHASE_NUM}" -le 3 ]; then
    log "═══ Phase 3: Router RL Training ═══"

    # Use best joint checkpoint
    JOINT_CKPT="${SAVE_DIR}/orchestranet_best.pt"
    if [ ! -f "${JOINT_CKPT}" ]; then
        JOINT_CKPT="${SAVE_DIR}/orchestranet_final.pt"
    fi

    python training/train_router.py \
        --weights "${JOINT_CKPT}" \
        --data-root "${DATA_ROOT}" \
        --epochs 20 \
        --batch-size 8 \
        --device "${DEVICE}" \
        --save-dir "${SAVE_DIR}/router" \
        --log-dir "${LOG_DIR}/router" \
        --num-workers "${NUM_WORKERS}" \
        2>&1 | tee -a "${PIPELINE_LOG}"

    log "✅ Phase 3 complete: Router training done"
    echo ""
fi

# ============================================================================
# Phase 4: Evaluation
# ============================================================================
if [ "${START_PHASE_NUM}" -le 4 ]; then
    log "═══ Phase 4: Evaluation ═══"

    EVAL_CKPT="${SAVE_DIR}/orchestranet_best.pt"
    if [ ! -f "${EVAL_CKPT}" ]; then
        EVAL_CKPT="${SAVE_DIR}/orchestranet_final.pt"
    fi

    python training/evaluate.py \
        --weights "${EVAL_CKPT}" \
        --data-root "${DATA_ROOT}" \
        --device "${DEVICE}" \
        --save-results "./results" \
        2>&1 | tee -a "${PIPELINE_LOG}"

    # Ablation studies (disable each model one at a time)
    for ablate_model in m2 m3 m4 m5 m6 m7; do
        log "   Running ablation: disable ${ablate_model}..."
        python training/evaluate.py \
            --weights "${EVAL_CKPT}" \
            --data-root "${DATA_ROOT}" \
            --device "${DEVICE}" \
            --ablate "${ablate_model}" \
            --save-results "./results" \
            --num-images 500 \
            2>&1 | tee -a "${PIPELINE_LOG}"
    done

    log "✅ Phase 4 complete: Evaluation and ablations done"
    echo ""
fi

# ============================================================================
# Phase 5: Export
# ============================================================================
if [ "${START_PHASE_NUM}" -le 5 ]; then
    log "═══ Phase 5: Model Export ═══"

    EXPORT_CKPT="${SAVE_DIR}/orchestranet_best.pt"
    if [ ! -f "${EXPORT_CKPT}" ]; then
        EXPORT_CKPT="${SAVE_DIR}/orchestranet_final.pt"
    fi

    # Full pipeline ONNX
    python -m orchestranet.utils.export \
        --weights "${EXPORT_CKPT}" \
        --format onnx \
        --output-dir ./exports \
        --device cpu \
        2>&1 | tee -a "${PIPELINE_LOG}"

    # Individual model ONNX
    python -m orchestranet.utils.export \
        --weights "${EXPORT_CKPT}" \
        --format individual \
        --output-dir ./exports \
        --device cpu \
        2>&1 | tee -a "${PIPELINE_LOG}"

    log "✅ Phase 5 complete: Models exported"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "🎼 ═══════════════════════════════════════════════════════════"
echo "🎼   Pipeline Complete!"
echo "🎼 ═══════════════════════════════════════════════════════════"
echo ""
echo "📁 Outputs:"
echo "   Checkpoints: ${SAVE_DIR}/"
echo "   Logs:        ${LOG_DIR}/"
echo "   Exports:     ./exports/"
echo "   Results:     ./results/"
echo "   Pipeline log: ${PIPELINE_LOG}"
echo ""
echo "📊 View TensorBoard:"
echo "   tensorboard --logdir ${LOG_DIR} --bind_all"
echo ""
echo "🔄 Sync to S3:"
echo "   bash aws/sync_checkpoints.sh"
