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
DATA_ROOT="./data"
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"
DEVICE="cuda"
BATCH_SIZE=32
NUM_WORKERS=4
AMP_DTYPE="bf16"
DRIVE_SAVE_DIR=""
# Auto-detect if Colab Google Drive is mounted
if [ -d "/content/drive/MyDrive/ANVRiksh Project/OrchestraNET" ]; then
    DRIVE_SAVE_DIR="/content/drive/MyDrive/ANVRiksh Project/OrchestraNET"
fi
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
        --amp-dtype)
            AMP_DTYPE="$2"
            shift 2
            ;;
        --drive-save-dir)
            DRIVE_SAVE_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Activate environment
source ~/orchestranet_env/bin/activate 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "🎼 ═══════════════════════════════════════════════════════════"
echo "🎼   OrchestraNet — Full Training Pipeline"
echo "🎼 ═══════════════════════════════════════════════════════════"
echo ""
echo "   Data:       ${DATA_ROOT}"
echo "   Device:     ${DEVICE}"
echo "   Batch size: ${BATCH_SIZE}"
echo "   Start from: Phase ${START_PHASE_NUM}"
if [ -n "${DRIVE_SAVE_DIR}" ]; then
    echo "   Drive Sync: ${DRIVE_SAVE_DIR} (Enabled)"
else
    echo "   Drive Sync: Disabled (Local only)"
fi
echo ""

# Resolve COCO dataset path
COCO_DATA="${DATA_ROOT}"
if [ -d "${DATA_ROOT}/coco/coco" ]; then
    COCO_DATA="${DATA_ROOT}/coco/coco"
elif [ -d "${DATA_ROOT}/coco" ]; then
    COCO_DATA="${DATA_ROOT}/coco"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "${PIPELINE_LOG}"
}

sync_to_drive_all() {
    if [ -n "${DRIVE_SAVE_DIR}" ]; then
        log "☁️ Syncing checkpoints, logs, and results to Google Drive..."
        mkdir -p "${DRIVE_SAVE_DIR}/checkpoints" "${DRIVE_SAVE_DIR}/logs" "${DRIVE_SAVE_DIR}/results"
        cp -r -u "${SAVE_DIR}"/* "${DRIVE_SAVE_DIR}/checkpoints/" 2>/dev/null || true
        cp -r -u "${LOG_DIR}"/* "${DRIVE_SAVE_DIR}/logs/" 2>/dev/null || true
        if [ -d "./results" ]; then
            cp -r -u ./results/* "${DRIVE_SAVE_DIR}/results/" 2>/dev/null || true
        fi
        log "   ✅ Google Drive backup synced"
    fi
}

# ============================================================================
# Phase 1: Individual Model Pre-training
# ============================================================================
if [ "${START_PHASE_NUM}" -le 1 ]; then
    log "═══ Phase 1: Individual Model Pre-training ═══"

    # Pre-training epochs per model (calibrated for convergence)
    declare -A MODEL_EPOCHS=(
        ["m1"]=35 ["m2"]=30 ["m3"]=30
        ["m4"]=30 ["m5"]=20 ["m6"]=30
    )

    for model_id in m1 m2 m3 m4 m5 m6; do
        epochs=${MODEL_EPOCHS[$model_id]}

        # Map correct dataset directory per micro-model
        model_data="${DATA_ROOT}"
        if [ "${model_id}" = "m1" ] || [ "${model_id}" = "m3" ]; then
            model_data="${COCO_DATA}"
        elif [ "${model_id}" = "m2" ] || [ "${model_id}" = "m6" ]; then
            # M2 (Occlusion) & M6 (Amodal) use KINS for real occlusion masks (finishes in ~18 & ~26 mins)
            if [ -d "${DATA_ROOT}/KINS" ]; then
                model_data="${DATA_ROOT}/KINS"
            else
                model_data="${COCO_DATA}"
            fi
        elif [ "${model_id}" = "m4" ]; then
            if [ -d "${DATA_ROOT}/kitti" ]; then
                model_data="${DATA_ROOT}/kitti"
            else
                log "   ⚠️ KITTI depth dataset not found at ${DATA_ROOT}/kitti. Skipping M4 individual pre-training (will use default head weights in joint training)."
                continue
            fi
        elif [ "${model_id}" = "m5" ]; then
            if [ -d "${DATA_ROOT}/places365" ]; then
                model_data="${DATA_ROOT}/places365"
            else
                log "   ⚠️ Places365 dataset not found at ${DATA_ROOT}/places365. Skipping M5 individual pre-training (will use default head weights in joint training)."
                continue
            fi
        fi

        # Check for resume checkpoint (e.g. continuing m1 from epoch 8)
        resume_args=()
        if [ -f "${SAVE_DIR}/individual/${model_id}_latest.pt" ]; then
            resume_args=(--resume "${SAVE_DIR}/individual/${model_id}_latest.pt")
            log "   Found existing checkpoint: ${SAVE_DIR}/individual/${model_id}_latest.pt (Resuming)"
        fi

        # Extra flags for M1 (boost classification loss for fast convergence)
        extra_args=()
        if [ "${model_id}" = "m1" ]; then
            extra_args=(--cls-loss-weight 2.0 --conf-thresh 0.05)
        fi

        # Drive sync args
        drive_args=()
        if [ -n "${DRIVE_SAVE_DIR}" ]; then
            drive_args=(--drive-save-dir "${DRIVE_SAVE_DIR}/checkpoints/individual")
        fi

        log "▶ Training ${model_id} for ${epochs} epochs (data: ${model_data})..."

        python training/train_individual.py \
            --model "${model_id}" \
            --data-root "${model_data}" \
            --epochs "${epochs}" \
            --batch-size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --amp-dtype "${AMP_DTYPE}" \
            --save-dir "${SAVE_DIR}/individual" \
            --log-dir "${LOG_DIR}/individual" \
            --num-workers "${NUM_WORKERS}" \
            --use-ema \
            "${resume_args[@]}" \
            "${extra_args[@]}" \
            "${drive_args[@]}" \
            2>&1 | tee -a "${PIPELINE_LOG}"

        log "   ✅ ${model_id} pre-training complete"
        sync_to_drive_all
        echo ""
    done

    log "✅ Phase 1 complete: All individual models pre-trained"
    sync_to_drive_all
    echo ""
fi

# ============================================================================
# Phase 2: Joint End-to-End Training
# ============================================================================
if [ "${START_PHASE_NUM}" -le 2 ]; then
    log "═══ Phase 2: Joint Training ═══"

    joint_data="${COCO_DATA}"

    joint_drive_args=()
    if [ -n "${DRIVE_SAVE_DIR}" ]; then
        joint_drive_args=(--drive-save-dir "${DRIVE_SAVE_DIR}/checkpoints")
    fi

    joint_resume_args=()
    if [ -f "${SAVE_DIR}/orchestranet_latest.pt" ]; then
        joint_resume_args=(--resume "${SAVE_DIR}/orchestranet_latest.pt")
        log "   Found existing joint checkpoint: ${SAVE_DIR}/orchestranet_latest.pt (Resuming)"
    elif [ -f "${SAVE_DIR}/orchestranet_best.pt" ]; then
        joint_resume_args=(--resume "${SAVE_DIR}/orchestranet_best.pt")
        log "   Found existing joint checkpoint: ${SAVE_DIR}/orchestranet_best.pt (Resuming)"
    else
        latest_epoch_ckpt=$(ls -v "${SAVE_DIR}"/orchestranet_epoch*.pt 2>/dev/null | tail -n 1)
        if [ -n "${latest_epoch_ckpt}" ] && [ -f "${latest_epoch_ckpt}" ]; then
            joint_resume_args=(--resume "${latest_epoch_ckpt}")
            log "   Found existing joint checkpoint: ${latest_epoch_ckpt} (Resuming)"
        fi
    fi

    python training/train_joint.py \
        --data-root "${joint_data}" \
        --epochs 35 \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --amp-dtype "${AMP_DTYPE}" \
        --save-dir "${SAVE_DIR}" \
        --log-dir "${LOG_DIR}/joint" \
        --num-workers "${NUM_WORKERS}" \
        --load-individual "${SAVE_DIR}/individual" \
        --use-ema \
        --warmup-epochs 3 \
        --save-freq 5 \
        "${joint_resume_args[@]}" \
        "${joint_drive_args[@]}" \
        2>&1 | tee -a "${PIPELINE_LOG}"

    log "✅ Phase 2 complete: Joint training done"
    sync_to_drive_all
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

    router_drive_args=()
    if [ -n "${DRIVE_SAVE_DIR}" ]; then
        router_drive_args=(--drive-save-dir "${DRIVE_SAVE_DIR}/checkpoints/router")
    fi

    python training/train_router.py \
        --weights "${JOINT_CKPT}" \
        --data-root "${COCO_DATA}" \
        --epochs 20 \
        --batch-size 8 \
        --device "${DEVICE}" \
        --save-dir "${SAVE_DIR}/router" \
        --log-dir "${LOG_DIR}/router" \
        --num-workers "${NUM_WORKERS}" \
        --acc-weight 1.3 \
        --lat-weight 0.2 \
        "${router_drive_args[@]}" \
        2>&1 | tee -a "${PIPELINE_LOG}"

    log "✅ Phase 3 complete: Router training done"
    sync_to_drive_all
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
        --data-root "${COCO_DATA}" \
        --device "${DEVICE}" \
        --save-results "./results" \
        2>&1 | tee -a "${PIPELINE_LOG}"

    # Ablation studies (disable each model one at a time)
    for ablate_model in m2 m3 m4 m5 m6 m7; do
        log "   Running ablation: disable ${ablate_model}..."
        python training/evaluate.py \
            --weights "${EVAL_CKPT}" \
            --data-root "${COCO_DATA}" \
            --device "${DEVICE}" \
            --ablate "${ablate_model}" \
            --save-results "./results" \
            --num-images 500 \
            2>&1 | tee -a "${PIPELINE_LOG}"
    done

    log "✅ Phase 4 complete: Evaluation and ablations done"
    sync_to_drive_all
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
    sync_to_drive_all
    echo ""
fi

# Final overall Drive sync
sync_to_drive_all

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
if [ -n "${DRIVE_SAVE_DIR}" ]; then
    echo "   Drive Backup: ${DRIVE_SAVE_DIR}/"
fi
echo ""
echo "📊 View TensorBoard:"
echo "   tensorboard --logdir ${LOG_DIR} --bind_all"
echo ""
echo "🔄 Continuous Sync to Google Drive:"
echo "   bash aws/sync_gdrive.sh --watch"
echo ""
echo "🔄 Sync to S3 (Optional):"
echo "   bash aws/sync_checkpoints.sh"
