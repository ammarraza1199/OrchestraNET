"""
Diagnostic script for OrchestraNet checkpoints, training curves, and logs.

Usage:
  python scripts/diagnose_server_checkpoints.py
"""

import os
import sys
import glob
import json
from pathlib import Path
import torch

# Ensure UTF-8 output on all consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def audit_checkpoints():
    print("=" * 70)
    print("1. CHECKPOINT AUDIT (./checkpoints)")
    print("=" * 70)
    
    ckpt_files = sorted(glob.glob("checkpoints/*.pt") + glob.glob("checkpoints/**/*.pt"))
    if not ckpt_files:
        print("  ⚠️ No checkpoint files found in ./checkpoints")
        return

    for path in ckpt_files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", "N/A")
            train_loss = ckpt.get("avg_loss", ckpt.get("loss", "N/A"))
            val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", "N/A"))
            has_ema = "ema_state_dict" in ckpt
            has_model = "model_state_dict" in ckpt

            # Count parameters / check components
            components = []
            if has_model:
                keys = list(ckpt["model_state_dict"].keys())
                for comp in ["backbone", "fpn", "router", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]:
                    if any(k.startswith(f"{comp}.") or k.startswith(f"models.{comp}.") for k in keys):
                        components.append(comp)

            t_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else str(train_loss)
            v_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)

            print(f"[*] {path}")
            print(f"   Size: {size_mb:.1f} MB | Epoch: {epoch} | Train Loss: {t_loss_str} | Val Loss: {v_loss_str} | EMA: {has_ema}")
            if components:
                print(f"   Components: {', '.join(components)}")
            print("-" * 70)
        except Exception as e:
            print(f"[*] {path} ({size_mb:.1f} MB) - Error reading: {e}")
            print("-" * 70)


def audit_joint_log():
    print("\n" + "=" * 70)
    print("2. JOINT TRAINING LOG (logs/joint/training_log.jsonl)")
    print("=" * 70)

    candidates = [
        Path("logs/joint/training_log.jsonl"),
        Path("logs/training_log.jsonl"),
    ]
    log_file = None
    for c in candidates:
        if c.exists():
            log_file = c
            break

    if not log_file:
        print("  [!] No joint training_log.jsonl found.")
        return

    try:
        lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        print(f"  Total epochs recorded: {len(lines)}")
        print(f"  {'Epoch':6s} | {'Train Loss':12s} | {'Val Loss':16s} | {'LR':12s}")
        print("  " + "-" * 55)
        for entry in lines:
            ep = entry.get("epoch", "?")
            t_loss = entry.get("train_loss", entry.get("avg_loss", "N/A"))
            v_loss = entry.get("val_loss", "N/A")
            lr = entry.get("lr", "N/A")
            t_str = f"{t_loss:.4f}" if isinstance(t_loss, (int, float)) else str(t_loss)
            v_str = f"{v_loss:.4f}" if isinstance(v_loss, (int, float)) else str(v_loss)
            lr_str = f"{lr:.6f}" if isinstance(lr, (int, float)) else str(lr)
            print(f"  {str(ep):6s} | {t_str:12s} | {v_str:16s} | {lr_str:12s}")
    except Exception as e:
        print(f"  Error parsing joint log: {e}")


def audit_m1_log():
    print("\n" + "=" * 70)
    print("3. M1 STANDALONE TRAINING LOG (logs/individual/m1/training_log.jsonl)")
    print("=" * 70)

    log_file = Path("logs/individual/m1/training_log.jsonl")
    if not log_file.exists():
        for candidate in Path("logs").glob("**/training_log.jsonl"):
            if "m1" in str(candidate):
                log_file = candidate
                break

    if not log_file or not log_file.exists():
        print("  [!] No M1 training_log.jsonl found.")
        return

    try:
        lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        print(f"  Total epochs recorded: {len(lines)}")
        print(f"  {'Epoch':6s} | {'Total Loss':12s} | {'mAP@50':10s} | {'mAP@50:95':10s} | {'AP_large':10s}")
        print("  " + "-" * 60)
        for entry in lines:
            ep = entry.get("epoch", "?")
            t_loss = entry.get("total_loss", entry.get("avg_loss", "N/A"))
            map50 = entry.get("mAP@50", "N/A")
            map95 = entry.get("mAP@50:95", "N/A")
            apl = entry.get("AP_large", "N/A")
            t_str = f"{t_loss:.4f}" if isinstance(t_loss, (int, float)) else str(t_loss)
            m50_str = f"{map50:.4f}" if isinstance(map50, (int, float)) else str(map50)
            m95_str = f"{map95:.4f}" if isinstance(map95, (int, float)) else str(map95)
            apl_str = f"{apl:.4f}" if isinstance(apl, (int, float)) else str(apl)
            print(f"  {str(ep):6s} | {t_str:12s} | {m50_str:10s} | {m95_str:10s} | {apl_str:10s}")
    except Exception as e:
        print(f"  Error parsing M1 log: {e}")


def audit_results():
    print("\n" + "=" * 70)
    print("4. EVALUATION RESULTS AUDIT (./results/*.json)")
    print("=" * 70)
    
    result_files = sorted(glob.glob("results/*.json"))
    if not result_files:
        print("  [!] No result json files found in ./results")
        return

    for path in result_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m50 = data.get("mAP@50", "N/A")
            m95 = data.get("mAP@50:95", "N/A")
            fps = data.get("speed", {}).get("fps", "N/A") if isinstance(data.get("speed"), dict) else "N/A"
            routing = data.get("routing_distribution", {})
            m50_str = f"{m50:.4f}" if isinstance(m50, (int, float)) else str(m50)
            m95_str = f"{m95:.4f}" if isinstance(m95, (int, float)) else str(m95)
            fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
            print(f"[*] {path}")
            print(f"   mAP@50: {m50_str} | mAP@50:95: {m95_str} | FPS: {fps_str}")
            if routing:
                print(f"   Routing: {routing}")
            print("-" * 70)
        except Exception as e:
            print(f"[*] {path} - Error reading: {e}")


def main():
    print("=== OrchestraNet Server Diagnostic Report ===")
    print(f"Working Directory: {os.getcwd()}")
    audit_checkpoints()
    audit_joint_log()
    audit_m1_log()
    audit_results()
    print("\n[+] Diagnostic extraction complete.")


if __name__ == "__main__":
    main()
