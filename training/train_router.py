"""
Router RL Training for OrchestraNet.

Uses REINFORCE to train the Adaptive Compute Router. Reward balances
accuracy against latency. Router learns which models to activate per scene.

Reward = accuracy_component - latency_penalty + entropy_bonus

Usage:
  python training/train_router.py --weights checkpoints/orchestranet_final.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestranet.orchestrator import OrchestraNet
from orchestranet.data.datasets import COCODetectionDataset
from orchestranet.data.transforms import get_train_transforms
from orchestranet.utils.logger import TrainingLogger, AverageMeter

LATENCY_BUDGET = {"simple": 2.0, "medium": 5.0, "complex": 12.0}
MAX_LATENCY = 15.0


class RouterRLTrainer:
    """
    REINFORCE-based router training.

    Reward components:
      1. Accuracy proxy: detection confidence of M1 outputs (higher = better)
      2. Latency penalty: penalize complex routing when scene is simple
      3. Efficiency bonus: fewer models = faster
    """

    def __init__(self, model, device="cuda", acc_w=1.0, lat_w=0.3, eff_w=0.1):
        self.model = model.to(device)
        self.device = device
        self.acc_w, self.lat_w, self.eff_w = acc_w, lat_w, eff_w

        # Freeze everything except router
        for name, p in self.model.named_parameters():
            p.requires_grad = "router" in name

        self.baseline = 0.0
        self.route_counts = {"simple": 0, "medium": 0, "complex": 0}

    def train_step(self, images, targets):
        images = images.to(self.device)
        targets_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }

        self.model.train()

        # Forward through backbone+FPN (no grad needed)
        with torch.no_grad():
            fpn_features = self.model.fpn(self.model.backbone(images))

        # Router forward (with grad)
        self.model.router.train()
        routing = self.model.router(fpn_features)
        level = routing["routing_level"]
        pred_level = routing["level_idx"]
        complexity = routing["complexity_score"]

        # === Ground-Truth Scene Complexity & Matching Score ===
        B = images.shape[0]
        gt_levels = []
        target_scores = []
        for b in range(B):
            n_objs = int(targets_device["num_objects"][b].item()) if "num_objects" in targets_device else 0
            if n_objs <= 2:
                gt_lvl = 0  # simple
                t_score = 0.15
            else:
                boxes_b = targets_device["boxes"][b][:n_objs]
                widths = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0)
                heights = (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
                areas = widths * heights
                small_count = (areas < (32 * 32)).sum().item()

                has_overlap = False
                if n_objs >= 2:
                    x1 = boxes_b[:, 0]
                    y1 = boxes_b[:, 1]
                    x2 = boxes_b[:, 2]
                    y2 = boxes_b[:, 3]
                    inter_x1 = torch.max(x1.unsqueeze(1), x1.unsqueeze(0))
                    inter_y1 = torch.max(y1.unsqueeze(1), y1.unsqueeze(0))
                    inter_x2 = torch.min(x2.unsqueeze(1), x2.unsqueeze(0))
                    inter_y2 = torch.min(y2.unsqueeze(1), y2.unsqueeze(0))
                    inter_w = (inter_x2 - inter_x1).clamp(min=0)
                    inter_h = (inter_y2 - inter_y1).clamp(min=0)
                    inter_area = inter_w * inter_h
                    diag_mask = torch.eye(n_objs, dtype=torch.bool, device=boxes_b.device)
                    inter_area = inter_area.masked_fill(diag_mask, 0)
                    union_area = areas.unsqueeze(1) + areas.unsqueeze(0) - inter_area
                    ious = inter_area / union_area.clamp(min=1e-6)
                    has_overlap = bool((ious.max() > 0.20).item()) if ious.numel() > 0 else False

                if n_objs >= 8 or (n_objs >= 4 and (small_count >= 2 or has_overlap)):
                    gt_lvl = 2  # complex
                    t_score = 0.85
                else:
                    gt_lvl = 1  # medium
                    t_score = 0.50

            gt_levels.append(gt_lvl)
            target_scores.append(t_score)

        # Match matrix: [gt_level, pred_level]
        # Prevents collapse by penalizing under-allocation on complex scenes
        # and rewarding full ensemble deployment when occlusions/crowds exist.
        match_matrix = [
            [1.00, 0.75, 0.40],  # gt=simple: simple optimal (+1.0), complex is wasteful (-0.6)
            [0.40, 1.00, 0.80],  # gt=medium: simple under-allocates (-0.6), medium optimal
            [0.10, 0.60, 1.30],  # gt=complex: simple severely penalized (-0.9), complex gets bonus!
        ]
        match_scores = [match_matrix[gt_l][pred_level] for gt_l in gt_levels]
        avg_match = sum(match_scores) / len(match_scores)

        # === Compute Reward ===
        # 1. Accuracy proxy with scene-match multiplier
        with torch.no_grad():
            m1_out = self.model.models["m1"](fpn_features)
            obj_scores = torch.sigmoid(m1_out["objectness"]).squeeze(-1)
            topk = min(50, obj_scores.shape[1])
            top_scores = obj_scores.topk(topk, dim=1).values
            accuracy_reward = top_scores.mean().item()

        # 2. Latency penalty
        latency = LATENCY_BUDGET.get(level, MAX_LATENCY)
        latency_reward = 1.0 - latency / MAX_LATENCY

        # 3. Efficiency
        num_models = len(routing["active_models"])
        efficiency_reward = 1.0 - num_models / 7.0

        reward = (
            self.acc_w * (accuracy_reward * avg_match)
            + self.lat_w * latency_reward
            + self.eff_w * efficiency_reward
        )
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)

        # Update baseline (running average)
        self.baseline = 0.99 * self.baseline + 0.01 * reward.item()
        advantage = reward - self.baseline

        # Policy gradient loss
        log_prob = F.log_softmax(
            self.model.router.route_classifier(complexity), dim=-1
        )
        policy_loss = -advantage * log_prob[:, routing["level_idx"]].mean()

        # Complexity alignment loss: guides complexity_estimator to align with ground truth
        target_tensor = torch.tensor(target_scores, device=self.device, dtype=torch.float32).unsqueeze(1)
        complexity_loss = F.mse_loss(complexity, target_tensor)

        # Entropy bonus (encourage balanced exploration and prevent mode collapse)
        probs = F.softmax(
            self.model.router.route_classifier(complexity), dim=-1
        )
        entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()

        # Combined loss
        total_loss = policy_loss + 0.5 * complexity_loss - 0.05 * entropy

        self.route_counts[level] += 1

        return {
            "total_loss": total_loss,
            "reward": reward,
            "accuracy_reward": accuracy_reward,
            "match_score": avg_match,
            "routing_level": level,
            "complexity_score": complexity.mean().item(),
            "entropy": entropy.item(),
        }


def sync_file_to_drive(
    src_path: str | Path,
    drive_dir: str | Path,
    logger: TrainingLogger | None = None,
) -> str | None:
    """Safely copy/sync a file to Google Drive using atomic write."""
    if not drive_dir:
        return None
    try:
        import shutil
        src_path = Path(src_path)
        drive_dir = Path(drive_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)

        dest_path = drive_dir / src_path.name
        tmp_dest = drive_dir / f"{src_path.name}.tmp"

        shutil.copyfile(src_path, tmp_dest)
        try:
            with open(tmp_dest, "a+b") as f:
                f.flush()
                os.fsync(f.fileno())
        except (OSError, IOError):
            pass

        os.replace(tmp_dest, dest_path)
        if logger:
            logger.info(f"  ☁️ Drive sync complete: {dest_path}")
        return str(dest_path)
    except Exception as e:
        msg = f"Drive synchronisation failed for {src_path} -> {drive_dir}: {type(e).__name__}: {e}"
        if logger:
            logger.error(msg)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./checkpoints/router")
    parser.add_argument("--log-dir", default="./logs/router")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--drive-save-dir", default=None,
                        help="Google Drive directory to synchronise checkpoints and logs to")
    parser.add_argument("--acc-weight", type=float, default=1.3,
                        help="Weight for accuracy reward (default: 1.3)")
    parser.add_argument("--lat-weight", type=float, default=0.2,
                        help="Weight for latency penalty (default: 0.2)")
    parser.add_argument("--eff-weight", type=float, default=0.05,
                        help="Weight for efficiency bonus (default: 0.05)")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logger = TrainingLogger(log_dir=args.log_dir, tb_enabled=True)

    logger.info("🎼 OrchestraNet — Router RL Training")
    logger.info("=" * 50)

    model = OrchestraNet(num_classes=80, pretrained_backbone=False)
    state = torch.load(args.weights, map_location=args.device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))

    ann_file = os.path.join(args.data_root, "annotations", "instances_train2017.json")
    train_root = os.path.join(args.data_root, "train2017")
    if not os.path.exists(ann_file):
        for sub in ["coco", "coco/coco"]:
            candidate_ann = os.path.join(args.data_root, sub, "annotations", "instances_train2017.json")
            if os.path.exists(candidate_ann):
                ann_file = candidate_ann
                train_root = os.path.join(args.data_root, sub, "train2017")
                break

    dataset = COCODetectionDataset(
        root=train_root,
        ann_file=ann_file,
        transforms=get_train_transforms(img_size=640),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    trainer = RouterRLTrainer(
        model,
        device=args.device,
        acc_w=args.acc_weight,
        lat_w=args.lat_weight,
        eff_w=args.eff_weight,
    )
    optimizer = torch.optim.Adam(
        [p for p in model.router.parameters() if p.requires_grad],
        lr=args.lr,
    )

    global_step = 0
    for epoch in range(args.epochs):
        trainer.route_counts = {"simple": 0, "medium": 0, "complex": 0}
        reward_meter = AverageMeter("reward")

        for bi, (images, targets) in enumerate(loader):
            optimizer.zero_grad()
            result = trainer.train_step(images, targets)
            result["total_loss"].backward()
            optimizer.step()

            reward_meter.update(result["reward"].item())
            global_step += 1

            logger.log_scalar("router/reward", result["reward"].item(), global_step)
            logger.log_scalar("router/complexity", result["complexity_score"], global_step)
            logger.log_scalar("router/entropy", result["entropy"], global_step)

            if bi % 20 == 0:
                print(
                    f"  [{bi}/{len(loader)}] "
                    f"R={result['reward'].item():.3f} "
                    f"AccR={result['accuracy_reward']:.3f} "
                    f"Match={result['match_score']:.2f} "
                    f"Route={result['routing_level']}"
                )

        model.router.update_temperature(0.85)

        tot = sum(trainer.route_counts.values()) or 1
        dist = {k: f"{v/tot*100:.0f}%" for k, v in trainer.route_counts.items()}
        logger.info(
            f"Epoch {epoch} | Avg R: {reward_meter.avg:.3f} | Dist: {dist}"
        )
        logger.log_epoch(epoch, {"reward": reward_meter.avg, "dist": dist})

    router_save_path = os.path.join(args.save_dir, "router_trained.pt")
    router_state = {"router_state_dict": model.router.state_dict()}
    torch.save(router_state, router_save_path)
    logger.info(f"✅ Saved router weights: {router_save_path}")

    # Also update orchestranet_best.pt if present
    parent_dir = Path(args.save_dir).parent
    for candidate_name in ["orchestranet_best.pt", "orchestranet_final.pt"]:
        cand_path = parent_dir / candidate_name
        if cand_path.exists():
            try:
                ckpt = torch.load(cand_path, map_location="cpu", weights_only=False)
                # Update model_state_dict router keys
                if "model_state_dict" in ckpt:
                    for k, v in model.router.state_dict().items():
                        ckpt["model_state_dict"][f"router.{k}"] = v.cpu()
                if "ema_state_dict" in ckpt:
                    for k, v in model.router.state_dict().items():
                        ckpt["ema_state_dict"][f"router.{k}"] = v.cpu()
                torch.save(ckpt, cand_path)
                logger.info(f"✅ Updated {cand_path} with trained router weights")
            except Exception as e:
                logger.warning(f"Could not update {cand_path}: {e}")

    if args.drive_save_dir:
        sync_file_to_drive(router_save_path, args.drive_save_dir, logger=logger)
        drive_log_dir = Path(args.drive_save_dir).parent / "logs" / "router"
        if logger.json_path.exists():
            sync_file_to_drive(logger.json_path, drive_log_dir, logger=None)
        if logger.log_file_path.exists():
            sync_file_to_drive(logger.log_file_path, drive_log_dir, logger=None)

    logger.flush()
    logger.close()
    logger.info("✅ Router training complete.")


if __name__ == "__main__":
    main()
