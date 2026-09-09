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
        complexity = routing["complexity_score"]

        # === Compute Reward ===
        # 1. Accuracy proxy: run M1 and measure detection quality
        with torch.no_grad():
            m1_out = self.model.models["m1"](fpn_features)
            obj_scores = torch.sigmoid(m1_out["objectness"]).squeeze(-1)
            # Use mean top-k objectness as quality proxy
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
            self.acc_w * accuracy_reward
            + self.lat_w * latency_reward
            + self.eff_w * efficiency_reward
        )
        reward = torch.tensor(reward, device=self.device)

        # Update baseline (running average)
        self.baseline = 0.99 * self.baseline + 0.01 * reward.item()
        advantage = reward - self.baseline

        # Policy gradient loss
        log_prob = F.log_softmax(
            self.model.router.route_classifier(complexity), dim=-1
        )
        policy_loss = -advantage * log_prob[:, routing["level_idx"]].mean()

        # Entropy bonus (encourage exploration)
        probs = F.softmax(
            self.model.router.route_classifier(complexity), dim=-1
        )
        entropy = -(probs * probs.log().clamp(min=-10)).sum(-1).mean()

        total_loss = policy_loss - 0.01 * entropy

        self.route_counts[level] += 1

        return {
            "total_loss": total_loss,
            "reward": reward,
            "accuracy_reward": accuracy_reward,
            "routing_level": level,
            "complexity_score": complexity.mean().item(),
            "entropy": entropy.item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./checkpoints/router")
    parser.add_argument("--log-dir", default="./logs/router")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logger = TrainingLogger(log_dir=args.log_dir, tb_enabled=True)

    logger.info("🎼 OrchestraNet — Router RL Training")
    logger.info("=" * 50)

    model = OrchestraNet(num_classes=80, pretrained_backbone=False)
    state = torch.load(args.weights, map_location=args.device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))

    dataset = COCODetectionDataset(
        root=os.path.join(args.data_root, "train2017"),
        ann_file=os.path.join(args.data_root, "annotations", "instances_train2017.json"),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    trainer = RouterRLTrainer(model, device=args.device)
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
            model.router.update_temperature(0.99)

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
                    f"Route={result['routing_level']}"
                )

        tot = sum(trainer.route_counts.values()) or 1
        dist = {k: f"{v/tot*100:.0f}%" for k, v in trainer.route_counts.items()}
        logger.info(
            f"Epoch {epoch} | Avg R: {reward_meter.avg:.3f} | Dist: {dist}"
        )
        logger.log_epoch(epoch, {"reward": reward_meter.avg, "dist": dist})

    torch.save(
        {"router_state_dict": model.router.state_dict()},
        os.path.join(args.save_dir, "router_trained.pt"),
    )
    logger.flush()
    logger.close()
    logger.info("✅ Router training complete.")


if __name__ == "__main__":
    main()
