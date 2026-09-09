"""
Individual Micro-Model Pre-training Script for OrchestraNet.

Pre-trains each micro-model on its dedicated task before joint fine-tuning.
This establishes strong per-task baselines and speeds up joint convergence.

Supported models:
  m1 — Object detection on COCO
  m2 — Occlusion prediction on COCOA/KINS
  m3 — Small object detection on VisDrone
  m4 — Depth estimation on NYU Depth V2
  m5 — Scene classification on Places365
  m6 — Amodal completion on COCOA
  m7 — Confidence calibration (post-hoc, no pre-training needed)

Usage:
  python training/train_individual.py --model m1 --data-root ./data/coco --epochs 50
  python training/train_individual.py --model m2 --data-root ./data/cocoa --epochs 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.models import (
    M1PrimaryDetector, M2OcclusionAnalyzer, M3SmallObjectEnhancer,
    M4DepthEstimator, M5SemanticContext, M6AmodalCompleter,
)
from orchestranet.data.datasets import COCODetectionDataset
from orchestranet.data.synthetic_occlusion import SyntheticOcclusionGenerator
from orchestranet.data.transforms import get_train_transforms, get_val_transforms
from orchestranet.losses.self_supervised_loss import (
    SelfSupervisedOcclusionLoss, OcclusionMaskGenerator,
)
from orchestranet.utils.config import Config
from orchestranet.utils.ema import ModelEMA
from orchestranet.utils.logger import TrainingLogger, AverageMeter


MODEL_REGISTRY = {
    "m1": {"cls": M1PrimaryDetector, "task": "Object Detection", "dataset": "COCO"},
    "m2": {"cls": M2OcclusionAnalyzer, "task": "Occlusion Prediction", "dataset": "COCOA"},
    "m3": {"cls": M3SmallObjectEnhancer, "task": "Small Object Enhancement", "dataset": "VisDrone"},
    "m4": {"cls": M4DepthEstimator, "task": "Depth Estimation", "dataset": "NYU Depth V2"},
    "m5": {"cls": M5SemanticContext, "task": "Scene Classification", "dataset": "Places365"},
    "m6": {"cls": M6AmodalCompleter, "task": "Amodal Completion", "dataset": "COCOA"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="OrchestraNet Individual Model Training")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="Which model to train")
    parser.add_argument("--config", default="configs/training/individual_pretrain.yaml")
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./checkpoints/individual")
    parser.add_argument("--log-dir", default="./logs/individual")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone during training")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true", default=True)
    return parser.parse_args()


class IndividualTrainer:
    """
    Trains a single micro-model with shared backbone feature extraction.

    Architecture during individual training:
      Input → Backbone → FPN → [Single Micro-Model] → Task Loss
    """

    def __init__(self, model_id, device, fpn_channels=128, freeze_backbone=False):
        self.model_id = model_id
        self.device = device
        info = MODEL_REGISTRY[model_id]

        # Shared backbone + FPN
        self.backbone = MobileNetV4Backbone(pretrained=True).to(device)
        backbone_channels = self.backbone.get_out_channels()
        self.fpn = LightweightFPN(
            in_channels=backbone_channels, out_channels=fpn_channels
        ).to(device)

        # Target micro-model
        self.model = info["cls"](in_channels=fpn_channels).to(device)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("   🔒 Backbone frozen")

        # Collect all trainable parameters
        self.all_params = (
            list(self.backbone.parameters())
            + list(self.fpn.parameters())
            + list(self.model.parameters())
        )

        # For M2: self-supervised components
        self.ss_loss = None
        self.mask_gen = None
        if model_id == "m2":
            self.ss_loss = SelfSupervisedOcclusionLoss().to(device)
            self.mask_gen = OcclusionMaskGenerator().to(device)

    def train_step(self, images, targets):
        """Single training step."""
        images = images.to(self.device)

        # Feature extraction
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)

        # Model forward
        predictions = self.model(fpn_features)

        # Task loss
        targets_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }
        losses = self.model.get_loss(predictions, targets_device)

        # Self-supervised loss for M2
        if self.ss_loss is not None and self.mask_gen is not None:
            masked_feats, gt_mask = self.mask_gen(fpn_features[0])
            ss_pred = self.model([masked_feats] + fpn_features[1:])
            ss_losses = self.ss_loss(ss_pred["occlusion_map"], gt_mask)
            losses["ss_loss"] = ss_losses["total"]
            losses["total_loss"] = losses["total_loss"] + 0.5 * ss_losses["total"]

        return losses

    def count_parameters(self):
        total = sum(p.numel() for p in self.all_params)
        trainable = sum(p.numel() for p in self.all_params if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        fpn = sum(p.numel() for p in self.fpn.parameters())
        model = sum(p.numel() for p in self.model.parameters())
        return {
            "total": total, "trainable": trainable,
            "backbone": backbone, "fpn": fpn, "model_head": model,
        }


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load config (merge with CLI args)
    cfg = Config.from_yaml(args.config).merge_args(args)

    info = MODEL_REGISTRY[args.model]

    # Setup logger
    log_dir = os.path.join(args.log_dir, args.model)
    logger = TrainingLogger(log_dir=log_dir, tb_enabled=True)

    logger.info("🎼 OrchestraNet — Individual Model Training")
    logger.info("=" * 60)
    logger.info(f"   Model:   {args.model} ({info['task']})")
    logger.info(f"   Dataset: {info['dataset']}")
    logger.info(f"   Device:  {args.device}")

    # Build trainer
    trainer = IndividualTrainer(
        args.model, args.device,
        freeze_backbone=args.freeze_backbone,
    )
    params = trainer.count_parameters()
    logger.info(f"\n📊 Parameters:")
    logger.info(f"   Backbone:   {params['backbone']:>10,}")
    logger.info(f"   FPN:        {params['fpn']:>10,}")
    logger.info(f"   Model Head: {params['model_head']:>10,}")
    logger.info(f"   Trainable:  {params['trainable']:>10,}")

    # Dataset
    occ_aug = SyntheticOcclusionGenerator() if args.model == "m2" else None
    train_transforms = get_train_transforms(img_size=640)
    is_kins = "kins" in args.data_root.lower()
    root_path = os.path.join(args.data_root, "training/image_2" if is_kins else "train2017")
    ann_path = os.path.join(args.data_root, "update_train_2020.json" if is_kins else "annotations/instances_train2017.json")

    dataset = COCODetectionDataset(
        root=root_path,
        ann_file=ann_path,
        transforms=train_transforms,
        occlusion_aug=occ_aug,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Optimizer
    trainable_params = [p for p in trainer.all_params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Scheduler with warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs * len(loader)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs * len(loader)],
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    # EMA
    ema = None
    if args.use_ema:
        ema = ModelEMA(trainer.model)

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        logger.info(f"   Resumed from epoch {start_epoch}")

    logger.info(f"\n🚀 Training {args.model} for {args.epochs} epochs...")

    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        trainer.backbone.train()
        trainer.fpn.train()
        trainer.model.train()

        loss_meter = AverageMeter("loss")
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(loader):
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
                losses = trainer.train_step(images, targets)

            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update EMA
            if ema:
                ema.update(trainer.model)

            loss_val = losses["total_loss"].item()
            loss_meter.update(loss_val, images.shape[0])
            global_step += 1

            # Log to TensorBoard
            logger.log_scalar(f"train/{args.model}/total_loss", loss_val, global_step)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and k != "total_loss":
                    logger.log_scalar(f"train/{args.model}/{k}", v.item(), global_step)
            logger.log_lr(optimizer.param_groups[0]["lr"], global_step)

            if batch_idx % args.print_freq == 0:
                loss_strs = " | ".join(
                    f"{k}: {v.item():.4f}" for k, v in losses.items()
                    if isinstance(v, torch.Tensor)
                )
                print(f"  [{batch_idx}/{len(loader)}] {loss_strs}")

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{args.epochs} | Avg Loss: {loss_meter.avg:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.1f}s"
        )
        logger.log_epoch(epoch, {"avg_loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]})

        # Save checkpoint
        is_best = loss_meter.avg < best_loss
        best_loss = min(best_loss, loss_meter.avg)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1 or is_best:
            ckpt_data = {
                "epoch": epoch,
                "model_id": args.model,
                "model_state_dict": trainer.model.state_dict(),
                "backbone_state_dict": trainer.backbone.state_dict(),
                "fpn_state_dict": trainer.fpn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": loss_meter.avg,
                "best_loss": best_loss,
            }
            if ema:
                ckpt_data["ema_state_dict"] = ema.state_dict()

            path = os.path.join(args.save_dir, f"{args.model}_epoch{epoch}.pt")
            torch.save(ckpt_data, path)
            logger.info(f"  💾 Saved: {path}")

            if is_best:
                best_path = os.path.join(args.save_dir, f"{args.model}_best.pt")
                torch.save(ckpt_data, best_path)
                logger.info(f"  🏆 New best model: {best_path}")

    logger.flush()
    logger.close()
    logger.info(f"\n✅ {args.model} pre-training complete!")


if __name__ == "__main__":
    main()
