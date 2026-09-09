"""
Joint Training Script for OrchestraNet.

Trains the full OrchestraNet pipeline end-to-end with:
  1. All micro-models jointly on COCO
  2. Self-supervised occlusion pretext task
  3. Occlusion curriculum (progressive difficulty)
  4. Router training with Gumbel-Softmax
  5. TensorBoard logging + EMA + validation

Usage:
  python training/train_joint.py --config configs/training/joint_finetune.yaml
  python training/train_joint.py --data-root ./data/coco --epochs 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestranet.orchestrator import OrchestraNet
from orchestranet.data.datasets import COCODetectionDataset
from orchestranet.data.synthetic_occlusion import SyntheticOcclusionGenerator
from orchestranet.data.curriculum import OcclusionCurriculum
from orchestranet.data.transforms import get_train_transforms, get_val_transforms
from orchestranet.losses.self_supervised_loss import (
    SelfSupervisedOcclusionLoss,
    OcclusionMaskGenerator,
)
from orchestranet.utils.config import Config
from orchestranet.utils.ema import ModelEMA
from orchestranet.utils.logger import TrainingLogger, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="OrchestraNet Joint Training")
    parser.add_argument("--config", default="configs/training/joint_finetune.yaml")
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs/joint")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--load-individual", default=None,
                        help="Directory of individual pretrained checkpoints to load")
    return parser.parse_args()


def build_model(args) -> OrchestraNet:
    """Build the OrchestraNet model."""
    model = OrchestraNet(
        num_classes=80,
        backbone_name="mobilenetv4_hybrid_medium",
        fpn_channels=128,
        pretrained_backbone=True,
    )
    return model.to(args.device)


def load_individual_checkpoints(model: OrchestraNet, ckpt_dir: str, device: str):
    """Load individually pretrained model weights into the full OrchestraNet."""
    ckpt_dir = Path(ckpt_dir)
    loaded = []

    for model_id in ["m1", "m2", "m3", "m4", "m5", "m6"]:
        # Try best checkpoint first, then latest
        for pattern in [f"{model_id}_best.pt", f"{model_id}_epoch*.pt"]:
            matches = sorted(ckpt_dir.glob(pattern))
            if matches:
                ckpt_path = matches[-1]  # Latest
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.models[model_id].load_state_dict(ckpt["model_state_dict"])
                loaded.append(model_id)
                print(f"   ✅ Loaded {model_id} from {ckpt_path}")

                # Load backbone/FPN from M1 (most general)
                if model_id == "m1":
                    if "backbone_state_dict" in ckpt:
                        model.backbone.load_state_dict(ckpt["backbone_state_dict"])
                        print(f"   ✅ Loaded backbone from {ckpt_path}")
                    if "fpn_state_dict" in ckpt:
                        model.fpn.load_state_dict(ckpt["fpn_state_dict"])
                        print(f"   ✅ Loaded FPN from {ckpt_path}")
                break

    if loaded:
        print(f"   📦 Loaded individual checkpoints: {loaded}")
    return loaded


def build_dataset(args, curriculum, is_train=True):
    """Build training/val dataset with occlusion augmentation."""
    is_kins = "kins" in args.data_root.lower()

    if is_train:
        occ_aug = SyntheticOcclusionGenerator(
            max_occlusion_ratio=curriculum.get_params(0)["max_ratio"],
        )
        transforms = get_train_transforms(img_size=640)
        root_path = os.path.join(args.data_root, "training/image_2" if is_kins else "train2017")
        ann_path = os.path.join(args.data_root, "update_train_2020.json" if is_kins else "annotations/instances_train2017.json")
    else:
        occ_aug = None
        transforms = get_val_transforms(img_size=640)
        root_path = os.path.join(args.data_root, "testing/image_2" if is_kins else "val2017")
        ann_path = os.path.join(args.data_root, "update_test_2020.json" if is_kins else "annotations/instances_val2017.json")

    dataset = COCODetectionDataset(
        root=root_path,
        ann_file=ann_path,
        transforms=transforms,
        occlusion_aug=occ_aug,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return dataset, loader


def train_one_epoch(
    model: OrchestraNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    args,
    ss_loss_fn: SelfSupervisedOcclusionLoss,
    mask_generator: OcclusionMaskGenerator,
    curriculum: OcclusionCurriculum,
    logger: TrainingLogger,
    global_step: int,
    ema: ModelEMA | None = None,
):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter("loss")
    use_cuda = args.device == "cuda"

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)
        targets = {k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                   for k, v in targets.items()}

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_cuda):
            # Main forward pass
            outputs = model(images, targets)
            losses = outputs["losses"]

            # Aggregate all model losses
            total_loss = sum(
                v for k, v in losses.items()
                if "total" in k and isinstance(v, torch.Tensor)
            )

            # Self-supervised occlusion loss
            if curriculum.should_apply_occlusion(epoch) and "m2" in outputs["model_outputs"]:
                backbone_features = model.backbone(images)
                fpn_features = model.fpn(backbone_features)
                masked_features, gt_mask = mask_generator(fpn_features[0])
                m2_output = model.models["m2"]([masked_features] + fpn_features[1:])
                ss_losses = ss_loss_fn(m2_output["occlusion_map"], gt_mask)
                total_loss = total_loss + 0.5 * ss_losses["total"]
                losses["ss_total"] = ss_losses["total"]

        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        # EMA update
        if ema:
            ema.update(model)

        loss_val = total_loss.item()
        loss_meter.update(loss_val, images.shape[0])
        global_step += 1

        # Log to TensorBoard
        logger.log_scalar("train/total_loss", loss_val, global_step)
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                logger.log_scalar(f"train/{k}", v.item(), global_step)
        logger.log_lr(optimizer.param_groups[0]["lr"], global_step)

        # Routing stats
        routing = outputs["routing"]
        logger.log_scalar(
            "train/complexity_score",
            routing["complexity_score"].mean().item(),
            global_step,
        )

        if batch_idx % args.print_freq == 0:
            print(
                f"  [{batch_idx}/{len(loader)}] "
                f"Loss: {loss_val:.4f} (avg: {loss_meter.avg:.4f}) | "
                f"Route: {routing['routing_level']} | "
                f"Active: {routing['active_models']}"
            )

    return loss_meter.avg, global_step


@torch.no_grad()
def validate(model: OrchestraNet, loader: DataLoader, device: str) -> dict:
    """Run validation and return metrics."""
    model.eval()
    val_loss_meter = AverageMeter("val_loss")
    route_dist = {"simple": 0, "medium": 0, "complex": 0}

    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in targets.items()}
        outputs = model(images, targets)
        losses = outputs["losses"]

        total_loss = sum(
            v for k, v in losses.items()
            if "total" in k and isinstance(v, torch.Tensor)
        )
        val_loss_meter.update(total_loss.item(), images.shape[0])
        route_dist[outputs["routing"]["routing_level"]] += 1

    return {"val_loss": val_loss_meter.avg, "routing_distribution": route_dist}


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load config
    cfg = Config.from_yaml(args.config).merge_args(args)

    # Setup logger
    logger = TrainingLogger(log_dir=args.log_dir, tb_enabled=True)

    logger.info("=" * 70)
    logger.info("🎼 OrchestraNet — Joint Training")
    logger.info("=" * 70)

    # Build components
    model = build_model(args)
    curriculum = OcclusionCurriculum()

    # Load individual pretrained checkpoints
    if args.load_individual:
        load_individual_checkpoints(model, args.load_individual, args.device)

    # Build datasets
    train_dataset, train_loader = build_dataset(args, curriculum, is_train=True)
    val_dataset, val_loader = build_dataset(args, curriculum, is_train=False)

    # Parameter count
    params = model.count_all_parameters()
    logger.info("\n📊 Model Parameters:")
    for name, counts in params.items():
        logger.info(f"  {name:20s}: {counts['total']:>10,} total, {counts['trainable']:>10,} trainable")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # Warmup + cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01,
        total_iters=args.warmup_epochs * len(train_loader),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(args.epochs - args.warmup_epochs), eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs * len(train_loader)],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    # Self-supervised components
    ss_loss_fn = SelfSupervisedOcclusionLoss().to(args.device)
    mask_generator = OcclusionMaskGenerator().to(args.device)

    # EMA
    ema = ModelEMA(model) if args.use_ema else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        global_step = checkpoint.get("global_step", 0)
        if ema and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info(f"\n🚀 Starting training for {args.epochs} epochs...")
    logger.info(f"   Train dataset: {len(train_dataset)} images")
    logger.info(f"   Val dataset:   {len(val_dataset)} images")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Device: {args.device}")
    logger.info("")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Update curriculum
        curr_params = curriculum.get_params(epoch)
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Curriculum: occ_prob={curr_params['occlusion_prob']:.1f}, "
            f"max_ratio={curr_params['max_ratio']:.1f}"
        )

        # Update router temperature
        model.router.update_temperature(decay=0.95)

        # Train
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, args,
            ss_loss_fn, mask_generator, curriculum, logger, global_step, ema,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Validate
        if ema:
            ema.apply_shadow(model)
        val_metrics = validate(model, val_loader, args.device)
        if ema:
            ema.restore(model)

        val_loss = val_metrics["val_loss"]
        is_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        logger.log_scalar("val/loss", val_loss, epoch)
        logger.log_epoch(epoch, {
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        logger.info(
            f"  → Epoch {epoch} complete | "
            f"Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best or epoch == args.epochs - 1:
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "best_val_loss": best_val_loss,
                "curriculum_params": curr_params,
                "global_step": global_step,
            }
            if ema:
                ckpt_data["ema_state_dict"] = ema.state_dict()

            checkpoint_path = os.path.join(args.save_dir, f"orchestranet_epoch{epoch}.pt")
            torch.save(ckpt_data, checkpoint_path)
            logger.info(f"  💾 Saved checkpoint: {checkpoint_path}")

            if is_best:
                best_path = os.path.join(args.save_dir, "orchestranet_best.pt")
                torch.save(ckpt_data, best_path)
                logger.info(f"  🏆 New best model: {best_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, "orchestranet_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.flush()
    logger.close()
    logger.info(f"\n✅ Training complete. Final model saved: {final_path}")


if __name__ == "__main__":
    main()
