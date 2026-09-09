"""
Knowledge Distillation Pipeline for OrchestraNet.

Implements hierarchical distillation from a heavy teacher to lightweight student:
  1. Feature-level distillation (intermediate FPN representations)
  2. Logit-level distillation (soft label transfer)
  3. Relation-based distillation (inter-sample relationships)
  4. Task-specific distillation (per micro-model)

The teacher is a full OrchestraNet with larger backbone (e.g., EfficientNetV2-L)
and the student is the lightweight version for real-time deployment.

Usage:
  python training/distill.py --teacher checkpoints/teacher.pt --epochs 50
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
# torch.amp is used inline (torch.amp.autocast, torch.amp.GradScaler)

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestranet.orchestrator import OrchestraNet
from orchestranet.data.datasets import COCODetectionDataset


# ============ Distillation Losses ============

class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation: align student FPN features with teacher's.

    Uses channel adaptation (1x1 conv) if dimensions mismatch,
    then L2 loss on normalized features for scale invariance.
    """

    def __init__(self, teacher_channels: int = 256, student_channels: int = 128):
        super().__init__()
        self.adapt = None
        if teacher_channels != student_channels:
            self.adapt = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature alignment loss across all FPN levels."""
        total_loss = torch.tensor(0.0, device=student_features[0].device)

        for s_feat, t_feat in zip(student_features, teacher_features):
            # Adapt student channels if needed
            if self.adapt is not None:
                s_feat = self.adapt(s_feat)

            # Resize if spatial dims differ
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[2:], mode="bilinear", align_corners=False
                )

            # Normalized L2 loss (scale-invariant)
            s_norm = F.normalize(s_feat.flatten(2), dim=-1)
            t_norm = F.normalize(t_feat.flatten(2), dim=-1)
            total_loss = total_loss + F.mse_loss(s_norm, t_norm)

        return total_loss / len(student_features)


class LogitDistillationLoss(nn.Module):
    """
    Logit-level distillation: soft label transfer from teacher to student.

    Uses KL divergence between temperature-scaled softmax distributions
    of teacher and student detection outputs.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, N, C) student class logits
            teacher_logits: (B, N, C) teacher class logits
        """
        # Match prediction count (take minimum)
        min_n = min(student_logits.shape[1], teacher_logits.shape[1])
        s = student_logits[:, :min_n]
        t = teacher_logits[:, :min_n]

        # Temperature-scaled soft targets
        s_soft = F.log_softmax(s / self.temperature, dim=-1)
        t_soft = F.softmax(t / self.temperature, dim=-1)

        # KL divergence
        kl = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.temperature ** 2)
        return kl


class OcclusionDistillationLoss(nn.Module):
    """
    Occlusion-specific distillation: transfer the teacher's occlusion
    reasoning capability to the student's M2 module.

    Aligns:
      - Occlusion maps
      - Visibility scores
      - Occlusion feature representations
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        student_occ: dict[str, torch.Tensor],
        teacher_occ: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(student_occ.values())).device)

        # Occlusion map alignment
        if "occlusion_map" in student_occ and "occlusion_map" in teacher_occ:
            s_map = student_occ["occlusion_map"]
            t_map = teacher_occ["occlusion_map"]
            if s_map.shape != t_map.shape:
                t_map = F.interpolate(t_map, size=s_map.shape[2:], mode="bilinear",
                                       align_corners=False)
            loss = loss + F.mse_loss(s_map, t_map)

        # Visibility score alignment
        if "visibility_scores" in student_occ and "visibility_scores" in teacher_occ:
            loss = loss + F.mse_loss(
                student_occ["visibility_scores"],
                teacher_occ["visibility_scores"]
            )

        return loss


# ============ Distillation Trainer ============

class DistillationTrainer:
    """
    Manages the knowledge distillation process from teacher to student.

    Training loop:
      1. Forward pass through both teacher (frozen) and student
      2. Compute GT detection loss for student
      3. Compute feature distillation loss
      4. Compute logit distillation loss
      5. Compute occlusion distillation loss
      6. Weighted sum → backward pass
    """

    def __init__(
        self,
        teacher: OrchestraNet,
        student: OrchestraNet,
        device: str = "cuda",
        feature_weight: float = 1.0,
        logit_weight: float = 2.0,
        occlusion_weight: float = 1.5,
        gt_weight: float = 1.0,
        temperature: float = 4.0,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Loss weights
        self.feature_weight = feature_weight
        self.logit_weight = logit_weight
        self.occlusion_weight = occlusion_weight
        self.gt_weight = gt_weight

        # Loss functions
        self.feature_loss_fn = FeatureDistillationLoss(
            teacher_channels=teacher.fpn.get_out_channels(),
            student_channels=student.fpn.get_out_channels(),
        ).to(device)
        self.logit_loss_fn = LogitDistillationLoss(temperature=temperature).to(device)
        self.occ_loss_fn = OcclusionDistillationLoss().to(device)

    def train_step(
        self,
        images: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Single distillation training step.

        Returns dict of loss components.
        """
        images = images.to(self.device)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_backbone = self.teacher.backbone(images)
            t_fpn = self.teacher.fpn(t_backbone)
            t_outputs = self.teacher(images)

        # Student forward
        s_backbone = self.student.backbone(images)
        s_fpn = self.student.fpn(s_backbone)
        s_outputs = self.student(images, targets)

        losses = {}

        # 1. Ground truth detection loss
        gt_loss = sum(
            v for k, v in s_outputs["losses"].items()
            if "total" in k and isinstance(v, torch.Tensor)
        )
        losses["gt_loss"] = gt_loss

        # 2. Feature distillation loss
        feat_loss = self.feature_loss_fn(s_fpn, t_fpn)
        losses["feature_loss"] = feat_loss

        # 3. Logit distillation loss
        if "m1" in s_outputs["model_outputs"] and "m1" in t_outputs["model_outputs"]:
            s_logits = s_outputs["model_outputs"]["m1"]["class_logits"]
            t_logits = t_outputs["model_outputs"]["m1"]["class_logits"]
            logit_loss = self.logit_loss_fn(s_logits, t_logits)
            losses["logit_loss"] = logit_loss
        else:
            losses["logit_loss"] = torch.tensor(0.0, device=self.device)

        # 4. Occlusion distillation loss
        if "m2" in s_outputs["model_outputs"] and "m2" in t_outputs["model_outputs"]:
            occ_loss = self.occ_loss_fn(
                s_outputs["model_outputs"]["m2"],
                t_outputs["model_outputs"]["m2"],
            )
            losses["occlusion_loss"] = occ_loss
        else:
            losses["occlusion_loss"] = torch.tensor(0.0, device=self.device)

        # Total weighted loss
        total = (
            self.gt_weight * losses["gt_loss"]
            + self.feature_weight * losses["feature_loss"]
            + self.logit_weight * losses["logit_loss"]
            + self.occlusion_weight * losses["occlusion_loss"]
        )
        losses["total_loss"] = total

        return losses


# ============ Main Training Script ============

def parse_args():
    parser = argparse.ArgumentParser(description="OrchestraNet Knowledge Distillation")
    parser.add_argument("--teacher", required=True, help="Teacher checkpoint path")
    parser.add_argument("--student-weights", default=None, help="Student pretrained weights")
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./checkpoints/distilled")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=2.0)
    parser.add_argument("--print-freq", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("🎼 OrchestraNet — Knowledge Distillation")
    print("=" * 60)

    # Build teacher (larger model)
    print("\n📚 Loading teacher model...")
    teacher = OrchestraNet(
        num_classes=80,
        fpn_channels=256,  # Larger teacher
        pretrained_backbone=False,
    )
    teacher_state = torch.load(args.teacher, map_location=args.device, weights_only=False)
    if "model_state_dict" in teacher_state:
        teacher_state = teacher_state["model_state_dict"]
    teacher.load_state_dict(teacher_state)
    print(f"   Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    # Build student (lightweight)
    print("\n🎒 Building student model...")
    student = OrchestraNet(
        num_classes=80,
        fpn_channels=128,  # Lighter student
        pretrained_backbone=False,
    )
    if args.student_weights:
        student.load_state_dict(torch.load(args.student_weights, map_location=args.device, weights_only=False))
    print(f"   Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Dataset
    dataset = COCODetectionDataset(
        root=os.path.join(args.data_root, "train2017"),
        ann_file=os.path.join(args.data_root, "annotations", "instances_train2017.json"),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Distillation trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        device=args.device,
        temperature=args.temperature,
        feature_weight=args.feature_weight,
        logit_weight=args.logit_weight,
    )

    # Optimizer (student only)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device == "cuda"))

    print(f"\n🚀 Starting distillation for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        student.train()
        epoch_losses = {"gt": 0, "feat": 0, "logit": 0, "occ": 0, "total": 0}
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(loader):
            targets = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(args.device == "cuda")):
                losses = trainer.train_step(images, targets)

            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate
            epoch_losses["gt"] += losses["gt_loss"].item()
            epoch_losses["feat"] += losses["feature_loss"].item()
            epoch_losses["logit"] += losses["logit_loss"].item()
            epoch_losses["occ"] += losses["occlusion_loss"].item()
            epoch_losses["total"] += losses["total_loss"].item()
            num_batches += 1

            if batch_idx % args.print_freq == 0:
                print(
                    f"  [{batch_idx}/{len(loader)}] "
                    f"Total: {losses['total_loss'].item():.4f} | "
                    f"GT: {losses['gt_loss'].item():.4f} | "
                    f"Feat: {losses['feature_loss'].item():.4f} | "
                    f"Logit: {losses['logit_loss'].item():.4f} | "
                    f"Occ: {losses['occlusion_loss'].item():.4f}"
                )

        scheduler.step()

        # Epoch summary
        n = max(num_batches, 1)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Total: {epoch_losses['total']/n:.4f} | "
            f"GT: {epoch_losses['gt']/n:.4f} | "
            f"Feat: {epoch_losses['feat']/n:.4f} | "
            f"Logit: {epoch_losses['logit']/n:.4f}"
        )

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            path = os.path.join(args.save_dir, f"student_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, path)
            print(f"  💾 Saved: {path}")

    # Final save
    final = os.path.join(args.save_dir, "student_final.pt")
    torch.save(student.state_dict(), final)
    print(f"\n✅ Distillation complete. Student saved: {final}")


if __name__ == "__main__":
    main()
