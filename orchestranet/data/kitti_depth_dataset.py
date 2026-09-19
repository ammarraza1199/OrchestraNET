"""
KITTI Monocular Depth Dataset for OrchestraNet M4 Depth Estimator.

Supports:
  - split="train": Sequence-level KITTI ground truth depth maps paired with
                   discovered raw RGB images across standard KITTI layouts.
  - split="val":   Canonical val_selection_cropped (1,000 frames) with robust
                   filename-identity pairing (<seq>_image_<frame>_<cam>.png <->
                   <seq>_groundtruth_depth_<frame>_<cam>.png).
"""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEPTH_SCALE = 256.0
MAX_DEPTH = 80.0


class KITTIDepthDataset(Dataset):
    """
    KITTI Depth Dataset loader for OrchestraNet M4.

    Args:
        root: Path to KITTI dataset root (e.g., /content/data/KITTI/extracted).
        split: 'train' or 'val'.
        img_size: Target square canvas resolution (default: 640).
        raw_root: Optional explicit path to KITTI raw sequence images.
        max_depth: Maximum depth in meters for normalization (default: 80.0).
        depth_scale: Scaling factor for 16-bit PNG (default: 256.0).
        transforms: Optional torchvision transforms for RGB image.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 640,
        raw_root: Optional[str | Path] = None,
        max_depth: float = MAX_DEPTH,
        depth_scale: float = DEPTH_SCALE,
        transforms: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split.lower()
        self.img_size = img_size
        self.raw_root = Path(raw_root) if raw_root is not None else None
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.transforms = transforms

        # Diagnostics
        self.total_depth_files = 0
        self.paired_samples = 0
        self.missing_rgb = 0
        self.missing_gt = 0
        self.image_02_count = 0
        self.image_03_count = 0

        self.pairs: List[Dict[str, Any]] = []
        self._discover_pairs()

    def _discover_pairs(self) -> None:
        """Discover RGB-depth pairs according to the split."""
        if self.split == "val":
            self._discover_val_pairs()
        elif self.split == "train":
            self._discover_train_pairs()
        else:
            raise ValueError(f"Unsupported split: '{self.split}'. Expected 'train' or 'val'.")

        self.paired_samples = len(self.pairs)

    def _discover_val_pairs(self) -> None:
        """
        Discover validation pairs in val_selection_cropped.
        Filenames follow:
          image:             <seq>_image_<frame>_<cam>.png
          groundtruth_depth: <seq>_groundtruth_depth_<frame>_<cam>.png
        """
        # Support both root=/.../depth_selection/val_selection_cropped and root=/.../extracted
        candidates = [
            self.root / "depth_selection" / "val_selection_cropped",
            self.root / "val_selection_cropped",
            self.root,
        ]
        val_dir = next((c for c in candidates if (c / "image").is_dir() or (c / "groundtruth_depth").is_dir()), None)

        if val_dir is None:
            # Check if there is a val sequence folder as fallback
            val_seq_dir = self.root / "val"
            if val_seq_dir.is_dir():
                self._discover_train_like_pairs(val_seq_dir)
                return
            return

        img_dir = val_dir / "image"
        gt_dir = val_dir / "groundtruth_depth"

        gt_pattern = re.compile(r"^(.+)_groundtruth_depth_(\d+)_(image_\d\d)\.png$")
        img_pattern = re.compile(r"^(.+)_image_(\d+)_(image_\d\d)\.png$")

        gt_map = {}
        if gt_dir.is_dir():
            for gt_path in gt_dir.glob("*.png"):
                match = gt_pattern.match(gt_path.name)
                if match:
                    gt_map[match.groups()] = gt_path

        img_map = {}
        if img_dir.is_dir():
            for img_path in img_dir.glob("*.png"):
                match = img_pattern.match(img_path.name)
                if match:
                    img_map[match.groups()] = img_path

        self.total_depth_files = len(gt_map)
        all_keys = set(gt_map.keys()) | set(img_map.keys())

        for key in sorted(all_keys):
            seq, frame, cam = key
            has_gt = key in gt_map
            has_img = key in img_map

            if has_gt and has_img:
                self.pairs.append({
                    "rgb_path": img_map[key],
                    "depth_path": gt_map[key],
                    "sequence": seq,
                    "frame": frame,
                    "camera": cam,
                })
                if cam == "image_02":
                    self.image_02_count += 1
                elif cam == "image_03":
                    self.image_03_count += 1
            elif has_img and not has_gt:
                self.missing_gt += 1
            elif has_gt and not has_img:
                self.missing_rgb += 1

    def _discover_train_pairs(self) -> None:
        """Discover training pairs in sequence-level KITTI structure."""
        train_dir = self.root / "train"
        if not train_dir.is_dir():
            # Check if root is already the train directory
            if (self.root / "proj_depth").is_dir() or any(self.root.glob("*/proj_depth")):
                train_dir = self.root

        self._discover_train_like_pairs(train_dir)

    def _discover_train_like_pairs(self, base_dir: Path) -> None:
        """Search sequence-level directories for proj_depth/groundtruth."""
        if not base_dir.is_dir():
            return

        # Pattern: base_dir/<sequence>/proj_depth/groundtruth/<camera>/*.png
        gt_paths = sorted(base_dir.glob("*/proj_depth/groundtruth/image_*/*.png"))
        self.total_depth_files = len(gt_paths)

        for gt_path in gt_paths:
            cam = gt_path.parent.name            # image_02 or image_03
            frame = gt_path.stem                 # 0000000005
            seq_dir = gt_path.parents[3]         # <sequence_dir>
            seq_name = seq_dir.name              # 2011_09_26_drive_0001_sync
            date = seq_name[:10] if len(seq_name) >= 10 else ""

            rgb_path = self._resolve_train_rgb(seq_dir, seq_name, date, cam, gt_path.name)

            if rgb_path is not None and rgb_path.is_file():
                self.pairs.append({
                    "rgb_path": rgb_path,
                    "depth_path": gt_path,
                    "sequence": seq_name,
                    "frame": frame,
                    "camera": cam,
                })
                if cam == "image_02":
                    self.image_02_count += 1
                elif cam == "image_03":
                    self.image_03_count += 1
            else:
                self.missing_rgb += 1

    def _resolve_train_rgb(
        self,
        seq_dir: Path,
        seq_name: str,
        date: str,
        cam: str,
        file_name: str,
    ) -> Optional[Path]:
        """
        Attempt to locate the corresponding RGB image across supported KITTI layouts:
          1. <seq_dir>/<cam>/data/<file_name>
          2. <seq_dir>/<cam>/<file_name>
          3. <raw_root>/<date>/<seq_name>/<cam>/data/<file_name>
          4. <raw_root>/<seq_name>/<cam>/data/<file_name>
          5. <raw_root>/<seq_name>/<cam>/<file_name>
          6. <root>/raw/<date>/<seq_name>/<cam>/data/<file_name>
          7. <root>/raw/<seq_name>/<cam>/data/<file_name>
          8. <root>/raw_data/<date>/<seq_name>/<cam>/data/<file_name>
          9. <root>/<date>/<seq_name>/<cam>/data/<file_name>
          10. <root>/training/image_2/<file_name> (if cam == image_02)
        """
        candidates: List[Path] = [
            seq_dir / cam / "data" / file_name,
            seq_dir / cam / file_name,
        ]

        if self.raw_root is not None:
            if date:
                candidates.append(self.raw_root / date / seq_name / cam / "data" / file_name)
            candidates.append(self.raw_root / seq_name / cam / "data" / file_name)
            candidates.append(self.raw_root / seq_name / cam / file_name)

        if date:
            candidates.append(self.root / "raw" / date / seq_name / cam / "data" / file_name)
            candidates.append(self.root / "raw_data" / date / seq_name / cam / "data" / file_name)
            candidates.append(self.root / date / seq_name / cam / "data" / file_name)

        candidates.append(self.root / "raw" / seq_name / cam / "data" / file_name)

        # KINS / KITTI object fallback
        if cam == "image_02":
            candidates.append(self.root / "training" / "image_2" / file_name)
            candidates.append(self.root / "image_2" / file_name)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.pairs[index]
        rgb_path = sample["rgb_path"]
        depth_path = sample["depth_path"]

        # Load RGB
        rgb_img = Image.open(rgb_path).convert("RGB")
        w, h = rgb_img.size

        # Load 16-bit Depth PNG without precision loss
        # PIL in 'I;16' or 'I' mode or numpy conversion preserves uint16
        with Image.open(depth_path) as depth_img:
            depth_raw = np.array(depth_img)

        # Ensure 2D uint16 array
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        # ------------------------------------------------------------
        # Geometrically identical aspect-ratio preserving letterbox
        # ------------------------------------------------------------
        target_size = self.img_size
        scale = min(target_size / w, target_size / h)

        new_w = round(w * scale)
        new_h = round(h * scale)

        pad_x = (target_size - new_w) / 2
        pad_y = (target_size - new_h) / 2

        # 1. Resize RGB with BILINEAR
        resample_bilinear = getattr(Image, "Resampling", Image).BILINEAR
        rgb_resized = rgb_img.resize((new_w, new_h), resample_bilinear)

        # 2. Resize Depth with NEAREST (CRITICAL to avoid synthetic boundary values)
        # Using PIL nearest or nearest coordinate indexing
        resample_nearest = getattr(Image, "Resampling", Image).NEAREST
        depth_pil = Image.fromarray(depth_raw)
        depth_resized_raw = np.array(depth_pil.resize((new_w, new_h), resample_nearest))

        # 3. Pad RGB to canvas (black padding)
        padded_rgb = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        padded_rgb.paste(rgb_resized, (int(pad_x), int(pad_y)))

        # 4. Pad Depth to canvas (0.0 invalid padding)
        depth_canvas_raw = np.zeros((target_size, target_size), dtype=np.float32)
        depth_canvas_raw[int(pad_y):int(pad_y) + new_h, int(pad_x):int(pad_x) + new_w] = depth_resized_raw.astype(np.float32)

        # ------------------------------------------------------------
        # Depth Conversion & Valid Mask
        # ------------------------------------------------------------
        # KITTI official encoding: depth in meters = raw_uint16 / 256.0
        depth_meters = depth_canvas_raw / self.depth_scale

        # Valid mask: depth > 1mm and finite
        valid_mask = (depth_meters > 1e-3) & np.isfinite(depth_meters)

        # Normalized depth in [0, 1] for M4 sigmoid output: depth_m / 80.0
        depth_normalized = np.zeros_like(depth_meters)
        depth_normalized[valid_mask] = np.clip(depth_meters[valid_mask] / self.max_depth, 0.0, 1.0)

        # Convert RGB to Tensor (with transforms or standard normalization)
        if self.transforms is not None:
            image_tensor = self.transforms(padded_rgb)
        else:
            # Default ImageNet normalization
            img_arr = np.array(padded_rgb, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_arr = (img_arr - mean) / std
            image_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

        targets = {
            "depth_gt": torch.from_numpy(depth_normalized).unsqueeze(0).float(),
            "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0).bool(),
            "depth_meters": torch.from_numpy(depth_meters).unsqueeze(0).float(),
        }

        return image_tensor, targets
