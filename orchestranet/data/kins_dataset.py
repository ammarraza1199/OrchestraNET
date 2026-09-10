import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils


class KINSAmodalDataset(Dataset):
    """
    KINS dataset adapter for OrchestraNet M6 amodal completion.

    Returns:
        image:
            Tensor [3, 640, 640]

        targets:
            boxes:
                [100, 4] visible boxes
            labels:
                [100]
            num_objects:
                scalar

            amodal_boxes:
                [50, 4] amodal-vs-visible bbox offsets in resized
                image coordinate units

            amodal_masks:
                [50, 28, 28] object-aligned amodal masks

            is_occluded:
                [50] binary occlusion targets
    """

    def __init__(
        self,
        root,
        ann_file,
        transforms=None,
        max_objects=50,
        max_detection_objects=100,
        image_size=640,
    ):
        self.root = Path(root)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        self.max_objects = max_objects
        self.max_detection_objects = max_detection_objects
        self.image_size = image_size

        with open(self.ann_file, "r") as f:
            data = json.load(f)

        self.images = {
            int(img["id"]): img
            for img in data["images"]
        }

        self.annotations_by_image = {}

        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            self.annotations_by_image.setdefault(image_id, []).append(ann)

        # Deterministic ordering is important because M6 currently
        # aligns prediction query i with target i.
        for image_id in self.annotations_by_image:
            self.annotations_by_image[image_id].sort(
                key=lambda x: int(x.get("id", 0))
            )

        self.ids = sorted(self.images.keys())

        # KINS categories
        categories = sorted(
            data.get("categories", []),
            key=lambda x: int(x["id"])
        )

        self.cat_to_id = {
            int(cat["id"]): i
            for i, cat in enumerate(categories)
        }

    def __len__(self):
        return len(self.ids)

    def _resolve_image_path(self, file_name):
        """
        KINS JSON normally references image_2/<filename>.
        Try several layouts so the loader is robust to the extracted archive.
        """
        candidates = [
            self.root / file_name,
            self.root / "training" / "image_2" / Path(file_name).name,
            self.root / "testing" / "image_2" / Path(file_name).name,
            self.root / "training" / "image_2" / file_name,
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"KINS image not found: {file_name}\n"
            f"Tried:\n" +
            "\n".join(str(p) for p in candidates)
        )

    @staticmethod
    def _decode_polygon(segmentation, height, width):
        """
        Decode KINS polygon/RLE segmentation into a binary mask.
        """
        if segmentation is None:
            return np.zeros((height, width), dtype=np.uint8)

        try:
            if isinstance(segmentation, list):
                # Polygon format
                rles = mask_utils.frPyObjects(
                    segmentation,
                    height,
                    width,
                )

                if isinstance(rles, list):
                    rle = mask_utils.merge(rles)
                else:
                    rle = rles

            elif isinstance(segmentation, dict):
                # RLE format
                rle = segmentation

                # JSON may store counts as a list
                if isinstance(rle.get("counts"), list):
                    rle = mask_utils.frPyObjects(
                        rle,
                        height,
                        width,
                    )

            else:
                return np.zeros(
                    (height, width),
                    dtype=np.uint8
                )

            decoded = mask_utils.decode(rle)

            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)

            return decoded.astype(np.uint8)

        except Exception as e:
            raise RuntimeError(
                f"Failed to decode KINS segmentation: {e}"
            ) from e

    @staticmethod
    def _bbox_xyxy(bbox):
        """
        KINS bbox format:
            [x, y, width, height]
        """
        x, y, w, h = [float(v) for v in bbox]

        return np.array(
            [x, y, x + w, y + h],
            dtype=np.float32,
        )

    def _resize_mask_crop(
        self,
        mask,
        bbox,
        output_size=28,
    ):
        """
        Extract an object-aligned amodal mask crop and resize it to 28x28.

        We use the amodal bbox as the spatial reference because M6 predicts
        an object-level 28x28 mask rather than a full-image mask.
        """
        x1, y1, x2, y2 = bbox

        h, w = mask.shape

        x1i = max(0, int(np.floor(x1)))
        y1i = max(0, int(np.floor(y1)))
        x2i = min(w, int(np.ceil(x2)))
        y2i = min(h, int(np.ceil(y2)))

        if x2i <= x1i or y2i <= y1i:
            return torch.zeros(
                (output_size, output_size),
                dtype=torch.float32,
            )

        crop = mask[y1i:y2i, x1i:x2i]

        crop_tensor = torch.from_numpy(
            crop.astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        crop_tensor = F.interpolate(
            crop_tensor,
            size=(output_size, output_size),
            mode="nearest",
        )

        return crop_tensor[0, 0]

    def __getitem__(self, index):
        image_id = self.ids[index]
        image_info = self.images[image_id]

        file_name = image_info["file_name"]

        image_path = self._resolve_image_path(file_name)

        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        annotations = self.annotations_by_image.get(
            image_id,
            []
        )

        # ------------------------------------------------------------
        # Keep deterministic valid annotations
        # ------------------------------------------------------------
        valid_annotations = []

        for ann in annotations:
            if "i_bbox" not in ann:
                continue

            if "a_bbox" not in ann:
                continue

            valid_annotations.append(ann)

        # M6 only supports max_objects=50.
        m6_annotations = valid_annotations[:self.max_objects]

        # Detection branch supports 100 objects.
        detection_annotations = valid_annotations[
            :self.max_detection_objects
        ]

        # ------------------------------------------------------------
        # Resize image to the same 640x640 geometry used by project
        # ------------------------------------------------------------
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        sx = self.image_size / float(original_width)
        sy = self.image_size / float(original_height)

        # ------------------------------------------------------------
        # Detection boxes
        # ------------------------------------------------------------
        boxes = []
        labels = []

        for ann in detection_annotations:
            bbox = self._bbox_xyxy(ann["i_bbox"])

            bbox[[0, 2]] *= sx
            bbox[[1, 3]] *= sy

            boxes.append(bbox.tolist())

            labels.append(
                self.cat_to_id.get(
                    int(ann["category_id"]),
                    0,
                )
            )

        padded_boxes = torch.zeros(
            (self.max_detection_objects, 4),
            dtype=torch.float32,
        )

        padded_labels = torch.zeros(
            (self.max_detection_objects,),
            dtype=torch.int64,
        )

        actual_objects = min(
            len(boxes),
            self.max_detection_objects,
        )

        if actual_objects > 0:
            padded_boxes[:actual_objects] = torch.tensor(
                boxes[:actual_objects],
                dtype=torch.float32,
            )

            padded_labels[:actual_objects] = torch.tensor(
                labels[:actual_objects],
                dtype=torch.int64,
            )

        # ------------------------------------------------------------
        # M6 targets
        # ------------------------------------------------------------
        amodal_boxes = torch.zeros(
            (self.max_objects, 4),
            dtype=torch.float32,
        )

        amodal_masks = torch.zeros(
            (self.max_objects, 28, 28),
            dtype=torch.float32,
        )

        is_occluded = torch.zeros(
            (self.max_objects,),
            dtype=torch.float32,
        )

        for i, ann in enumerate(m6_annotations):

            visible_bbox = self._bbox_xyxy(
                ann["i_bbox"]
            )

            amodal_bbox = self._bbox_xyxy(
                ann["a_bbox"]
            )

            # Scale both boxes into the resized image coordinate system.
            visible_bbox[[0, 2]] *= sx
            visible_bbox[[1, 3]] *= sy

            amodal_bbox[[0, 2]] *= sx
            amodal_bbox[[1, 3]] *= sy

            # --------------------------------------------------------
            # M6 bbox target
            # --------------------------------------------------------
            # M6 predicts an offset relative to the visible bbox.
            # Keep the target in the same resized-image coordinate
            # units as the bbox geometry.
            # --------------------------------------------------------
            delta = amodal_bbox - visible_bbox

            amodal_boxes[i] = torch.from_numpy(
                delta.astype(np.float32)
            )

            # --------------------------------------------------------
            # Amodal segmentation
            # --------------------------------------------------------
            segmentation = ann.get("a_segm")

            mask = self._decode_polygon(
                segmentation,
                original_height,
                original_width,
            )

            # Resize mask to the same geometry as the image.
            mask_tensor = torch.from_numpy(
                mask.astype(np.float32)
            ).unsqueeze(0).unsqueeze(0)

            mask_tensor = F.interpolate(
                mask_tensor,
                size=(self.image_size, self.image_size),
                mode="nearest",
            )

            resized_mask = mask_tensor[0, 0].numpy()

            # Transform amodal bbox to resized coordinates.
            mask_bbox = amodal_bbox

            object_mask = self._resize_mask_crop(
                resized_mask,
                mask_bbox,
                output_size=28,
            )

            amodal_masks[i] = object_mask

            # --------------------------------------------------------
            # KINS does not contain explicit isoccluded in the
            # validated annotation files. Use area relation:
            #
            # visible area < amodal area => occluded
            # --------------------------------------------------------
            a_area = float(ann.get("a_area", 0))
            i_area = float(ann.get("i_area", 0))

            is_occluded[i] = float(
                i_area < a_area
            )

        # ------------------------------------------------------------
        # Convert image
        # ------------------------------------------------------------
        if self.transforms is not None:
            # We already resized geometrically above.
            # To avoid resizing a second time, convert directly if
            # the supplied transform is the project's standard transform.
            try:
                image_tensor = self.transforms(image)
            except Exception:
                image_tensor = (
                    torch.from_numpy(
                        np.asarray(image)
                    )
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
        else:
            image_tensor = (
                torch.from_numpy(
                    np.asarray(image)
                )
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        targets = {
            "boxes": padded_boxes,
            "labels": padded_labels,
            "num_objects": torch.tensor(
                actual_objects,
                dtype=torch.int64,
            ),
            "amodal_boxes": amodal_boxes,
            "amodal_masks": amodal_masks,
            "is_occluded": is_occluded,
        }

        return image_tensor, targets