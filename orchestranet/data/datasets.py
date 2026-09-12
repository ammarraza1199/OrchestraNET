import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

class COCODetectionDataset(Dataset):
    def __init__(
        self,
        root,
        ann_file,
        transforms=None,
        occlusion_aug=None,
        img_size=640,
        image_size=None,
        max_objs=100,
    ):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.occlusion_aug = occlusion_aug
        target_size = image_size if image_size is not None else img_size
        self.img_size = target_size
        self.image_size = target_size
        self.max_objs = max_objs
        
        # Mapping COCO categories to 0-79 (since COCO has 80 classes but IDs go up to 90)
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_to_id = {cat['id']: i for i, cat in enumerate(categories)}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        w, h = img.size

        # ------------------------------------------------------------
        # Aspect-ratio-preserving scaling and padding
        # ------------------------------------------------------------
        target_size = self.image_size
        scale = min(target_size / w, target_size / h)

        new_w = round(w * scale)
        new_h = round(h * scale)

        pad_x = (target_size - new_w) / 2
        pad_y = (target_size - new_h) / 2

        # Resize image using aspect-ratio-preserving scaling
        resample_filter = getattr(Image, "Resampling", Image).BILINEAR
        resized_img = img.resize((new_w, new_h), resample_filter)

        # Pad to target square canvas (e.g. 640x640)
        padded_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        padded_img.paste(resized_img, (int(pad_x), int(pad_y)))
        img = padded_img

        num_objs = len(coco_annotation)
        
        # Get bounding boxes and labels with identical geometric transformation
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]

            x1_prime = xmin * scale + pad_x
            y1_prime = ymin * scale + pad_y
            x2_prime = xmax * scale + pad_x
            y2_prime = ymax * scale + pad_y

            boxes.append([x1_prime, y1_prime, x2_prime, y2_prime])
            labels.append(self.cat_to_id.get(coco_annotation[i]['category_id'], 0))

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            
        if self.occlusion_aug is not None:
            img, boxes = self.occlusion_aug(img, boxes)

        if self.transforms is not None:
            img = self.transforms(img)
            
        # Pad boxes to maximum possible objects for batching
        max_objs = self.max_objs
        padded_boxes = torch.zeros((max_objs, 4), dtype=torch.float32)
        padded_labels = torch.zeros((max_objs,), dtype=torch.int64)
        
        actual_objs = min(len(boxes), max_objs)
        if actual_objs > 0:
            padded_boxes[:actual_objs] = boxes[:actual_objs]
            padded_labels[:actual_objs] = labels[:actual_objs]
            
        targets = {
            "boxes": padded_boxes,
            "labels": padded_labels,
            "num_objects": torch.tensor(actual_objs, dtype=torch.int64)
        }

        return img, targets

    def __len__(self):
        return len(self.ids)
