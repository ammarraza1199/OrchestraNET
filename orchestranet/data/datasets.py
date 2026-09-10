import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

class COCODetectionDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None, occlusion_aug=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.occlusion_aug = occlusion_aug
        
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

        num_objs = len(coco_annotation)
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat_to_id[coco_annotation[i]['category_id']])

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
        max_objs = 100
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
