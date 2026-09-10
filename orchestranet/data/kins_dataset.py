import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

class KINSAmodalDataset(Dataset):
    def __init__(
        self,
        root,
        ann_file,
        transforms=None,
        max_objects=50,
        max_detection_objects=100,
        image_size=640,
    ):
        self.root = root
        self.transforms = transforms
        self.max_objects = max_objects
        self.max_detection_objects = max_detection_objects
        self.image_size = image_size
        
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        
    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = np.zeros((self.max_detection_objects, 4), dtype=np.float32)
        labels = np.zeros(self.max_detection_objects, dtype=np.int64)
        num_objects = 0
        
        amodal_boxes = np.zeros((self.max_objects, 4), dtype=np.float32)
        amodal_masks = np.zeros((self.max_objects, 28, 28), dtype=np.float32)
        is_occluded = np.zeros(self.max_objects, dtype=np.float32)
        
        for i, ann in enumerate(anns):
            if i >= self.max_detection_objects:
                break
                
            num_objects += 1
            
            i_bbox = ann.get('i_bbox', ann.get('bbox', [0, 0, 0, 0]))
            a_bbox = ann.get('a_bbox', [0, 0, 0, 0])
            
            i_x, i_y, i_w, i_h = i_bbox
            a_x, a_y, a_w, a_h = a_bbox
            
            i_x *= scale_x
            i_y *= scale_y
            i_w *= scale_x
            i_h *= scale_y
            
            a_x *= scale_x
            a_y *= scale_y
            a_w *= scale_x
            a_h *= scale_y
            
            i_x1, i_y1 = i_x, i_y
            i_x2, i_y2 = i_x + i_w, i_y + i_h
            
            boxes[i] = [i_x1, i_y1, i_x2, i_y2]
            labels[i] = ann.get('category_id', 0)
            
            if i < self.max_objects:
                visible_bbox = np.array([i_x, i_y, i_w, i_h], dtype=np.float32)
                amodal_bbox = np.array([a_x, a_y, a_w, a_h], dtype=np.float32)
                
                delta = amodal_bbox - visible_bbox
                amodal_boxes[i] = delta.astype(np.float32)
                
                if 'a_segm' in ann and ann['a_segm'] is not None:
                    if isinstance(ann['a_segm'], list):
                        rles = maskUtils.frPyObjects(ann['a_segm'], orig_h, orig_w)
                        rle = maskUtils.merge(rles)
                    elif isinstance(ann['a_segm'], dict) and 'counts' in ann['a_segm'] and isinstance(ann['a_segm']['counts'], list):
                        rle = maskUtils.frPyObjects([ann['a_segm']], orig_h, orig_w)[0]
                    else:
                        rle = ann['a_segm']
                        
                    mask = maskUtils.decode(rle)
                    
                    if len(mask.shape) > 2:
                        mask = np.max(mask, axis=2)
                        
                    mask_resized = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                    
                    ax1 = max(0, int(a_x))
                    ay1 = max(0, int(a_y))
                    ax2 = min(self.image_size, int(a_x + a_w))
                    ay2 = min(self.image_size, int(a_y + a_h))
                    
                    crop = mask_resized[ay1:ay2, ax1:ax2]
                    
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_NEAREST)
                        amodal_masks[i] = crop_resized.astype(np.float32)
                        
                i_area = i_w * i_h
                a_area = a_w * a_h
                is_occluded[i] = float(i_area < a_area)
                
        targets = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "num_objects": torch.tensor(num_objects, dtype=torch.int64),
            "amodal_boxes": torch.from_numpy(amodal_boxes),
            "amodal_masks": torch.from_numpy(amodal_masks),
            "is_occluded": torch.from_numpy(is_occluded),
        }
        
        return image, targets
