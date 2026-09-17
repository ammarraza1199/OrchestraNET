import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from training.train_individual import IndividualTrainer, validate_m1

class MockValDataset(Dataset):
    def __init__(self, num_images=100):
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Image of size 640x640
        img = torch.randn(3, 640, 640)
        # 5 GT boxes per image
        boxes = torch.tensor([
            [50.0, 60.0, 150.0, 180.0],
            [200.0, 200.0, 350.0, 320.0],
            [10.0, 10.0, 80.0, 80.0],
            [400.0, 420.0, 580.0, 590.0],
            [100.0, 300.0, 250.0, 450.0],
        ] + [[0.0, 0.0, 0.0, 0.0]] * 95, dtype=torch.float32)
        labels = torch.tensor([1, 15, 0, 44, 79] + [0] * 95, dtype=torch.long)
        num_objects = torch.tensor(5, dtype=torch.long)
        return img, {"boxes": boxes, "labels": labels, "num_objects": num_objects}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running 100-Image Validation Verification on device: {device}")

    trainer = IndividualTrainer(model_id="m1", device=device, freeze_backbone=True)
    dataset = MockValDataset(num_images=100)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = validate_m1(
        trainer=trainer,
        val_loader=loader,
        device=device,
        conf_thresh=0.25,
        iou_thresh=0.5,
        max_detections=300,
        num_images=100,
    )

    diag = results["diagnostics"]
    print("=== Validation Diagnostics ===")
    print(f"Ground Truth Boxes:              {diag['gt_count']}")
    print(f"Raw Predictions:                  {diag['raw_preds']:,}")
    print(f"Raw Predictions per image:        {diag['raw_preds'] / 100:,.1f}")
    print(f"Predictions After Conf (> 0.25):  {diag['after_conf']:,}")
    print(f"Predictions After NMS:            {diag['after_nms']:,}")
    print(f"Raw Score min/mean/max:           {diag['raw_score_min']:.4f} / {diag['raw_score_mean']:.4f} / {diag['raw_score_max']:.4f}")
    print(f"mAP@50:                           {results['mAP@50']:.4f}")
    print(f"mAP@50:95:                        {results['mAP@50:95']:.4f}")
    print(f"AP_small:                         {results['AP_small']:.4f}")
    print(f"AP_medium:                        {results['AP_medium']:.4f}")
    print(f"AP_large:                         {results['AP_large']:.4f}")
    print(f"Raw Box Bounds:                   {diag['raw_box_bounds']}")
    print(f"Kept Box Bounds:                  {diag['kept_box_bounds']}")

    assert diag['raw_preds'] == 2520000, f"Expected 2,520,000 raw preds, got {diag['raw_preds']}"
    assert diag['raw_preds'] / 100 == 25200, f"Expected 25,200 raw preds/image, got {diag['raw_preds'] / 100}"
    print("SUCCESS: Exactly 25,200 predictions per image verified!")

if __name__ == "__main__":
    main()
