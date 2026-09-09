"""
Evaluation Script for OrchestraNet.

Evaluates on COCO val2017 with:
  - Standard metrics: mAP@50, mAP@50:95, AP_small/medium/large
  - Occlusion-specific metrics (if occlusion annotations available)
  - Speed: FPS, per-route latency
  - Ablation support: disable individual models

Usage:
  python training/evaluate.py --weights checkpoints/orchestranet_final.pt --data-root ./data/coco
  python training/evaluate.py --weights checkpoints/orchestranet_final.pt --ablate m2
"""

import argparse, os, sys, time, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestranet.orchestrator import OrchestraNet
from orchestranet.data.datasets import COCODetectionDataset
from orchestranet.utils.metrics import DetectionMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="OrchestraNet Evaluation")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--ablate", default=None, help="Model to disable for ablation (e.g., m2)")
    parser.add_argument("--save-results", default="./results")
    parser.add_argument("--num-images", type=int, default=None, help="Limit eval images")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, conf_thresh=0.25, ablate=None, num_images=None):
    """Run full evaluation."""
    model.eval()
    metrics = DetectionMetrics(num_classes=80)
    latencies = []
    route_dist = {"simple": 0, "medium": 0, "complex": 0}

    # Disable ablated model
    if ablate and ablate in model.models:
        print(f"⚠️  ABLATION: {ablate} disabled")
        model.models[ablate].is_active = False

    count = 0
    total = num_images or len(loader)
    pbar = tqdm(loader, total=min(total, len(loader)), desc="Evaluating")

    for images, targets in pbar:
        if num_images and count >= num_images:
            break

        images = images.to(device)

        # Timed inference
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        # Track routing
        route_dist[outputs["routing"]["routing_level"]] += 1

        # Extract detections
        detections = outputs["detections"]
        if isinstance(detections, list):
            for b in range(len(detections)):
                det = detections[b]
                if det["boxes"].shape[0] == 0:
                    metrics.update([], [], [], 
                                   targets["boxes"][b][:targets["num_objects"][b]].numpy(),
                                   targets["labels"][b][:targets["num_objects"][b]].numpy(),
                                   image_id=count)
                else:
                    mask = det["scores"] > conf_thresh
                    pred_boxes = det["boxes"][mask].cpu().numpy()
                    pred_scores = det["scores"][mask].cpu().numpy()
                    pred_labels = det["labels"][mask].cpu().numpy()
                    n = targets["num_objects"][b].item()
                    gt_boxes = targets["boxes"][b][:n].numpy()
                    gt_labels = targets["labels"][b][:n].numpy()
                    metrics.update(pred_boxes, pred_scores, pred_labels,
                                   gt_boxes, gt_labels, image_id=count)

        count += 1
        pbar.set_postfix({"fps": 1000.0 / np.mean(latencies[-10:])})

    # Compute final metrics
    results = metrics.compute()

    # Speed stats
    lat = np.array(latencies)
    results["speed"] = {
        "mean_ms": float(lat.mean()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "fps": float(1000.0 / lat.mean()),
    }
    results["routing_distribution"] = route_dist
    results["num_images"] = count

    return results


def print_results(results, ablate=None):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    title = "OrchestraNet Evaluation Results"
    if ablate:
        title += f" (ABLATION: {ablate} disabled)"
    print(f"🎼 {title}")
    print("=" * 60)

    print("\n📊 Detection Metrics:")
    print(f"   mAP@50:      {results.get('mAP@50', 0):.4f}")
    print(f"   mAP@50:95:   {results.get('mAP@50:95', 0):.4f}")
    print(f"   AP_small:     {results.get('AP_small', 0):.4f}")
    print(f"   AP_medium:    {results.get('AP_medium', 0):.4f}")
    print(f"   AP_large:     {results.get('AP_large', 0):.4f}")

    if "speed" in results:
        s = results["speed"]
        print(f"\n⚡ Speed:")
        print(f"   Mean latency: {s['mean_ms']:.2f} ms")
        print(f"   Median:       {s['median_ms']:.2f} ms")
        print(f"   P95:          {s['p95_ms']:.2f} ms")
        print(f"   FPS:          {s['fps']:.1f}")

    if "routing_distribution" in results:
        rd = results["routing_distribution"]
        total = sum(rd.values()) or 1
        print(f"\n🔀 Routing Distribution:")
        for level, count in rd.items():
            print(f"   {level:10s}: {count:5d} ({count/total*100:.1f}%)")

    print(f"\n📷 Images evaluated: {results.get('num_images', 0)}")


def main():
    args = parse_args()
    os.makedirs(args.save_results, exist_ok=True)

    print("🎼 OrchestraNet — Evaluation")
    print("=" * 50)

    model = OrchestraNet(num_classes=80, pretrained_backbone=False)
    if args.weights and Path(args.weights).exists():
        state = torch.load(args.weights, map_location=args.device)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"✅ Loaded: {args.weights}")
    else:
        print("⚠️  Using random weights")
    model = model.to(args.device)

    dataset = COCODetectionDataset(
        root=os.path.join(args.data_root, "val2017"),
        ann_file=os.path.join(args.data_root, "annotations", "instances_val2017.json"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = evaluate(model, loader, args.device, args.conf_thresh,
                       ablate=args.ablate, num_images=args.num_images)
    print_results(results, ablate=args.ablate)

    # Save results
    suffix = f"_ablate_{args.ablate}" if args.ablate else ""
    results_path = os.path.join(args.save_results, f"eval_results{suffix}.json")
    # Convert numpy types for JSON
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                for kk, vv in v.items()}
        elif isinstance(v, (float, np.floating)):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n💾 Results saved: {results_path}")


if __name__ == "__main__":
    main()
