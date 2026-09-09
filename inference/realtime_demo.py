"""
Real-Time Demo for OrchestraNet.

Runs OrchestraNet on a webcam feed or video file, showing:
- Detection bounding boxes with class labels
- Occlusion percentage per detection
- Routing decision (which models are active)
- FPS counter
- Occlusion map overlay (toggle with 'o' key)

Usage:
  python inference/realtime_demo.py --weights checkpoints/orchestranet_final.pt --source webcam
  python inference/realtime_demo.py --weights checkpoints/orchestranet_final.pt --source video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestranet.orchestrator import OrchestraNet
from orchestranet.utils.visualization import (
    draw_detections,
    draw_occlusion_map,
    draw_routing_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OrchestraNet Real-Time Demo")
    parser.add_argument("--weights", default=None, help="Model weights path")
    parser.add_argument("--source", default="webcam", help="'webcam' or video path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--show-occlusion", action="store_true")
    return parser.parse_args()


def preprocess(frame: np.ndarray, img_size: int = 640) -> torch.Tensor:
    """Preprocess frame for model input."""
    # Letterbox resize
    h, w = frame.shape[:2]
    scale = min(img_size / h, img_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))

    # Pad to square
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    top = (img_size - nh) // 2
    left = (img_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized

    # To tensor
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dim
    return tensor, scale, (top, left)


def main():
    args = parse_args()

    print("🎼 OrchestraNet Real-Time Demo")
    print("=" * 50)

    # Build model
    model = OrchestraNet(
        num_classes=80,
        backbone_name="mobilenetv4_hybrid_medium",
        fpn_channels=128,
        pretrained_backbone=False,
    )

    if args.weights and Path(args.weights).exists():
        state_dict = torch.load(args.weights, map_location=args.device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights: {args.weights}")
    else:
        print("⚠️  Running with random weights (no checkpoint loaded)")

    model = model.to(args.device)
    model.eval()

    # Print model info
    params = model.count_all_parameters()
    total = params["TOTAL"]["total"]
    print(f"📊 Total parameters: {total:,}")

    # Open video source
    if args.source == "webcam":
        cap = cv2.VideoCapture(0)
        print("📷 Using webcam")
    else:
        cap = cv2.VideoCapture(args.source)
        print(f"📹 Using video: {args.source}")

    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    show_occlusion = args.show_occlusion
    fps_history = []

    print("\nControls:")
    print("  'o' — Toggle occlusion map overlay")
    print("  'q' — Quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        t_start = time.perf_counter()
        input_tensor, scale, padding = preprocess(frame, args.img_size)
        input_tensor = input_tensor.to(args.device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)

        t_end = time.perf_counter()
        inference_time = (t_end - t_start) * 1000  # ms
        fps = 1000.0 / inference_time if inference_time > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Draw results
        display = frame.copy()
        routing = outputs["routing"]

        # Draw routing info
        display = draw_routing_info(
            display,
            routing["routing_level"],
            routing["complexity_score"][0, 0].item(),
            routing["active_models"],
            fps=avg_fps,
        )

        # Draw detections
        detections = outputs["detections"]
        if isinstance(detections, list) and len(detections) > 0:
            det = detections[0]
            if det["boxes"].shape[0] > 0:
                boxes = det["boxes"].cpu().numpy()
                scores = det["scores"].cpu().numpy()
                labels = det["labels"].cpu().numpy()

                # Scale boxes back to original frame size
                top, left = padding
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale

                display = draw_detections(
                    display, boxes, scores, labels,
                    score_threshold=args.conf_thresh,
                )

        # Draw occlusion map overlay
        if show_occlusion and "m2" in outputs["model_outputs"]:
            occ_map = outputs["model_outputs"]["m2"]["occlusion_map"][0, 0].cpu().numpy()
            display = draw_occlusion_map(display, occ_map, alpha=0.3)

        cv2.imshow("OrchestraNet Demo", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            show_occlusion = not show_occlusion
            print(f"Occlusion overlay: {'ON' if show_occlusion else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == "__main__":
    main()
