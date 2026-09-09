"""
Visualization utilities for OrchestraNet.

Draws detection results with occlusion information overlay:
- Bounding boxes with class labels
- Occlusion percentage per detection
- Depth ordering visualization
- Occlusion map overlay
"""

import cv2
import numpy as np
import torch


# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Color palette for visualization (distinct colors for up to 20 classes)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    occlusion_pcts: np.ndarray | None = None,
    depth_values: np.ndarray | None = None,
    class_names: list[str] | None = None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw detection results on an image with occlusion information.

    Args:
        image: (H, W, 3) BGR image
        boxes: (N, 4) xyxy bounding boxes
        scores: (N,) confidence scores
        labels: (N,) class indices
        occlusion_pcts: (N,) occlusion percentage [0,1] (optional)
        depth_values: (N,) depth values (optional)
        class_names: List of class names
        score_threshold: Minimum score to display

    Returns:
        Annotated image
    """
    class_names = class_names or COCO_CLASSES
    result = image.copy()

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        label_idx = int(labels[i]) % len(class_names)
        color = COLORS[label_idx % len(COLORS)]

        # Draw bounding box
        thickness = 2
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        class_name = class_names[label_idx]
        label_text = f"{class_name} {scores[i]:.2f}"

        if occlusion_pcts is not None:
            occ_pct = occlusion_pcts[i] * 100
            label_text += f" | Occ:{occ_pct:.0f}%"
            
            # Color-code box based on occlusion level
            if occ_pct > 50:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), thickness + 1)
            elif occ_pct > 25:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 165, 255), thickness)

        if depth_values is not None:
            label_text += f" | D:{depth_values[i]:.2f}"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            result, label_text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return result


def draw_occlusion_map(
    image: np.ndarray,
    occlusion_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay occlusion probability map on image.

    Red = high occlusion, transparent = no occlusion.
    """
    H, W = image.shape[:2]
    occ_resized = cv2.resize(occlusion_map, (W, H))

    # Create heatmap
    heatmap = np.zeros((H, W, 3), dtype=np.uint8)
    heatmap[:, :, 2] = (occ_resized * 255).astype(np.uint8)  # Red channel

    result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return result


def draw_routing_info(
    image: np.ndarray,
    routing_level: str,
    complexity_score: float,
    active_models: list[str],
    fps: float = 0.0,
) -> np.ndarray:
    """Draw routing decision info on the image."""
    result = image.copy()
    H, W = result.shape[:2]

    # Info panel background
    panel_h = 80
    cv2.rectangle(result, (0, 0), (W, panel_h), (0, 0, 0), -1)

    # Routing level with color
    level_colors = {"simple": (0, 255, 0), "medium": (0, 165, 255), "complex": (0, 0, 255)}
    color = level_colors.get(routing_level, (255, 255, 255))

    cv2.putText(result, f"Route: {routing_level.upper()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(result, f"Complexity: {complexity_score:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(result, f"Models: {', '.join(active_models)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if fps > 0:
        cv2.putText(result, f"FPS: {fps:.1f}", (W - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result
