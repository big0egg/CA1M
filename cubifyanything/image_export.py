# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""Render camera-space 3D predictions onto the original RGB frame."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Corner ordering follows GeneralInstance3DBoxes.corners.
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
COLORS = ((255, 80, 80), (60, 220, 100), (70, 160, 255),
          (255, 200, 50), (220, 100, 255), (40, 220, 220))


def _clip_image_line(start, end, width, height):
    """Clip a projected line to the image rectangle (Liang–Barsky)."""
    x, y = start
    dx, dy = end - start
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, x), (dx, width - 1 - x),
                 (-dy, y), (dy, height - 1 - y)):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return (x + lower * dx, y + lower * dy,
            x + upper * dx, y + upper * dy)


def project_box_edge(start, end, intrinsics, width, height, near=1e-3):
    """Clip an edge at the camera near plane before perspective projection."""
    points = np.array([start, end], dtype=np.float64)
    if not np.isfinite(points).all() or (points[:, 2] < near).all():
        return None
    if points[0, 2] < near:
        fraction = (near - points[0, 2]) / (points[1, 2] - points[0, 2])
        points[0] += fraction * (points[1] - points[0])
    elif points[1, 2] < near:
        fraction = (near - points[1, 2]) / (points[0, 2] - points[1, 2])
        points[1] += fraction * (points[0] - points[1])
    projected = points @ intrinsics.T
    if not np.isfinite(projected).all() or (projected[:, 2] <= 0).any():
        return None
    pixels = projected[:, :2] / projected[:, 2:3]
    return _clip_image_line(pixels[0], pixels[1], width, height)


def save_prediction_image(image, instances, intrinsics, output_dir, video_id,
                          timestamp, frame_index, image_format="png"):
    """Save one RGB frame with projected cuboids and confidence labels.

    The model already returns boxes in camera coordinates, including its
    predicted rotation. Use the original image intrinsics without applying
    the gravity transform a second time. Boxes behind the camera are skipped.
    """
    if image_format not in ("png", "jpg"):
        raise ValueError("image_format must be png or jpg")
    if hasattr(intrinsics, "detach"):
        intrinsics = intrinsics.detach().cpu().numpy()
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Expected finite 3x3 image intrinsics")

    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(max(width, height) / 400))
    corners = instances.pred_boxes_3d.corners.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()
    for index, (box, score) in enumerate(zip(corners, scores)):
        if not np.isfinite(box).all() or not np.isfinite(score):
            continue
        color = COLORS[index % len(COLORS)]
        segments = []
        for first, second in BOX_EDGES:
            segment = project_box_edge(box[first], box[second], intrinsics,
                                       width, height)
            if segment is not None:
                draw.line(segment, fill=color, width=line_width)
                segments.append(segment)
        if not segments:
            continue
        label = f"#{index + 1} {score:.2f}"
        bbox = draw.textbbox((0, 0), label)
        label_width, label_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(0, min(min(s[0] for s in segments), width - label_width - 4))
        y = max(0, min(min(s[1] for s in segments), height - label_height - 4))
        draw.rectangle((x, y, x + label_width + 4, y + label_height + 4), fill=color)
        draw.text((x + 2 - bbox[0], y + 2 - bbox[1]), label, fill=(0, 0, 0))

    scene_dir = Path(output_dir) / str(int(video_id))
    scene_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = round(float(timestamp) * 1e9)
    output_path = scene_dir / f"frame_{frame_index:06d}_{timestamp_ns}.{image_format}"
    options = {"quality": 95} if image_format == "jpg" else {}
    canvas.save(output_path, **options)
    return output_path
