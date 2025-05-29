#!/usr/bin/env python3
import cv2
import os
import numpy as np
import argparse
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import json
import re
from typing import Dict, List, Optional, Union

def sample_frames(video_path: str, output_dir: str, num_samples: int):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Total frame count
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Failed to read frame count from: {video_path}")
    print(f"Total frames in video: {total}")

    # Choose frame indices uniformly over [0, total-1]
    indices = np.linspace(0, total-1, num_samples, dtype=int)

    # Prepare output folder
    os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Warning: could not read frame {idx}")
            continue
        fname = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(fname, frame)
        print(f"Saved frame {idx} → {fname}")

    cap.release()
    print("Done.")



def display_frame_with_bbox_and_score(frame_number, video_path, pkl_path):
    """
    Display a specific frame with its bounding box and score from tracking data.

    Args:
        frame_number (int): zero-based index of the frame to display.
        video_path (str): path to the video file.
        pkl_path (str): path to the pickle file containing tracking outputs.
    """
    # Load tracking outputs
    with open(pkl_path, 'rb') as f:
        outputs_list = pickle.load(f)

    # Validate frame index
    if frame_number < 0 or frame_number >= len(outputs_list):
        raise IndexError(f"Frame {frame_number} out of range (0 to {len(outputs_list)-1}).")

    # Read the specified frame (RGB) via imageio
    reader = iio.imiter(video_path)
    for idx, frame in enumerate(reader):
        if idx == frame_number:
            img = frame
            break
    else:
        raise ValueError(f"Couldn’t read frame {frame_number} from {video_path}.")

    # Extract bbox and score
    data = outputs_list[frame_number]
    bbox = data.get('bbox')
    score = data.get('best_score', None)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Frame {frame_number}")

    if bbox:
        x, y, w, h = bbox
        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # Display the score above the box
        score_text = f"score: {score:.3f}"
        ax.text(x, y - 8, score_text, color='yellow', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.6, pad=2))
    else:
        ax.text(0.5, 0.5, 'No bbox for this frame', color='yellow',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)

    plt.show()

def load_via_annotations(json_path: str) -> Dict[int, Optional[List[int]]]:
    """
    Load VIA annotations mapping frame IDs to boxes or None.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)
    via_data = raw.get('_via_img_metadata', raw) if isinstance(raw, dict) else raw
    annotations = {}
    for entry in (via_data.values() if isinstance(via_data, dict) else via_data):
        fname = entry['filename']
        frame_id = int(re.search(r'\d+', fname).group())
        regions = entry.get('regions', [])
        if regions:
            shape = regions[0]['shape_attributes']
            annotations[frame_id] = [
                shape['x'], shape['y'], shape['width'], shape['height']
            ]
        else:
            annotations[frame_id] = None
    return annotations

def load_pred_with_scores(pkl_path: str) -> Dict[int, Optional[Tuple[List[float], float]]]:
    """
    Load tracker predictions (list of dicts) preserving bbox and best_score.
    Returns mapping frame_id -> (bbox, score) or None.
    """
    raw = pickle.load(open(pkl_path, 'rb'))
    preds = {}
    for idx, entry in enumerate(raw):
        if entry is None:
            preds[idx] = None
        elif isinstance(entry, dict) and 'bbox' in entry and 'best_score' in entry:
            bbox = [float(coord) for coord in entry['bbox']]
            score = float(entry['best_score'])
            preds[idx] = (bbox, score)
        else:
            preds[idx] = None
    return preds

def compute_kpi_with_score(
    gt: Dict[int, Optional[List[int]]],
    preds: Dict[int, Optional[Tuple[List[float], float]]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.95
) -> Dict[str, Union[int, float]]:
    """
    Compute KPI including frames with no GT. Apply score threshold to predictions.
    Returns TP, FP, FN, TN, low_IoU counts and metrics.
    """
    def iou(a, b):
        xa, ya, wa, ha = a
        xb, yb, wb, hb = b
        xa2, ya2 = xa + wa, ya + ha
        xb2, yb2 = xb + wb, yb + hb
        xi1, yi1 = max(xa, xb), max(ya, yb)
        xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = wa*ha + wb*hb - inter
        return inter/union if union > 0 else 0.0

    frames = sorted(gt.keys())  # GT frames include both present and None
    TP = FP = FN = TN = low_iou = 0

    for f in frames:
        gt_box = gt[f]
        pred_entry = preds.get(f)
        # treat low-score or missing as no prediction
        if pred_entry is None or pred_entry[1] < score_thresh:
            pred_box = None
        else:
            pred_box = pred_entry[0]

        if gt_box is None:
            if pred_box is None:
                TN += 1
            else:
                FP += 1
        else:
            # GT present
            if pred_box is None:
                FN += 1
            else:
                overlap = iou(gt_box, pred_box)
                if overlap >= iou_thresh:
                    TP += 1
                else:
                    low_iou += 1
                    FP += 1
                    FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    success   = (TP + TN) / len(frames) if frames else 0.0

    return {
        "total_frames": len(frames),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "low_IoU": low_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "success_rate": success,
    }
