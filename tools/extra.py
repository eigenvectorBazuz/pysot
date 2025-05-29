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


def load_via_or_coco_annotations(json_path: str) -> Dict[str, Optional[List[int]]]:
    """
    Load annotations from either VIA native JSON or COCO-flavored VIA export.
    
    Returns a dict mapping filename -> [x, y, width, height] or None.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)
    
    # Detect COCO-style export
    if isinstance(raw, dict) and 'images' in raw and 'annotations' in raw:
        # Build image_id -> filename map
        id2fname = {img['id']: img['file_name'] for img in raw['images']}
        # Initialize all to None
        annotations: Dict[str, Optional[List[int]]] = {fname: None for fname in id2fname.values()}
        # For each annotation, assign bbox
        for ann in raw['annotations']:
            img_id = ann['image_id']
            bbox = ann.get('bbox', [])
            if bbox and img_id in id2fname:
                # COCO bbox: [x, y, w, h]
                annotations[id2fname[img_id]] = bbox
        return annotations
    
    # Detect VIA native export under _via_img_metadata
    if isinstance(raw, dict) and '_via_img_metadata' in raw:
        via_data = raw['_via_img_metadata']
    else:
        via_data = raw  # assume top-level mapping
    
    annotations: Dict[str, Optional[List[int]]] = {}
    for entry in via_data.values():
        if not isinstance(entry, dict):
            continue
        filename = entry.get('filename')
        regions = entry.get('regions', [])
        
        if regions:
            shape = regions[0].get('shape_attributes', {})
            box = [
                shape.get('x', 0),
                shape.get('y', 0),
                shape.get('width', 0),
                shape.get('height', 0),
            ]
            annotations[filename] = box
        else:
            annotations[filename] = None
    
    return annotations

def filename_to_frame_id(filename: str) -> Union[int, str]:
    """Extract integer from filename like 'frame_000123.jpg'."""
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else filename

def build_frame_indexed_annotations(json_path: str) -> Dict[int, Optional[List[int]]]:
    """
    Load annotations and return dict mapping frame index -> box or None.
    """
    raw = load_via_or_coco_annotations(json_path)
    indexed: Dict[int, Optional[List[int]]] = {}
    for fname, box in raw.items():
        fid = filename_to_frame_id(fname)
        if isinstance(fid, int):
            indexed[fid] = box
    return indexed

# Example usage:
# annotations = load_via_or_coco_annotations('via_project_29May2025_18h19m_coco.json')
# indexed = build_frame_indexed_annotations('via_project_29May2025_18h19m_coco.json')
# print(indexed.get(0), indexed.get(189))




def load_pred_pkl_list_of_dicts(pkl_path: str) -> Dict[int, Optional[List[float]]]:
    """
    Load tracker predictions saved as a list of dicts from pickle.
    Each element is expected to be a dict with a 'bbox' key (list of 4 numbers)
    or to be None.

    Returns a dict mapping frame index (list index) -> [x, y, w, h] or None.
    """
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in pickle, got {type(raw)}")
    
    preds: Dict[int, Optional[List[float]]] = {}
    for idx, entry in enumerate(raw):
        if entry is None:
            preds[idx] = None
        elif isinstance(entry, dict) and 'bbox' in entry:
            # Extract the bbox and convert to floats
            try:
                bbox = entry['bbox']
                preds[idx] = [float(coord) for coord in bbox]
            except Exception:
                preds[idx] = None
        else:
            preds[idx] = None
    return preds

# Example usage:
# pred_indexed = load_pred_pkl_list_of_dicts('/content/drive/MyDrive/samp/tracking_data.pkl')
# print(pred_indexed.get(0), pred_indexed.get(189))


# Example:
# pred_indexed = load_pred_pkl('/content/tracking_data.pkl')
# print(pred_indexed.get(0), pred_indexed.get(189))


# Example usage:
# pred_indexed = load_pred_pkl('tracker_results.pkl')
# print(pred_indexed.get(0), pred_indexed.get(189))




def compute_tracker_kpi_ignore_no_gt(
    gt: Dict[int, Optional[List[int]]],
    preds: Dict[int, Optional[List[float]]],
    iou_thresh: float = 0.5
) -> Dict[str, Union[int, float]]:
    """
    Compute tracker KPI metrics, **ignoring** frames where GT is None.
    
    Only considers frames with a GT bounding box present.
    
    Parameters
    ----------
    gt : dict[int, [x,y,w,h] or None]
    preds : dict[int, [x,y,w,h] or None]
    iou_thresh : float
        IoU threshold to count a true positive.
    
    Returns
    -------
    Dict[str, Union[int, float]]
        Metrics: total_gt_frames, TP, FP, FN, low_IoU, precision, recall, f1
    """
    def iou(boxA, boxB):
        xa, ya, wa, ha = boxA
        xb, yb, wb, hb = boxB
        xa2, ya2 = xa + wa, ya + ha
        xb2, yb2 = xb + wb, yb + hb

        xi1, yi1 = max(xa, xb), max(ya, yb)
        xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
        inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
        inter = inter_w * inter_h
        union = wa * ha + wb * hb - inter
        return inter / union if union > 0 else 0.0

    # Only consider frames where GT exists
    eval_frames = [f for f, box in gt.items() if box is not None]

    TP = FP = FN = low_iou = 0

    for f in eval_frames:
        g = gt[f]
        p = preds.get(f)

        if p is None:
            # missed detection
            FN += 1
        else:
            # both present → check IoU
            overlap = iou(g, p)
            if overlap >= iou_thresh:
                TP += 1
            else:
                low_iou += 1
                FP += 1
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_gt_frames": len(eval_frames),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "low_IoU": low_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Example usage:
# metrics = compute_tracker_kpi_ignore_no_gt(gt_indexed, pred_indexed, iou_thresh=0.5)
# print(metrics)


# Example usage:
# gt_indexed = build_frame_indexed_annotations('via_annotations.json')
# pred_indexed = load_pred_pkl('tracker_results.pkl')
# metrics = compute_tracker_kpi(gt_indexed, pred_indexed, iou_thresh=0.5)
# print(metrics)





