#!/usr/bin/env python3
import cv2
import os
import numpy as np
import argparse
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from typing import Optional



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



def annotate_video(
    video_path: str,
    pkl_path: str,
    output_path: str,
    score_thresh: float = 0.95,
    codec: str = "mp4v",
    fps: Optional[float] = None
):
    """
    Create an annotated video overlaying bboxes and scores on each frame.
    Frames with best_score < score_thresh are left unannotated.
    
    Args:
        video_path:     path to input video
        pkl_path:       path to pickle list of dicts with 'bbox' and 'best_score'
        output_path:    path for the output mp4
        score_thresh:   only show annotations if best_score >= this
        codec:          fourcc code for VideoWriter (default 'mp4v')
        fps:            if None, will use input video's fps
    """
    # Load predictions
    with open(pkl_path, "rb") as f:
        preds = pickle.load(f)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps if fps is not None else in_fps
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # overlay if prediction exists and score >= thresh
        entry = preds[frame_idx] if frame_idx < len(preds) else None
        if isinstance(entry, dict):
            bbox  = entry.get("bbox", None)
            score = entry.get("best_score", -1.0)
            if bbox and score >= score_thresh:
                x, y, bw, bh = map(int, bbox)
                # draw box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                # draw score
                txt = f"{score:.3f}"
                cv2.putText(
                    frame, txt, (x, max(y-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
        
        writer.write(frame)
        frame_idx += 1
    
    cap.release()
    writer.release()
    print(f"Annotated video saved to {output_path}")


def display_frame_with_bbox_and_score_cands(frame_number, video_path, pkl_path):
    """
    Display a specific frame with its bounding box and score from tracking data,
    plus the top-k candidate boxes.

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

    # Read the specified frame via imageio (RGB)
    reader = iio.imiter(video_path)
    for idx, frame in enumerate(reader):
        if idx == frame_number:
            img = frame
            break
    else:
        raise ValueError(f"Couldn’t read frame {frame_number} from {video_path}.")

    # Extract data
    data  = outputs_list[frame_number]
    bbox  = data.get('bbox')
    best_score = data.get('best_score', None)
    pscore_all = data.get('pscore', None)
    cands = data.get('candidates', [])
    cand_ps  = data.get('candidate_score', [])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Frame {frame_number}")

    # Draw the other top-k candidates in green
    for (x, y, w, h), ps in zip(cands, cand_ps):
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1.5, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 6, f"ps: {ps:.3f}",
                color='white', fontsize=10,
                bbox=dict(facecolor='green', alpha=0.6, pad=1))

    # Draw the best box in red, with both raw and pscore
    if bbox is not None and best_score is not None and pscore_all is not None:
        x, y, w, h = bbox
        # find the corresponding pscore for the best box
        # if you have stored indices, you could use that; otherwise approximate:
        # here we assume best_score corresponds to the max of raw scores
        # and pscore_all[np.argmax(pscore_all)] is its pscore
        best_ps = pscore_all.max()

        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        txt = f"s: {best_score:.3f}, ps: {best_ps:.3f}"
        ax.text(x, y - 12, txt,
                color='yellow', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.7, pad=2))
    else:
        ax.text(0.5, 0.5, 'No bbox for this frame',
                color='yellow', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.show()




