#!/usr/bin/env python3
import cv2
import os
import numpy as np
import argparse
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

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



