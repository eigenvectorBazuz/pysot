#!/usr/bin/env python3
import cv2
import os
import numpy as np
import argparse

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



    # sample_frames(args.video, args.out, args.num)
