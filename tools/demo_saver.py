# pysot/tools/demo.py
# Patched to save tracking outputs to a data file instead of video/display

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import pickle

import cv2
import torch
import numpy as np
from glob import glob

import easyocr

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

def get_text_mask(img_shape, detections):
    """
    Build a binary mask that zeros out all detected text regions.

    Args:
        img_shape (tuple): shape of the image as (H, W[, C])
        detections (list): List of (box, text, score), where
            box is a list of 4 [x, y] corner points.

    Returns:
        mask (np.ndarray): uint8 mask of shape (H, W), with 0 in text areas and 1 elsewhere.
    """
    H, W = img_shape[:2]
    mask = np.ones((H, W), dtype=np.uint8)

    for box, _, _ in detections:
        # box is [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 0)

    return mask


def get_frames(video_name):
    """
    A combined loader: tries imageio-ffmpeg, then falls back to OpenCV.
    Always yields BGR frames.
    """
    from pathlib import Path
    path = Path(video_name)

    # Try imageio-ffmpeg first
    if path.is_file():
        try:
            import imageio.v3 as iio
            reader = iio.imiter(path, plugin="ffmpeg")
            print(f"[get_frames] ▶ imageio-ffmpeg opened {path.name}")
            for idx, rgb in enumerate(reader):
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                yield frame
            print("[get_frames] ✓ imageio iteration complete.")
            return
        except Exception as e:
            print(f"[get_frames] ⚠ imageio failed ({e}), falling back to OpenCV")

    # Fallback: OpenCV VideoCapture
    print(f"[get_frames] ▶ cv2.VideoCapture opening {video_name!r}")
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print(f"[get_frames] ✖ cv2 failed to open {video_name!r}")
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[get_frames] ⛔ cv2 no more frames or decode error")
            break
        yield frame
        idx += 1

    cap.release()
    print("[get_frames] 🗑️ cv2 capture released")


def main():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--init_rect', type=str, default='',
                        help='x,y,w,h of the target in the first frame')
    parser.add_argument(
        '--data_output', type=str, default='tracking_data.pkl',
        help='where to save the tracking outputs as a pickle'
    )
    parser.add_argument('--filter_hud',    action='store_true',
                    help='whether to zero out any detected HUD/text regions')
    args = parser.parse_args()

    # load config and model
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    model = ModelBuilder()
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    tracker = build_tracker(model)

    if args.filter_hud:
        reader = easyocr.Reader(['en'], gpu=True)
        print('Loaded the OCR reader')

    # prepare for tracking
    outputs_list = []
    first_frame = True

    i = 0
    for frame in get_frames(args.video_name):
        if first_frame:
            # initial bounding box
            if args.init_rect:
                x, y, w, h = map(int, args.init_rect.split(','))
                init_rect = (x, y, w, h)
            else:
                init_rect = cv2.selectROI('demo', frame,
                                          showCrosshair=False,
                                          fromCenter=False)
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            if i > 1 and args.filter_hud and outputs['best_score'] < 0.7 and i & 50 == 0:
                ocr_dets = reader.readtext(np.array(frame))
                mask = get_text_mask(frame.shape, ocr_dets)
                print(f'frame {i}: running HUD filter')
            else:
                mask = None
            outputs = tracker.track(frame, mask)
            outputs_list.append(outputs)
        i += 1
        if i % 100 == 0:
            print(i)

    # save the tracking data
    with open(args.data_output, 'wb') as f:
        pickle.dump(outputs_list, f)
    print(f"[demo.py] tracking data saved to {args.data_output}")


if __name__ == '__main__':
    main()
