from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--init_rect', type=str, default='',
                    help='x,y,w,h of the target in the first frame')
parser.add_argument('--output', type=str, default='tracking_out.mp4',
                    help='where to save the annotated video (MP4)')
args = parser.parse_args()


# def get_frames(video_name):
#     if not video_name:
#         cap = cv2.VideoCapture(0)
#         # warmup
#         for i in range(5):
#             cap.read()
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 yield frame
#             else:
#                 break
#     elif video_name.endswith('avi') or \
#         video_name.endswith('mp4'):
#         cap = cv2.VideoCapture(args.video_name)
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 yield frame
#             else:
#                 break
#     else:
#         images = glob(os.path.join(video_name, '*.jp*'))
#         images = sorted(images,
#                         key=lambda x: int(x.split('/')[-1].split('.')[0]))
#         for img in images:
#             frame = cv2.imread(img)
#             yield frame

def get_frames(video_name):
    import cv2
    print(f"[get_frames] trying to open {video_name!r}")
    cap = cv2.VideoCapture(video_name)
    print("[get_frames] isOpened:", cap.isOpened())
    if not cap.isOpened():
        print("[get_frames] ❌ Failed to open video. Exiting generator.")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        # print(f"[get_frames] read frame {frame_idx}: ret={ret}")
        if not ret:
            print("[get_frames] ⛔ No more frames (or decode error).")
            break
        yield frame
        frame_idx += 1

    cap.release()
    print("[get_frames] released capture.")



def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    writer = None
    i = 0
    for frame in get_frames(args.video_name):
        if first_frame:
            if args.init_rect:                       # <-- NEW
                x, y, w, h = map(int, args.init_rect.split(','))
                init_rect = (x, y, w, h)
            else:                                    # fallback to GUI
                try:
                  init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                  exit()
            
            tracker.init(frame, init_rect)
            # -------- open MP4 writer --------------------------------
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                args.output, fourcc, 30,           # FPS
                (frame.shape[1], frame.shape[0])   # width, height
            )
            # ----------------------------------------------------------
            first_frame = False
        else:
            outputs = tracker.track(frame)
            score = outputs['best_score']
            # print(i, score)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                if score>0.8:
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)
            writer.write(frame)             # save annotated frame
            cv2.imshow(video_name, frame)
            cv2.waitKey(1)
            i += 1
            if i % 100 == 0:
              print(i)
  
    if writer is not None:
      writer.release()
      print(f"[demo.py] video saved to {args.output}")


if __name__ == '__main__':
    main()
