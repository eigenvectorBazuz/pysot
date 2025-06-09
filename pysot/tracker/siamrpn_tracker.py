# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _deltas_to_bboxes(self, deltas, scale_z, center_pos, img_shape):
        """
        deltas: np.array shape (4, N) of [dx, dy, w, h] in RPN frame
        scale_z: float
        center_pos: np.array([cx, cy])
        img_shape: tuple (H, W)
        returns: list of N [x, y, w, h] in image coords
        """
        raw = deltas / scale_z        # undo search/exemplar scaling
        dx, dy, dw, dh = raw

        # compute absolute centers
        cx = dx + center_pos[0]
        cy = dy + center_pos[1]

        # to top-left
        x = cx - dw / 2
        y = cy - dh / 2

        # clip into image bounds
        H, W = img_shape
        x  = np.clip(x, 0, W)
        y  = np.clip(y, 0, H)
        dw = np.clip(dw, 1, W)
        dh = np.clip(dh, 1, H)

        boxes = np.stack([x, y, dw, dh], axis=1)  # (N,4)
        return boxes.tolist()

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img, mask=None):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs   = self.model.track(x_crop)
        score     = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        # penalties
        def change(r): return np.maximum(r, 1. / r)
        def sz(w, h): return np.sqrt((w + (w+h)*0.5) * (h + (w+h)*0.5))

        s_c     = change(sz(pred_bbox[2], pred_bbox[3]) /
                         sz(self.size[0]*scale_z, self.size[1]*scale_z))
        r_c     = change((self.size[0]/self.size[1]) /
                         (pred_bbox[2]/pred_bbox[3]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)

        # penalized & windowed score
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE

        # 1) single best
        best_idx = int(np.argmax(pscore))

        # 2) top-k candidates
        k = 400
        topk_inds     = np.argsort(pscore)[-k:][::-1]
        topk_bboxes   = self._deltas_to_bboxes(
                            pred_bbox[:, topk_inds],
                            scale_z,
                            self.center_pos,
                            img.shape[:2]
                        )
        topk_scores   = pscore[topk_inds].tolist()

        # convert that single best delta → image-space bbox
        best_delta = pred_bbox[:, best_idx:best_idx+1]  # shape (4,1)
        bx, by, bw, bh = self._deltas_to_bboxes(
                            best_delta, scale_z, self.center_pos, img.shape[:2]
                        )[0]

        # smooth & clip as before
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        cx = bx + bw/2
        cy = by + bh/2

        width, height = (
            self.size[0] * (1 - lr) + bw * lr,
            self.size[1] * (1 - lr) + bh * lr
        )
        cx, cy, width, height = self._bbox_clip(
            cx, cy, width, height, img.shape[:2]
        )

        # update state
        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])

        return {
            'bbox':            [cx - width/2, cy - height/2, width, height],
            'best_score':      float(score[best_idx]),
            'candidates':      topk_bboxes,
            'candidate_score': topk_scores
        }

