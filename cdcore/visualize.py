# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2
import torch

from .paths import RESULTS_DIR
from .config import Config
from .postprocess import connected_boxes
from .evaluate import find_ckpt_dir

CHANGE_COLOR = (255, 60, 60)   # RGB 红
GT_COLOR = (80, 230, 120)      # RGB 绿


def overlay_mask(rgb, mask, color=CHANGE_COLOR, alpha=0.5):
    out = rgb.copy()
    colored = np.zeros_like(out)
    colored[mask > 0] = color
    out = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)
    out[(mask > 0)] = (0.5 * out[(mask > 0)] + 0.5 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 1)
    return out


def draw_boxes(rgb, mask, color, min_area=64):
    out = rgb.copy()
    for b in connected_boxes((mask > 0).astype(np.uint8) * 255, min_area):
        x, y, w, h = b["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
    return out


def _label_head(panel, text):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


def tri_panel(B, pred, gt=None):
    mask_view = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    ov = overlay_mask(B, pred)
    if gt is not None:
        ov = draw_boxes(ov, gt, GT_COLOR)
    panels = [_label_head(B, "t2 image"),
              _label_head(mask_view, "change mask"),
              _label_head(ov, "overlay (pred=red, GT=green)")]
    gap = np.full((B.shape[0], 4, 3), 255, np.uint8)
    return np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)


def five_panel(A, B, gt, pred):
    ov = overlay_mask(B, pred)
    gtv = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    pv = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    panels = [_label_head(A, "A t1"), _label_head(B, "B t2"), _label_head(gtv, "GT"),
              _label_head(pv, "pred"), _label_head(ov, "overlay")]
    gap = np.full((B.shape[0], 3, 3), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, gap, p], axis=1)
    return row


def visualize_tag(tag, n=12, split="test", dataset_override=None):
    ckpt = torch.load(os.path.join(find_ckpt_dir(tag), "best_ckpt.pt"), map_location="cpu")
    ck_cfg = ckpt.get("cfg", {})
    cfg = Config(**ck_cfg)
    if dataset_override:
        cfgd = dict(cfg.__dict__); cfgd["dataset"] = dataset_override
        cfg = Config(**cfgd)
        if ck_cfg.get("norm_mean") is not None:
            cfg.norm_mean = list(ck_cfg["norm_mean"])
            cfg.norm_std = list(ck_cfg["norm_std"])
    raw = cfg.data_root_raw
    pred_sub = "pred_post" if not dataset_override else "pred_post_" + dataset_override
    vis_sub = "vis" if not dataset_override else "vis_" + dataset_override
    pred_dir = os.path.join(str(RESULTS_DIR), tag, pred_sub)
    out_dir = os.path.join(str(RESULTS_DIR), tag, vis_sub)
    os.makedirs(out_dir, exist_ok=True)

    names = [x.strip() for x in open(os.path.join(raw, "list", split + ".txt"),
                                     encoding="utf-8-sig") if x.strip()][:n]
    for name in names:
        A = cv2.cvtColor(cv2.imread(os.path.join(raw, "A", name)), cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(cv2.imread(os.path.join(raw, "B", name)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(raw, "label", name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_dir, name), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        cv2.imwrite(os.path.join(out_dir, "tri_" + name),
                    cv2.cvtColor(tri_panel(B, pred, gt), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "five_" + name),
                    cv2.cvtColor(five_panel(A, B, gt, pred), cv2.COLOR_RGB2BGR))

    tris = []
    for name in names[:6]:
        p = os.path.join(out_dir, "tri_" + name)
        if os.path.exists(p):
            tris.append(cv2.imread(p))
    if tris:
        sheet = np.concatenate(tris, axis=0)
        overview = "tri_overview.png" if not dataset_override \
            else "tri_overview_%s.png" % dataset_override
        cv2.imwrite(os.path.join(str(RESULTS_DIR), tag, overview), sheet)
    print("visualization saved to", out_dir, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="uav_mitb2")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--split", default="test")
    p.add_argument("--dataset", default=None, help="数据集覆盖，如 UAVtest256")
    a = p.parse_args()
    visualize_tag(a.tag, a.n, a.split, dataset_override=a.dataset)


if __name__ == "__main__":
    main()
