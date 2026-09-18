# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from .paths import CHECKPOINTS_DIR, RESULTS_DIR
from .config import Config
from .datasets import ChangeDataset
from .model_io import load_model
from .postprocess import refine_mask
from .metric_tool import ConfuseMatrixMeter
from .evaluate import find_ckpt_dir


@torch.no_grad()
def collect_raw(tag, split):
    ck = find_ckpt_dir(tag)
    ckpt = torch.load(os.path.join(ck, "best_ckpt.pt"), map_location="cpu")
    cfg = Config(**ckpt.get("cfg", {}))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, dev = load_model(os.path.join(ck, "best_ckpt.pt"), dev)
    ds = ChangeDataset(cfg.data_root, split=split, img_size=cfg.img_size,
                       is_train=False, use_augment=False,
                       mean=cfg.norm_mean, std=cfg.norm_std)
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    raws, gts = [], []
    for b in dl:
        prob = torch.softmax(net(b["A"].to(dev), b["B"].to(dev))[-1], 1)[0, 1].cpu().numpy()
        raws.append((prob > 0.5).astype(np.uint8)); gts.append(b["L"].numpy()[0])
    return raws, gts


def score(raw_list, gt_list, **pp):
    m = ConfuseMatrixMeter(2)
    for raw, gt in zip(raw_list, gt_list):
        post = (refine_mask((raw * 255).astype(np.uint8), **pp) > 0).astype(np.uint8)
        m.update_cm(post[None], gt[None])
    s = m.get_scores()
    return s["miou"], s["iou_1"], s["recall_1"], s["precision_1"], s["F1_1"]


def grid_search(raws, gts):
    grid = [(ok, ck, ma, fh) for ok in [0, 3] for ck in [0, 5]
            for ma in [0, 32, 64] for fh in [True]]
    best = None
    for ok, ck, ma, fh in grid:
        miou, iou, rec, prec, f1 = score(raws, gts, open_k=ok, close_k=ck,
                                         min_area=ma, fill_holes=fh)
        if best is None or miou > best[1]["mIoU"]:
            best = ((ok, ck, ma, fh), dict(mIoU=miou, IoU_change=iou,
                                           Recall_change=rec, Precision_change=prec,
                                           F1_change=f1))
    return best, grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="uav_mitb2")
    a = p.parse_args()
    print("collect raw predictions on val...", flush=True)
    va_raw, va_gt = collect_raw(a.tag, "val")
    (params, va_score), _ = grid_search(va_raw, va_gt)
    ok, ck, ma, fh = params
    print("best val params: open=%d close=%d min_area=%d fill=%s" % (ok, ck, ma, fh),
          va_score, flush=True)

    te_raw, te_gt = collect_raw(a.tag, "test")
    base = ConfuseMatrixMeter(2)
    for raw, gt in zip(te_raw, te_gt):
        base.update_cm(raw[None], gt[None])
    sb = base.get_scores()
    te_raw_score = dict(mIoU=sb["miou"], IoU_change=sb["iou_1"],
                        Recall_change=sb["recall_1"], Precision_change=sb["precision_1"],
                        F1_change=sb["F1_1"], OA=sb["acc"])
    miou, iou, rec, prec, f1 = score(te_raw, te_gt, open_k=ok, close_k=ck,
                                     min_area=ma, fill_holes=fh)
    te_post_score = dict(mIoU=miou, IoU_change=iou, Recall_change=rec,
                         Precision_change=prec, F1_change=f1)
    out = {"tag": a.tag,
           "best_params": {"open_k": ok, "close_k": ck, "min_area": ma, "fill_holes": fh},
           "val_score": va_score, "test_raw": te_raw_score, "test_post": te_post_score}
    od = os.path.join(str(RESULTS_DIR), a.tag); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "post_tuning.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
