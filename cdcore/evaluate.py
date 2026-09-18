# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from .paths import CHECKPOINTS_DIR, RESULTS_DIR, PROJECT_ROOT
from .config import Config
from .datasets import ChangeDataset
from .model_io import load_model
from .postprocess import refine_mask
from .metric_tool import ConfuseMatrixMeter, scores_from_meter


def find_ckpt_dir(tag):
    ms = [d for d in os.listdir(str(CHECKPOINTS_DIR)) if ("_" + tag + "_ed256") in d]
    if not ms:
        raise FileNotFoundError("no checkpoint dir for tag %s under %s"
                                % (tag, CHECKPOINTS_DIR))
    return os.path.join(str(CHECKPOINTS_DIR), sorted(ms)[-1])


@torch.no_grad()
def evaluate_tag(tag, ckpt_name="best_ckpt.pt", save_pred=True, split="test",
                 dataset_override=None, max_save=100000):
    ckpt_dir = find_ckpt_dir(tag)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ck_cfg = ckpt.get("cfg", {})
    cfgd = dict(ck_cfg)
    if dataset_override:
        cfgd["dataset"] = dataset_override
    cfg = Config(**cfgd)
    # 独立测试集 meta 不含通道统计：归一化必须沿用 checkpoint 内的训练集口径
    if dataset_override and ck_cfg.get("norm_mean") is not None:
        cfg.norm_mean = list(ck_cfg["norm_mean"])
        cfg.norm_std = list(ck_cfg["norm_std"])
    cfg.num_workers = 0

    # 后处理参数：优先验证集调优结果
    pp_open, pp_close, pp_min_area, pp_fill = (cfg.pp_open, cfg.pp_close,
                                               cfg.pp_min_area, cfg.pp_fill_holes)
    tune_path = os.path.join(str(RESULTS_DIR), tag, "post_tuning.json")
    if os.path.exists(tune_path):
        bp = json.load(open(tune_path, encoding="utf-8")).get("best_params", {})
        pp_open = bp.get("open_k", pp_open); pp_close = bp.get("close_k", pp_close)
        pp_min_area = bp.get("min_area", pp_min_area); pp_fill = bp.get("fill_holes", pp_fill)
    pp = dict(open_k=pp_open, close_k=pp_close, min_area=pp_min_area, fill_holes=pp_fill)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, device = load_model(ckpt_path, device)

    ds = ChangeDataset(cfg.data_root, split=split, img_size=cfg.img_size,
                       is_train=False, use_augment=False,
                       mean=cfg.norm_mean, std=cfg.norm_std)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    m_raw = ConfuseMatrixMeter(2); m_pp = ConfuseMatrixMeter(2)
    out_dir = os.path.join(str(RESULTS_DIR), tag)
    raw_sub = "pred_raw" if not dataset_override else "pred_raw_" + dataset_override
    post_sub = "pred_post" if not dataset_override else "pred_post_" + dataset_override
    if save_pred:
        os.makedirs(os.path.join(out_dir, raw_sub), exist_ok=True)
        os.makedirs(os.path.join(out_dir, post_sub), exist_ok=True)

    for i, batch in enumerate(dl):
        A = batch["A"].to(device); B = batch["B"].to(device); L = batch["L"].numpy()[0]
        prob = torch.softmax(net(A, B)[-1], 1)[0, 1].cpu().numpy()
        raw = (prob > 0.5).astype(np.uint8)
        post = (refine_mask((raw * 255).astype(np.uint8), **pp) > 0).astype(np.uint8)
        m_raw.update_cm(raw[None], L[None]); m_pp.update_cm(post[None], L[None])
        if save_pred and i < max_save:
            name = batch["name"][0]
            cv2.imwrite(os.path.join(out_dir, raw_sub, name), raw * 255)
            cv2.imwrite(os.path.join(out_dir, post_sub, name), post * 255)

    res = {"tag": tag, "dataset": cfg.dataset, "split": split,
           "ckpt": os.path.relpath(ckpt_path, str(PROJECT_ROOT)),
           "n_samples": len(ds), "postprocess": pp,
           "raw": scores_from_meter(m_raw), "postprocessed": scores_from_meter(m_pp)}
    os.makedirs(out_dir, exist_ok=True)
    metrics_name = ("test_metrics.json" if not dataset_override
                    else "test_metrics_%s.json" % dataset_override)
    json.dump(res, open(os.path.join(out_dir, metrics_name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="uav_mitb2")
    p.add_argument("--ckpt_name", default="best_ckpt.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--dataset", default=None,
                   help="数据集覆盖：用同一 checkpoint 在独立测试集（如 UAVtest256）上评估")
    a = p.parse_args()
    evaluate_tag(a.tag, a.ckpt_name, split=a.split, dataset_override=a.dataset)


if __name__ == "__main__":
    main()
