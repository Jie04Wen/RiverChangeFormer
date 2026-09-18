# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
import numpy as np
import cv2

from .paths import DATASETS_DIR, resolve
from .preprocess import enhance, local_contrast
from .config import build_config


def _d(name):
    return os.path.join(str(DATASETS_DIR), name)


def read_list(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [x.strip() for x in f if x.strip()]


def fg_ratio(label_dir, names, thr=127):
    rs, no_change = [], 0
    for n in names:
        l = cv2.imread(os.path.join(label_dir, n), cv2.IMREAD_GRAYSCALE)
        r = float((l > thr).mean()); rs.append(r)
        no_change += int(r == 0)
    rs = np.asarray(rs)
    return {"n": int(len(rs)), "fg_ratio_mean": float(rs.mean()),
            "fg_ratio_median": float(np.median(rs)),
            "no_change_ratio": float(no_change / max(1, len(rs)))}


def channel_stats(root, names, max_n=600, seed=2026):

    rng = np.random.RandomState(seed)
    if len(names) > max_n:
        names = [names[i] for i in sorted(rng.choice(len(names), max_n, replace=False))]
    means, stds = [], []
    for n in names:
        for sub in ("A", "B"):
            im = cv2.cvtColor(cv2.imread(os.path.join(root, sub, n)), cv2.COLOR_BGR2RGB)
            x = im.astype(np.float32) / 255.0
            means.append(x.reshape(-1, 3).mean(0)); stds.append(x.reshape(-1, 3).std(0))
    return [float(x) for x in np.stack(means).mean(0)], \
        [float(x) for x in np.stack(stds).mean(0)]


def register(name, img_size=256, force=False):
    root = _d(name)
    meta_path = os.path.join(root, "meta.json")
    if os.path.exists(meta_path) and not force:
        print("[register] meta.json exists, skip:", name); return
    lists = {s: read_list(os.path.join(root, "list", s + ".txt"))
             for s in ("train", "val", "test")}
    for s, names in lists.items():
        miss = [n for n in names if not all(
            os.path.exists(os.path.join(root, sub, n)) for sub in ("A", "B", "label"))]
        print("[%s] %d names, missing-triplet: %d" % (s, len(names), len(miss)))
        if miss:
            raise RuntimeError("missing triplet e.g. %s" % miss[:3])
    tr, va, te = set(lists["train"]), set(lists["val"]), set(lists["test"])
    overlap = {"val_eq_test": bool(va == te), "train_val": len(tr & va),
               "train_test": len(tr & te), "union": len(tr | va | te)}
    print("split overlap:", overlap)
    mean, std = channel_stats(root, lists["train"])
    meta = {
        "dataset": name,
        "img_size": img_size,
        "label": "binary mask, white=255 change, threshold L>127 at load",
        "splits": {s: len(v) for s, v in lists.items()},
        "split_overlap": overlap,
        "fg_ratio": {s: fg_ratio(os.path.join(root, "label"), v)
                     for s, v in lists.items()},
        "channel_stats_sample": {"mean": mean, "std": std},
    }
    json.dump(meta, open(meta_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[register] wrote meta.json ->", meta_path)
    print("mean=%s\nstd=%s" % (mean, std))


def _resolve(folder, stem, exts):
    for e in exts:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return None


def build_test(src, dst, size=256, force=False):
    src, dst = _d(src), _d(dst)
    if os.path.exists(os.path.join(dst, "meta.json")) and not force:
        print("[build-test] exists, skip:", dst); return
    for sub in ("A", "B", "label", "list"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(src, "A")))
    kept = []
    for st in stems:
        pa = _resolve(os.path.join(src, "A"), st, (".jpg", ".png"))
        pb = _resolve(os.path.join(src, "B"), st, (".jpg", ".png"))
        pl = _resolve(os.path.join(src, "label"), st, (".png", ".jpg"))
        if not (pa and pb and pl):
            print("skip incomplete triplet:", st); continue
        a = cv2.imread(pa); b = cv2.imread(pb)
        l = cv2.imread(pl, cv2.IMREAD_GRAYSCALE)
        if a.shape[:2] != (size, size):
            a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR)
        if l.shape[:2] != (size, size):
            l = cv2.resize(l, (size, size), interpolation=cv2.INTER_NEAREST)
        name = st + ".png"
        cv2.imwrite(os.path.join(dst, "A", name), a)
        cv2.imwrite(os.path.join(dst, "B", name), b)
        cv2.imwrite(os.path.join(dst, "label", name), l)
        kept.append(name)
    with open(os.path.join(dst, "list", "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    open(os.path.join(dst, "list", "train.txt"), "w").close()
    open(os.path.join(dst, "list", "val.txt"), "w").close()

    meta = {"dataset": os.path.basename(dst),
            "source": "independent test resized to %d" % size,
            "img_size": size,
            "splits": {"train": 0, "val": 0, "test": len(kept)},
            "label": "binary mask, threshold L>127 at load",
            "norm_note": "no channel_stats; normalization inherited from training meta",
            "fg_ratio": {"test": fg_ratio(os.path.join(dst, "label"), kept)}}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[build-test] ->", dst, "n=", len(kept))


def preprocess(name, mode="lab_clahe", force=False):
    src = _d(name); dst = _d(name + "_pre")
    if os.path.exists(dst) and os.listdir(os.path.join(dst, "A")) and not force:
        print("[preprocess] exists, skip:", dst); return
    for sub in ("A", "B", "label", "list"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    for f in os.listdir(os.path.join(src, "list")):
        shutil.copy(os.path.join(src, "list", f), os.path.join(dst, "list", f))
    if os.path.exists(os.path.join(src, "meta.json")):
        shutil.copy(os.path.join(src, "meta.json"), os.path.join(dst, "meta.json"))
    names = sorted(os.listdir(os.path.join(src, "A")))
    before, after = [], []
    for i, name in enumerate(names):
        for sub in ("A", "B"):
            im = cv2.cvtColor(cv2.imread(os.path.join(src, sub, name)), cv2.COLOR_BGR2RGB)
            before.append(local_contrast(im))
            e = enhance(im, mode)
            after.append(local_contrast(e))
            cv2.imwrite(os.path.join(dst, sub, name), cv2.cvtColor(e, cv2.COLOR_RGB2BGR))
        shutil.copy(os.path.join(src, "label", name), os.path.join(dst, "label", name))
        if (i + 1) % 200 == 0:
            print("preprocessed %d/%d" % (i + 1, len(names)), flush=True)
    b = np.array(before); af = np.array(after)
    gain = {"mode": mode, "n_images": len(names) * 2,
            "L_std_before": float(b[:, 0].mean()), "L_std_after": float(af[:, 0].mean()),
            "grad_before": float(b[:, 1].mean()), "grad_after": float(af[:, 1].mean()),
            "L_std_gain_pct": float(100 * (af[:, 0].mean() / b[:, 0].mean() - 1)),
            "grad_gain_pct": float(100 * (af[:, 1].mean() / b[:, 1].mean() - 1))}
    json.dump(gain, open(os.path.join(dst, "preprocess_gain.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(gain, ensure_ascii=False, indent=2))
    print("[preprocess] ->", dst)


def run_all(config_path, force=False):
    cfg = build_config(config_path)
    register(cfg.dataset, cfg.img_size, force=force)
    test_src = getattr(cfg, "test_source", None)
    if test_src:
        build_test(test_src, cfg.test_dataset, cfg.img_size, force=force)
    preprocess(cfg.dataset, cfg.preprocess_mode, force=force)
    if test_src:
        preprocess(cfg.test_dataset, cfg.preprocess_mode, force=force)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("register"); pa.add_argument("--name", required=True)
    pa.add_argument("--img_size", type=int, default=256); pa.add_argument("--force", action="store_true")

    pb = sub.add_parser("build-test")
    pb.add_argument("--src", required=True); pb.add_argument("--dst", required=True)
    pb.add_argument("--size", type=int, default=256); pb.add_argument("--force", action="store_true")

    pc = sub.add_parser("preprocess"); pc.add_argument("--name", required=True)
    pc.add_argument("--mode", default="lab_clahe"); pc.add_argument("--force", action="store_true")

    pall = sub.add_parser("all"); pall.add_argument("--config", default="configs/uav_mitb2.yaml")
    pall.add_argument("--force", action="store_true")

    a = p.parse_args()
    if a.cmd == "register":
        register(a.name, a.img_size, a.force)
    elif a.cmd == "build-test":
        build_test(a.src, a.dst, a.size, a.force)
    elif a.cmd == "preprocess":
        preprocess(a.name, a.mode, a.force)
    elif a.cmd == "all":
        run_all(a.config, a.force)


if __name__ == "__main__":
    main()
