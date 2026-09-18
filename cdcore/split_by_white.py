#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_by_white.py

基于 label 的“白色像素占比/数量”进行数据集划分。
- 只读取 label 做统计；A/B 仅用于写出 train.txt/test.txt
- 支持两种策略：
  1) 固定张数：--n_test 500
  2) 限额抽测：--cap_ratio 0.30（当 n_test==0 时启用）

输出：
- out_dir/train.txt
- out_dir/test.txt
每行：<A_path> <B_path> <L_path>

示例（固定抽 500 张测试图，不分组）：
python split_by_white.py \
  --A_dir /data/A --B_dir /data/B --L_dir /data/label \
  --out_dir /data/splits --n_test 500 --group_by_prefix 0

示例（限额抽测，测试集白像素总量≤30%，按前缀分组防泄漏）：
python split_by_white.py \
  --A_dir /data/A --B_dir /data/B --L_dir /data/label \
  --out_dir /data/splits --cap_ratio 0.30 --group_by_prefix 1
"""

import os
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import random
from typing import List, Tuple, Dict

ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_ids(label_dir: str, exts=ALLOWED_EXTS) -> List[str]:
    ids = []
    for e in exts:
        for p in glob.glob(os.path.join(label_dir, f"*{e}")):
            ids.append(Path(p).stem)
    return sorted(set(ids))


def read_mask_as_binary(p: str) -> np.ndarray:
    """
    灰度读入，>0 视为白（变化），返回 uint8 二值图（0/1）
    """
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"cannot read label: {p}")
    return (m > 0).astype(np.uint8)


def find_file_with_stem(root: str, stem: str, exts=ALLOWED_EXTS) -> str:
    """
    在 root 下按给定 stem 匹配文件，按 exts 优先级返回第一个存在的路径。
    """
    for e in exts:
        p = os.path.join(root, stem + e)
        if os.path.exists(p):
            return p
    # 再做一次模糊匹配（兼容奇怪后缀）
    pats = [os.path.join(root, stem + ".*")]
    for pat in pats:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return ""


def summarize(stats: List[Tuple[str, float, int, int, int]],
              ids: List[str]) -> Dict:
    subset = [t for t in stats if t[0] in ids]
    if not subset:
        return {"count": 0, "white_pixels": 0, "white_ratio_mean": 0.0, "white_share_of_all": 0.0}
    total_wp = int(sum(t[2] for t in subset))
    total_pos_all = int(sum(t[2] for t in stats))
    mean_wr = float(np.mean([t[1] for t in subset]))
    share = (total_wp / total_pos_all) if total_pos_all > 0 else 0.0
    return {"count": len(subset), "white_pixels": total_wp,
            "white_ratio_mean": mean_wr, "white_share_of_all": share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_dir", required=True, type=str)
    ap.add_argument("--B_dir", required=True, type=str)
    ap.add_argument("--L_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    # 固定测试张数（>0 时生效，优先于 cap_ratio）
    ap.add_argument("--n_test", type=int, default=0,
                    help="固定抽取的测试集张数（>0 时生效，忽略 cap_ratio）")

    # 限额抽测参数（当 n_test==0 时生效）
    ap.add_argument("--cap_ratio", type=float, default=0.30,
                    help="测试集白像素总量 ≤ 全体白像素的这个比例")
    ap.add_argument("--min_test", type=int, default=0,
                    help="测试集最少样本数兜底（只在 n_test==0 时使用）")
    ap.add_argument("--max_test", type=int, default=10**9,
                    help="测试集最多样本数上限（只在 n_test==0 时使用）")

    # 分组与随机性
    ap.add_argument("--group_by_prefix", type=int, default=0,
                    help="1=按文件名前缀（第一个下划线前）成组防泄漏；0=不分组")
    ap.add_argument("--dont_break_group", type=int, default=0,
                    help="在 group_by_prefix=1 且 n_test>0 时，是否不拆组（可能导致测试数>n_test）。默认0=允许拆组确保正好 n_test。")
    ap.add_argument("--seed", type=int, default=123)

    # 输出格式
    ap.add_argument("--output_mode",
                    choices=["triplet", "filenames3", "label_filename", "id"],
                    default="triplet",
                    help="输出格式：triplet=三元组全路径；filenames3=三元组仅文件名；label_filename=仅label文件名；id=仅stem不带后缀")

    # 额外输出
    ap.add_argument("--save_stats_csv", type=int, default=1,
                    help="是否在 out_dir 保存 stats.csv")
    args = ap.parse_args()


    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # 1) 枚举 label ID
    ids = list_ids(args.L_dir)
    if not ids:
        raise RuntimeError(f"label 目录为空：{args.L_dir}")

    # 2) 统计每张白像素/占比
    stats = []  # (id, white_ratio, white_pixels, H, W)
    total_pos = 0
    missing_labels = []
    for img_id in ids:
        lp = find_file_with_stem(args.L_dir, img_id)
        if not lp:
            missing_labels.append(img_id)
            continue
        m = read_mask_as_binary(lp)
        wp = int(m.sum())
        H, W = m.shape[:2]
        wr = float(wp) / (H * W)
        stats.append((img_id, wr, wp, H, W))
        total_pos += wp

    if missing_labels:
        print(f"[warn] 下列 ID 在 label 目录中未找到具体文件，已跳过：{missing_labels[:10]}{' ...' if len(missing_labels)>10 else ''}")

    if not stats:
        raise RuntimeError("统计结果为空，请检查 label 路径/扩展名是否匹配")

    # 3) 排序（优先选择“有白”的、且占比高的）
    nonzero = [t for t in stats if t[2] > 0]
    zero = [t for t in stats if t[2] == 0]
    nonzero.sort(key=lambda x: (x[1], x[2]), reverse=True)  # 先按占比，再按白像素
    zero.sort(key=lambda x: (x[1], x[2]), reverse=True)
    stats_sorted = nonzero + zero  # 有白在前

    # 小工具：分组 key
    def group_key(stem: str) -> str:
        return stem.split("_")[0] if "_" in stem else stem

    # 4) 选择测试集
    if args.n_test and args.n_test > 0:
        # 固定张数：top-K
        if args.group_by_prefix:
            # 组 -> 列表
            groups: Dict[str, List[Tuple[str, float, int, int, int]]] = {}
            for rec in stats_sorted:
                g = group_key(rec[0])
                groups.setdefault(g, []).append(rec)
            # 按组排序（组代表值：组内最大 wr，再最大 wp）
            group_list = []
            for g, items in groups.items():
                top_wr = max(t[1] for t in items)
                top_wp = max(t[2] for t in items)
                group_list.append((g, top_wr, top_wp, items))
            group_list.sort(key=lambda x: (x[1], x[2]), reverse=True)

            test_ids: List[str] = []
            for g, _, _, items in group_list:
                if len(test_ids) >= args.n_test:
                    break
                # 如果不拆组（可能导致超过 n_test）
                if args.dont_break_group:
                    test_ids.extend([t[0] for t in items])
                else:
                    remain = args.n_test - len(test_ids)
                    if remain <= 0:
                        break
                    if len(items) <= remain:
                        test_ids.extend([t[0] for t in items])
                    else:
                        # 拆组只取前 remain 个（组内已按样本排序）
                        test_ids.extend([t[0] for t in items[:remain]])

            # 若不拆组且超过 n_test，就保留超出的结果并提示
            if args.dont_break_group and len(test_ids) > args.n_test:
                print(f"[info] dont_break_group=1 导致 test 数={len(test_ids)} 超过 n_test={args.n_test}（为防泄漏不拆组）")
            # 若拆组，确保正好 n_test
            if not args.dont_break_group and len(test_ids) > args.n_test:
                test_ids = test_ids[:args.n_test]

        else:
            # 不分组：直接 top-K
            test_ids = [t[0] for t in stats_sorted[:args.n_test]]

        train_ids = [i for (i, *_rest) in stats_sorted if i not in test_ids]

    else:
        # 限额抽测：cap_ratio、生效时可配 min/max 数量
        cap = int(args.cap_ratio * total_pos)
        print(f"[info] images={len(stats_sorted)}, total_white_pixels={total_pos}, cap={cap} ({args.cap_ratio:.2%})")

        if args.group_by_prefix:
            groups: Dict[str, List[Tuple[str, float, int, int, int]]] = {}
            for rec in stats_sorted:
                g = group_key(rec[0])
                groups.setdefault(g, []).append(rec)
            group_list = []
            for g, items in groups.items():
                top_wr = max(t[1] for t in items)
                top_wp = max(t[2] for t in items)
                sum_wp = sum(t[2] for t in items)
                group_list.append((g, top_wr, top_wp, sum_wp, items))
            group_list.sort(key=lambda x: (x[1], x[2]), reverse=True)

            test_ids: List[str] = []
            cum = 0
            for g, _, _, grp_wp, items in group_list:
                if cum + grp_wp > cap and cap > 0:
                    continue
                if len(test_ids) + len(items) > args.max_test:
                    continue
                test_ids.extend([t[0] for t in items])
                cum += grp_wp

            # 兜底：补到 min_test
            if len(test_ids) < args.min_test:
                flat = [t[0] for _, _, _, _, items in group_list for t in items if t[0] not in test_ids]
                for x in flat:
                    if len(test_ids) >= args.min_test:
                        break
                    test_ids.append(x)

        else:
            test_ids: List[str] = []
            cum = 0
            for img_id, wr, wp, H, W in stats_sorted:
                if cap > 0 and cum + wp > cap:
                    break
                if len(test_ids) >= args.max_test:
                    break
                test_ids.append(img_id)
                cum += wp
            # 兜底：补到 min_test
            if len(test_ids) < args.min_test:
                for img_id, wr, wp, H, W in stats_sorted:
                    if img_id in test_ids:
                        continue
                    test_ids.append(img_id)
                    if len(test_ids) >= args.min_test:
                        break

        train_ids = [i for (i, *_rest) in stats_sorted if i not in test_ids]

    # 去重并保持顺序
    test_ids = list(dict.fromkeys(test_ids))
    train_ids = list(dict.fromkeys(train_ids))

    # 5) 写出 train.txt / test.txt
    def write_split(path_txt: str, ids_list: List[str]):
        miss = 0
        with open(path_txt, "w") as f:
            for i in ids_list:
                ap = find_file_with_stem(args.A_dir, i)
                bp = find_file_with_stem(args.B_dir, i)
                lp = find_file_with_stem(args.L_dir, i)
                if not ap or not bp or not lp:
                    miss += 1
                    print(f"[warn] 跳过：A/B/L 之一缺失（{i}）")
                    continue

                if args.output_mode == "triplet":
                    line = f"{ap} {bp} {lp}\n"
                elif args.output_mode == "filenames3":
                    line = f"{os.path.basename(ap)} {os.path.basename(bp)} {os.path.basename(lp)}\n"
                elif args.output_mode == "label_filename":
                    line = f"{os.path.basename(lp)}\n"  # 例如：10019.png
                elif args.output_mode == "id":
                    line = f"{Path(lp).stem}\n"  # 例如：10019
                else:
                    line = f"{ap} {bp} {lp}\n"  # 兜底

                f.write(line)
        if miss:
            print(f"[warn] {os.path.basename(path_txt)} 有 {miss} 行因路径缺失被跳过")

    train_txt = os.path.join(args.out_dir, "train.txt")
    test_txt = os.path.join(args.out_dir, "test.txt")
    write_split(train_txt, train_ids)
    write_split(test_txt, test_ids)

    # 6) 打印统计 + 可选保存 CSV
    tr_sum = summarize(stats, train_ids)
    te_sum = summarize(stats, test_ids)
    print("[train]", json.dumps(tr_sum, ensure_ascii=False))
    print("[test ]", json.dumps(te_sum, ensure_ascii=False))
    print(f"[done] wrote: {train_txt}  &  {test_txt}")

    if args.save_stats_csv:
        import csv
        csv_path = os.path.join(args.out_dir, "stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "white_ratio", "white_pixels", "H", "W"])
            # 用原始排序（有白优先）
            for r in stats_sorted:
                w.writerow([r[0], f"{r[1]:.6f}", r[2], r[3], r[4]])
        print(f"[done] wrote: {csv_path}")


if __name__ == "__main__":
    main()
