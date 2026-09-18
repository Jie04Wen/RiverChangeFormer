# -*- coding: utf-8 -*-
"""
  * A/B/label 同步几何增强（水平/垂直翻转、90° 旋转），label 最近邻；
  * A、B 独立颜色抖动（亮度/对比度/轻微模糊），学习对光照差异的不变性。
颜色空间 LAB-CLAHE+HSV 增强已离线固化到 <name>_pre 平行数据集，训练直接读取。
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


class ChangeDataset(Dataset):
    def __init__(self, root, split="train", img_size=256, is_train=True,
                 use_augment=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_train = is_train
        self.use_augment = use_augment and is_train
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)
        list_file = os.path.join(root, "list", split + ".txt")
        with open(list_file, "r", encoding="utf-8-sig") as f:
            self.names = [x.strip() for x in f if x.strip()]

    def __len__(self):
        return len(self.names)

    # ---- A/B/label 同步几何增强 ----
    @staticmethod
    def _geo_aug(A, B, L, rng):
        if rng.random() < 0.5:                 # 水平翻转
            A, B, L = A[:, ::-1], B[:, ::-1], L[:, ::-1]
        if rng.random() < 0.5:                 # 垂直翻转
            A, B, L = A[::-1], B[::-1], L[::-1]
        k = int(rng.integers(0, 4))            # 90° 旋转
        if k:
            A = np.rot90(A, k); B = np.rot90(B, k); L = np.rot90(L, k)
        return np.ascontiguousarray(A), np.ascontiguousarray(B), np.ascontiguousarray(L)

    # ---- A、B 独立光度抖动（模拟双时相光照差异）----
    @staticmethod
    def _photo_aug(img, rng):
        img = img.astype(np.float32)
        a = rng.uniform(0.85, 1.15)            # 亮度增益
        b = rng.uniform(-12, 12)               # 亮度偏置
        c = rng.uniform(0.85, 1.15)            # 对比度
        m = img.reshape(-1, 3).mean(0)
        img = (img - m) * c * a + m + b
        if rng.random() < 0.3:                 # 轻微高斯模糊（离焦差异）
            img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.8))
        return np.clip(img, 0, 255).astype(np.uint8)

    def _to_tensor(self, A, B, L):
        A = (A.astype(np.float32) / 255.0 - self.mean) / self.std
        B = (B.astype(np.float32) / 255.0 - self.mean) / self.std
        A = torch.from_numpy(A.transpose(2, 0, 1)).float()
        B = torch.from_numpy(B.transpose(2, 0, 1)).float()
        L = torch.from_numpy((L > 127).astype(np.int64))   # {0,1}
        return A, B, L

    def __getitem__(self, idx):
        name = self.names[idx]
        A = read_rgb(os.path.join(self.root, "A", name))
        B = read_rgb(os.path.join(self.root, "B", name))
        L = cv2.imread(os.path.join(self.root, "label", name), cv2.IMREAD_GRAYSCALE)
        if L is None:
            raise FileNotFoundError(os.path.join(self.root, "label", name))
        if self.img_size and A.shape[0] != self.img_size:
            A = cv2.resize(A, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            B = cv2.resize(B, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            L = cv2.resize(L, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng()
        if self.use_augment:
            A, B, L = self._geo_aug(A, B, L, rng)
            A = self._photo_aug(A, rng); B = self._photo_aug(B, rng)
        A, B, L = self._to_tensor(A, B, L)
        return {"A": A, "B": B, "L": L, "name": name}


def build_loaders(cfg, shuffle_test=False):
    common = dict(img_size=cfg.img_size, mean=cfg.norm_mean, std=cfg.norm_std)
    train = ChangeDataset(cfg.data_root, "train", is_train=True,
                          use_augment=cfg.use_augment, **common)
    val = ChangeDataset(cfg.data_root, "val", is_train=False, use_augment=False, **common)
    test = ChangeDataset(cfg.data_root, "test", is_train=False, use_augment=False, **common)
    nw = cfg.num_workers
    mk = dict(persistent_workers=(nw > 0)) if nw > 0 else {}
    pf = dict(prefetch_factor=4) if nw > 0 else {}
    return {
        "train": DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=nw, drop_last=False, pin_memory=True, **mk, **pf),
        "val": DataLoader(val, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True, **mk, **pf),
        "test": DataLoader(test, batch_size=1, shuffle=shuffle_test,
                           num_workers=0, pin_memory=True),
    }
