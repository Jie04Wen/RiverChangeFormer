# -*- coding: utf-8 -*-


import numpy as np
import cv2


def lab_clahe_enhance(rgb, clip_limit=2.0, tile_grid=(8, 8), s_gain=1.15, v_gain=1.05):
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_gain, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_gain, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def enhance(rgb, mode="lab_clahe", **kw):
    if mode in (None, "none", "None", ""):
        return rgb
    if mode == "lab_clahe":
        return lab_clahe_enhance(rgb, **kw)
    raise ValueError("unknown preprocess mode: %s" % mode)


def local_contrast(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32)
    gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    return float(l.std()), float(grad)
