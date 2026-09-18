# -*- coding: utf-8 -*-
import cv2
import numpy as np


def refine_mask(mask, open_k=3, close_k=5, min_area=0, fill_holes=True):
    """mask: 0/1 或 0/255 的 uint8 二值图，返回同口径 0/255 uint8。"""
    m = (mask > 0).astype(np.uint8) * 255
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if fill_holes:
        m = _fill_holes(m)
    if min_area and min_area > 0:
        m = _remove_small(m, min_area)
    return m


def _fill_holes(m):
    """填充各前景连通域内部孔洞"""
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def _remove_small(m, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def connected_boxes(m, min_area=0):
    """返回后处理后各变化连通域的外接框与面积，用于巡检定位/告警。"""
    n, labels, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append({"bbox": [int(x), int(y), int(w), int(h)],
                      "area": area, "centroid": [float(cent[i][0]), float(cent[i][1])]})
    return boxes
