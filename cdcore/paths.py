# -*- coding: utf-8 -*-
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = PROJECT_ROOT / "datasets"        # 数据集（A/B/label/list）
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"    # MiT-B2 ImageNet 预训练权重
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"   # 训练权重 / history
RESULTS_DIR = OUTPUTS_DIR / "results"           # 指标 / 预测 / 可视化
CONFIGS_DIR = PROJECT_ROOT / "configs"

# MiT-B2 ImageNet-1k 预训练权重自动下载
MIT_B2_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
    "segformer/mit_b2_20220624-66e8bf70.pth"
)
MIT_B2_NAME = "mit_b2_imagenet.pth"
MIT_B2_PATH = PRETRAINED_DIR / MIT_B2_NAME

for _d in (DATASETS_DIR, PRETRAINED_DIR, OUTPUTS_DIR,
           CHECKPOINTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def resolve(path):
    if path is None or path == "":
        return path
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def dataset_root(name, preprocessed=False):
    return str(DATASETS_DIR / (name + ("_pre" if preprocessed else "")))
