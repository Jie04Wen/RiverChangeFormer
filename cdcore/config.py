# -*- coding: utf-8 -*-
import os
import json

from .paths import DATASETS_DIR, PROJECT_ROOT, resolve, dataset_root

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover - 未装 yaml 时退化为 json
    yaml = None


class Config:
    # ---------- data ----------
    dataset = "mytraindata"
    test_dataset = "UAVtest256"
    img_size = 256
    num_workers = 4
    use_preprocess = True
    preprocess_mode = "lab_clahe"
    use_augment = True
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

    # ---------- model (ChangeFormerV6) ----------
    net_G = "ChangeFormerV6"
    embed_dim = 256
    n_class = 2

    # ---------- train ----------
    batch_size = 16
    lr = 1e-4
    optimizer = "adamw"
    weight_decay = 0.01
    max_epochs = 120
    lr_policy = "linear"
    loss = "ce"                  # ce | dicece
    multi_scale_train = True
    multi_pred_weights = [0.5, 0.5, 0.5, 0.8, 1.0]
    ce_class_weight = (1.0, 2.0)   # 变化（前景）小样本加权
    seed = 2026
    save_every = 10
    patience = 30                # 验证 mIoU 连续 patience 轮不提升则早停；0=不早停
    val_every = 2                # 每 N 轮验证一次
    val_subset = 0               # 训练期监控验证子集大小；0=完整 val
    resume = False               # True：从同 tag 的 last_ckpt.pt 断点续训
    pretrain = "pretrained/mit_b2_imagenet.pth"  # MiT-B2 ImageNet 预训练编码器；置 "" 则从零训练

    # ---------- postprocess ----------
    pp_open = 3
    pp_close = 5
    pp_min_area = 0
    pp_fill_holes = True

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._resolve_paths()

    def _resolve_paths(self):
        self.data_root_raw = dataset_root(self.dataset, False)
        self.data_root_pre = dataset_root(self.dataset, True)
        self.data_root = self.data_root_pre if self.use_preprocess else self.data_root_raw

        meta_path = os.path.join(self.data_root, "meta.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, encoding="utf-8"))
            st = meta.get("channel_stats_sample", {})
            if "mean" in st:
                self.norm_mean = tuple(st["mean"])
                self.norm_std = tuple(st["std"])

        if self.pretrain:
            self.pretrain = resolve(self.pretrain)

    # ---------- YAML / 序列化 ----------
    @classmethod
    def from_yaml(cls, path):
        path = resolve(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                if yaml is None:
                    raise RuntimeError("需要 PyYAML：pip install pyyaml")
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        return cls(**data)

    def project_name(self, tag="main"):
        return ("%s_%s_%s_ed%d_b%d_lr%s_%s_ep%d_pre%s_aug%s"
                % (self.dataset, self.net_G, tag, self.embed_dim, self.batch_size,
                   self.lr, self.optimizer, self.max_epochs,
                   int(self.use_preprocess), int(self.use_augment)))

    def serializable(self):
        d = dict(self.__dict__)
        return d


def _read_yaml_or_json(path):
    path = resolve(path)
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("需要 PyYAML：pip install pyyaml")
            return yaml.safe_load(f) or {}
        return json.load(f)


def build_config(config_path=None, overrides=None):

    data = {}
    if config_path:
        p = resolve(config_path)
        if p and os.path.exists(p):
            data = _read_yaml_or_json(p)
    for k, v in (overrides or {}).items():
        if v is not None:
            data[k] = v
    return Config(**data)
