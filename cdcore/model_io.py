# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from .models import ChangeFormerV6


def build_network(embed_dim=256, n_class=2):
    return ChangeFormerV6(input_nc=3, output_nc=n_class, embed_dim=embed_dim)


def load_model(ckpt_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    embed_dim = ckpt.get("cfg", {}).get("embed_dim", 256)
    n_class = ckpt.get("cfg", {}).get("n_class", 2)
    net = build_network(embed_dim, n_class).to(device)
    net.load_state_dict(ckpt["model_G_state_dict"])
    net.eval()
    return net, ckpt, device


@torch.no_grad()
def predict_prob(net, A, B, device, multi_scale_infer=False):
    """返回变化类概率图（H,W）float32，范围 0-1。A/B 为单样本 CHW tensor。"""
    A = A.unsqueeze(0).to(device); B = B.unsqueeze(0).to(device)
    outs = net(A, B)
    if multi_scale_infer:
        size = outs[-1].shape[-2:]
        prob = 0.0
        for o in outs:
            prob = prob + torch.softmax(F.interpolate(
                o, size=size, mode="bilinear", align_corners=False), 1)
        prob = prob / len(outs)
    else:
        prob = torch.softmax(outs[-1], 1)
    return prob[0, 1].cpu().numpy()
