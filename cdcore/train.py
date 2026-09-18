# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import CHECKPOINTS_DIR
from .config import build_config
from .datasets import build_loaders
from .pretrained import load_mit_b2
from .weights import ensure_mit_b2
from .models import ChangeFormerV6
from .metric_tool import ConfuseMatrixMeter


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(cfg, device):
    return ChangeFormerV6(input_nc=3, output_nc=cfg.n_class,
                          embed_dim=cfg.embed_dim).to(device)


def multiscale_ce(preds, gt, weights, class_weight=None):
    loss = 0.0
    for i, p in enumerate(preds):
        if p.shape[-1] != gt.shape[-1]:
            tgt = F.interpolate(gt.unsqueeze(1).float(), p.shape[-2:],
                                mode="nearest").squeeze(1).long()
        else:
            tgt = gt
        loss = loss + weights[i] * F.cross_entropy(p, tgt, weight=class_weight,
                                                    ignore_index=255)
    return loss


def dice_ce_loss(pred, gt, class_weight=None):
    ce = F.cross_entropy(pred, gt, weight=class_weight, ignore_index=255)
    prob = torch.softmax(pred, 1)[:, 1]
    m = (gt != 255)
    p = prob[m]; t = (gt[m] == 1).float()
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


@torch.no_grad()
def evaluate(net, loader, device, n_class=2):
    net.eval()
    meter = ConfuseMatrixMeter(n_class=n_class)
    for batch in loader:
        A = batch["A"].to(device); B = batch["B"].to(device); L = batch["L"].to(device)
        pr = net(A, B)[-1].argmax(1).cpu().numpy()
        meter.update_cm(pr, L.cpu().numpy())
    return meter.get_scores()


def train(cfg, tag="main", verbose=True):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[env] device =", device, flush=True)
    loaders = build_loaders(cfg)
    if getattr(cfg, "val_subset", 0) and cfg.val_subset > 0:
        val_ds = loaders["val"].dataset
        rng = np.random.RandomState(cfg.seed)
        idx = sorted(rng.choice(len(val_ds), min(cfg.val_subset, len(val_ds)),
                                replace=False).tolist())
        nw = cfg.num_workers
        loaders["val"] = DataLoader(Subset(val_ds, idx), batch_size=cfg.batch_size,
                                    shuffle=False, num_workers=nw, pin_memory=True,
                                    persistent_workers=(nw > 0))
        print("[train] monitoring on fixed val subset n=%d" % len(idx), flush=True)

    net = build_model(cfg, device)
    cw = None
    if cfg.loss == "ce" and cfg.ce_class_weight is not None:
        cw = torch.tensor(cfg.ce_class_weight, device=device)
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                                weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    max_ep = int(cfg.max_epochs)
    sched = lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: max(0.0, (max_ep - e) / float(max_ep)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_dir = os.path.join(str(CHECKPOINTS_DIR), cfg.project_name(tag))
    if getattr(cfg, "resume", False) and not os.path.exists(os.path.join(ckpt_dir, "last_ckpt.pt")):
        cand = [os.path.join(str(CHECKPOINTS_DIR), d)
                for d in os.listdir(str(CHECKPOINTS_DIR))
                if ("_" + tag + "_ed" + str(cfg.embed_dim)) in d
                and os.path.exists(os.path.join(str(CHECKPOINTS_DIR), d, "last_ckpt.pt"))]
        if cand:
            ckpt_dir = sorted(cand)[-1]
            print("[resume] exact run dir missing, reuse:", os.path.basename(ckpt_dir), flush=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # MiT-B2 pre-encoder init
    resume_path = os.path.join(ckpt_dir, "last_ckpt.pt")
    will_resume = getattr(cfg, "resume", False) and os.path.exists(resume_path)
    if getattr(cfg, "pretrain", "") and not will_resume:
        weights = ensure_mit_b2(cfg.pretrain)
        report = load_mit_b2(net.Tenc_x2, weights)
        report["weights"] = weights
        json.dump(report, open(os.path.join(ckpt_dir, "pretrain_report.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)
    elif getattr(cfg, "pretrain", "") and will_resume:
        print("[pretrain] --resume 且存在 last_ckpt，跳过预训练加载（由检查点覆盖）",
              flush=True)

    history = {"epoch": [], "train_loss": [], "val_miou": [], "val_iou_change": [],
               "val_recall_change": [], "val_f1_change": [], "val_epoch": [], "lr": []}
    best_miou, best_epoch = -1.0, 0
    n_train = len(loaders["train"])
    val_every = max(1, getattr(cfg, "val_every", 1))
    start_epoch = 0

    # resume
    if getattr(cfg, "resume", False) and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        net.load_state_dict(ck["model_G_state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        best_miou = float(ck.get("best_val_miou", -1.0))
        best_epoch = int(ck.get("best_epoch", start_epoch - 1))
        if "optimizer_G_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_G_state_dict"])
            scaler.load_state_dict(ck["scaler_state_dict"])
            sched.load_state_dict(ck["scheduler_state_dict"])
            if isinstance(ck.get("history"), dict):
                history = ck["history"]; history.setdefault("val_epoch", [])
            print("[resume] full-state resume at epoch %d (best %.4f @ep%d)"
                  % (start_epoch - 1, best_miou, best_epoch), flush=True)
        else:
            sched = lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda e: max(0.0, (max_ep - e) / float(max_ep)),
                last_epoch=start_epoch - 1)
            hp = os.path.join(ckpt_dir, "history.json")
            if os.path.exists(hp):
                history = json.load(open(hp, encoding="utf-8")); history.setdefault("val_epoch", [])
            print("[resume] weights-only resume at epoch %d" % (start_epoch - 1), flush=True)
        if start_epoch >= cfg.max_epochs:
            print("[resume] already at max_epochs=%d" % cfg.max_epochs)
            plot_history(history, ckpt_dir)
            return {"tag": tag, "best_val_miou": best_miou, "best_epoch": best_epoch,
                    "epochs_run": start_epoch, "ckpt_dir": ckpt_dir, "resumed": True}

    for epoch in range(start_epoch, cfg.max_epochs):
        net.train(); t0 = time.time(); run_loss = 0.0
        for batch in loaders["train"]:
            A = batch["A"].to(device, non_blocking=True)
            B = batch["B"].to(device, non_blocking=True)
            L = batch["L"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                preds = net(A, B)
                if cfg.loss == "dicece":
                    loss = dice_ce_loss(preds[-1], L, cw)
                    if cfg.multi_scale_train:
                        loss = loss + 0.5 * multiscale_ce(
                            preds[:-1], L,
                            [w / sum(cfg.multi_pred_weights) for w in cfg.multi_pred_weights[:-1]], cw)
                else:
                    loss = (multiscale_ce(preds, L, tuple(cfg.multi_pred_weights), cw)
                            if cfg.multi_scale_train else F.cross_entropy(preds[-1], L, weight=cw))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += float(loss.detach())
        sched.step()
        t_train = time.time() - t0

        do_val = ((epoch + 1) % val_every == 0) or (epoch == cfg.max_epochs - 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(run_loss / n_train)
        history["lr"].append(opt.param_groups[0]["lr"])

        if do_val:
            tv0 = time.time()
            scores = evaluate(net, loaders["val"], device, cfg.n_class)
            t_val = time.time() - tv0
            miou = float(scores["miou"]); iou_c = float(scores["iou_1"])
            rec_c = float(scores["recall_1"]); f1_c = float(scores["F1_1"])
            history["val_miou"].append(miou); history["val_iou_change"].append(iou_c)
            history["val_recall_change"].append(rec_c); history["val_f1_change"].append(f1_c)
            history["val_epoch"].append(epoch)
            if verbose:
                print("[%s] ep %3d/%d loss %.4f | val mIoU %.4f IoU-c %.4f "
                      "Recall-c %.4f F1-c %.4f | train %.0fs val %.0fs"
                      % (tag, epoch, cfg.max_epochs - 1, run_loss / n_train, miou,
                         iou_c, rec_c, f1_c, t_train, t_val), flush=True)
            if miou > best_miou:
                best_miou, best_epoch = miou, epoch
                torch.save({"epoch": epoch, "model_G_state_dict": net.state_dict(),
                            "best_val_miou": best_miou, "best_epoch": best_epoch,
                            "cfg": cfg.serializable()},
                           os.path.join(ckpt_dir, "best_ckpt.pt"))
        else:
            for k in ("val_miou", "val_iou_change", "val_recall_change", "val_f1_change"):
                history[k].append(float("nan"))
            if verbose:
                print("[%s] ep %3d/%d loss %.4f | train %.0fs (skip val)"
                      % (tag, epoch, cfg.max_epochs - 1, run_loss / n_train, t_train),
                      flush=True)

        json.dump(history, open(os.path.join(ckpt_dir, "history.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
        torch.save({"epoch": epoch, "model_G_state_dict": net.state_dict(),
                    "optimizer_G_state_dict": opt.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "best_val_miou": best_miou, "best_epoch": best_epoch,
                    "history": history, "cfg": cfg.serializable()},
                   os.path.join(ckpt_dir, "last_ckpt.pt"))

        if do_val and getattr(cfg, "patience", 0) and (epoch - best_epoch) >= cfg.patience:
            print("[%s] early stop at ep %d: no mIoU gain for %d epochs (best %.4f @ep%d)"
                  % (tag, epoch, cfg.patience, best_miou, best_epoch), flush=True)
            break

    plot_history(history, ckpt_dir)
    epochs_run = (history["epoch"][-1] + 1) if history["epoch"] else start_epoch
    summary = {"tag": tag, "best_val_miou": best_miou, "best_epoch": best_epoch,
               "epochs_run": epochs_run, "ckpt_dir": ckpt_dir}
    json.dump(summary, open(os.path.join(ckpt_dir, "train_summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    print("DONE", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def plot_history(h, out_dir):
    ve = [e for e, v in zip(h["epoch"], h["val_miou"]) if not np.isnan(v)]
    vm = [v for v in h["val_miou"] if not np.isnan(v)]
    vi = [v for v in h["val_iou_change"] if not np.isnan(v)]
    vr = [v for v in h["val_recall_change"] if not np.isnan(v)]
    vf = [v for v in h["val_f1_change"] if not np.isnan(v)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], color="k"); ax[0].set_title("train loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(ve, vm, "-o", ms=3, label="mIoU")
    ax[1].plot(ve, vi, "-o", ms=3, label="IoU(change)")
    ax[1].plot(ve, vr, "-o", ms=3, label="recall(change)")
    ax[1].plot(ve, vf, "-o", ms=3, label="F1(change)")
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("val metrics"); ax[1].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=130)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/uav_mitb2.yaml")
    p.add_argument("--tag", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--embed_dim", type=int, default=None)
    p.add_argument("--loss", default=None)
    p.add_argument("--no_preprocess", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_multiscale", action="store_true")
    p.add_argument("--no_pretrain", action="store_true", help="禁用 MiT-B2 预训练（改为从零训练）")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--val_subset", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    overrides = dict(dataset=a.dataset, max_epochs=a.max_epochs,
                     batch_size=a.batch_size, lr=a.lr, embed_dim=a.embed_dim,
                     loss=a.loss, num_workers=a.num_workers, patience=a.patience,
                     val_every=a.val_every, val_subset=a.val_subset)
    if a.no_preprocess: overrides["use_preprocess"] = False
    if a.no_augment: overrides["use_augment"] = False
    if a.no_multiscale: overrides["multi_scale_train"] = False
    if a.no_pretrain: overrides["pretrain"] = ""
    if a.resume: overrides["resume"] = True
    cfg = build_config(a.config, overrides)
    tag = a.tag or "uav_mitb2"
    train(cfg, tag)


if __name__ == "__main__":
    main()
