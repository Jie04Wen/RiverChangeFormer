# RiverChangeFormer

A pixel-level change detection project for drone aerial survey of rivers/ground surface inspection. Based on the Twin Transformer change detection network **ChangeFormerV6**, it integrates the **MiT-B2 (SegFormer-B2) ImageNet pre-trained encoder**,
along with color space preprocessing, dual-phase data augmentation, multi-scale deep supervision, OpenCV post-processing, and complete evaluation/visualization.

---

## 1. Introduction

- **MiT-B2 Pre-trained Encoder Transfer**: Convert and load the MixVisionTransformer weights of the mmSeg/HuggingFace version of SegFormer-B2 into the ChangeFormer twin encoder (attention qkv splitting, 3×3 → 7×7 patch convolution with zero-padding at the center, automatic depth difference cropping), and remove the random initialization.
  
- **Color Space Preprocessing**: LAB-L channel CLAHE + HSV-S/V stretching, offline and fixed to `<dataset>_pre`, 
to alleviate the issues of uneven illumination, low contrast, and double-phase white balance drift in drone images (measured local contrast +24.6%, gradient +76.5%).
  
- **Dual-phase Data Augmentation**: Synchronous geometric transformation (flipping/90° rotation) for A/B/label, independent photometric jitter and slight blurring for A and B,
learning invariance to illumination/viewpoint differences.
  
- **Training**: Multi-scale deep supervision weighted cross-entropy, foreground class weighting (1,2), AdamW + linear decay, AMP mixed precision, select the best based on validation mIoU, early stopping, `--resume` for resuming training from a checkpoint.
  
- **Evaluation and Post-processing**: OA / mIoU / Modified IoU for Different Classes / Recall / Precision / F1 / Kappa;
OpenCV morphological processing combined with connected component post-processing. The parameters are only optimized through grid search on the **validation set**, and are only evaluated once on the test set.
  
- **Visualization**: Generate a three-panel and a five-panel combined image (with the original image, the change mask, and the overlay, labeled with green boxes for GT and red overlays for predictions), and assemble the table.
---


## 2. Setting up conda environment:

Create a virtual conda environment named rivercd with the following command:

```bash
# conda create -n rivercd python=3.10 -y
conda activate rivercd
```

Installation

```bash
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# pip install -r requirements.txt
```

---

## 3. Data Preprocess

Put the data in the `datasets/` directory, and then execute:

```bash
python -m cdcore.prepare_data all --config configs/uav_mitb2.yaml
```

This command is executed in sequence: training set verification and channel statistics (write `meta.json`), uniformly scaling the independent test set to 256 (as `UAVtest256`), and offline generating `_pre` enhanced data for both the training set and the test set.

Data directory convention: `A/` (t1), `B/` (t2), `label/` (white = 255 variation), 
`list/{train,val,test}.txt` (dataset division)

---

## 4. Dataset Description
 
The experimental data belongs to the UAV dataset of the research group. 

According to the laboratory's confidentiality regulations, it is not possible to provide the data in its entirety externally. 

You can create your own dataset following the same structure.

---


## 5. Quick Start

```bash
# 1) Training
python -m cdcore.train --config configs/uav_mitb2.yaml --tag uav_mitb2

# 2) Adjustment of post-processing parameters on the validation set
python -m cdcore.tune_postprocess --tag uav_mitb2

# 3) Internal test set + Independent test set evaluation
python -m cdcore.evaluate --tag uav_mitb2
python -m cdcore.evaluate --tag uav_mitb2 --dataset UAVtest256

# 4) Visualization
python -m cdcore.visualize --tag uav_mitb2 --n 12
python -m cdcore.visualize --tag uav_mitb2 --dataset UAVtest256 --n 12
```

The result is saved to：

- `outputs/checkpoints/<run>/`：`best_ckpt.pt`、`last_ckpt.pt`、`history.json`、
  `training_curve.png`、`pretrain_report.json`、`train_summary.json`
  
- `outputs/results/uav_mitb2/`：`test_metrics.json`（Internal validation set）、
  `test_metrics_UAVtest256.json`（Independent validation set）、`post_tuning.json`、
  `pred_raw*/`、`pred_post*/`、`vis*/`、`tri_overview*.png`

---


## 6. Disclaimer

Appreciate the work from the following repositories:
https://github.com/wgcban/ChangeFormer.git
