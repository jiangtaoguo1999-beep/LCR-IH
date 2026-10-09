"""
train_stegformer_crop40_finetune_v3.py

在 v2 基础上新增：
  ★ deblur 裁剪课程（Deblur Crop Curriculum）
    Phase 1 不再从第一天就强制 crop_max=0.40，
    而是在 deblur_ramp_epochs 个 epoch 内从 crop_min 线性增长到 crop_max，
    让 deblur 模块先学简单修复再逐步攻克大裁剪。

其余训练逻辑与 v2 完全一致。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import timm.scheduler
from tensorboardX import SummaryWriter
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from datasets import *  # noqa: F401,F403
from stegformer_crop_model1 import (
    FreqLoss,
    StegFormerCropRobust,
    build_crop_robust_model,
    load_pretrained_base,
    set_requires_grad,
)


class L1CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((x - y) ** 2 + self.eps).mean()


class RestrictLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = F.relu(x - 1.0)
        low = F.relu(-x)
        return 0.5 * (high.pow(2).mean() + low.pow(2).mean())


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> float:
    x = torch.clamp(x.detach(), 0, 1)
    y = torch.clamp(y.detach(), 0, 1)
    mse = F.mse_loss(x, y).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def masked_l1(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1, -1)
    denom = mask.sum().clamp_min(eps)
    return (torch.abs(x - y) * mask).sum() / denom


def boundary_mask_from_valid(valid_mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    invalid = 1.0 - valid_mask[:, :1]
    if invalid.max() <= 0:
        return torch.zeros_like(invalid)
    k = radius * 2 + 1
    dilated = F.max_pool2d(invalid, kernel_size=k, stride=1, padding=radius)
    return (dilated - invalid).clamp(0.0, 1.0)


def simulate_crop(img: torch.Tensor, crop_min: float, crop_max: float, fill_mode: str = "mean") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, c, h, w = img.shape
    device = img.device
    dtype = img.dtype
    valid_mask = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    bboxes = []

    crop_min = max(0.0, float(crop_min))
    crop_max = max(crop_min, min(float(crop_max), 0.95))

    for i in range(b):
        area_ratio = random.uniform(crop_min, crop_max)
        area = max(1.0, area_ratio * h * w)
        aspect = random.uniform(0.5, 2.0)
        crop_h = max(1, min(h - 1, int(round(math.sqrt(area / aspect)))))
        crop_w = max(1, min(w - 1, int(round(math.sqrt(area * aspect)))))
        y1 = random.randint(0, h - crop_h)
        x1 = random.randint(0, w - crop_w)
        y2, x2 = y1 + crop_h, x1 + crop_w
        valid_mask[i, :, y1:y2, x1:x2] = 0.0
        bboxes.append([x1, y1, x2, y2])

    fill_mode = str(fill_mode).lower()
    if fill_mode == "zero":
        fill = torch.zeros_like(img)
    elif fill_mode == "half":
        fill = torch.full_like(img, 0.5)
    elif fill_mode == "mean":
        mean = img.detach().mean(dim=(2, 3), keepdim=True)
        fill = mean.expand_as(img)
    elif fill_mode == "noise":
        fill = torch.rand_like(img)
    else:
        raise ValueError(f"Unknown fill_mode={fill_mode!r}")

    padded = img * valid_mask + fill * (1.0 - valid_mask)
    bbox = torch.tensor(bboxes, device=device, dtype=torch.float32)
    return padded, valid_mask, bbox


def crop_ratio_to_key(ratio: float) -> str:
    return f"crop{int(round(float(ratio) * 100)):02d}"


def args_to_dict(args) -> Dict:
    out = {}
    for k in dir(args):
        if k.startswith("_"):
            continue
        v = getattr(args, k)
        if callable(v):
            continue
        if isinstance(v, torch.device):
            v = str(v)
        out[k] = v
    return out


def get_phase(epoch: int, args) -> int:
    return 0 if epoch < args.deblur_start_epoch else 1


AUX_MODULE_NAMES = ("aux_encoder", "aux_embedder", "aux_extractor", "prior_decoder", "fusion")


def _set_aux_grad(model, requires_grad: bool) -> None:
    for name in AUX_MODULE_NAMES:
        m = getattr(model, name, None)
        if m is not None:
            set_requires_grad(m, requires_grad)


def _set_aux_train(model, train_mode: bool) -> None:
    for name in AUX_MODULE_NAMES:
        m = getattr(model, name, None)
        if m is not None:
            m.train(train_mode)


def apply_phase(model: StegFormerCropRobust, phase: int, args) -> None:
    if phase == 0:
        _set_aux_grad(model, True)
        model.freeze_deblur()
    else:
        model.freeze_base()
        _set_aux_grad(model, False)
        model.unfreeze_deblur()
        model.base_encoder.eval()
        model.base_decoder.eval()
        _set_aux_train(model, False)
        torch.cuda.empty_cache()


def enable_deblur_grad_checkpoint(model: StegFormerCropRobust) -> None:
    if not (model.use_deblur and model.deblur is not None):
        return
    unet = getattr(model.deblur, "unet", None)
    if unet is None:
        print("[grad_checkpoint] 未找到 model.deblur.unet，跳过")
        return

    patched = 0
    for layer_name in ("enc1", "enc2", "enc3", "bot"):
        layer = getattr(unet, layer_name, None)
        if layer is None:
            continue
        orig = layer.forward

        def make_cp(orig_fwd):
            def _forward(x):
                return grad_checkpoint(orig_fwd, x, use_reentrant=False)
            return _forward

        layer.forward = make_cp(orig)
        patched += 1

    print(f"[grad_checkpoint] deblur UNet {patched} 层已启用梯度检查点")


def crop_curriculum(epoch: int, args, last_val_psnr_secret: float | None) -> Tuple[float, float, float]:
    if epoch < args.warmup_epochs:
        return 0.0, 0.0, 0.0
    if args.start_crop_psnr > 0 and last_val_psnr_secret is not None:
        if last_val_psnr_secret < args.start_crop_psnr:
            return 0.0, 0.0, 0.0
    t = 1.0 if args.ramp_epochs <= 0 else min(1.0, max(0.0, (epoch - args.warmup_epochs) / float(args.ramp_epochs)))
    return args.crop_prob * t, args.crop_min * t, args.crop_max * t


def deblur_crop_curriculum(epoch: int, args) -> Tuple[float, float]:
    ramp_epochs = int(getattr(args, "deblur_ramp_epochs", 40))
    if ramp_epochs <= 0:
        return float(args.crop_min), float(args.crop_max)

    deblur_epochs_done = epoch - args.deblur_start_epoch
    t = min(1.0, max(0.0, deblur_epochs_done / float(ramp_epochs)))
    cur_crop_max = float(args.crop_min) + t * (float(args.crop_max) - float(args.crop_min))
    cur_crop_min = float(args.crop_min)
    return cur_crop_min, cur_crop_max


def build_optimizer(model: StegFormerCropRobust, args) -> torch.optim.Optimizer:
    base_params = list(model.base_encoder.parameters()) + list(model.base_decoder.parameters())
    aux_params = (
        list(model.aux_encoder.parameters())
        + list(model.aux_embedder.parameters())
        + list(model.aux_extractor.parameters())
        + list(model.prior_decoder.parameters())
        + list(model.fusion.parameters())
    )
    param_groups = [
        {"params": base_params, "lr": args.lr_base, "name": "base"},
        {"params": aux_params, "lr": args.lr_aux, "name": "aux"},
    ]
    if model.use_deblur and model.deblur is not None:
        deblur_params = list(model.deblur.parameters())
        param_groups.append({"params": deblur_params, "lr": args.lr_deblur, "name": "deblur"})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def merge_config_and_cli():
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config_module", type=str, default="config_finetune3_6")
    known, _ = p0.parse_known_args()

    cfg = importlib.import_module(known.config_module)
    base = cfg.Args()

    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--config_module", type=str, default=known.config_module)
    p.add_argument("--path", type=str, default=getattr(base, "path", "."))
    p.add_argument("--model_name", type=str, default=getattr(base, "model_name", "StegFormer-S_crop40_deblur"))
    p.add_argument("--use_model", type=str, default=getattr(base, "use_model", "StegFormer-S"))
    p.add_argument("--image_size_train", type=int, default=getattr(base, "image_size_train", 256))
    p.add_argument("--num_secret", type=int, default=getattr(base, "num_secret", 1))
    p.add_argument("--output_act", type=str, default=getattr(base, "output_act", None))
    p.add_argument("--norm_train", type=str, default=getattr(base, "norm_train", "clamp"))
    p.add_argument("--device", type=str, default=getattr(base, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--epochs", type=int, default=getattr(base, "epochs", 300))
    p.add_argument("--val_freq", type=int, default=getattr(base, "val_freq", 10))
    p.add_argument("--save_freq", type=int, default=getattr(base, "save_freq", 50))
    p.add_argument("--warm_up_epoch", type=int, default=getattr(base, "warm_up_epoch", 20))
    p.add_argument("--warm_up_lr_init", type=float, default=getattr(base, "warm_up_lr_init", 5e-6))
    p.add_argument("--pretrained_path", type=str, default=getattr(base, "pretrained_path", ""))
    p.add_argument("--resume", type=str, default=getattr(base, "resume", ""))

    p.add_argument("--aux_ch", type=int, default=getattr(base, "aux_ch", 32))
    p.add_argument("--aux_sz", type=int, default=getattr(base, "aux_sz", 8))
    p.add_argument("--tile_size", type=int, default=getattr(base, "tile_size", 32))
    p.add_argument("--embed_strength", type=float, default=getattr(base, "embed_strength", 0.02))
    p.add_argument("--learnable_strength", action="store_true", default=getattr(base, "learnable_strength", False))
    p.add_argument("--invalid_fill", type=str, default=getattr(base, "invalid_fill", "mean"), choices=["zero", "half", "mean", "noise"])

    p.add_argument("--use_deblur", action="store_true", default=getattr(base, "use_deblur", True))
    p.add_argument("--no_deblur", dest="use_deblur", action="store_false")
    p.add_argument("--deblur_feat_c", type=int, default=getattr(base, "deblur_feat_c", 64))
    p.add_argument("--deblur_unet_c", type=int, default=getattr(base, "deblur_unet_c", 32))
    p.add_argument("--deblur_res_scale", type=float, default=getattr(base, "deblur_res_scale", 0.25))
    p.add_argument("--deblur_start_epoch", type=int, default=getattr(base, "deblur_start_epoch", 100))
    p.add_argument("--deblur_ramp_epochs", type=int, default=getattr(base, "deblur_ramp_epochs", 40))

    p.add_argument("--lr_base", type=float, default=getattr(base, "lr_base", 1e-5))
    p.add_argument("--lr_aux", type=float, default=getattr(base, "lr_aux", 1e-4))
    p.add_argument("--lr_deblur", type=float, default=getattr(base, "lr_deblur", 2e-4))
    p.add_argument("--weight_decay", type=float, default=getattr(base, "weight_decay", 1e-4))
    p.add_argument("--freeze_encoder_epochs", type=int, default=getattr(base, "freeze_encoder_epochs", 40))
    p.add_argument("--freeze_decoder_epochs", type=int, default=getattr(base, "freeze_decoder_epochs", 5))

    p.add_argument("--warmup_epochs", type=int, default=getattr(base, "warmup_epochs", 25))
    p.add_argument("--ramp_epochs", type=int, default=getattr(base, "ramp_epochs", 60))
    p.add_argument("--start_crop_psnr", type=float, default=getattr(base, "start_crop_psnr", 27.0))
    p.add_argument("--crop_prob", type=float, default=getattr(base, "crop_prob", 0.8))
    p.add_argument("--crop_min", type=float, default=getattr(base, "crop_min", 0.02))
    p.add_argument("--crop_max", type=float, default=getattr(base, "crop_max", 0.40))

    p.add_argument("--val_crop_ratios", type=float, nargs="+", default=getattr(base, "val_crop_ratios", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]))
    p.add_argument("--val_crop_repeats", type=int, default=getattr(base, "val_crop_repeats", 1))
    p.add_argument("--val_max_batches", type=int, default=getattr(base, "val_max_batches", 0))
    p.add_argument("--train_max_batches", type=int, default=getattr(base, "train_max_batches", 0))

    p.add_argument("--lam_hide", type=float, default=getattr(base, "lam_hide", 3.0))
    p.add_argument("--lam_hide_mse", type=float, default=getattr(base, "lam_hide_mse", 30.0))
    p.add_argument("--lam_full", type=float, default=getattr(base, "lam_full", 5.0))
    p.add_argument("--lam_rec", type=float, default=getattr(base, "lam_rec", 10.0))
    p.add_argument("--lam_aux", type=float, default=getattr(base, "lam_aux", 4.0))
    p.add_argument("--lam_boundary", type=float, default=getattr(base, "lam_boundary", 0.5))
    p.add_argument("--invalid_weight", type=float, default=getattr(base, "invalid_weight", 2.0))
    p.add_argument("--lam_restrict", type=float, default=getattr(base, "lam_restrict", 0.1))

    p.add_argument("--lam_deblur_rec", type=float, default=getattr(base, "lam_deblur_rec", 14.0))
    p.add_argument("--lam_freq", type=float, default=getattr(base, "lam_freq", 4.0))
    p.add_argument("--freq_high_boost", type=float, default=getattr(base, "freq_high_boost", 2.0))
    p.add_argument("--freq_boost_ratio", type=float, default=getattr(base, "freq_boost_ratio", 0.5))
    p.add_argument("--lam_deblur_valid", type=float, default=getattr(base, "lam_deblur_valid", 0.3))

    p.add_argument("--use_grad_checkpoint", action="store_true", default=getattr(base, "use_grad_checkpoint", True))
    p.add_argument("--no_grad_checkpoint", dest="use_grad_checkpoint", action="store_false")
    p.add_argument("--seed", type=int, default=getattr(base, "seed", 42))

    cli, _ = p.parse_known_args()
    for k, v in vars(cli).items():
        setattr(base, k, v)

    base.crop_max = min(max(float(base.crop_max), float(base.crop_min)), 0.40)
    base.val_crop_ratios = [float(x) for x in base.val_crop_ratios]
    return base


def train_one_epoch(model, loader_cover, loader_secret, optimizer, loss_fn, restrict_loss_fn, freq_loss_fn, writer, args, epoch, last_val_psnr_secret):
    model.train()
    phase = get_phase(epoch, args)
    apply_phase(model, phase, args)

    if phase == 0:
        set_requires_grad(model.base_encoder, epoch >= args.freeze_encoder_epochs)
        set_requires_grad(model.base_decoder, epoch >= args.freeze_decoder_epochs)

    if phase == 0:
        crop_prob, crop_min, crop_max = crop_curriculum(epoch, args, last_val_psnr_secret)
    else:
        crop_prob = 1.0
        crop_min, crop_max = deblur_crop_curriculum(epoch, args)

    meters = {
        "loss": [],
        "hide": [],
        "full": [],
        "crop": [],
        "aux": [],
        "deblur_rec": [],
        "freq": [],
        "psnr_cover": [],
        "psnr_secret_full": [],
        "psnr_secret_train": [],
        "psnr_enhanced_train": [],
        "deblur_valid": [],
        "deblur_crop_max": [],
    }

    for batch_idx, (cover, secret) in enumerate(zip(loader_cover, loader_secret)):
        if getattr(args, "train_max_batches", 0) and batch_idx >= args.train_max_batches:
            break

        cover = cover.to(args.device, non_blocking=True)
        secret = secret.to(args.device, non_blocking=True)
        b, _, h, w = cover.shape
        valid_full = torch.ones(b, 1, h, w, device=args.device, dtype=cover.dtype)

        if phase == 1:
            with torch.no_grad():
                stego, A, stego_base = model.encode(cover, secret)
                dec_full = model.decode(stego, valid_full, run_deblur=False)
            stego = stego.detach()
            A = A.detach()
            loss_hide = loss_fn(stego, cover).detach()
            loss_hide_mse = F.mse_loss(stego, cover).detach()
            loss_full = loss_fn(dec_full.recovered.detach(), secret).detach()
            loss_aux = F.mse_loss(dec_full.A_hat.detach(), A).detach()
            loss_restrict = restrict_loss_fn(stego).detach()
        else:
            stego, A, stego_base = model.encode(cover, secret)
            dec_full = model.decode(stego, valid_full, run_deblur=False)
            loss_hide_mse = F.mse_loss(stego, cover).detach()
            loss_hide = loss_fn(stego, cover)
            loss_full = loss_fn(dec_full.recovered, secret)
            loss_aux = F.mse_loss(dec_full.A_hat, A.detach())
            loss_restrict = restrict_loss_fn(stego)

        use_crop = (phase == 1) or (crop_prob > 0 and random.random() < crop_prob)
        loss_deblur_valid = torch.zeros((), device=args.device)

        if use_crop:
            padded, mask, _ = simulate_crop(stego, crop_min, crop_max, fill_mode=args.invalid_fill)
            inv_mask = 1.0 - mask

            run_db = (phase == 1 and model.use_deblur and model.deblur is not None)
            dec_crop = model.decode(padded, mask, run_deblur=run_db)

            valid_loss = masked_l1(dec_crop.recovered, secret, mask)
            invalid_loss = masked_l1(dec_crop.recovered, secret, inv_mask)
            crop_full_loss = loss_fn(dec_crop.recovered, secret)
            bmask = boundary_mask_from_valid(mask)
            boundary_loss = masked_l1(dec_crop.recovered, secret, bmask) if bmask.sum() > 0 else torch.zeros((), device=args.device)
            loss_crop = valid_loss + args.invalid_weight * invalid_loss + 0.5 * crop_full_loss + args.lam_boundary * boundary_loss

            if phase == 0:
                loss_aux = loss_aux + F.mse_loss(dec_crop.A_hat, A.detach())

            if run_db:
                loss_deblur_rec = masked_l1(dec_crop.enhanced, secret, inv_mask)
                if bmask.sum() > 0:
                    loss_deblur_rec = loss_deblur_rec + 0.5 * masked_l1(dec_crop.enhanced, secret, bmask)
                lam_valid = float(getattr(args, "lam_deblur_valid", 0.3))
                loss_deblur_valid = masked_l1(dec_crop.enhanced, dec_crop.recovered.detach(), mask)
                loss_deblur_rec = loss_deblur_rec + lam_valid * loss_deblur_valid
                loss_freq = freq_loss_fn(dec_crop.enhanced, secret, mask=inv_mask) if args.lam_freq > 0 else torch.zeros((), device=args.device)
                enhanced_for_metric = dec_crop.enhanced
            else:
                loss_deblur_rec = torch.zeros((), device=args.device)
                loss_freq = torch.zeros((), device=args.device)
                enhanced_for_metric = dec_crop.recovered
            recovered_for_metric = dec_crop.recovered
        else:
            loss_crop = torch.zeros((), device=args.device)
            loss_deblur_rec = torch.zeros((), device=args.device)
            loss_freq = torch.zeros((), device=args.device)
            recovered_for_metric = dec_full.recovered
            enhanced_for_metric = dec_full.recovered

        if phase == 1:
            total_loss = args.lam_deblur_rec * loss_deblur_rec + args.lam_freq * loss_freq
        else:
            total_loss = (
                args.lam_hide * loss_hide
                + args.lam_hide_mse * loss_hide_mse
                + args.lam_full * loss_full
                + args.lam_rec * loss_crop
                + args.lam_aux * loss_aux
                + args.lam_restrict * loss_restrict
            )

        optimizer.zero_grad(set_to_none=True)
        if not total_loss.requires_grad:
            continue
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        meters["loss"].append(float(total_loss.detach().cpu()))
        meters["hide"].append(float(loss_hide.detach().cpu()))
        meters["full"].append(float(loss_full.detach().cpu()))
        meters["crop"].append(float(loss_crop.detach().cpu()))
        meters["aux"].append(float(loss_aux.detach().cpu()))
        meters["deblur_rec"].append(float(loss_deblur_rec.detach().cpu()))
        meters["freq"].append(float(loss_freq.detach().cpu()))
        meters["deblur_valid"].append(float(loss_deblur_valid.detach().cpu()))
        meters["deblur_crop_max"].append(float(crop_max))
        meters["psnr_cover"].append(psnr(stego, cover))
        meters["psnr_secret_full"].append(psnr(dec_full.recovered, secret))
        meters["psnr_secret_train"].append(psnr(recovered_for_metric, secret))
        meters["psnr_enhanced_train"].append(psnr(enhanced_for_metric, secret))

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in meters.items()}
    out.update({
        "crop_prob_now": float(crop_prob),
        "crop_min_now": float(crop_min),
        "crop_max_now": float(crop_max),
        "train_phase": float(phase),
    })

    for k, v in out.items():
        writer.add_scalar(f"train/{k}", v, epoch)
    return out


@torch.no_grad()
def validate(model, loader_cover, loader_secret, args, epoch):
    model.eval()
    meters = {"psnr_cover": [], "psnr_secret_full": []}
    crop_keys = []
    for ratio in args.val_crop_ratios:
        k_rec = f"psnr_secret_{crop_ratio_to_key(ratio)}"
        k_enh = f"psnr_enhanced_{crop_ratio_to_key(ratio)}"
        meters[k_rec] = []
        meters[k_enh] = []
        crop_keys.append((ratio, k_rec, k_enh))

    repeats = max(1, int(getattr(args, "val_crop_repeats", 1)))
    run_db_val = bool(model.use_deblur and model.deblur is not None)

    for idx, (cover, secret) in enumerate(zip(loader_cover, loader_secret)):
        if args.val_max_batches and idx >= args.val_max_batches:
            break

        cover = cover.to(args.device, non_blocking=True)
        secret = secret.to(args.device, non_blocking=True)
        b, _, h, w = cover.shape
        valid_full = torch.ones(b, 1, h, w, device=args.device, dtype=cover.dtype)

        stego, _, _ = model.encode(cover, secret)
        dec_full = model.decode(stego, valid_full, run_deblur=False)

        meters["psnr_cover"].append(psnr(stego, cover))
        meters["psnr_secret_full"].append(psnr(dec_full.recovered, secret))

        for ratio, k_rec, k_enh in crop_keys:
            rec_scores, enh_scores = [], []
            for _ in range(repeats):
                padded, mask, _ = simulate_crop(stego, ratio, ratio, fill_mode=args.invalid_fill)
                dec_crop = model.decode(padded, mask, run_deblur=run_db_val)
                rec_scores.append(psnr(dec_crop.recovered, secret))
                enh_scores.append(psnr(dec_crop.enhanced, secret))
            meters[k_rec].append(float(np.mean(rec_scores)))
            meters[k_enh].append(float(np.mean(enh_scores)))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in meters.items()}


def save_checkpoint(path, model, optimizer, epoch, args, best_metric):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "encoder": model.base_encoder.state_dict(),
        "decoder": model.base_decoder.state_dict(),
        "opt": optimizer.state_dict(),
        "best_metric": best_metric,
        "args": args_to_dict(args),
    }
    if model.use_deblur and model.deblur is not None:
        ckpt["deblur"] = model.deblur.state_dict()
    torch.save(ckpt, path)


def main():
    args = merge_config_and_cli()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.device = str(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    save_dir = Path(args.path) / "checkpoint" / args.model_name
    log_dir = Path(args.path) / "tensorboard_log" / args.model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    model = build_crop_robust_model(args).to(device)
    if args.pretrained_path:
        load_pretrained_base(model, args.pretrained_path, device="cpu")
    if getattr(args, "use_grad_checkpoint", True):
        enable_deblur_grad_checkpoint(model)

    optimizer = build_optimizer(model, args)
    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer=optimizer,
        t_initial=args.epochs,
        lr_min=0,
        warmup_t=args.warm_up_epoch,
        warmup_lr_init=args.warm_up_lr_init,
    )

    loss_fn = L1CharbonnierLoss().to(device)
    restrict_loss_fn = RestrictLoss().to(device)
    freq_loss_fn = FreqLoss(
        high_freq_boost=float(getattr(args, "freq_high_boost", 2.0)),
        boost_ratio=float(getattr(args, "freq_boost_ratio", 0.5)),
    ).to(device)

    start_epoch = 0
    best_crop40 = -1.0

    if getattr(args, "resume", ""):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            print(f"[resume] model: missing={len(missing)}, unexpected={len(unexpected)}")
        if "deblur" in ckpt and model.use_deblur and model.deblur is not None:
            missing, unexpected = model.deblur.load_state_dict(ckpt["deblur"], strict=False)
            print(f"[resume] deblur: missing={len(missing)}, unexpected={len(unexpected)}")
        if "opt" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["opt"])
            except Exception as e:
                print(f"[resume] optimizer 未完整恢复: {e}")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_crop40 = float(ckpt.get("best_metric", -1.0))

    model.freeze_deblur()

    train_cover_loader = globals().get("DIV2K_train_cover_loader")
    train_secret_loader = globals().get("DIV2K_train_secret_loader")
    val_cover_loader = globals().get("DIV2K_val_cover_loader")
    val_secret_loader = globals().get("DIV2K_val_secret_loader")
    if train_cover_loader is None or train_secret_loader is None:
        raise RuntimeError("Cannot find DIV2K_train_cover_loader / DIV2K_train_secret_loader")
    if val_cover_loader is None or val_secret_loader is None:
        raise RuntimeError("Cannot find DIV2K_val_cover_loader / DIV2K_val_secret_loader")

    print("Training start")
    last_val_psnr_secret = None
    history = []
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        train_stats = train_one_epoch(model, train_cover_loader, train_secret_loader, optimizer, loss_fn, restrict_loss_fn, freq_loss_fn, writer, args, epoch, last_val_psnr_secret)
        if epoch % args.val_freq == 0:
            val_stats = validate(model, val_cover_loader, val_secret_loader, args, epoch)
            for k, v in val_stats.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            last_val_psnr_secret = val_stats["psnr_secret_full"]
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}})
        with open(save_dir / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(save_dir / f"{args.model_name}.pt", model, optimizer, args.epochs - 1, args, best_crop40)
    writer.close()
    print(f"Training done. Best crop40 metric = {best_crop40:.4f}")


if __name__ == "__main__":
    main()
