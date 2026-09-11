"""
train_gamus.py - Fine-tune Depth Anything V2 to regress ABSOLUTE height (metres).

WHAT THIS CHANGES
=================
The stock backbone predicts *relative inverse depth*: scale- and shift-invariant,
trained on egocentric ground-level photography. Two consequences for our problem:

  1. Domain gap - nadir aerial scenes have no vanishing points or horizon cues, so
     the features it relies on are largely absent.
  2. No absolute scale - it cannot output metres, which is the actual deliverable.

GAMUS gives us LiDAR-derived nDSM (AGL, above-ground-level height in metres) paired
with RGB. Training against that directly attacks both problems at once: the model
learns the aerial domain AND learns to emit metres, removing the need to back out
scale from SRTM for every scene.

This trains the model to predict AGL (height above local terrain), NOT absolute
elevation above sea level. That is deliberate - it composes cleanly with the
existing SRTM path:

    absolute DSM  =  SRTM terrain elevation  +  predicted AGL

which is a far better-posed problem than regressing sea-level elevation from a
single RGB image, and it sidesteps the DEM-vs-DSM mismatch documented in
docs/LIMITATIONS.md.

HARDWARE NOTES (tuned for a 6 GB RTX 3060 Laptop)
=================================================
ViT-S full fine-tune at 448x448 with AMP and batch 2 + gradient accumulation fits
in ~4-5 GB. If you OOM: lower --crop to 392, or --batch to 1 and raise --accum.

USAGE
=====
    python tools/train_gamus.py --manifest <train_manifest.json> \
        --val-manifest <val_manifest.json> --epochs 12 --out checkpoints/agl_vits.pt
"""
 
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ImageNet statistics - the DINOv2 backbone was normalised with these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HEIGHT_CLAMP = (-5.0, 120.0)   # metres AGL; water sits slightly negative


def read_h5(path):
    with h5py.File(path, "r") as f:
        return f[list(f.keys())[0]][:]


def to_rgb_uint8(arr):
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip((arr - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
    return arr


class GamusAGL(Dataset):
    """Random crops of (RGB, AGL-in-metres) pairs with light geometric augmentation."""

    def __init__(self, manifest, crop=448, train=True):
        self.items = json.load(open(manifest, encoding="utf-8"))
        self.crop = crop
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            rgb = to_rgb_uint8(read_h5(it["rgb"]))
            agl = read_h5(it["agl"]).astype(np.float32)
        except Exception:
            # Corrupt tile: fall back to a neighbour rather than killing the epoch.
            return self[(i + 1) % len(self)]

        H, W = agl.shape[:2]
        c = min(self.crop, H, W)
        if self.train:
            y0 = random.randint(0, H - c)
            x0 = random.randint(0, W - c)
        else:
            y0, x0 = (H - c) // 2, (W - c) // 2
        rgb = rgb[y0:y0 + c, x0:x0 + c]
        agl = agl[y0:y0 + c, x0:x0 + c]

        if self.train:
            # Flips and 90-degree rotations are safe for nadir imagery (no gravity
            # prior in the image plane). Photometric jitter is deliberately NOT
            # applied: cast-shadow length is a primary height cue and aggressive
            # brightness augmentation destroys exactly the signal we want learned.
            if random.random() < 0.5:
                rgb, agl = rgb[:, ::-1], agl[:, ::-1]
            if random.random() < 0.5:
                rgb, agl = rgb[::-1], agl[::-1]
            k = random.randint(0, 3)
            if k:
                rgb, agl = np.rot90(rgb, k), np.rot90(agl, k)

        valid = np.isfinite(agl)
        agl = np.nan_to_num(agl, nan=0.0)
        agl = np.clip(agl, *HEIGHT_CLAMP)

        x = (np.ascontiguousarray(rgb).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(x.transpose(2, 0, 1)),
            torch.from_numpy(np.ascontiguousarray(agl)),
            torch.from_numpy(np.ascontiguousarray(valid)),
        )


def gradient_matching_loss(pred, gt, mask, scales=3):
    """
    Penalise disagreement in spatial gradients at several scales.

    WHY: a pure L1 loss is happy with soft, blurry roofs - the error it saves by
    hedging at a building edge outweighs the error of a crisp wrong edge. Matching
    gradients forces genuinely vertical facades, which is what makes the resulting
    mesh look like architecture instead of melted mounds.
    """
    total = pred.new_tensor(0.0)
    p, g, m = pred, gt, mask.float()
    for _ in range(scales):
        if p.shape[-1] < 4 or p.shape[-2] < 4:
            break
        dpx, dgx = p[:, :, 1:] - p[:, :, :-1], g[:, :, 1:] - g[:, :, :-1]
        mx = m[:, :, 1:] * m[:, :, :-1]
        dpy, dgy = p[:, 1:, :] - p[:, :-1, :], g[:, 1:, :] - g[:, :-1, :]
        my = m[:, 1:, :] * m[:, :-1, :]
        if mx.sum() > 0:
            total = total + ((dpx - dgx).abs() * mx).sum() / mx.sum().clamp(min=1)
        if my.sum() > 0:
            total = total + ((dpy - dgy).abs() * my).sum() / my.sum().clamp(min=1)
        p, g, m = p[:, ::2, ::2], g[:, ::2, ::2], m[:, ::2, ::2]
    return total


def masked_l1(pred, gt, mask):
    m = mask.float()
    denom = m.sum().clamp(min=1)
    return ((pred - gt).abs() * m).sum() / denom


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(pixel_values=x).predicted_depth
        out = out.float()
        if out.shape[-2:] != y.shape[-2:]:
            out = F.interpolate(out.unsqueeze(1), size=y.shape[-2:],
                                mode="bilinear", align_corners=False).squeeze(1)
        d = (out - y)[v]
        se += float((d ** 2).sum()); ae += float(d.abs().sum()); n += int(v.sum())
    model.train()
    return math.sqrt(se / max(1, n)), ae / max(1, n)



def save_checkpoint(obj, path, retries=5):
    """
    Write a checkpoint atomically, retrying on transient locks.

    WHY: on Windows a real-time AV scanner frequently still holds the freshly
    written ~100 MB .pt file when the next epoch tries to overwrite it, and
    torch.save dies with "File ... cannot be opened". Writing to a temp file and
    renaming makes the swap atomic, and the retry rides out the scan window. An
    overnight run must not lose hours of training to that.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"  [save retry {attempt+1}/{retries}] {type(e).__name__}: {e}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    print("  [WARN] checkpoint save failed; continuing training", file=sys.stderr, flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 for metric AGL")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--model", default="small", choices=["small", "base"])
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "checkpoints" / "agl_model.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA not available - this will be extremely slow on CPU.", file=sys.stderr)
    else:
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

    from transformers import AutoModelForDepthEstimation
    ckpt = f"depth-anything/Depth-Anything-V2-{args.model.capitalize()}-hf"
    model = AutoModelForDepthEstimation.from_pretrained(ckpt).to(device)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.train()

    # Backbone keeps its pretrained geometry priors (low LR); the head has to learn a
    # brand-new output space (metres instead of relative disparity) so it moves faster.
    head_p, back_p = [], []
    for n, p in model.named_parameters():
        (back_p if "backbone" in n or "encoder" in n else head_p).append(p)
    print(f"[INFO] head params {sum(p.numel() for p in head_p)/1e6:.1f}M | "
          f"backbone {sum(p.numel() for p in back_p)/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(
        [{"params": head_p, "lr": args.lr_head},
         {"params": back_p, "lr": args.lr_backbone}], weight_decay=0.01)

    tr = DataLoader(GamusAGL(args.manifest, args.crop, True), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    va = None
    if args.val_manifest and os.path.exists(args.val_manifest):
        va = DataLoader(GamusAGL(args.val_manifest, args.crop, False),
                        batch_size=args.batch, shuffle=False, num_workers=args.workers)

    steps = max(1, len(tr) // args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_head, args.lr_backbone], total_steps=steps, pct_start=0.15)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    print(f"[INFO] {len(tr.dataset)} train tiles | {steps} optimiser steps", flush=True)

    for ep in range(1, args.epochs + 1):
        t0, run, seen = time.perf_counter(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, v) in enumerate(tr):
            x, y, v = x.to(device, non_blocking=True), y.to(device), v.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(pixel_values=x).predicted_depth
            out = out.float()
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out.unsqueeze(1), size=y.shape[-2:],
                                    mode="bilinear", align_corners=False).squeeze(1)
            loss = masked_l1(out, y, v) + 0.5 * gradient_matching_loss(out, y, v)
            (loss / args.accum).backward()

            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
            run += float(loss); seen += 1
            if (i + 1) % 40 == 0:
                mem = torch.cuda.max_memory_allocated()/1e9 if device.type == "cuda" else 0
                print(f"  ep{ep} {i+1}/{len(tr)} loss={run/seen:.3f} "
                      f"vram={mem:.1f}GB", flush=True)

        msg = f"[EPOCH {ep}] train_loss={run/max(1,seen):.4f} ({time.perf_counter()-t0:.0f}s)"
        score = run / max(1, seen)
        if va:
            rmse, mae = validate(model, va, device)
            msg += f" | val RMSE={rmse:.3f}m MAE={mae:.3f}m"
            score = rmse
        print(msg, flush=True)

        if score < best:
            best = score
            if save_checkpoint({"state_dict": model.state_dict(), "checkpoint": ckpt,
                                "model_size": args.model, "crop": args.crop,
                                "target": "AGL_metres", "epoch": ep, "score": score}, args.out):
                print(f"  saved -> {args.out} (best={best:.4f})", flush=True)

    print(f"\n[DONE] best={best:.4f} | checkpoint: {args.out}")


if __name__ == "__main__":
    main()
