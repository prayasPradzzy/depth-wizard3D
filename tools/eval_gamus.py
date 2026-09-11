"""
eval_gamus.py - Independent accuracy benchmark against LiDAR ground truth.

WHY THIS EXISTS
===============
Until now the only accuracy numbers DepthWizard could report were computed against
the very SRTM tile used to fit the calibration regression. That is circular: it
measures self-consistency, not accuracy, and docs/LIMITATIONS.md flags it as
Limitation #1.

This script replaces that with a genuinely independent measurement. The GAMUS
benchmark ships LiDAR-derived nDSM (above-ground-level height, in metres) for every
tile, and none of it is used anywhere in our calibration path.

EVALUATION PROTOCOL (important - read before quoting these numbers)
==================================================================
The backbone predicts *relative* inverse depth, which is invariant to scale and
shift by construction. Comparing it to metres directly would measure an arbitrary
unit mismatch rather than geometric quality. We therefore use the standard
scale-invariant protocol from the monocular-depth literature (MiDaS, Depth Anything):

    per tile, solve least-squares  a,b  minimising  || a*pred + b - gt ||
    then report errors on the aligned prediction.

This answers "how correct is the predicted *shape* of the surface, given a correct
scale". It does NOT demonstrate that the system recovers absolute height unaided -
that is what the SRTM/GCP calibration stage is for. Every number written by this
script is tagged `protocol: "per-tile affine aligned (scale-invariant)"` so the
distinction travels with the results.

Reported: overall RMSE / MAE / Pearson r / delta accuracies, plus RMSE broken down
by land-cover class, which is what the problem statement's "stability across urban,
sparse, hilly, forested landscapes" criterion actually asks for.

USAGE
=====
    python tools/eval_gamus.py --manifest <gamus_test_manifest.json> [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_extract import DepthEstimator

# GAMUS land-cover codes. Order follows the benchmark paper (ground, low-vegetation,
# building, water, road, tree); 0 is unlabelled. The measured per-class median
# heights printed by this script are a sanity check on this mapping.
CLASS_NAMES = {
    0: "Unlabelled",
    1: "Ground",
    2: "Low vegetation",
    3: "Building",
    4: "Water",
    5: "Road",
    6: "Tree",
}
SKIP_CLASSES = {0}


def read_h5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[list(f.keys())[0]][:]


def affine_align(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Least-squares scale+shift taking pred into gt's units. Returns aligned pred."""
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    if p.size < 64 or np.std(p) < 1e-9:
        return None
    A = np.vstack([p, np.ones_like(p)]).T
    (a, b), *_ = np.linalg.lstsq(A, g, rcond=None)
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return a * pred.astype(np.float64) + b


def main():
    ap = argparse.ArgumentParser(description="Benchmark DepthWizard against GAMUS LiDAR nDSM")
    ap.add_argument("--manifest", required=True, help="JSON manifest from the GAMUS fetch step")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate at most N tiles (0 = all)")
    ap.add_argument("--model", default="small", choices=["small", "base", "large"])
    ap.add_argument("--tiled", default="off", choices=["off", "auto", "on"])
    ap.add_argument("--checkpoint", default=None,
                    help="Fine-tuned AGL checkpoint from tools/train_gamus.py. When given, "
                         "metrics are reported BOTH in absolute metres (no alignment) and "
                         "affine-aligned, so the result is comparable with the zero-shot run.")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "benchmark.json"))
    args = ap.parse_args()

    tiles = json.load(open(args.manifest, encoding="utf-8"))
    if args.limit:
        tiles = tiles[: args.limit]
    print(f"[EVAL] {len(tiles)} tiles | model={args.model} | tiled={args.tiled}", flush=True)

    tiled_arg = {"off": False, "on": True, "auto": "auto"}[args.tiled]
    ft_model = None
    if args.checkpoint:
        import torch
        from transformers import AutoModelForDepthEstimation
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ft_model = AutoModelForDepthEstimation.from_pretrained(ck["checkpoint"])
        ft_model.load_state_dict(ck["state_dict"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ft_model.to(dev).eval()
        print(f"[EVAL] fine-tuned checkpoint: {args.checkpoint} "
              f"(epoch {ck.get('epoch')}, score {ck.get('score'):.4f}) on {dev}", flush=True)
        MEAN = np.array([0.485, 0.456, 0.406], np.float32)
        STD = np.array([0.229, 0.224, 0.225], np.float32)

        def predict(rgb_u8):
            x = (rgb_u8.astype(np.float32) / 255.0 - MEAN) / STD
            t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = ft_model(pixel_values=t).predicted_depth
            return o.float().squeeze().cpu().numpy()
    else:
        est = DepthEstimator(model_size=args.model, device="auto")

        def predict(rgb_u8):
            return est.predict(rgb_u8, robust=True, tiled=tiled_arg).normalized_depth

    sq_err, abs_err, n_px = 0.0, 0.0, 0
    abs_sq, abs_abs = 0.0, 0.0        # metrics with NO alignment (fine-tuned model only)
    all_p, all_g = [], []
    d1 = d2 = d3 = 0
    per_class = {k: {"sq": 0.0, "abs": 0.0, "n": 0, "gt_sum": 0.0} for k in CLASS_NAMES}
    cities, done, skipped = set(), 0, 0
    # Per-city accumulators: the three GAMUS cities are genuinely different built
    # regimes (NYC high-rise / DC low-rise / PHL rowhouse), so this is the most
    # direct evidence for the 'stability across landscapes' criterion.
    per_city = {}

    for i, t in enumerate(tiles, 1):
        try:
            rgb = read_h5(t["rgb"])
            gt = read_h5(t["agl"]).astype(np.float32)
            cls = read_h5(t["cls"]).astype(np.int32)
        except Exception as e:
            skipped += 1
            continue

        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):      # CHW -> HWC
            rgb = np.transpose(rgb, (1, 2, 0))
        rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            lo, hi = np.percentile(rgb, [1, 99])
            rgb = np.clip((rgb - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)

        valid = np.isfinite(gt)
        if valid.sum() < 4096:
            skipped += 1
            continue

        pred = predict(rgb)
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.float32), mode="F")
                .resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )

        if ft_model is not None:
            dabs = (pred - gt)[valid]
            abs_sq += float(np.sum(dabs ** 2)); abs_abs += float(np.sum(np.abs(dabs)))

        aligned = affine_align(pred, gt, valid)
        if aligned is None:
            skipped += 1
            continue

        diff = (aligned - gt)[valid]
        sq_err += float(np.sum(diff ** 2))
        abs_err += float(np.sum(np.abs(diff)))
        n_px += int(valid.sum())

        # delta-accuracy on above-ground pixels (ratio is meaningless near h=0)
        ag = valid & (gt > 2.0)
        if ag.sum() > 0:
            ratio = np.maximum(
                np.abs(aligned[ag]) / np.maximum(gt[ag], 1e-3),
                np.maximum(gt[ag], 1e-3) / np.maximum(np.abs(aligned[ag]), 1e-3),
            )
            d1 += int(np.sum(ratio < 1.25)); d2 += int(np.sum(ratio < 1.25 ** 2))
            d3 += int(np.sum(ratio < 1.25 ** 3)); per_class["_ag_n"] = per_class.get("_ag_n", 0)
            per_class.setdefault("_ag_total", 0)
            per_class["_ag_total"] += int(ag.sum())

        for k in CLASS_NAMES:
            m = valid & (cls == k)
            if m.sum() == 0:
                continue
            dk = (aligned - gt)[m]
            per_class[k]["sq"] += float(np.sum(dk ** 2))
            per_class[k]["abs"] += float(np.sum(np.abs(dk)))
            per_class[k]["n"] += int(m.sum())
            per_class[k]["gt_sum"] += float(np.sum(gt[m]))

        pc = per_city.setdefault(t.get("city", "?"), {"sq": 0.0, "abs": 0.0, "n": 0, "tiles": 0})
        pc["sq"] += float(np.sum(diff ** 2)); pc["abs"] += float(np.sum(np.abs(diff)))
        pc["n"] += int(valid.sum()); pc["tiles"] += 1

        sub = np.random.RandomState(i).choice(int(valid.sum()), size=min(4000, int(valid.sum())), replace=False)
        all_p.append(aligned[valid][sub]); all_g.append(gt[valid][sub])
        cities.add(t.get("city", "?")); done += 1
        if i % 10 == 0:
            print(f"  {i}/{len(tiles)}  running RMSE={np.sqrt(sq_err/max(1,n_px)):.3f} m", flush=True)

    if n_px == 0:
        print("[EVAL] No tiles evaluated.", file=sys.stderr); sys.exit(1)

    P = np.concatenate(all_p); G = np.concatenate(all_g)
    ag_total = per_class.get("_ag_total", 0)
    result = {
        "dataset": "GAMUS test split",
        "protocol": "per-tile affine aligned (scale-invariant)",
        "ground_truth": "LiDAR-derived nDSM (above-ground level, metres)",
        "model": (f"Depth Anything V2 {args.model} fine-tuned on GAMUS (metric AGL)"
                  if ft_model is not None else f"Depth Anything V2 {args.model} (zero-shot)"),
        "absolute_metric": ({"rmse": float(np.sqrt(abs_sq / n_px)),
                             "mae": float(abs_abs / n_px),
                             "note": "no alignment - direct metric output in metres"}
                            if ft_model is not None else None),
        "tiled_inference": args.tiled,
        "n_tiles": done,
        "n_tiles_skipped": skipped,
        "n_pixels": n_px,
        "cities": sorted(cities),
        "overall": {
            "rmse": float(np.sqrt(sq_err / n_px)),
            "mae": float(abs_err / n_px),
            "pearson_r": float(np.corrcoef(P, G)[0, 1]),
            "delta1": float(d1 / ag_total) if ag_total else None,
            "delta2": float(d2 / ag_total) if ag_total else None,
            "delta3": float(d3 / ag_total) if ag_total else None,
        },
        "per_city": [
            {"city": c, "rmse": float(np.sqrt(v["sq"] / v["n"])), "mae": float(v["abs"] / v["n"]),
             "n_tiles": v["tiles"], "n_px": v["n"]}
            for c, v in sorted(per_city.items())
        ],
        "per_class": [
            {
                "id": k,
                "name": CLASS_NAMES[k],
                "rmse": float(np.sqrt(v["sq"] / v["n"])),
                "mae": float(v["abs"] / v["n"]),
                "mean_gt_height_m": float(v["gt_sum"] / v["n"]),
                "n_px": v["n"],
                "n_px_m": f"{v['n']/1e6:.1f}M",
            }
            for k, v in per_class.items()
            if isinstance(k, int) and k not in SKIP_CLASSES and v["n"] > 10000
        ],
    }
    result["per_class"].sort(key=lambda c: -c["n_px"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2)

    o = result["overall"]
    print("\n" + "=" * 66)
    print(f"GAMUS LiDAR BENCHMARK  ({done} tiles, {'/'.join(result['cities'])})")
    print(f"protocol: {result['protocol']}")
    print("=" * 66)
    print(f"  RMSE {o['rmse']:.3f} m   MAE {o['mae']:.3f} m   Pearson r {o['pearson_r']:.4f}")
    if result.get("absolute_metric"):
        a = result["absolute_metric"]
        print(f"  ABSOLUTE (no alignment):  RMSE {a['rmse']:.3f} m   MAE {a['mae']:.3f} m")
    if o["delta1"]:
        print(f"  delta<1.25 {o['delta1']:.1%}   <1.25^2 {o['delta2']:.1%}   <1.25^3 {o['delta3']:.1%}")
    print("-" * 66)
    print(f"  {'city':<18}{'RMSE':>9}{'MAE':>9}{'tiles':>9}")
    for c in result["per_city"]:
        print(f"  {c['city']:<18}{c['rmse']:>8.2f}m{c['mae']:>8.2f}m{c['n_tiles']:>9}")
    print("-" * 66)
    print(f"  {'class':<18}{'RMSE':>9}{'MAE':>9}{'mean GT h':>12}{'pixels':>10}")
    for c in result["per_class"]:
        print(f"  {c['name']:<18}{c['rmse']:>8.2f}m{c['mae']:>8.2f}m{c['mean_gt_height_m']:>11.2f}m{c['n_px_m']:>10}")
    print("=" * 66)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
