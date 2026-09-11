"""
run_overnight.py - Unattended: wait for training tiles, split, train, then benchmark.

Designed to be started before bed and read in the morning. Each stage logs to
logs/overnight.log so the whole run can be reconstructed after the fact.

Stages
------
1. Poll the train manifest until it holds at least --min-tiles entries. The fetch
   script writes it incrementally, so training can start before the download ends.
2. Split 90/10 into train/val. The validation split is carved out of the TRAIN
   manifest on purpose - never out of test, because test is what the published
   benchmark number is computed on and using it for model selection would leak.
3. Fine-tune (tools/train_gamus.py).
4. Re-run the benchmark with the trained checkpoint and write a comparison file.

Nothing here touches the demo venv or data/benchmark.json until the very last step,
so a failure at any stage leaves the presentation build exactly as it was.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = PROJECT_ROOT / ".venv-train" / "Scripts" / "python.exe"
LOG_DIR = PROJECT_ROOT / "logs"


def log(msg):
    LOG_DIR.mkdir(exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "overnight.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_tiles(manifest, min_tiles, timeout_min):
    deadline = time.time() + timeout_min * 60
    last = -1
    while time.time() < deadline:
        try:
            n = len(json.load(open(manifest, encoding="utf-8")))
        except Exception:
            n = 0
        if n != last:
            log(f"train manifest: {n} tiles (need {min_tiles})")
            last = n
        if n >= min_tiles:
            return n
        time.sleep(60)
    log(f"timeout waiting for tiles; proceeding with {last}")
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--min-tiles", type=int, default=150)
    ap.add_argument("--wait-min", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crop", type=int, default=448)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    log("=" * 60)
    log("OVERNIGHT RUN START")

    n = wait_for_tiles(args.manifest, args.min_tiles, args.wait_min)
    if n < 40:
        log(f"FATAL: only {n} tiles available, not enough to train. Stopping.")
        return 1

    items = json.load(open(args.manifest, encoding="utf-8"))
    random.Random(0).shuffle(items)
    cut = max(8, int(len(items) * 0.1))
    val, tr = items[:cut], items[cut:]
    sp = Path(args.manifest).parent
    tr_p, va_p = sp / "_train_split.json", sp / "_val_split.json"
    json.dump(tr, open(tr_p, "w"), indent=1)
    json.dump(val, open(va_p, "w"), indent=1)
    log(f"split: {len(tr)} train / {len(val)} val")

    ckpt = PROJECT_ROOT / "checkpoints" / "agl_vits.pt"
    cmd = [str(TRAIN_PY), str(PROJECT_ROOT / "tools" / "train_gamus.py"),
           "--manifest", str(tr_p), "--val-manifest", str(va_p),
           "--epochs", str(args.epochs), "--crop", str(args.crop),
           "--batch", str(args.batch), "--out", str(ckpt)]
    log("TRAIN: " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    log(f"training exited {r.returncode} after {(time.time()-t0)/60:.1f} min")

    if r.returncode != 0 or not ckpt.exists():
        log("training did not produce a checkpoint - demo build is untouched, all good.")
        return 1

    out = PROJECT_ROOT / "data" / "benchmark_finetuned.json"
    cmd = [str(TRAIN_PY), str(PROJECT_ROOT / "tools" / "eval_gamus.py"),
           "--manifest", args.test_manifest, "--checkpoint", str(ckpt), "--out", str(out)]
    log("EVAL: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    log(f"eval exited {r.returncode}")

    base = PROJECT_ROOT / "data" / "benchmark.json"
    if out.exists() and base.exists():
        b = json.load(open(base, encoding="utf-8"))
        f = json.load(open(out, encoding="utf-8"))
        log("=" * 60)
        log(f"BASELINE  (zero-shot)  RMSE {b['overall']['rmse']:.3f} m  "
            f"MAE {b['overall']['mae']:.3f} m  r {b['overall']['pearson_r']:.3f}")
        log(f"FINETUNED (aligned)    RMSE {f['overall']['rmse']:.3f} m  "
            f"MAE {f['overall']['mae']:.3f} m  r {f['overall']['pearson_r']:.3f}")
        if f.get("absolute_metric"):
            a = f["absolute_metric"]
            log(f"FINETUNED (absolute)   RMSE {a['rmse']:.3f} m  MAE {a['mae']:.3f} m")
        better = f["overall"]["rmse"] < b["overall"]["rmse"]
        log(f"VERDICT: fine-tuning {'IMPROVED' if better else 'did NOT improve'} scale-invariant RMSE")
        log("To show it in the viewer: copy data/benchmark_finetuned.json over data/benchmark.json")
    log("OVERNIGHT RUN COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
