"""
run_day1.py - CLI for Stage 1 Relative Digital Surface Model (rDSM) Extraction.

USAGE:
    python run_day1.py <image_path_or_dir> [--model small|base|large] [--device auto|cpu|cuda] [--outdir data/output/]

OUTPUTS GENERATED PER IMAGE:
    {stem}_raw.npy        - Raw 32-bit float disparity array
    {stem}_heightmap.png  - 16-bit grayscale PNG normalized heightmap (Three.js displacement ready)
    {stem}_turbo.png      - Colored 8-bit visualization with Turbo colormap
    {stem}_compare.png    - Side-by-side composite panel: [Input RGB | Turbo Depth]

FAILURE HANDLING & QUALITY GATES:
    - Trips flat_warning if std < 0.05 (severe failure: image predicted as flat plane).
    - Trips Stage 1 Gate Warning if std < 0.10 (insufficient terrain relief contrast).
"""

import argparse
import sys
import time
from pathlib import Path
from PIL import Image

from src.depth_extract import DepthEstimator
from src.enhance import enhance_shadows
from src.utils import (
    depth_stats,
    save_colormap,
    save_comparison,
    save_grayscale,
    save_raw,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def process_single_image(
    estimator: DepthEstimator,
    image_path: Path,
    out_dir: Path,
    tiled: "bool | str" = False,
    enhance: bool = False,
) -> bool:
    """Process a single image through Stage 1 inference and export routines."""
    stem = image_path.stem
    print(f"\n[INFO] Processing: {image_path.name}")
    start_time = time.perf_counter()

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}", file=sys.stderr)
        return False

    try:
        infer_img = enhance_shadows(pil_img) if enhance else pil_img
        # Run monocular depth extraction
        result = estimator.predict(infer_img, robust=True, tiled=tiled)
        infer_time = time.perf_counter() - start_time

        # Calculate statistics
        norm_depth = result.normalized_depth
        stats = depth_stats(norm_depth)

        # File outputs
        raw_path = out_dir / f"{stem}_raw.npy"
        hmap_path = out_dir / f"{stem}_heightmap.png"
        turbo_path = out_dir / f"{stem}_turbo.png"
        comp_path = out_dir / f"{stem}_compare.png"

        save_raw(norm_depth, raw_path)
        save_grayscale(norm_depth, hmap_path)
        save_colormap(norm_depth, turbo_path)
        save_comparison(pil_img, norm_depth, comp_path)

        total_time = time.perf_counter() - start_time

        # Report stats
        print(f"       Resolution : {result.source_size[1]}x{result.source_size[0]}")
        print(f"       Inference  : {infer_time:.2f}s (Total: {total_time:.2f}s)")
        print(f"       Stats      : min={stats['min']:.3f}, max={stats['max']:.3f}, "
              f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, p05={stats['p05']:.3f}, p95={stats['p95']:.3f}")

        # Quality Gates & Warnings
        if stats["flat_warning"]:
            print(
                f"[CRITICAL WARNING] Near-flat prediction detected! std={stats['std']:.4f} < 0.05. "
                f"The model failed to identify meaningful surface relief in this scene.",
                file=sys.stderr,
            )
        elif stats["std"] < 0.10:
            print(
                f"[GATE WARNING] Stage 1 Gate Warning: std={stats['std']:.4f} < 0.10 threshold. "
                f"Check {comp_path.name} to verify whether structural relief is visible.",
                file=sys.stderr,
            )
        else:
            print(f"       [PASS] Stage 1 Quality Gate Passed (std={stats['std']:.4f} >= 0.10).")

        print(f"       Outputs saved to: {out_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed during processing {image_path.name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="DepthWizard Stage 1 - Relative Digital Surface Model (rDSM) Extraction"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an input image file or directory containing images",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Depth Anything V2 model checkpoint size (default: small)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cpu, or cuda (default: auto)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/output",
        help="Destination directory for output products (default: data/output)",
    )
    parser.add_argument(
        "--tiled",
        type=str,
        choices=["off", "auto", "on"],
        default="off",
        help="Multi-scale tiled inference for large images: off, auto (>~1000px), or on (default: off)",
    )
    parser.add_argument(
        "--enhance-shadows",
        action="store_true",
        help="Apply CLAHE shadow enhancement to the input before depth inference",
    )

    args = parser.parse_args()
    tiled_arg = {"off": False, "on": True, "auto": "auto"}[args.tiled]

    input_path = Path(args.input_path)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Collect files
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_files:
            print(f"[WARNING] No supported image files found in directory: {input_path}")
            sys.exit(0)
    else:
        print(f"[ERROR] Invalid path type: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"============================================================")
    print(f"DepthWizard Stage 1: Relative Depth Extraction (Day 1)")
    print(f"Target Checkpoint : {args.model}")
    print(f"Compute Device    : {args.device}")
    print(f"Input Images      : {len(image_files)} file(s)")
    print(f"Output Directory  : {out_dir}")
    print(f"============================================================")

    # Initialize DepthEstimator (lazy load)
    estimator = DepthEstimator(model_size=args.model, device=args.device)
    print(f"[INFO] Initialized DepthEstimator on device: {estimator.device}")

    successes = 0
    failures = 0

    for img_file in image_files:
        success = process_single_image(
            estimator, img_file, out_dir, tiled=tiled_arg, enhance=args.enhance_shadows
        )
        if success:
            successes += 1
        else:
            failures += 1

    print(f"\n============================================================")
    print(f"Batch Processing Summary: {successes} succeeded, {failures} failed (Total: {len(image_files)})")
    print(f"============================================================")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
