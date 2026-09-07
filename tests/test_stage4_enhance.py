"""
test_stage4_enhance.py - Verification for radiometric enhancement and tiled-inference math.

PURPOSE & WHY:
1. CLAHE shadow enhancement must lift contrast inside dark regions without destroying
   the colour of well-exposed areas or changing image dimensions.
2. The tiled-inference affine solver must recover a known scale/shift exactly, since the
   whole point of tiling is aligning scale/shift-invariant tiles to one global map.
3. The feather window and stitched blend must reconstruct a smooth field with no seams.
4. The RGB water heuristic must fire on a synthetic flat blue lake and stay off dry land.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_extract import _feather_window, _resize_bilinear, _solve_affine
from src.enhance import detect_water_mask, enhance_shadows, flatten_water


def test_clahe_shadow_enhancement():
    print("\n[1] CLAHE Shadow Enhancement:")
    rng = np.random.RandomState(0)

    # Bright half (well exposed) + dark half (crushed shadow) with faint detail in both.
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    img[:, :80] = np.clip(180 + rng.randn(160, 80, 1) * 8, 0, 255).astype(np.uint8)
    img[:, 80:] = np.clip(18 + rng.randn(160, 80, 1) * 4, 0, 255).astype(np.uint8)

    out = np.asarray(enhance_shadows(img, clip_limit=3.0))
    assert out.shape == img.shape, "CLAHE changed image dimensions!"

    dark_before = img[:, 80:].astype(np.float32).std()
    dark_after = out[:, 80:].astype(np.float32).std()
    bright_before = img[:, :80].astype(np.float32).mean()
    bright_after = out[:, :80].astype(np.float32).mean()

    print(f"    Shadow region std : {dark_before:.2f} -> {dark_after:.2f} (contrast lifted)")
    print(f"    Bright region mean: {bright_before:.2f} -> {bright_after:.2f} (preserved)")
    assert dark_after > dark_before * 1.5, "CLAHE did not expand shadow contrast!"
    assert abs(bright_after - bright_before) < 40, "CLAHE blew out the highlights!"


def test_affine_solver_recovers_scale_shift():
    print("\n[2] Tiled-Inference Affine Alignment:")
    rng = np.random.RandomState(42)
    base = rng.rand(64, 64).astype(np.float32)
    true_a, true_b = 3.7, -1.25
    tile = (base - true_b) / true_a  # so that a*tile + b == base

    a, b = _solve_affine(tile, base)
    print(f"    True (a, b)      : ({true_a:.4f}, {true_b:.4f})")
    print(f"    Recovered (a, b) : ({a:.4f}, {b:.4f})")
    assert abs(a - true_a) < 1e-3 and abs(b - true_b) < 1e-3, "Affine solver failed to recover parameters!"

    recon = a * tile + b
    assert np.allclose(recon, base, atol=1e-3), "Aligned tile does not match target!"


def test_feathered_blend_is_seamless():
    print("\n[3] Feathered Overlap Blend:")
    field = np.fromfunction(lambda y, x: np.sin(x / 9.0) + np.cos(y / 7.0), (80, 120), dtype=np.float32)

    acc = np.zeros_like(field)
    wsum = np.zeros_like(field)
    tile, step = 48, 32
    ys = sorted(set(list(range(0, 80 - tile + 1, step)) + [80 - tile]))
    xs = sorted(set(list(range(0, 120 - tile + 1, step)) + [120 - tile]))
    for y0 in ys:
        for x0 in xs:
            win = _feather_window(tile, tile)
            acc[y0:y0 + tile, x0:x0 + tile] += field[y0:y0 + tile, x0:x0 + tile] * win
            wsum[y0:y0 + tile, x0:x0 + tile] += win

    stitched = acc / np.maximum(wsum, 1e-6)
    max_err = float(np.max(np.abs(stitched - field)))
    print(f"    Max reconstruction error: {max_err:.6f}")
    assert max_err < 1e-4, "Feathered blend introduced seams / errors!"


def test_resize_bilinear_roundtrip():
    print("\n[4] Bilinear Resize Helper:")
    a = np.linspace(0, 1, 32 * 40, dtype=np.float32).reshape(32, 40)
    up = _resize_bilinear(a, (64, 80))
    down = _resize_bilinear(up, (32, 40))
    assert up.shape == (64, 80) and down.shape == (32, 40)
    print(f"    32x40 -> 64x80 -> 32x40, mean abs drift {np.mean(np.abs(down - a)):.5f}")
    assert np.mean(np.abs(down - a)) < 0.02, "Resize round-trip drifted too far!"


def test_water_detection_and_flatten():
    print("\n[5] RGB Water Heuristic:")
    rng = np.random.RandomState(7)
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    # Dry land: textured green/brown.
    img[..., 0] = np.clip(90 + rng.randn(120, 120) * 25, 0, 255)
    img[..., 1] = np.clip(120 + rng.randn(120, 120) * 25, 0, 255)
    img[..., 2] = np.clip(70 + rng.randn(120, 120) * 20, 0, 255)
    # A calm blue lake, low texture, blue-dominant.
    img[70:110, 20:90] = [30, 55, 120]

    mask = detect_water_mask(img)
    lake_hits = mask[70:110, 20:90].mean()
    land_hits = mask[:60, :].mean()
    print(f"    Lake coverage detected : {lake_hits:.1%}")
    print(f"    Dry-land false positives: {land_hits:.1%}")
    assert lake_hits > 0.6, "Water heuristic missed an obvious lake!"
    assert land_hits < 0.05, "Water heuristic fired on dry land!"

    rel = np.full((120, 120), 0.5, dtype=np.float32)
    rel[70:110, 20:90] = rng.rand(40, 70).astype(np.float32)  # noisy water disparity
    flat = flatten_water(rel, mask)
    assert flat[mask].std() < 1e-4, "flatten_water left residual noise on water!"


def main():
    print("=" * 70)
    print("DepthWizard Stage 4: Radiometric Enhancement & Tiled-Inference Verification")
    print("=" * 70)
    test_clahe_shadow_enhancement()
    test_affine_solver_recovers_scale_shift()
    test_feathered_blend_is_seamless()
    test_resize_bilinear_roundtrip()
    test_water_detection_and_flatten()
    print("\n" + "=" * 70)
    print("STAGE 4 TEST PASSED! Enhancement and tiled-inference math verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
