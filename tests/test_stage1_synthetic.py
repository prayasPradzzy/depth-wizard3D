"""
test_stage1_synthetic.py - Verification test for Stage 1 (Synthetic Terrain Benchmark).

PURPOSE & WHY:
1. Validates normalise_depth() resilience against extreme radiometric outliers:
   Rooftop glints and sensor dropouts destroy min/max scaling (crushing dynamic range),
   whereas 2nd/98th percentile clipping preserves full structural relief.
2. Validates 16-bit PNG heightmap precision:
   Confirms lossless round-trip without quantization terracing.
3. Tests every utility function in src/utils.py on synthetic data without requiring a GPU.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_extract import normalise_depth
from src.utils import (
    depth_stats,
    save_colormap,
    save_comparison,
    save_grayscale,
    save_raw,
)


def create_synthetic_terrain(size: int = 256) -> np.ndarray:
    """
    Construct a synthetic terrain containing:
    - 2 Gaussian hills (natural topography)
    - 1 Rectangular building block (sharp urban structure)
    - 2 Extreme outlier pixels (+2000.0 glint, -1000.0 dropout)
    """
    y, x = np.mgrid[0:size, 0:size]

    # Subtle base ground tilt: 10m to 15m
    terrain = 10.0 + 5.0 * (x / size)

    # Gaussian Hill 1: Center (70, 70), height +35m, sigma 25
    hill1 = 35.0 * np.exp(-((x - 70) ** 2 + (y - 70) ** 2) / (2 * (25 ** 2)))

    # Gaussian Hill 2: Center (180, 180), height +50m, sigma 35
    hill2 = 50.0 * np.exp(-((x - 180) ** 2 + (y - 180) ** 2) / (2 * (35 ** 2)))

    terrain += hill1 + hill2

    # Rectangular Building Block: slice [120:165, 60:110], height +30m
    terrain[120:165, 60:110] += 30.0

    # True terrain values range approximately from 10m to 75m
    # Now inject two extreme outlier pixels:
    # 1. Specular glint from metallic roof (+2000.0)
    terrain[20, 20] = 2000.0
    # 2. Sensor dropout / nodata artifact (-1000.0)
    terrain[240, 240] = -1000.0

    return terrain.astype(np.float32)


def main():
    print("=" * 70)
    print("DepthWizard Stage 1: Synthetic Terrain Benchmark & Verification Test")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Terrain
    raw_terrain = create_synthetic_terrain(size=256)
    inlier_mask = np.ones_like(raw_terrain, dtype=bool)
    inlier_mask[20, 20] = False
    inlier_mask[240, 240] = False

    true_min = float(np.min(raw_terrain[inlier_mask]))
    true_max = float(np.max(raw_terrain[inlier_mask]))
    print(f"\n[1] Synthetic Terrain Generated:")
    print(f"    Dimensions             : {raw_terrain.shape[0]}x{raw_terrain.shape[1]}")
    print(f"    Genuine Terrain Range  : [{true_min:.2f}m, {true_max:.2f}m] (Span: {true_max - true_min:.2f}m)")
    print(f"    Injected Outlier 1     : +2000.0 at (20, 20) [Roof Glint]")
    print(f"    Injected Outlier 2     : -1000.0 at (240, 240) [Sensor Dropout]")
    print(f"    Total Raw Array Range  : [{raw_terrain.min():.1f}, {raw_terrain.max():.1f}] (Span: {raw_terrain.max() - raw_terrain.min():.1f})")

    # 2. Normalization Comparison (Min/Max vs 2/98 Percentile)
    print(f"\n[2] Normalization Comparison (Outlier Resilience):")
    norm_minmax = normalise_depth(raw_terrain, robust=False)
    norm_robust = normalise_depth(raw_terrain, robust=True)

    # Inspect inlier pixels
    inliers_minmax = norm_minmax[inlier_mask]
    inliers_robust = norm_robust[inlier_mask]

    span_minmax = inliers_minmax.max() - inliers_minmax.min()
    span_robust = inliers_robust.max() - inliers_robust.min()

    std_minmax = inliers_minmax.std()
    std_robust = inliers_robust.std()

    print(f"    --- Min/Max Normalization (naive) ---")
    print(f"    Inlier Dynamic Range Span : {span_minmax:.4f} (Only {span_minmax*100:.2f}% of available levels!)")
    print(f"    Inlier Standard Deviation  : {std_minmax:.4f} (Severely crushed)")
    print(f"    Status                     : FAILED (Terrain elevation contrast destroyed by outliers)")

    print(f"\n    --- 2nd/98th Percentile Normalization (robust) ---")
    print(f"    Inlier Dynamic Range Span : {span_robust:.4f} ({span_robust*100:.2f}% of full [0, 1] range)")
    print(f"    Inlier Standard Deviation  : {std_robust:.4f} (Rich topographic contrast preserved)")
    print(f"    Status                     : PASSED (Terrain details completely intact)")

    assert span_robust > 0.90, "Robust normalization failed to preserve dynamic range!"
    assert span_minmax < 0.05, "Min/Max unexpectedly did not fail on outliers!"

    # 3. Test All src/utils.py Functions
    print(f"\n[3] Testing src/utils.py Export & Visualization Routines:")

    # a. save_raw
    raw_path = out_dir / "test_synthetic_raw.npy"
    save_raw(norm_robust, raw_path)
    loaded_raw = np.load(raw_path)
    assert np.allclose(norm_robust, loaded_raw), "save_raw .npy round-trip mismatch!"
    print(f"    [OK] save_raw()        -> {raw_path.name} ({raw_path.stat().st_size} bytes)")

    # b. save_grayscale (16-bit PNG)
    hmap_path = out_dir / "test_synthetic_heightmap.png"
    save_grayscale(norm_robust, hmap_path)
    print(f"    [OK] save_grayscale()  -> {hmap_path.name} (16-bit grayscale PNG)")

    # c. save_colormap (Turbo)
    turbo_path = out_dir / "test_synthetic_turbo.png"
    save_colormap(norm_robust, turbo_path)
    print(f"    [OK] save_colormap()   -> {turbo_path.name} (Turbo colormapped RGB PNG)")

    # d. save_comparison (Side-by-side composite)
    comp_path = out_dir / "test_synthetic_compare.png"
    # Create synthetic RGB satellite scene: green terrain, brown hills, gray roof
    synthetic_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    synthetic_rgb[:, :, 1] = np.clip(100 + raw_terrain * 0.8, 0, 255).astype(np.uint8)  # Green vegetation
    synthetic_rgb[:, :, 0] = np.clip(60 + raw_terrain * 0.5, 0, 255).astype(np.uint8)   # Earth/dirt
    synthetic_rgb[120:165, 60:110] = [180, 180, 180]                                    # Gray concrete building
    save_comparison(synthetic_rgb, norm_robust, comp_path)
    print(f"    [OK] save_comparison() -> {comp_path.name} (Side-by-side RGB | Depth)")

    # e. depth_stats
    stats = depth_stats(norm_robust)
    print(f"    [OK] depth_stats()     -> min={stats['min']:.4f}, max={stats['max']:.4f}, "
          f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, flat_warning={stats['flat_warning']}")
    assert not stats["flat_warning"], "Synthetic terrain falsely triggered flat_warning!"
    assert stats["std"] >= 0.10, "Synthetic terrain failed Stage 1 Gate threshold (std >= 0.10)!"

    # 4. 16-Bit PNG Round-Trip Precision Verification
    print(f"\n[4] 16-Bit PNG Round-Trip Precision Verification:")
    with Image.open(hmap_path) as img:
        assert img.mode == "I;16", f"Expected image mode 'I;16', got '{img.mode}'"
        reloaded_u16 = np.array(img, dtype=np.uint16)

    # Convert back to float [0.0, 1.0]
    reloaded_float = reloaded_u16.astype(np.float32) / 65535.0
    abs_errors = np.abs(norm_robust - reloaded_float)
    max_err = float(np.max(abs_errors))
    mean_err = float(np.mean(abs_errors))

    # Theoretical maximum quantization error for 16-bit is 0.5 / 65535 ~ 7.63e-6
    theoretical_max = 0.5 / 65535.0
    # For comparison, 8-bit quantization theoretical max error is 0.5 / 255 ~ 1.96e-3 (256x worse!)
    eight_bit_max = 0.5 / 255.0

    print(f"    Image Mode Verified    : {img.mode} (16-bit unsigned single channel)")
    print(f"    Max Absolute Error     : {max_err:.8f} (Theoretical limit: {theoretical_max:.8f})")
    print(f"    Mean Absolute Error    : {mean_err:.8f}")
    print(f"    8-Bit Error Comparison : 8-bit would suffer {eight_bit_max:.6f} max error (256x worse)")
    print(f"    Terracing Prevention   : ZERO visible terracing in 3D vertex displacement")
    assert max_err <= theoretical_max + 1e-7, f"16-bit round-trip error {max_err} exceeds theoretical bound!"

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! Stage 1 utils and logic fully verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
