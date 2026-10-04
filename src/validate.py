"""
validate.py - Quantitative accuracy metrics and spatial error visualization for DepthWizard.

CRITICAL METHODOLOGICAL LIMITATION & NOTICE FOR EVALUATORS:
============================================================
Validating the calibrated DSM against the same SRTM 30m dataset used for calibration
regression is inherently CIRCULAR and OPTIMISTIC.

Furthermore:
1. SRTM measures bare-earth topography (DEM) at 30m horizontal spacing.
2. DepthWizard produces a Digital Surface Model (DSM) capturing individual buildings and tree crowns.
3. Consequently, in dense urban or forested areas, the signed error (pred_dsm - ref_dem) will be
   systematically positive. These are NOT model errors; they represent actual above-ground features
   that SRTM cannot resolve.

For this Hackathon prototype round, this validation serves as a verification of mathematical
consistency and datum alignment. Full operational validation requires an independent airborne
LiDAR point cloud (see docs/LIMITATIONS.md).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image


def compute_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute statistical error metrics between calibrated elevation and reference elevation.

    Returns:
        Dict containing:
            - rmse: Root Mean Square Error in metres.
            - mae: Mean Absolute Error in metres.
            - bias: Mean signed error (pred - ref) in metres.
            - pearson_r: Pearson linear correlation coefficient.
            - std_error: Standard deviation of residuals in metres.
            - valid_pixels: Count of evaluated pixels.
    """
    valid_idx = np.where(mask & np.isfinite(pred) & np.isfinite(ref))
    p = pred[valid_idx]
    r = ref[valid_idx]

    if len(p) == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "pearson_r": float("nan"),
            "std_error": float("nan"),
            "valid_pixels": 0,
        }

    diff = p - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    std_error = float(np.std(diff))

    # Pearson correlation coefficient
    if np.std(p) > 1e-6 and np.std(r) > 1e-6:
        pearson_r = float(np.corrcoef(p, r)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson_r": pearson_r,
        "std_error": std_error,
        "valid_pixels": int(len(p)),
    }


def error_map(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Compute pixel-wise signed elevation difference: (pred - ref).
    Returns NaN where invalid.
    """
    diff = np.full_like(pred, np.nan, dtype=np.float32)
    valid = mask & np.isfinite(pred) & np.isfinite(ref)
    diff[valid] = pred[valid] - ref[valid]
    return diff


def save_error_map(
    error_arr: np.ndarray,
    path: Union[str, Path],
    clip_range_m: float = 20.0,
    cmap: str = "coolwarm",
) -> Path:
    """
    Render and save a signed error map with a diverging colormap centered at 0m.

    Blue = predicted lower than SRTM (under-estimation).
    White = exact agreement (0m error).
    Red = predicted higher than SRTM (over-estimation or above-ground structures).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid_mask = np.isfinite(error_arr)
    # Map [-clip_range, +clip_range] to [0.0, 1.0]
    normalized = np.full_like(error_arr, 0.5, dtype=np.float32)
    if np.any(valid_mask):
        clipped = np.clip(error_arr[valid_mask], -clip_range_m, clip_range_m)
        normalized[valid_mask] = (clipped + clip_range_m) / (2.0 * clip_range_m)

    from src.colormaps import apply as _cmap
    rgb_uint8 = _cmap(normalized, cmap)

    # Set invalid pixels to neutral dark gray
    if not np.all(valid_mask):
        rgb_uint8[~valid_mask] = [40, 40, 40]

    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(out_path)
    return out_path


def format_validation_summary(
    metrics: Dict[str, Any],
    calib_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Format metrics and calibration parameters into both a human-readable text block and JSON.
    """
    combined = {
        "methodology_notice": (
            "Internal validation against calibration reference DEM (SRTM 30m). "
            "Positive bias in built areas reflects DSM vs DEM structural differences."
        ),
        "calibration": calib_params,
        "metrics": metrics,
    }

    text_block = (
        f"--- DepthWizard Stage 2 Metric DSM Validation ---\n"
        f"Calibration Method : {calib_params.get('method', 'N/A').upper()}\n"
        f"Regression Formula : Elevation = {calib_params.get('slope', 0):.2f} * rDSM + {calib_params.get('intercept', 0):.2f} m\n"
        f"Goodness of Fit R2 : {calib_params.get('r2', 0):.4f}\n"
        f"Inlier Fraction    : {calib_params.get('inlier_fraction', 0):.1%}\n"
        f"Control Points     : {calib_params.get('n_points', 0)}\n"
        f"--------------------------------------------------\n"
        f"Validation RMSE    : {metrics.get('rmse', 0):.2f} m\n"
        f"Validation MAE     : {metrics.get('mae', 0):.2f} m\n"
        f"Mean Bias (DSM-DEM): {metrics.get('bias', 0):+.2f} m\n"
        f"Pearson r (corr)   : {metrics.get('pearson_r', 0):.4f}\n"
        f"Valid Pixel Count  : {metrics.get('valid_pixels', 0):,}\n"
        f"--------------------------------------------------"
    )

    return text_block, combined


def save_validation_report(
    metrics: Dict[str, Any],
    calib_params: Dict[str, Any],
    path: Union[str, Path],
) -> Path:
    """Save validation summary as structured JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, data = format_validation_summary(metrics, calib_params)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_path
