"""
utils.py - Model-independent I/O, serialization, and visualization for DepthWizard.

WHY THIS MODULE EXISTS:
1. Separation of Concerns: Model inference (PyTorch/Transformers) is cleanly decoupled
   from file formatting and visualization. This allows the testing of all I/O and
   image processing logic on any system without requiring GPU or heavyweight ML runtimes.
2. 16-bit PNG Heightmaps (Crucial for 3D Mesh Generation):
   Standard 8-bit images offer only 256 discrete elevation steps. Displacing Three.js
   geometry with an 8-bit heightmap results in severe stair-stepping (terracing artifacts).
   A 16-bit PNG provides 65,536 elevation discrete steps, yielding continuous, smooth
   topography in the browser viewer.
3. Perceptually Uniform Color Mapping (Turbo):
   Google's Turbo colormap provides high dynamic range and smooth luminance transitions,
   avoiding the false gradients created by Jet while retaining visual clarity for judges.
4. Failure Detection (flat_warning):
   Depth models trained on terrestrial photography can struggle with nadir aerial images.
   Calculating depth statistics with a flag at std < 0.05 catches degenerate predictions early
   before they propagate to downstream calibration or 3D rendering.
"""

from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm


def save_raw(data: np.ndarray, path: Union[str, Path]) -> Path:
    """
    Save floating point depth/elevation array as a raw binary NumPy .npy file.

    WHY:
    Preserves exact IEEE 754 float32 precision without quantization, allowing lossless
    re-use in calibration, statistical evaluation, or offline analysis.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data.astype(np.float32))
    return out_path


def save_grayscale(data: np.ndarray, path: Union[str, Path]) -> Path:
    """
    Save a normalized [0, 1] depth/elevation array as a 16-bit single-channel PNG.

    WHY 16-BIT:
    Directly displaces vertices in Three.js. 16-bit (0..65535) provides 256x finer vertical
    resolution than standard 8-bit images, preventing vertex terracing.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure data is strictly bounded within [0, 1]
    clipped = np.clip(data, 0.0, 1.0)
    # Scale to 16-bit unsigned integer range [0, 65535]
    u16_data = np.round(clipped * 65535.0).astype(np.uint16)

    img = Image.fromarray(u16_data, mode="I;16")
    img.save(out_path)
    return out_path


def save_colormap(
    data: np.ndarray, path: Union[str, Path], cmap: str = "turbo"
) -> Path:
    """
    Save a normalized [0, 1] array as an RGB image colored with the specified colormap.

    WHY TURBO:
    Turbo provides a perceptually uniform, high-contrast spectrum that highlights subtle
    variations in elevation (e.g., roads vs sidewalks vs building roofs).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(data, 0.0, 1.0)
    colormap_fn = mpl.colormaps[cmap] if hasattr(mpl, "colormaps") else cm.get_cmap(cmap)
    # colormap_fn returns RGBA floats in [0, 1]
    rgba = colormap_fn(clipped)
    rgb_uint8 = (rgba[:, :, :3] * 255.0).round().astype(np.uint8)

    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(out_path)
    return out_path


def save_comparison(
    rgb_image: Union[Image.Image, np.ndarray],
    depth_map: np.ndarray,
    path: Union[str, Path],
    cmap: str = "turbo",
) -> Path:
    """
    Save a side-by-side composite comparison panel: [RGB Input | Turbo Depth].

    WHY:
    Immediate visual verification for judges and operators. Allows instant spot-checking
    to confirm whether high-contrast rooftops correctly correlate with high relative height.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare RGB PIL Image
    if isinstance(rgb_image, np.ndarray):
        if rgb_image.dtype != np.uint8:
            rgb_uint8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_image
        if rgb_uint8.ndim == 2:  # Grayscale to RGB
            rgb_uint8 = np.stack([rgb_uint8] * 3, axis=-1)
        elif rgb_uint8.shape[2] == 4:  # Strip alpha
            rgb_uint8 = rgb_uint8[:, :, :3]
        pil_rgb = Image.fromarray(rgb_uint8, mode="RGB")
    elif isinstance(rgb_image, Image.Image):
        pil_rgb = rgb_image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(rgb_image)}")

    target_h, target_w = pil_rgb.height, pil_rgb.width

    # Prepare colored depth image matching RGB dimensions
    clipped_depth = np.clip(depth_map, 0.0, 1.0)
    colormap_fn = mpl.colormaps[cmap] if hasattr(mpl, "colormaps") else cm.get_cmap(cmap)
    rgba_depth = colormap_fn(clipped_depth)
    depth_uint8 = (rgba_depth[:, :, :3] * 255.0).round().astype(np.uint8)
    pil_depth = Image.fromarray(depth_uint8, mode="RGB")

    if pil_depth.size != (target_w, target_h):
        pil_depth = pil_depth.resize((target_w, target_h), Image.Resampling.BILINEAR)

    # Stitch side-by-side: [RGB | Depth]
    composite = Image.new("RGB", (target_w * 2, target_h))
    composite.paste(pil_rgb, (0, 0))
    composite.paste(pil_depth, (target_w, 0))
    composite.save(out_path)
    return out_path


def depth_stats(depth_map: np.ndarray) -> Dict[str, Any]:
    """
    Compute distribution statistics for a depth map and flag degenerate predictions.

    WHY:
    Monocular backbones can fail silently on low-contrast or nadir satellite imagery,
    outputting near-constant depth planes. If standard deviation is < 0.05,
    flat_warning is tripped to signal that relative relief extraction has failed.
    """
    clean_data = depth_map[np.isfinite(depth_map)]
    if clean_data.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "flat_warning": True,
        }

    d_min = float(np.min(clean_data))
    d_max = float(np.max(clean_data))
    d_mean = float(np.mean(clean_data))
    d_std = float(np.std(clean_data))
    p05, p95 = np.percentile(clean_data, [5.0, 95.0])

    flat_warning = bool(d_std < 0.05)

    return {
        "min": d_min,
        "max": d_max,
        "mean": d_mean,
        "std": d_std,
        "p05": float(p05),
        "p95": float(p95),
        "flat_warning": flat_warning,
    }
