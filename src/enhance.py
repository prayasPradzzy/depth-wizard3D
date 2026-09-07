"""
enhance.py - Radiometric pre-processing for monocular depth extraction.

WHY THIS MODULE EXISTS:
Depth Anything V2 is trained on consumer photography with balanced exposure. Nadir
aerial/satellite scenes routinely violate that assumption in ways that quietly wreck
relative-height extraction:

1. Deep cast shadows (north sides of tall buildings, canyon floors) clip to near-black.
   The backbone sees a flat dark blob and predicts a flat surface where there is really
   a vertical facade. CLAHE (Contrast Limited Adaptive Histogram Equalization) locally
   re-expands the tonal range inside those shadow regions without blowing out the
   already well-exposed rooftops.

2. Open water bodies have near-zero texture plus specular sun glints, so the backbone
   emits high-frequency noise that later becomes spiky garbage geometry. A conservative
   RGB-only water heuristic lets the pipeline clamp those pixels to the local ground
   datum instead.

Everything here is implemented in pure NumPy/Pillow/SciPy (already project dependencies)
so no OpenCV runtime is introduced.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from PIL import Image
import scipy.ndimage as ndi


# --------------------------------------------------------------------------------------
# CLAHE shadow enhancement
# --------------------------------------------------------------------------------------

def _clahe_channel(
    channel: np.ndarray,
    clip_limit: float,
    grid: Tuple[int, int],
) -> np.ndarray:
    """
    Apply CLAHE to a single 2-D uint8 channel and return a float array in [0, 255].

    Implementation: partition the image into `grid` tiles, build a clipped, redistributed
    CDF per tile, then bilinearly interpolate the four surrounding tile mappings for every
    pixel. This is the standard Zuiderveld (1994) formulation.
    """
    h, w = channel.shape
    gy, gx = grid
    gy = max(1, min(gy, h))
    gx = max(1, min(gx, w))

    tile_h = int(np.ceil(h / gy))
    tile_w = int(np.ceil(w / gx))
    n_bins = 256

    # Per-tile lookup tables: shape (gy, gx, 256)
    luts = np.empty((gy, gx, n_bins), dtype=np.float32)
    clip = max(1.0, clip_limit * (tile_h * tile_w) / n_bins)

    for ty in range(gy):
        for tx in range(gx):
            y0, y1 = ty * tile_h, min((ty + 1) * tile_h, h)
            x0, x1 = tx * tile_w, min((tx + 1) * tile_w, w)
            tile = channel[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.float32)

            # Clip and redistribute the excess uniformly across all bins.
            excess = np.maximum(hist - clip, 0.0).sum()
            hist = np.minimum(hist, clip) + excess / n_bins

            cdf = np.cumsum(hist)
            if cdf[-1] <= 0:
                luts[ty, tx] = np.arange(n_bins, dtype=np.float32)
            else:
                luts[ty, tx] = (cdf - cdf[0]) / max(1e-6, cdf[-1] - cdf[0]) * (n_bins - 1)

    # Bilinear interpolation of tile mappings across the full-resolution grid.
    ys = np.arange(h)
    xs = np.arange(w)
    fy = np.clip(ys / tile_h - 0.5, 0, gy - 1)
    fx = np.clip(xs / tile_w - 0.5, 0, gx - 1)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, gy - 1)
    x1 = np.minimum(x0 + 1, gx - 1)
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[None, :]

    idx = channel.astype(np.int64)
    # Gather each corner's mapping for every pixel.
    m00 = _gather(luts, y0, x0, idx)
    m01 = _gather(luts, y0, x1, idx)
    m10 = _gather(luts, y1, x0, idx)
    m11 = _gather(luts, y1, x1, idx)

    top = m00 * (1 - wx) + m01 * wx
    bot = m10 * (1 - wx) + m11 * wx
    return top * (1 - wy) + bot * wy


def _gather(luts: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    """Vector gather: out[y, x] = luts[row_idx[y], col_idx[x], val_idx[y, x]]."""
    rr = row_idx[:, None]
    cc = col_idx[None, :]
    return luts[rr, cc, val_idx]


def enhance_shadows(
    image: Union[str, Image.Image, np.ndarray],
    clip_limit: float = 2.5,
    tile_grid: Tuple[int, int] = (8, 8),
    strength: float = 0.85,
) -> Image.Image:
    """
    Lift detail out of shadowed regions using CLAHE on the luminance channel.

    Args:
        image: Path, PIL image, or HxWx3 uint8 array.
        clip_limit: CLAHE contrast clip (higher = more aggressive; 2-4 is typical).
        tile_grid: (rows, cols) of contextual regions.
        strength: Blend factor between the original luminance (0.0) and the fully
                  equalized luminance (1.0). 0.85 keeps highlights natural.

    Returns:
        A new RGB PIL image with shadow contrast restored and hue/saturation preserved.
    """
    if isinstance(image, str):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil = Image.fromarray(arr[:, :, :3], mode="RGB")
    else:
        pil = image.convert("RGB")

    rgb = np.asarray(pil, dtype=np.float32)
    # Rec. 601 luma.
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    luma_u8 = np.clip(luma, 0, 255).astype(np.uint8)

    equalized = _clahe_channel(luma_u8, clip_limit, tile_grid)
    new_luma = (1.0 - strength) * luma + strength * equalized

    # Scale each colour channel by the luminance gain to preserve chroma.
    gain = np.where(luma > 1.0, new_luma / np.maximum(luma, 1.0), 1.0)[..., None]
    out = np.clip(rgb * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# --------------------------------------------------------------------------------------
# RGB-only water detection
# --------------------------------------------------------------------------------------

def detect_water_mask(
    image: Union[Image.Image, np.ndarray],
    texture_percentile: float = 20.0,
) -> np.ndarray:
    """
    Estimate a boolean water mask from an optical RGB image (no NIR band required).

    Heuristic: water is simultaneously (a) low local texture and (b) either blue-dominant
    or very dark. This is deliberately conservative - it is better to miss a pond than to
    flatten a parking lot.

    Args:
        image: PIL image or HxWx3 uint8 array.
        texture_percentile: Pixels below this local-gradient percentile are "smooth".

    Returns:
        Boolean array (H, W), True where the pixel is likely open water.
    """
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        rgb = np.asarray(image, dtype=np.float32)[:, :, :3]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b

    # Local texture via gradient magnitude, smoothed.
    gx = ndi.sobel(luma, axis=1)
    gy = ndi.sobel(luma, axis=0)
    texture = ndi.uniform_filter(np.hypot(gx, gy), size=7)
    smooth = texture < np.percentile(texture, texture_percentile)

    blueish = (b >= g - 4) & (g >= r - 4) & (b > 25)
    dark = luma < np.percentile(luma, 15)

    mask = smooth & (blueish | dark)
    # Drop tiny speckles, keep only connected bodies >= ~0.05% of the frame.
    labels, n = ndi.label(mask)
    if n > 0:
        min_size = max(64, int(0.0005 * mask.size))
        sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        keep = {i + 1 for i, s in enumerate(sizes) if s >= min_size}
        mask = np.isin(labels, list(keep)) if keep else np.zeros_like(mask)
    return mask.astype(bool)


def flatten_water(
    rel_depth: np.ndarray,
    water_mask: np.ndarray,
    datum_percentile: float = 8.0,
) -> np.ndarray:
    """
    Replace relative-depth values inside `water_mask` with the local ground datum so open
    water renders as a flat sheet instead of glint noise.
    """
    if not np.any(water_mask):
        return rel_depth
    out = np.array(rel_depth, dtype=np.float32, copy=True)
    land = out[~water_mask]
    datum = float(np.percentile(land, datum_percentile)) if land.size else float(np.min(out))
    out[water_mask] = datum
    return out
