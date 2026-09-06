"""
mesh_export.py - Web-ready 3D asset generation (downsampled 16-bit heightmap + matched texture).

WHY THIS MODULE EXISTS:
1. Browser Performance & Anti-Aliasing:
   Modern aerial/satellite images can exceed 4000x4000 pixels (16 million pixels). Attempting
   to generate a Three.js PlaneGeometry with 16M vertices will immediately freeze or crash
   the browser's WebGL context. Downsampling to max_dim=512 (~260,000 vertices) provides
   fluid 60 FPS rendering on laptops and integrated GPUs while retaining structural relief.
2. Anti-Aliasing on Downsampling:
   Downsampling without filtering introduces high-frequency aliasing and jagged edges.
   We use high-quality Lanczos filtering for the RGB texture and bilinear filtering for
   the elevation array to preserve smooth topography.
3. Strict Aspect Ratio Preservation:
   Satellite images are rarely square. Non-square images forced into a square geometry
   produce stretched, distorted landscapes. We compute dimensions dynamically to preserve
   the physical aspect ratio.
4. 16-bit Heightmap Normalization:
   The elevation is normalized and saved as a 16-bit PNG (0..65535). The true physical
   minimum and maximum elevation in metres (or relative percentage) are written into a
   metadata manifest.json, enabling Three.js to recover the exact metric height at any vertex.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image


def regularize_architectural_elevations(
    elevation: np.ndarray,
    rgb_img: Optional[np.ndarray] = None,
    iterations: int = 3,
) -> np.ndarray:
    """
    Transform continuous monocular elevation ramps into crisp architectural cuboids.

    WHY THIS MATTERS:
    Monocular depth models predict smooth continuous sigmoidal transitions around structures.
    In 3D terrain meshes, this causes buildings to look like melted mounds or pyramids rather
    than upright architectural cuboids. This regularizer:
    1. Shock-Filters elevation gradients: Turns slanted ramps into near-vertical cliff walls.
    2. Rooftop Plateau Smoothing: Flattens local roof surfaces using median filtering on upper quantiles.
    3. RGB-Guided Edge Snapping: Uses luminance gradients to snap wall boundaries to actual visual edges.
    """
    import scipy.ndimage as ndi

    arr = np.asarray(elevation, dtype=np.float32).copy()
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return arr

    d_min = float(np.min(arr[finite_mask]))
    d_max = float(np.max(arr[finite_mask]))
    span = d_max - d_min
    if span < 1e-4:
        return arr

    # Compute RGB luminance edge map if available
    edge_weight = np.ones_like(arr, dtype=np.float32)
    if rgb_img is not None:
        try:
            rgb_arr = np.asarray(rgb_img, dtype=np.float32)
            if rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                lum = 0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]
                gx = ndi.sobel(lum, axis=1)
                gy = ndi.sobel(lum, axis=0)
                rgb_edge = np.hypot(gx, gy)
                p95 = float(np.percentile(rgb_edge, 95.0))
                if p95 > 1e-3:
                    edge_norm = np.clip(rgb_edge / p95, 0.0, 1.0)
                    edge_weight = 1.0 + 1.5 * edge_norm
        except Exception:
            pass

    # Iterative shock filtering to sharpen building walls
    dt = 0.18
    for _ in range(iterations):
        lap = ndi.laplace(arr)
        gx = ndi.sobel(arr, axis=1)
        gy = ndi.sobel(arr, axis=0)
        grad_mag = np.hypot(gx, gy)

        # Shock filter update: march toward local extrema based on Laplacian sign
        shock = -np.sign(lap) * np.minimum(grad_mag * edge_weight, span * 0.08)
        arr = arr + dt * shock
        arr = np.clip(arr, d_min, d_max)

    # Rooftop plateau regularization:
    # Identify high elevation regions (upper 35% of local dynamic range) and level them
    p65 = float(np.percentile(arr[finite_mask], 65.0))
    rooftop_mask = arr > p65
    if np.any(rooftop_mask):
        med_filtered = ndi.median_filter(arr, size=5)
        arr[rooftop_mask] = 0.75 * med_filtered[rooftop_mask] + 0.25 * arr[rooftop_mask]

    return np.clip(arr, d_min, d_max).astype(np.float32)


def export_for_web(
    elevation_arr: np.ndarray,
    rgb_input: Union[str, Path, Image.Image, np.ndarray],
    out_dir: Union[str, Path],
    max_dim: int = 512,
    is_georeferenced: bool = False,
    crs_str: Optional[str] = None,
    gsd_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Export elevation array and RGB texture downsampled and formatted for Three.js.

    Args:
        elevation_arr: 2D float array of elevation values (metric metres or relative [0, 1]).
        rgb_input: Path, PIL Image, or numpy array of the source RGB image.
        out_dir: Destination folder for web assets.
        max_dim: Maximum dimension (width or height) for the 3D mesh grid (default 512).
        is_georeferenced: Whether elevation_arr represents true metric metres.
        crs_str: Coordinate reference system identifier if available.
        gsd_m: Ground sample distance in metres per pixel if available.

    Returns:
        Dict containing asset metadata, dimensions, and file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Standardize RGB Image
    if isinstance(rgb_input, (str, Path)):
        pil_rgb = Image.open(rgb_input).convert("RGB")
    elif isinstance(rgb_input, np.ndarray):
        if rgb_input.dtype != np.uint8:
            uint8_arr = np.clip(rgb_input * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_arr = rgb_input
        if uint8_arr.ndim == 2:
            uint8_arr = np.stack([uint8_arr] * 3, axis=-1)
        elif uint8_arr.shape[2] == 4:
            uint8_arr = uint8_arr[:, :, :3]
        pil_rgb = Image.fromarray(uint8_arr, mode="RGB")
    elif isinstance(rgb_input, Image.Image):
        pil_rgb = rgb_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported rgb_input type: {type(rgb_input)}")

    orig_w, orig_h = pil_rgb.size
    elev_h, elev_w = elevation_arr.shape[:2]

    # Ensure elevation array matches RGB size before web downsampling
    if (elev_h, elev_w) != (orig_h, orig_w):
        elev_img = Image.fromarray(elevation_arr.astype(np.float32), mode="F")
        elev_img = elev_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        elevation_arr = np.array(elev_img, dtype=np.float32)

    # 2. Compute Downsampled Dimensions (Preserve Aspect Ratio)
    aspect_ratio = orig_w / orig_h
    if max(orig_w, orig_h) > max_dim:
        if orig_w >= orig_h:
            target_w = max_dim
            target_h = max(16, int(round(max_dim / aspect_ratio)))
        else:
            target_h = max_dim
            target_w = max(16, int(round(max_dim * aspect_ratio)))
    else:
        target_w = orig_w
        target_h = orig_h

    # 3. Resample RGB Texture (Lanczos for crisp anti-aliased texture)
    resized_rgb = pil_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
    texture_path = out_path / "texture.png"
    resized_rgb.save(texture_path, format="PNG", optimize=True)

    # 4. Resample Elevation Array (Bilinear interpolation)
    elev_float_img = Image.fromarray(elevation_arr.astype(np.float32), mode="F")
    resized_elev_img = elev_float_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    downsampled_elev = np.array(resized_elev_img, dtype=np.float32)

    # 5. Generate Regularized Sharp Cuboid Elevation Array
    sharp_elev = regularize_architectural_elevations(
        downsampled_elev,
        rgb_img=np.array(resized_rgb),
        iterations=3,
    )

    # 6. Extract Elevation Span and Normalize to 16-bit Grayscale PNG
    finite_mask = np.isfinite(downsampled_elev)
    if np.any(finite_mask):
        min_elev = float(np.min(downsampled_elev[finite_mask]))
        max_elev = float(np.max(downsampled_elev[finite_mask]))
    else:
        min_elev = 0.0
        max_elev = 1.0

    span = max_elev - min_elev
    if span < 1e-6:
        norm_elev = np.zeros_like(downsampled_elev, dtype=np.float32)
        norm_sharp = np.zeros_like(sharp_elev, dtype=np.float32)
    else:
        norm_elev = (downsampled_elev - min_elev) / span
        norm_sharp = (sharp_elev - min_elev) / span

    # Clip to [0, 1] and quantize to 16-bit unsigned integer [0, 65535]
    u16_elev = np.round(np.clip(norm_elev, 0.0, 1.0) * 65535.0).astype(np.uint16)
    heightmap_img = Image.fromarray(u16_elev, mode="I;16")
    heightmap_path = out_path / "heightmap.png"
    heightmap_img.save(heightmap_path)

    u16_sharp = np.round(np.clip(norm_sharp, 0.0, 1.0) * 65535.0).astype(np.uint16)
    heightmap_sharp_img = Image.fromarray(u16_sharp, mode="I;16")
    heightmap_sharp_path = out_path / "heightmap_sharp.png"
    heightmap_sharp_img.save(heightmap_sharp_path)

    # Save raw float32 binary buffers for lossless browser WebGL displacement
    bin_path = out_path / "heightmap.bin"
    downsampled_elev.astype(np.float32).tofile(bin_path)

    bin_sharp_path = out_path / "heightmap_sharp.bin"
    sharp_elev.astype(np.float32).tofile(bin_sharp_path)

    # 7. Compute Ground Sample Distance at Resampled Resolution
    effective_gsd = None
    if gsd_m is not None:
        effective_gsd = float(gsd_m * (orig_w / target_w))

    # 8. Write Manifest Metadata
    manifest = {
        "width": target_w,
        "height": target_h,
        "original_width": orig_w,
        "original_height": orig_h,
        "aspect_ratio": round(aspect_ratio, 4),
        "vertex_count": target_w * target_h,
        "min_elevation_m": round(min_elev, 2),
        "max_elevation_m": round(max_elev, 2),
        "elevation_span_m": round(span, 2),
        "is_georeferenced": bool(is_georeferenced),
        "crs": crs_str if crs_str else "Local (Non-georeferenced)",
        "gsd_m": round(effective_gsd, 3) if effective_gsd else None,
        "heightmap_file": "heightmap.png",
        "heightmap_bin_file": "heightmap.bin",
        "heightmap_sharp_file": "heightmap_sharp.png",
        "heightmap_sharp_bin_file": "heightmap_sharp.bin",
        "texture_file": "texture.png",
    }

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "heightmap_path": heightmap_path,
        "heightmap_sharp_path": heightmap_sharp_path,
        "texture_path": texture_path,
    }
