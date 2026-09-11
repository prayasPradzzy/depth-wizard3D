"""
depth_extract.py - Stage 1 Monocular Relative Depth Extraction for DepthWizard.

WHY THIS MODULE EXISTS:
1. Monocular Elevation Estimation (Single-View Problem):
   ISRO PS 26175 demands 3D reconstruction from a SINGLE optical RGB image without stereo
   pairs or LiDAR. We leverage Depth Anything V2, a state-of-the-art vision foundation model
   trained on massive diverse datasets.
2. Disparity-to-Elevation Physics:
   Depth Anything V2 predicts metric-invariant INVERSE depth (disparity, 1/distance).
   For a nadir (overhead) satellite or aerial sensor looking down at the earth, pixels
   closer to the camera correspond physically to points with HIGHER ground elevation.
   Therefore, normalized disparity maps directly to relative Digital Surface Model (rDSM) height.
   (Limitation: This assumption degrades at high obliquity / off-nadir angles, which must
   be noted in project limitations).
3. 2nd / 98th Percentile Normalization:
   Aerial scenes contain extreme radiometric outliers: specular reflections from tin roofs,
   solar glints on water, or deep cast shadows. Raw min/max normalization gets pinned to
   these rare outlier pixels, crushing the elevation contrast of all genuine terrain.
   Robust clipping between the 2nd and 98th percentiles preserves the dynamic range across
   the entire actual landscape.
4. Lazy Loading & Device Fallback:
   Models are loaded on the first predict() invocation rather than on module import.
   This prevents unnecessary GPU/RAM allocation during testing, CLI parsing, or utility calls.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


# Approximate ground sample distance of the aerial imagery the backbone performs
# best on. Monocular depth is scale-dependent: the network learned what a building
# edge, a tree crown and a road width look like *in pixels*. Hand it imagery at a
# very different GSD and those learned priors no longer match, which is a large part
# of why cross-sensor performance degrades.
NATIVE_GSD_M = 0.5
GSD_TOLERANCE = (0.6, 1.7)     # ratios inside this band are close enough to leave alone
GSD_MAX_EDGE = 2048            # never blow an image up past this after rescaling


def _resize_bilinear(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D float array to (H, W) with high-precision bilinear interpolation."""
    target_h, target_w = size_hw
    if arr.shape[:2] == (target_h, target_w):
        return arr.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float()
    out = torch.nn.functional.interpolate(
        t, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return out.squeeze().numpy().astype(np.float32)


def _solve_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares scale/shift so that ``a * src + b`` best matches ``dst``.

    WHY: Depth Anything V2 output is scale- and shift-invariant, so each tile is
    predicted in its own arbitrary range. Aligning every tile to a single coarse
    global prediction is what makes the stitched high-resolution result consistent.
    """
    s = np.asarray(src, dtype=np.float64).ravel()
    d = np.asarray(dst, dtype=np.float64).ravel()
    finite = np.isfinite(s) & np.isfinite(d)
    if finite.sum() < 16:
        return 1.0, 0.0
    A = np.vstack([s[finite], np.ones(finite.sum())]).T
    sol, *_ = np.linalg.lstsq(A, d[finite], rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-8:
        return 1.0, 0.0
    return a, b


def _feather_window(h: int, w: int) -> np.ndarray:
    """Separable Hann window used to blend overlapping tiles without visible seams."""
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.clip(np.outer(wy, wx), 1e-3, None).astype(np.float32)


@dataclass
class DepthResult:
    """
    Dataclass encapsulating relative elevation extraction output.

    Attributes:
        normalized_depth: Relative height array bounded in [0.0, 1.0], shape (H, W), dtype float32.
                          0.0 indicates lowest terrain; 1.0 indicates highest roof/canopy.
        raw_prediction: Raw unnormalized disparity output from the model backbone.
        source_size: (Height, Width) of the original input image.
        model_name: HuggingFace model checkpoint identifier used.
    """
    normalized_depth: np.ndarray
    raw_prediction: np.ndarray
    source_size: Tuple[int, int]
    model_name: str


def normalise_depth(depth_map: np.ndarray, robust: bool = True) -> np.ndarray:
    """
    Normalize depth/disparity array to [0.0, 1.0].

    Args:
        depth_map: 2D numpy array of raw disparity values.
        robust: If True, uses 2nd and 98th percentile clipping to eliminate outliers.
                If False, applies standard min/max scaling.

    Returns:
        float32 array normalized to [0.0, 1.0].

    WHY ROBUST NORMALIZATION:
    In remote sensing imagery, high-reflectance features (e.g., metallic building roofs)
    or sensor noise produce severe disparity spikes. Min/max scaling compresses 99% of the
    terrain into a narrow 5-10% band. Percentile clipping ensures the true terrain relief
    spans the full [0, 1] dynamic range.
    """
    arr = np.asarray(depth_map, dtype=np.float32)

    # Filter non-finite values if any
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.float32)

    if robust:
        p2, p98 = np.percentile(arr[finite_mask], [2.0, 98.0])
        if p98 - p2 < 1e-6:
            # Flat prediction or zero dynamic range
            return np.zeros_like(arr, dtype=np.float32)
        clipped = np.clip(arr, p2, p98)
        norm = (clipped - p2) / (p98 - p2)
    else:
        d_min = float(np.min(arr[finite_mask]))
        d_max = float(np.max(arr[finite_mask]))
        if d_max - d_min < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        norm = (arr - d_min) / (d_max - d_min)

    return np.clip(norm, 0.0, 1.0).astype(np.float32)


class DepthEstimator:
    """
    Depth Anything V2 wrapper supporting lazy loading and device auto-detection.
    """

    MODEL_REGISTRY = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        """
        Initialize the estimator configuration. Model is NOT loaded until predict() is called.

        Args:
            model_size: One of 'small', 'base', 'large'. Default 'small'.
            device: 'cuda', 'mps', 'cpu', or None (auto-detect).
        """
        model_size = model_size.lower()
        if model_size not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_size '{model_size}'. Expected one of: {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model_size = model_size
        self.checkpoint = self.MODEL_REGISTRY[model_size]
        self.device = self._resolve_device(device)
        self._pipeline = None

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Auto-detect available compute device: cuda > mps > cpu."""
        if requested and requested.lower() != "auto":
            return requested.lower()
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazy load the HuggingFace depth-estimation pipeline."""
        if self._pipeline is not None:
            return

        from transformers import pipeline

        device_arg = 0 if self.device == "cuda" else (-1 if self.device == "cpu" else self.device)
        self._pipeline = pipeline(
            task="depth-estimation",
            model=self.checkpoint,
            device=device_arg,
        )

    @staticmethod
    def _to_pil(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Coerce any supported input into an RGB PIL image."""
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                uint8_img = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            else:
                uint8_img = image_input
            if uint8_img.ndim == 2:
                uint8_img = np.stack([uint8_img] * 3, axis=-1)
            elif uint8_img.shape[2] == 4:
                uint8_img = uint8_img[:, :, :3]
            return Image.fromarray(uint8_img, mode="RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _infer_raw(self, pil_image: Image.Image) -> np.ndarray:
        """Run the HF pipeline once and return a float32 disparity array at the image's own size."""
        self._load_model()
        orig_w, orig_h = pil_image.size
        result = self._pipeline(pil_image)

        if isinstance(result, dict) and "predicted_depth" in result:
            pred_tensor = result["predicted_depth"]
            if isinstance(pred_tensor, torch.Tensor):
                raw_pred = pred_tensor.squeeze().cpu().numpy()
            else:
                raw_pred = np.asarray(pred_tensor).squeeze()
        elif isinstance(result, dict) and "depth" in result:
            depth_item = result["depth"]
            if isinstance(depth_item, Image.Image):
                raw_pred = np.array(depth_item, dtype=np.float32)
            else:
                raw_pred = np.asarray(depth_item, dtype=np.float32).squeeze()
        else:
            raw_pred = np.asarray(result, dtype=np.float32).squeeze()

        return _resize_bilinear(raw_pred.astype(np.float32), (orig_h, orig_w))

    def _infer_tiled(
        self,
        pil_image: Image.Image,
        tile: int = 700,
        overlap: float = 0.33,
        base_max_dim: int = 768,
        max_tiles: int = 49,
    ) -> np.ndarray:
        """
        Multi-scale tiled inference for large images.

        1. Predict a coarse global disparity map (captures correct low-frequency shape).
        2. Predict each overlapping tile at native resolution (captures fine structure).
        3. Align every tile to the global map with a per-tile scale/shift (the backbone is
           scale/shift invariant, so this is required for the tiles to agree).
        4. Blend the aligned tiles with a feathered Hann window to remove seams.
        """
        orig_w, orig_h = pil_image.size

        # --- Stage 1: coarse global prediction ---
        scale = base_max_dim / max(orig_w, orig_h)
        base_w = max(32, int(round(orig_w * scale)))
        base_h = max(32, int(round(orig_h * scale)))
        base_small = self._infer_raw(pil_image.resize((base_w, base_h), Image.Resampling.LANCZOS))
        base = _resize_bilinear(base_small, (orig_h, orig_w))

        # --- Stage 2/3/4: tiled refinement ---
        step = max(1, int(round(tile * (1.0 - overlap))))
        ys = list(range(0, max(1, orig_h - tile) + 1, step)) or [0]
        xs = list(range(0, max(1, orig_w - tile) + 1, step)) or [0]
        if ys[-1] != max(0, orig_h - tile):
            ys.append(max(0, orig_h - tile))
        if xs[-1] != max(0, orig_w - tile):
            xs.append(max(0, orig_w - tile))

        if len(ys) * len(xs) > max_tiles:
            # Too many tiles for the CPU budget - fall back to the coarse map.
            return base

        acc = np.zeros((orig_h, orig_w), dtype=np.float32)
        wsum = np.zeros((orig_h, orig_w), dtype=np.float32)

        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + tile, orig_h)
                x1 = min(x0 + tile, orig_w)
                crop = pil_image.crop((x0, y0, x1, y1))
                td = self._infer_raw(crop)
                a, b = _solve_affine(td, base[y0:y1, x0:x1])
                aligned = a * td + b
                win = _feather_window(y1 - y0, x1 - x0)
                acc[y0:y1, x0:x1] += aligned * win
                wsum[y0:y1, x0:x1] += win

        combined = np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-6), base)
        return combined.astype(np.float32)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        robust: bool = True,
        tiled: Union[bool, str] = False,
        tile_size: int = 700,
        tile_overlap: float = 0.33,
        gsd_m: Optional[float] = None,
    ) -> DepthResult:
        """
        Estimate relative depth from a single RGB image.

        Args:
            image_input: Filepath, PIL Image, or numpy array (H, W, 3).
            robust: Whether to use 2/98 percentile normalization (default True).
            tiled: False (single pass), True (force tiled), or "auto" (tile only when the
                   longest edge exceeds ~1.4x the tile size, where detail is otherwise lost).
            tile_size: Native-resolution tile edge in pixels.
            tile_overlap: Fractional overlap between adjacent tiles (0-0.9).
            gsd_m: Ground sample distance of the input in metres per pixel. When given,
                   the image is rescaled to the backbone's native GSD before inference
                   and the result mapped back, which is what makes one model usable
                   across sensors of differing resolution.

        Returns:
            DepthResult containing normalized relative height and raw disparity.
        """
        pil_image = self._to_pil(image_input)
        orig_w, orig_h = pil_image.size

        # --- Ground-sample-distance normalisation -------------------------------
        # Resample so the scene is presented to the network at roughly the GSD it
        # was tuned for, then map the prediction back to the source grid. This is
        # what lets one model serve sensors with very different resolutions
        # (e.g. Cartosat-3 at 0.25 m vs Cartosat-2 at 0.65 m) instead of silently
        # degrading on whichever one it was not trained at.
        infer_image = pil_image
        gsd_note = None
        if gsd_m and gsd_m > 0:
            ratio = float(gsd_m) / NATIVE_GSD_M
            if ratio < GSD_TOLERANCE[0] or ratio > GSD_TOLERANCE[1]:
                tw, th = int(round(orig_w * ratio)), int(round(orig_h * ratio))
                longest = max(tw, th)
                if longest > GSD_MAX_EDGE:                # keep upsampling bounded
                    k = GSD_MAX_EDGE / longest
                    tw, th = int(tw * k), int(th * k)
                tw, th = max(64, tw), max(64, th)
                infer_image = pil_image.resize((tw, th), Image.Resampling.LANCZOS)
                gsd_note = f"{orig_w}x{orig_h} @ {gsd_m:.2f}m/px -> {tw}x{th} @ ~{NATIVE_GSD_M}m/px"
                print(f"[INFO] GSD normalisation: {gsd_note}")

        iw, ih = infer_image.size
        if tiled == "auto":
            use_tiled = max(iw, ih) > int(tile_size * 1.4)
        else:
            use_tiled = bool(tiled)

        if use_tiled:
            raw_pred = self._infer_tiled(infer_image, tile=tile_size, overlap=tile_overlap)
        else:
            raw_pred = self._infer_raw(infer_image)

        # Map the prediction back onto the source grid.
        if raw_pred.shape[:2] != (orig_h, orig_w):
            raw_pred = _resize_bilinear(raw_pred, (orig_h, orig_w))

        norm_depth = normalise_depth(raw_pred, robust=robust)

        return DepthResult(
            normalized_depth=norm_depth,
            raw_prediction=raw_pred,
            source_size=(orig_h, orig_w),
            model_name=self.checkpoint,
        )
