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

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        robust: bool = True,
    ) -> DepthResult:
        """
        Estimate relative depth from a single RGB image.

        Args:
            image_input: Filepath, PIL Image, or numpy array (H, W, 3).
            robust: Whether to use 2/98 percentile normalization (default True).

        Returns:
            DepthResult containing normalized relative height and raw disparity.
        """
        self._load_model()

        # Load & standardize image to RGB PIL Image
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                uint8_img = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            else:
                uint8_img = image_input
            if uint8_img.ndim == 2:
                uint8_img = np.stack([uint8_img] * 3, axis=-1)
            elif uint8_img.shape[2] == 4:
                uint8_img = uint8_img[:, :, :3]
            pil_image = Image.fromarray(uint8_img, mode="RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_image.size

        # Run pipeline inference
        result = self._pipeline(pil_image)

        # Extract predicted disparity array
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

        # Resize output back to source resolution with bilinear interpolation if mismatched
        pred_h, pred_w = raw_pred.shape[:2]
        if (pred_h, pred_w) != (orig_h, orig_w):
            # Use PyTorch functional interpolate for high-precision bilinear resizing
            tensor_in = torch.from_numpy(raw_pred).unsqueeze(0).unsqueeze(0).float()
            resized = torch.nn.functional.interpolate(
                tensor_in,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            raw_pred = resized.squeeze().numpy()

        # Normalize relative height (disparity -> relative height)
        norm_depth = normalise_depth(raw_pred, robust=robust)

        return DepthResult(
            normalized_depth=norm_depth,
            raw_prediction=raw_pred,
            source_size=(orig_h, orig_w),
            model_name=self.checkpoint,
        )
