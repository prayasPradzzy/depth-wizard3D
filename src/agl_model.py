"""
agl_model.py - Serve the GAMUS fine-tuned checkpoint: metric height above ground.

WHAT THIS ADDS
==============
The stock backbone returns relative inverse depth - useful for shape, useless for
measurement. Our fine-tuned checkpoint (tools/train_gamus.py) was trained against
LiDAR nDSM, so it emits **metres above ground level** directly, with no DEM anchor
and no scale fitting.

That is the capability the problem statement actually asks for, so it belongs in
the live request path rather than only in an offline benchmark.

HOW IT COMPOSES WITH THE REST OF THE PIPELINE
=============================================
AGL is height above the local terrain, not above sea level. It deliberately does
NOT replace the surface model used for the 3D geometry:

    surface (DSM)   <- relative depth, or SRTM/COP30-calibrated elevation
    AGL             <- this model: how far each pixel stands above its own ground
    ground (DTM)    =  surface - AGL

Rendering AGL on its own would flatten the landscape - a mountain would vanish and
leave only the huts on it. So the 3D surface stays as it was, and this supplies a
measured AGL layer that previously had to be *estimated* morphologically. Flood
screening then runs on a ground surface derived from a trained prediction instead
of from a shape heuristic.

SCALE DEPENDENCE - THE IMPORTANT CAVEAT
=======================================
The model learned what a storey looks like *in pixels*, at roughly the GAMUS ground
sample distance. Metres are therefore only trustworthy when the input's true GSD is
known and the image has been resampled to that native scale. A plain PNG carries no
GSD, so its output is reported as relative, never as metres. `predict()` returns an
explicit `metric` flag and callers must honour it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "agl_vits.pt"

# Matches tools/train_gamus.py. Changing either without the other silently
# de-calibrates every prediction.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NATIVE_GSD_M = 0.5           # approximate GSD the checkpoint was trained at
GSD_TOLERANCE = (0.6, 1.7)   # ratios inside this band need no resampling
MAX_EDGE = 1536              # cap after rescaling, to bound CPU inference cost
HEIGHT_CLAMP = (0.0, 120.0)  # metres AGL; negatives are not physical for height


@dataclass
class AGLResult:
    """
    agl:     height above ground. Metres when `metric` is True, else relative units.
    metric:  whether `agl` may be quoted in metres.
    reason:  why it is or is not metric - surfaced in the UI rather than hidden.
    """
    agl: np.ndarray
    metric: bool
    reason: str
    model_name: str


class AGLPredictor:
    """Lazy-loading wrapper around the fine-tuned AGL checkpoint."""

    def __init__(self, checkpoint: Union[str, Path, None] = None, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint or DEFAULT_CHECKPOINT)
        self.device = device
        self._model = None
        self._meta = {}

    @property
    def available(self) -> bool:
        return self.checkpoint_path.exists()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForDepthEstimation

        ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model = AutoModelForDepthEstimation.from_pretrained(ck["checkpoint"])
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        if missing or unexpected:
            # Loud rather than silent: a partially loaded head still produces
            # plausible-looking numbers, which is the worst possible failure here.
            raise RuntimeError(
                f"AGL checkpoint did not match the base model "
                f"({len(missing)} missing, {len(unexpected)} unexpected tensors). "
                f"Refusing to serve metres from a partially loaded model."
            )
        model.to(self.device).eval()
        self._model = model
        self._meta = {
            "base": ck.get("checkpoint"),
            "epoch": ck.get("epoch"),
            "val_rmse_m": round(float(ck.get("score", 0.0)), 3),
            "target": ck.get("target"),
        }

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return Image.fromarray(arr[:, :, :3], mode="RGB")
        return image.convert("RGB")

    def predict(self, image, gsd_m: Optional[float] = None) -> AGLResult:
        """
        Predict height above ground.

        Args:
            image: PIL image, path, or HxWx3 uint8 array.
            gsd_m: true ground sample distance in metres/pixel. Supply it whenever
                   known - without it the result cannot be quoted in metres.
        """
        import torch

        self._load()
        pil = self._to_pil(image)
        orig_w, orig_h = pil.size

        infer = pil
        if gsd_m and gsd_m > 0:
            ratio = float(gsd_m) / NATIVE_GSD_M
            if ratio < GSD_TOLERANCE[0] or ratio > GSD_TOLERANCE[1]:
                tw, th = int(round(orig_w * ratio)), int(round(orig_h * ratio))
                longest = max(tw, th)
                if longest > MAX_EDGE:
                    k = MAX_EDGE / longest
                    tw, th = int(tw * k), int(th * k)
                infer = pil.resize((max(64, tw), max(64, th)), Image.Resampling.LANCZOS)
            metric, reason = True, f"GSD {gsd_m:.2f} m/px known; resampled to ~{NATIVE_GSD_M} m/px"
        else:
            # Cap the work on an unscaled image; the output is relative regardless.
            if max(orig_w, orig_h) > MAX_EDGE:
                k = MAX_EDGE / max(orig_w, orig_h)
                infer = pil.resize((int(orig_w * k), int(orig_h * k)), Image.Resampling.LANCZOS)
            metric, reason = False, "No ground sample distance available; heights are relative, not metres"

        x = (np.asarray(infer, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            out = self._model(pixel_values=t).predicted_depth
        agl = out.float().squeeze().cpu().numpy()

        if agl.shape[:2] != (orig_h, orig_w):
            agl = np.array(
                Image.fromarray(agl.astype(np.float32), mode="F")
                .resize((orig_w, orig_h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        agl = np.clip(agl, *HEIGHT_CLAMP).astype(np.float32)

        return AGLResult(agl=agl, metric=metric, reason=reason,
                         model_name=f"Depth Anything V2 Small, fine-tuned on GAMUS "
                                    f"(val RMSE {self._meta.get('val_rmse_m')} m)")


_singleton: Optional[AGLPredictor] = None


def get_agl_predictor() -> Optional[AGLPredictor]:
    """Shared predictor, or None when the checkpoint is absent or disabled."""
    global _singleton
    if os.environ.get("DW_USE_FINETUNED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return None
    if _singleton is None:
        p = AGLPredictor()
        if not p.available:
            return None
        _singleton = p
    return _singleton
