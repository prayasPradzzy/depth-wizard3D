"""
colormaps.py - The three colour ramps this project needs, without matplotlib.

WHY THIS EXISTS
===============
matplotlib was pulled in solely to look up `turbo`, `terrain` and `coolwarm`. It
costs roughly 40 MB resident and drags in a font cache and a backend stack that a
headless API server never uses. On a 512 MB host that is a meaningful share of the
budget spent on three lookup tables.

Each ramp is stored as anchor points and interpolated linearly, which is exactly
how matplotlib builds a LinearSegmentedColormap from the same anchors. Output is
visually equivalent for our purpose - these drive preview rasters and an error map,
not quantitative colour matching.
"""

from __future__ import annotations

import numpy as np

# (position, R, G, B) with channels in 0..1.
_RAMPS = {
    # Google Turbo: perceptually ordered, high contrast, no false banding.
    "turbo": [
        (0.000, 0.190, 0.072, 0.232), (0.125, 0.275, 0.404, 0.870),
        (0.250, 0.125, 0.742, 0.863), (0.375, 0.188, 0.945, 0.529),
        (0.500, 0.553, 0.992, 0.235), (0.625, 0.867, 0.853, 0.188),
        (0.750, 0.992, 0.573, 0.137), (0.875, 0.922, 0.263, 0.047),
        (1.000, 0.478, 0.016, 0.012),
    ],
    # Hypsometric: water, lowland green, upland tan, snow.
    "terrain": [
        (0.00, 0.200, 0.200, 0.600), (0.15, 0.000, 0.600, 1.000),
        (0.25, 0.000, 0.800, 0.400), (0.50, 1.000, 1.000, 0.600),
        (0.75, 0.500, 0.360, 0.330), (1.00, 1.000, 1.000, 1.000),
    ],
    # Diverging, white at the midpoint - for signed error where zero must read neutral.
    "coolwarm": [
        (0.00, 0.230, 0.299, 0.754), (0.50, 0.865, 0.865, 0.865),
        (1.00, 0.706, 0.016, 0.150),
    ],
}


def apply(values: np.ndarray, name: str = "turbo") -> np.ndarray:
    """
    Map an array already normalised to [0, 1] onto a colour ramp.

    Returns uint8 RGB of shape (..., 3). Non-finite inputs are clamped to 0.
    """
    ramp = _RAMPS.get(name)
    if ramp is None:
        raise KeyError(f"Unknown colormap '{name}'. Available: {sorted(_RAMPS)}")

    v = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float32)), 0.0, 1.0)
    stops = np.array([r[0] for r in ramp], dtype=np.float32)
    cols = np.array([r[1:] for r in ramp], dtype=np.float32)

    # np.interp per channel is both simpler and faster here than hand-rolled
    # segment search, and matches matplotlib's piecewise-linear behaviour.
    out = np.empty(v.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, stops, cols[:, c])
    return (out * 255.0).round().astype(np.uint8)
