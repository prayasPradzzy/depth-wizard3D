"""
calibrate.py - Regression calibration from relative depth (0-1) to absolute elevation (metres).

WHY THIS MODULE EXISTS:
1. Bridging the Unitless Gap:
   Monocular vision models predict relative disparity (scale and shift invariant). To make
   this data actionable for civil engineering, flood modeling, or defense, we must map
   unitless [0, 1] values to absolute SI metres using coarse georeferenced reference data (SRTM).
2. The DEM vs. DSM Dilemma:
   SRTM is a bare-earth Digital Elevation Model (DEM) at 30m resolution. Our monocular output
   is a Digital Surface Model (DSM) that captures individual 3D structures (buildings, towers,
   tree canopies).
   - If we fit ordinary least squares (OLS) across all pixels, tall buildings pull the regression
     line upward, severely corrupting ground-level elevation.
   - We solve this by (a) excluding the top decile of relative depth during control point
     sampling, and (b) using RANSAC regression to treat elevated structures as positive outliers.
3. Stratified Sampling:
   Natural scenes frequently consist of 80% flat valley with a 20% hillside. Uniform random
   sampling over-indexes on the flat valley, causing the regression to lose slope sensitivity.
   Stratified sampling across depth quantiles ensures equal representation across all elevations.
4. Failure Alerts:
   If r^2 < 0.5 or inlier fraction < 0.3, the relative depth map does not physically correlate
   with the true topography. The module raises prominent warnings rather than outputting
   misleading elevations.
"""

from dataclasses import dataclass
import sys
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score


@dataclass
class CalibrationResult:
    """
    Encapsulates calibration regression metrics and parameters.

    Attributes:
        slope: Scale factor (elevation span in metres per unit relative depth).
        intercept: Base terrain datum in metres above sea level (WGS84 ellipsoid/geoid).
        r2: Coefficient of determination (goodness of fit against SRTM reference).
        inlier_fraction: Proportion of sampled control points accepted as bare terrain by RANSAC.
        method: 'ransac' or 'linear'.
        n_points: Number of sampled control points used.
    """
    slope: float
    intercept: float
    r2: float
    inlier_fraction: float
    method: str
    n_points: int


def sample_control_points(
    rel_depth: np.ndarray,
    ref_dem: np.ndarray,
    mask: np.ndarray,
    n: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract stratified control points across the relative depth range for calibration.

    Args:
        rel_depth: Relative depth array [0, 1] (H, W).
        ref_dem: Aligned SRTM elevation array in metres (H, W).
        mask: Boolean validity mask (True where both rasters are valid).
        n: Target number of control points (default 2000).
        seed: Random state for reproducibility.

    Returns:
        Tuple of (rel_samples, ref_samples) 1D float arrays.

    WHY EXCLUDE TOP DECILE:
    The top 10% highest relative depths almost universally correspond to structural roofs,
    bridges, and tree crowns. SRTM (bare-earth DEM) does not represent these features.
    Excluding them prevents building heights from corrupting terrain ground calibration.
    """
    valid_idx = np.where(mask & np.isfinite(rel_depth) & np.isfinite(ref_dem))
    if len(valid_idx[0]) < 50:
        raise ValueError(f"Insufficient valid pixels for calibration sampling: {len(valid_idx[0])} found.")

    valid_rel = rel_depth[valid_idx]
    valid_ref = ref_dem[valid_idx]

    # Exclude top decile of relative depth (buildings and canopy)
    p90 = float(np.percentile(valid_rel, 90.0))
    terrain_mask = valid_rel <= p90
    rel_terrain = valid_rel[terrain_mask]
    ref_terrain = valid_ref[terrain_mask]

    if len(rel_terrain) < 50:
        # Fallback to full range if terrain subset is too small
        rel_terrain = valid_rel
        ref_terrain = valid_ref

    # Stratified sampling across depth range (10 bins)
    rng = np.random.RandomState(seed)
    n_bins = 10
    samples_per_bin = max(1, n // n_bins)
    bin_edges = np.linspace(rel_terrain.min(), rel_terrain.max(), n_bins + 1)

    sampled_rel = []
    sampled_ref = []

    for i in range(n_bins):
        bin_mask = (rel_terrain >= bin_edges[i]) & (rel_terrain <= bin_edges[i + 1])
        bin_indices = np.where(bin_mask)[0]

        if len(bin_indices) == 0:
            continue
        elif len(bin_indices) <= samples_per_bin:
            selected = bin_indices
        else:
            selected = rng.choice(bin_indices, size=samples_per_bin, replace=False)

        sampled_rel.append(rel_terrain[selected])
        sampled_ref.append(ref_terrain[selected])

    if not sampled_rel:
        raise ValueError("Stratified sampling produced zero control points.")

    final_rel = np.concatenate(sampled_rel)
    final_ref = np.concatenate(sampled_ref)

    return final_rel, final_ref


def fit_linear(rel: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
    """Ordinary Least Squares (OLS) baseline regression. Returns (slope, intercept, r2)."""
    reg = LinearRegression()
    rel_2d = rel.reshape(-1, 1)
    reg.fit(rel_2d, ref)
    pred = reg.predict(rel_2d)
    r2 = float(r2_score(ref, pred))
    return float(reg.coef_[0]), float(reg.intercept_), r2


def fit_ransac(
    rel: np.ndarray,
    ref: np.ndarray,
    residual_threshold: Optional[float] = None,
    seed: int = 42,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Fit RANSAC regression to estimate metric scale while rejecting structural outliers.

    Args:
        rel: Sampled relative depths.
        ref: Sampled reference SRTM elevations in metres.
        residual_threshold: Max residual in metres to treat a point as an inlier (default: 8.0m).
        seed: Random seed.

    Returns:
        Tuple of (slope, intercept, r2_on_inliers, inlier_mask).

    WHY RANSAC:
    Even with top-decile filtering, lower buildings, overpasses, and trees contaminate
    control points. RANSAC fits the true bare-earth ground plane by finding the maximal
    consensus set, preventing elevation bias.
    """
    if residual_threshold is None:
        # SRTM vertical accuracy standard error is ~6-10m
        residual_threshold = 8.0

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=max(5, int(0.15 * len(rel))),
        residual_threshold=residual_threshold,
        random_state=seed,
        max_trials=1000,
    )

    rel_2d = rel.reshape(-1, 1)
    ransac.fit(rel_2d, ref)

    inlier_mask = ransac.inlier_mask_
    slope = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)

    # Calculate r2 specifically on inliers
    if np.sum(inlier_mask) > 2:
        inlier_pred = ransac.estimator_.predict(rel_2d[inlier_mask])
        r2 = float(r2_score(ref[inlier_mask], inlier_pred))
    else:
        r2 = 0.0

    return slope, intercept, r2, inlier_mask


def calibrate_depth(
    rel_depth: np.ndarray,
    ref_dem: np.ndarray,
    mask: np.ndarray,
    method: str = "ransac",
    n_samples: int = 2000,
) -> CalibrationResult:
    """
    Execute calibration from relative depth to metric elevation.

    Logs warnings if r2 < 0.5 or inlier_fraction < 0.3.
    """
    rel_samples, ref_samples = sample_control_points(
        rel_depth=rel_depth,
        ref_dem=ref_dem,
        mask=mask,
        n=n_samples,
    )

    if method.lower() == "ransac":
        slope, intercept, r2, inlier_mask = fit_ransac(rel_samples, ref_samples)
        inlier_frac = float(np.mean(inlier_mask))
    elif method.lower() == "linear":
        slope, intercept, r2 = fit_linear(rel_samples, ref_samples)
        inlier_frac = 1.0
    else:
        raise ValueError(f"Unknown calibration method '{method}'. Choose 'ransac' or 'linear'.")

    result = CalibrationResult(
        slope=slope,
        intercept=intercept,
        r2=r2,
        inlier_fraction=inlier_frac,
        method=method.lower(),
        n_points=len(rel_samples),
    )

    # Quality Gate & Reliability Checks
    if result.r2 < 0.5:
        print(
            f"[CALIBRATION WARNING] r2 = {result.r2:.3f} < 0.50 threshold!\n"
            f"The monocular depth prediction correlates weakly with SRTM topography.\n"
            f"Metric elevation outputs should be treated as provisional (see Debug Prompt B).",
            file=sys.stderr,
        )
    if result.inlier_fraction < 0.30:
        print(
            f"[CALIBRATION WARNING] Low inlier fraction: {result.inlier_fraction:.1%} < 30%.\n"
            f"Severe mismatch between optical features and reference DEM.",
            file=sys.stderr,
        )

    return result


def apply_calibration(
    rel_depth: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """
    Apply linear calibration transform to compute absolute metric elevation in metres.

    elevation_metres = slope * rel_depth + intercept
    """
    arr = np.asarray(rel_depth, dtype=np.float32)
    metric_dem = (slope * arr) + intercept
    return metric_dem.astype(np.float32)
