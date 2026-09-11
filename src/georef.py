"""
georef.py - Georeferencing, CRS verification, and SRTM alignment for DepthWizard.

WHY THIS MODULE EXISTS:
1. Routing & Multi-modal Input:
   ISRO PS 26175 requires handling both non-georeferenced images (JPG/PNG -> relative rDSM)
   and georeferenced GeoTIFFs (CRS embedded -> absolute metric DSM). This module provides
   is_georeferenced() as the system router.
2. CRS Validation & Loud Failures:
   A GeoTIFF without a CRS cannot be mapped to physical Earth coordinates. Rather than
   assuming a default CRS or failing silently downstream, read_geotiff() raises an explicit,
   actionable exception.
3. Reference DEM Alignment:
   SRTM provides global bare-earth elevations at 30m resolution (SRTMGL1). To calibrate
   sub-meter or high-resolution monocular depth against SRTM, the reference raster must be
   precisely reprojected onto the identical spatial grid, resolution, and coordinate system
   of the input image using bilinear resampling.
4. Metric vs. Geographic Precision:
   Pixel coordinates in degrees (EPSG:4326) vary in ground distance by latitude. Ground
   metrics must always be computed in projected metric units (e.g. UTM).
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pyproj
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
import requests


def is_georeferenced(path: str | Path) -> bool:
    """
    Determine whether a raster file contains valid georeferencing metadata.

    Returns:
        True if the file can be opened by rasterio and possesses both a valid CRS
        and non-identity geotransform; False otherwise.
    """
    try:
        with rasterio.open(path) as src:
            has_crs = src.crs is not None and bool(src.crs)
            has_transform = src.transform is not None and not src.transform.is_identity
            return bool(has_crs and has_transform)
    except Exception:
        return False


def read_geotiff(
    path: str | Path
) -> Tuple[np.ndarray, CRS, rasterio.Affine, rasterio.coords.BoundingBox, np.ndarray]:
    """
    Read a GeoTIFF and return its array, CRS, affine transform, spatial bounds, and nodata mask.

    Raises:
        ValueError: If the file lacks valid CRS or spatial transform metadata.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")

    with rasterio.open(file_path) as src:
        if src.crs is None or not bool(src.crs):
            raise ValueError(
                f"Input file '{file_path.name}' lacks valid Coordinate Reference System (CRS) metadata. "
                f"Cannot perform georeferenced calibration without a spatial reference."
            )
        if src.transform is None or src.transform.is_identity:
            raise ValueError(
                f"Input file '{file_path.name}' lacks spatial affine geotransform metadata."
            )

        data = src.read()
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        nodata = src.nodata

        # Construct boolean nodata mask (True where invalid/nodata)
        if nodata is not None:
            if np.isnan(nodata):
                nodata_mask = np.isnan(data).any(axis=0)
            else:
                nodata_mask = (data == nodata).any(axis=0)
        else:
            nodata_mask = np.zeros((data.shape[1], data.shape[2]), dtype=bool)

    return data, crs, transform, bounds, nodata_mask


# Global DEMs served by OpenTopography, with their practical trade-offs.
#
# SRTM is the historical default but it was flown in 2000 by C-band radar and has
# well-known VOIDS over steep Himalayan terrain and persistent snow - precisely the
# Indian topography this project most needs a reference for. Copernicus DEM 30m is
# derived from TanDEM-X, is void-filled, and is generally the better choice for
# Indian scenes. NASADEM is a reprocessed, partially void-filled SRTM.
DEM_SOURCES = {
    "COP30":    {"demtype": "COP30",    "desc": "Copernicus DEM 30m (TanDEM-X, void-filled; best for Himalayan terrain)"},
    "SRTMGL1":  {"demtype": "SRTMGL1",  "desc": "SRTM 30m GL1 (2000 C-band; voids over steep/snow terrain)"},
    "NASADEM":  {"demtype": "NASADEM",  "desc": "NASADEM 30m (reprocessed, partially void-filled SRTM)"},
    "AW3D30":   {"demtype": "AW3D30",   "desc": "ALOS World 3D 30m (JAXA optical stereo)"},
}
DEFAULT_DEM = os.environ.get("DW_DEM_TYPE", "COP30").strip().upper()


def fetch_srtm(
    bounds: rasterio.coords.BoundingBox,
    crs: CRS,
    out_dir: str | Path = "data/srtm",
    dem_type: Optional[str] = None,
) -> Path:
    """
    Fetch the 30m SRTM tile (SRTMGL1) covering the specified bounding box via OpenTopography API.
    Caches downloaded tiles in out_dir and reuses them across runs.

    Args:
        bounds: BoundingBox (left, bottom, right, top) in the raster's native CRS.
        crs: Coordinate Reference System of the bounding box.
        out_dir: Local directory for tile caching.

    Returns:
        Path to the cached SRTM GeoTIFF.

    Raises:
        RuntimeError: If OPENTOPO_API_KEY is not set or if download fails.
    """
    dem_type = (dem_type or DEFAULT_DEM).upper()
    if dem_type not in DEM_SOURCES:
        print(f"[WARNING] Unknown DEM type '{dem_type}', falling back to SRTMGL1.", file=sys.stderr)
        dem_type = "SRTMGL1"

    api_key = os.environ.get("OPENTOPO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENTOPO_API_KEY environment variable.\n"
            "OpenTopography requires a free API key to download 30m SRTM elevation data.\n"
            "Steps to resolve:\n"
            "  1. Register for free at: https://portal.opentopography.org/newUser\n"
            "  2. Generate your key under 'myOpenTopo' -> 'Authorizations'.\n"
            "  3. Set the key in your terminal:\n"
            "     PowerShell: $env:OPENTOPO_API_KEY = 'your_api_key_here'\n"
            "     Bash      : export OPENTOPO_API_KEY='your_api_key_here'"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # OpenTopography requires geographic coordinates (EPSG:4326 WGS84)
    if crs.to_epsg() != 4326:
        # Transform bounds to EPSG:4326
        wgs84_bounds = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
        west, south, east, north = wgs84_bounds
    else:
        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    # Add a 0.01 deg (~1 km) buffer to prevent edge nodata during resampling
    buffer = 0.01
    west -= buffer
    south -= buffer
    east += buffer
    north += buffer

    # Create deterministic cache filename based on spatial coordinates
    coord_str = f"{west:.4f}_{south:.4f}_{east:.4f}_{north:.4f}"
    cache_hash = hashlib.md5((dem_type + coord_str).encode("utf-8")).hexdigest()[:8]
    cached_file = out_path / f"{dem_type.lower()}_{coord_str}_{cache_hash}.tif"

    if cached_file.exists() and cached_file.stat().st_size > 1024:
        print(f"[INFO] Using cached SRTM reference tile: {cached_file.name}")
        return cached_file

    print(f"[INFO] Downloading {dem_type} ({DEM_SOURCES[dem_type]['desc']}) "
          f"for extent: [{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}]...")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": DEM_SOURCES[dem_type]["demtype"],
        "south": f"{south:.6f}",
        "north": f"{north:.6f}",
        "west": f"{west:.6f}",
        "east": f"{east:.6f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    try:
        response = requests.get(url, params=params, stream=True, timeout=60)
        if response.status_code == 401 or response.status_code == 403:
            raise RuntimeError(
                f"OpenTopography authentication failed (HTTP {response.status_code}). "
                f"Please verify that OPENTOPO_API_KEY is valid."
            )
        response.raise_for_status()

        # Write to temporary file first to avoid corrupted cache on interruption
        tmp_file = cached_file.with_suffix(".tmp")
        with open(tmp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)

        # Verify integrity with rasterio
        with rasterio.open(tmp_file) as test_src:
            if test_src.width < 2 or test_src.height < 2:
                raise ValueError("Downloaded SRTM raster has invalid dimensions.")

        tmp_file.replace(cached_file)
        print(f"[INFO] Successfully cached SRTM tile to: {cached_file.name} ({cached_file.stat().st_size // 1024} KB)")
        return cached_file

    except Exception as e:
        if 'tmp_file' in locals() and tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError(f"Failed to fetch SRTM elevation data from OpenTopography: {e}")


def align_to_reference(
    dst_shape: Tuple[int, int],
    dst_transform: rasterio.Affine,
    dst_crs: CRS,
    srtm_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample the SRTM raster onto the exact grid, resolution, and CRS of the target image.

    Args:
        dst_shape: Target image shape (height, width).
        dst_transform: Target image affine geotransform.
        dst_crs: Target image CRS.
        srtm_path: Path to the SRTM GeoTIFF.

    Returns:
        Tuple of:
            - aligned_dem: float32 array of shape (height, width) with aligned SRTM elevation in metres.
            - valid_mask: boolean array (True where SRTM elevation is valid and non-void).
    """
    dst_height, dst_width = dst_shape
    aligned_dem = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    with rasterio.open(srtm_path) as srtm_src:
        reproject(
            source=rasterio.band(srtm_src, 1),
            destination=aligned_dem,
            src_transform=srtm_src.transform,
            src_crs=srtm_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )

        nodata_val = srtm_src.nodata

    # Construct validity mask: finite, not nodata, and above Earth's lowest dry land (-430m Dead Sea)
    valid_mask = np.isfinite(aligned_dem)
    if nodata_val is not None:
        if np.isnan(nodata_val):
            valid_mask &= ~np.isnan(aligned_dem)
        else:
            valid_mask &= (aligned_dem != nodata_val)
    valid_mask &= (aligned_dem > -450.0)

    return aligned_dem, valid_mask


def to_metric_resolution(transform: rasterio.Affine, crs: CRS) -> float:
    """
    Compute ground sample distance (GSD) in metres per pixel.

    WHY:
    If CRS is geographic (EPSG:4326), pixel size is in degrees. 1 degree latitude ~ 111,320m,
    while longitude varies by cos(latitude). This helper resolves true ground scale in metres.
    """
    if crs.is_projected:
        # Affine scale elements (a, e) are already in projected metric units
        res_x = abs(transform.a)
        res_y = abs(transform.e)
        return float((res_x + res_y) / 2.0)
    else:
        # Geographic coordinates in degrees: compute approximate metric scale at center
        center_y = transform.yoff
        deg_lat_m = 111320.0
        deg_lon_m = 111320.0 * np.cos(np.radians(center_y))
        res_x = abs(transform.a) * deg_lon_m
        res_y = abs(transform.e) * deg_lat_m
        return float((res_x + res_y) / 2.0)
