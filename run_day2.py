"""
run_day2.py - CLI for Stage 2 Georeferencing and Metric Elevation Calibration.

USAGE:
    python run_day2.py <geotiff_path> [--model small|base|large] [--method ransac|linear] [--outdir data/output/]

FULL WORKFLOW:
    1. Check georeferencing metadata (CRS & Affine transform).
       -> If unreferenced, gracefully fall back to Stage 1 relative pipeline.
    2. Extract relative disparity / height using Stage 1 DepthEstimator.
    3. Query and fetch SRTM 30m reference DEM tile (SRTMGL1) from OpenTopography.
    4. Resample and align SRTM grid onto exact input image coordinates.
    5. Calibrate via Stratified RANSAC regression (rejecting structural outliers).
    6. Apply calibration to generate absolute metric Digital Surface Model (metres).
    7. Compute accuracy metrics (RMSE, MAE, Pearson r) and spatial error map.
    8. Write output products:
       - {stem}_dsm.tif        (GeoTIFF preserving native CRS and transform)
       - {stem}_dsm.npy        (Raw float32 elevation array in metres)
       - {stem}_error_map.png  (Diverging signed error map against SRTM)
       - {stem}_validation.json (Metrics, regression parameters, inlier ratio)
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import rasterio

from run_day1 import process_single_image as fallback_day1_process
from src.calibrate import apply_calibration, calibrate_depth
from src.depth_extract import DepthEstimator
from src.georef import align_to_reference, fetch_srtm, is_georeferenced, read_geotiff
from src.validate import compute_metrics, error_map, format_validation_summary, save_error_map, save_validation_report


def run_day2_pipeline(
    input_path: Path,
    out_dir: Path,
    model_size: str = "small",
    method: str = "ransac",
    device: str = "auto",
) -> bool:
    """Execute complete Stage 2 workflow on an input image."""
    stem = input_path.stem
    print(f"\n{'='*65}")
    print(f"DepthWizard Stage 2: Georeferenced Metric Calibration (Day 2)")
    print(f"Input File        : {input_path.name}")
    print(f"Calibration Method: {method.upper()}")
    print(f"Output Directory  : {out_dir}")
    print(f"{'='*65}")

    start_time = time.perf_counter()

    # 1. Routing Check: Georeferenced vs Non-georeferenced
    if not is_georeferenced(input_path):
        print(
            f"[NOTICE] File '{input_path.name}' contains no valid Coordinate Reference System (CRS).\n"
            f"[INFO] Gracefully routing down Stage 1 Relative Pipeline (unitless rDSM).",
            file=sys.stderr,
        )
        estimator = DepthEstimator(model_size=model_size, device=device)
        return fallback_day1_process(estimator, input_path, out_dir)

    # 2. Read GeoTIFF Metadata
    print(f"[1/7] Reading GeoTIFF metadata and raster bands...")
    try:
        data, crs, transform, bounds, nodata_mask = read_geotiff(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to read GeoTIFF: {e}", file=sys.stderr)
        return False

    # Extract RGB bands for depth model (if multi-band, take first 3 bands)
    if data.shape[0] >= 3:
        rgb_data = data[:3, :, :].transpose(1, 2, 0)
    elif data.shape[0] == 1:
        rgb_data = np.stack([data[0, :, :]] * 3, axis=-1)
    else:
        rgb_data = data[:3, :, :].transpose(1, 2, 0)

    # Normalize uint16/float image to uint8 for inference
    if rgb_data.dtype != np.uint8:
        rgb_min, rgb_max = np.percentile(rgb_data, [1.0, 99.0])
        rgb_uint8 = np.clip((rgb_data - rgb_min) / max(1e-5, rgb_max - rgb_min) * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb_data

    height, width = rgb_uint8.shape[:2]
    print(f"      Dimensions: {width}x{height} pixels | CRS: {crs.to_string()}")

    # 3. Stage 1 Relative Depth Extraction
    print(f"[2/7] Running monocular relative depth extraction (Depth Anything V2 {model_size})...")
    estimator = DepthEstimator(model_size=model_size, device=device)
    depth_res = estimator.predict(rgb_uint8, robust=True)
    rel_depth = depth_res.normalized_depth

    # 4. Fetch SRTM 30m Reference DEM
    print(f"[3/7] Sourcing SRTM 30m reference DEM tile...")
    try:
        srtm_path = fetch_srtm(bounds=bounds, crs=crs, out_dir="data/srtm")
    except Exception as e:
        print(f"[ERROR] Could not obtain SRTM reference data: {e}", file=sys.stderr)
        print(f"[INFO] Falling back to Stage 1 relative product save.", file=sys.stderr)
        return fallback_day1_process(estimator, input_path, out_dir)

    # 5. Resample and Align SRTM to Image Grid
    print(f"[4/7] Resampling & aligning SRTM onto image grid ({width}x{height})...")
    ref_dem, srtm_valid = align_to_reference(
        dst_shape=(height, width),
        dst_transform=transform,
        dst_crs=crs,
        srtm_path=srtm_path,
    )

    combined_mask = srtm_valid & (~nodata_mask) & np.isfinite(rel_depth)
    valid_count = int(np.sum(combined_mask))
    if valid_count < 100:
        print(f"[ERROR] Too few valid overlapping pixels ({valid_count}) between input and SRTM.", file=sys.stderr)
        return False
    print(f"      Aligned valid coverage: {valid_count:,} pixels ({valid_count / (height * width):.1%})")

    # 6. Fit Calibration Regression (RANSAC or Linear)
    print(f"[5/7] Calibrating relative depth to metric elevation via {method.upper()}...")
    try:
        calib_res = calibrate_depth(
            rel_depth=rel_depth,
            ref_dem=ref_dem,
            mask=combined_mask,
            method=method,
            n_samples=2000,
        )
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}", file=sys.stderr)
        return False

    # Apply calibration to produce absolute DSM
    print(f"[6/7] Applying calibration: Elevation = {calib_res.slope:.2f} * rDSM + {calib_res.intercept:.2f} m")
    pred_dsm = apply_calibration(rel_depth, calib_res.slope, calib_res.intercept)

    # 7. Compute Accuracy Metrics & Error Map
    print(f"[7/7] Computing validation metrics and rendering error map...")
    metrics = compute_metrics(pred_dsm, ref_dem, combined_mask)
    err_map = error_map(pred_dsm, ref_dem, combined_mask)

    summary_text, summary_dict = format_validation_summary(metrics, calib_res.__dict__)
    print(f"\n{summary_text}\n")

    # 8. Export Output Products
    dsm_tif_path = out_dir / f"{stem}_dsm.tif"
    dsm_npy_path = out_dir / f"{stem}_dsm.npy"
    err_png_path = out_dir / f"{stem}_error_map.png"
    val_json_path = out_dir / f"{stem}_validation.json"

    # Save GeoTIFF preserving CRS and transform
    with rasterio.open(
        dsm_tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(pred_dsm.astype(rasterio.float32), 1)

    # Save raw float numpy array
    np.save(dsm_npy_path, pred_dsm.astype(np.float32))

    # Save signed error map
    save_error_map(err_map, err_png_path, clip_range_m=25.0)

    # Save validation report
    save_validation_report(metrics, calib_res.__dict__, val_json_path)

    total_time = time.perf_counter() - start_time
    print(f"[SUCCESS] Stage 2 Pipeline Complete ({total_time:.2f}s total)")
    print(f"          Metric DSM GeoTIFF : {dsm_tif_path.name}")
    print(f"          Raw Elevation Array: {dsm_npy_path.name}")
    print(f"          Signed Error Map   : {err_png_path.name}")
    print(f"          Validation Report  : {val_json_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="DepthWizard Stage 2 - Georeferenced Metric Elevation Calibration (Day 2)"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an input GeoTIFF file (or image for auto-fallback)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Depth Anything V2 backbone size (default: small)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ransac", "linear"],
        default="ransac",
        help="Calibration regression algorithm (default: ransac)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cpu, cuda (default: auto)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/output",
        help="Destination directory for output products (default: data/output)",
    )

    args = parser.parse_args()
    input_path = Path(args.input_path)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    success = run_day2_pipeline(
        input_path=input_path,
        out_dir=out_dir,
        model_size=args.model,
        method=args.method,
        device=args.device,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
