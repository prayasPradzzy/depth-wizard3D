"""
main.py - FastAPI Application Server for DepthWizard (ISRO PS 26175).

WHY THIS MODULE EXISTS:
1. Zero Installation for Judges:
   Judges and evaluators should not have to install complex 3D rendering engines or Unity.
   A single command (python -m api.main or uvicorn) serves both the AI API endpoints and
   the Three.js browser viewer on a single port.
2. Synchronous & Direct Pipeline Routing:
   Auto-detects whether an uploaded file is a georeferenced GeoTIFF (routing to Stage 2 metric
   calibration) or a standard RGB image (routing to Stage 1 rDSM), then formats the assets
   for WebGL rendering via src/mesh_export.py.
3. Clean Asset Serving & State Management:
   Outputs are segregated per job_id with a fast 'latest' alias, allowing instant browser
   refresh and progressive loading of 16-bit heightmaps and textures.
"""

import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import rasterio

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_extract import DepthEstimator
from src.enhance import detect_water_mask, enhance_shadows, flatten_water
from src.georef import align_to_reference, fetch_srtm, is_georeferenced, read_geotiff, to_metric_resolution
from src.calibrate import apply_calibration, calibrate_depth
from src.mesh_export import export_for_web
from src.validate import compute_metrics, error_map, save_error_map
from src.utils import depth_stats


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Pipeline tuning via environment (see .env.example).
MESH_DIM = int(os.environ.get("DW_MESH_DIM", "1024"))   # web mesh grid + texture edge
MODEL_SIZE = os.environ.get("DW_MODEL_SIZE", "small").strip().lower()
TILED_MODE = os.environ.get("DW_TILED", "auto").strip().lower()          # auto | on | off
ENHANCE_SHADOWS = _env_flag("DW_ENHANCE_SHADOWS", False)
FLATTEN_WATER = _env_flag("DW_FLATTEN_WATER", False)
PREWARM = _env_flag("DW_PREWARM", True)   # load model at startup for fast first upload

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the initial demo terrain job and pre-warm the model on startup."""
    try:
        create_initial_demo_job()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARNING] Could not create initial demo job: {e}", file=sys.stderr)

    if PREWARM:
        # Load model weights + run one tiny inference now, so the first real upload
        # does not pay ~10 s of lazy initialisation. Matters for a live demo.
        try:
            t0 = time.perf_counter()
            get_estimator().predict(
                np.zeros((64, 64, 3), dtype=np.uint8), robust=True, tiled=False
            )
            print(f"[INFO] Model pre-warmed in {time.perf_counter() - t0:.1f}s "
                  f"(size={MODEL_SIZE}, tiled={TILED_MODE})")
        except Exception as e:
            print(f"[WARNING] Pre-warm failed (first upload will be slower): {e}", file=sys.stderr)
    yield


app = FastAPI(
    title="DepthWizard API",
    description="Single-View Height Estimation & 3D Flythrough (ISRO PS 26175)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: wildcard origin is incompatible with credentials, and the viewer is served
# same-origin anyway. Allow all origins WITHOUT credentials for easy local embedding.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = PROJECT_ROOT / "data" / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Keep only the N most recent job folders (plus any job_demo / job_nyc keepsakes).
MAX_RETAINED_JOBS = int(os.environ.get("DW_MAX_JOBS", "25"))
_KEEP_JOBS = {"job_demo", "job_nyc"}


def prune_old_jobs() -> None:
    """Delete stale job directories so data/web_jobs does not grow without bound."""
    try:
        jobs = [
            p for p in JOBS_DIR.iterdir()
            if p.is_dir() and p.name not in _KEEP_JOBS and (p / "manifest.json").exists()
        ]
        jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in jobs[MAX_RETAINED_JOBS:]:
            shutil.rmtree(stale, ignore_errors=True)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARNING] Job pruning failed: {e}", file=sys.stderr)

# Shared depth estimator instance
_estimator = None


def get_estimator() -> DepthEstimator:
    global _estimator
    if _estimator is None:
        size = MODEL_SIZE if MODEL_SIZE in {"small", "base", "large"} else "small"
        _estimator = DepthEstimator(model_size=size, device="auto")
    return _estimator


_TILED_ARG = {"on": True, "true": True, "off": False, "false": False}.get(TILED_MODE, "auto")


def get_job_directory(job_id: str) -> Path:
    """Resolve job directory, supporting 'latest' alias."""
    if job_id == "latest":
        # Find newest job folder
        jobs = [p for p in JOBS_DIR.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        if not jobs:
            raise HTTPException(status_code=404, detail="No processed jobs available yet.")
        jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return jobs[0]

    job_dir = JOBS_DIR / job_id
    if not job_dir.exists() or not (job_dir / "manifest.json").exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job_dir


@app.post("/api/process")
async def process_image_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accept an uploaded aerial/satellite image, process it through the ML/geospatial pipeline,
    and generate web-ready 3D mesh assets.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload.")

    suffix = Path(file.filename).suffix.lower()
    allowed_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported formats: GeoTIFF (.tif, .tiff) or optical RGB (.png, .jpg, .webp).",
        )

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / file.filename
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if input_path.stat().st_size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    print(f"[API] Received upload: {file.filename} ({input_path.stat().st_size} bytes) -> {job_id}")

    def _run() -> Dict[str, Any]:
        estimator = get_estimator()

        # Check if file is georeferenced
        if is_georeferenced(input_path):
            print(f"[API] GeoTIFF detected. Executing Stage 2 Metric DSM Pipeline...")
            data, crs, transform, bounds, nodata_mask = read_geotiff(input_path)

            if data.shape[0] >= 3:
                rgb_data = data[:3, :, :].transpose(1, 2, 0)
            elif data.shape[0] == 1:
                rgb_data = np.stack([data[0, :, :]] * 3, axis=-1)
            else:
                rgb_data = data[:3, :, :].transpose(1, 2, 0)

            if rgb_data.dtype != np.uint8:
                p1, p99 = np.percentile(rgb_data, [1.0, 99.0])
                rgb_uint8 = np.clip((rgb_data - p1) / max(1e-5, p99 - p1) * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_uint8 = rgb_data

            height, width = rgb_uint8.shape[:2]

            infer_rgb = rgb_uint8
            if ENHANCE_SHADOWS:
                print("[API] Applying CLAHE shadow enhancement before inference...")
                infer_rgb = np.asarray(enhance_shadows(rgb_uint8), dtype=np.uint8)

            depth_res = estimator.predict(infer_rgb, robust=True, tiled=_TILED_ARG)
            rel_depth = depth_res.normalized_depth

            if FLATTEN_WATER:
                water = detect_water_mask(rgb_uint8)
                if water.any():
                    print(f"[API] Flattening {int(water.sum()):,} water pixels ({water.mean():.1%}).")
                    rel_depth = flatten_water(rel_depth, water)

            # Attempt SRTM fetch & calibration if key is present
            has_calibrated = False
            pred_dsm = rel_depth
            val_metrics = None
            if "OPENTOPO_API_KEY" in os.environ:
                try:
                    srtm_path = fetch_srtm(bounds=bounds, crs=crs, out_dir="data/srtm")
                    ref_dem, srtm_valid = align_to_reference((height, width), transform, crs, srtm_path)
                    comb_mask = srtm_valid & (~nodata_mask) & np.isfinite(rel_depth)
                    calib_res = calibrate_depth(rel_depth, ref_dem, comb_mask, method="ransac")
                    pred_dsm = apply_calibration(rel_depth, calib_res.slope, calib_res.intercept)
                    has_calibrated = True

                    # Generate validation metrics & spatial error map
                    val_metrics = compute_metrics(pred_dsm, ref_dem, comb_mask)
                    val_metrics["inlier_fraction"] = round(calib_res.inlier_fraction, 3)
                    val_metrics["slope"] = round(calib_res.slope, 2)
                    val_metrics["intercept"] = round(calib_res.intercept, 2)
                    val_metrics["r2"] = round(calib_res.r2, 4)

                    err_arr = error_map(pred_dsm, ref_dem, comb_mask)
                    save_error_map(err_arr, job_dir / "error_map.png")
                except Exception as e:
                    print(f"[API WARNING] SRTM alignment/calibration skipped: {e}")

            # Export the DSM as a real GeoTIFF so the download panel serves an actual
            # geospatial product, not a screenshot. CRS + affine transform preserved.
            try:
                with rasterio.open(
                    job_dir / "metric_dsm.tif", "w", driver="GTiff",
                    height=height, width=width, count=1, dtype=rasterio.float32,
                    crs=crs, transform=transform, nodata=-9999.0, compress="deflate",
                ) as dst:
                    dst.write(pred_dsm.astype(np.float32), 1)
                with rasterio.open(
                    job_dir / "relative_dsm.tif", "w", driver="GTiff",
                    height=height, width=width, count=1, dtype=rasterio.float32,
                    crs=crs, transform=transform, nodata=-9999.0, compress="deflate",
                ) as dst:
                    dst.write(rel_depth.astype(np.float32), 1)
            except Exception as e:
                print(f"[API WARNING] GeoTIFF export failed: {e}", file=sys.stderr)

            gsd = to_metric_resolution(transform, crs)
            web_assets = export_for_web(
                elevation_arr=pred_dsm,
                rgb_input=rgb_uint8,
                out_dir=job_dir,
                max_dim=MESH_DIM,
                is_georeferenced=has_calibrated,
                crs_str=crs.to_string(),
                gsd_m=gsd,
            )

            manifest_file = job_dir / "manifest.json"
            with open(manifest_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            if val_metrics:
                m["validation"] = val_metrics
                m["error_map_file"] = "error_map.png"
            # WGS84 extent lets the viewer convert a cursor/camera position into
            # real latitude/longitude instead of unitless scene coordinates.
            try:
                from rasterio.warp import transform_bounds as _tb
                w_, s_, e_, n_ = _tb(crs, "EPSG:4326", *bounds)
                m["bounds_wgs84"] = [round(w_, 6), round(s_, 6), round(e_, 6), round(n_, 6)]
            except Exception:
                pass
            m["has_geotiff"] = (job_dir / "metric_dsm.tif").exists()
            m["has_crs"] = True
            m["metric_calibrated"] = bool(has_calibrated)
            if not has_calibrated:
                m["calibration_note"] = (
                    "CRS and ground sample distance are read from the file, but no SRTM "
                    "anchor was available, so heights remain relative. Set OPENTOPO_API_KEY "
                    "to enable absolute metric elevation."
                )
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)

        else:
            print(f"[API] Standard RGB image detected. Executing Stage 1 Relative Pipeline...")
            pil_img = Image.open(input_path).convert("RGB")

            infer_img = pil_img
            if ENHANCE_SHADOWS:
                print("[API] Applying CLAHE shadow enhancement before inference...")
                infer_img = enhance_shadows(pil_img)

            depth_res = estimator.predict(infer_img, robust=True, tiled=_TILED_ARG)
            rel_depth = depth_res.normalized_depth

            if FLATTEN_WATER:
                water = detect_water_mask(pil_img)
                if water.any():
                    print(f"[API] Flattening {int(water.sum()):,} water pixels ({water.mean():.1%}).")
                    rel_depth = flatten_water(rel_depth, water)

            stats = depth_stats(rel_depth)

            web_assets = export_for_web(
                elevation_arr=rel_depth,
                rgb_input=pil_img,
                out_dir=job_dir,
                max_dim=MESH_DIM,
                is_georeferenced=False,
            )

            # Attach relative stats to manifest
            manifest_file = job_dir / "manifest.json"
            with open(manifest_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["validation"] = {
                "type": "relative_rdsm",
                "std": stats["std"],
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"],
                "flat_warning": stats["flat_warning"],
            }
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)

        print(f"[API] Successfully generated web assets for job: {job_id}")
        return {"job_id": job_id, "status": "completed"}

    try:
        # Heavy ML + geospatial work runs in a worker thread so the event loop
        # stays free to serve other requests (health checks, asset downloads).
        result = await run_in_threadpool(_run)
        prune_old_jobs()
        return result
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        print(f"[API ERROR] Failed processing {file.filename}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/result/{job_id}")
async def get_job_result(job_id: str) -> Dict[str, Any]:
    """Retrieve metadata manifest for a specific job or 'latest'."""
    job_dir = get_job_directory(job_id)
    manifest_file = job_dir / "manifest.json"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    resolved_id = job_dir.name
    has_error_map = (job_dir / "error_map.png").exists()
    has_ao = (job_dir / "ao.png").exists()

    def _opt(name: str):
        return f"/api/assets/{job_dir.name}/{name}" if (job_dir / name).exists() else None
    has_sharp_bin = (job_dir / "heightmap_sharp.bin").exists()

    return {
        "job_id": resolved_id,
        "manifest": manifest,
        "assets": {
            "heightmap": f"/api/assets/{resolved_id}/heightmap.png",
            "heightmap_bin": f"/api/assets/{resolved_id}/heightmap.bin",
            "heightmap_sharp_bin": f"/api/assets/{resolved_id}/heightmap_sharp.bin" if has_sharp_bin else f"/api/assets/{resolved_id}/heightmap.bin",
            "texture": f"/api/assets/{resolved_id}/texture.png",
            "ao": f"/api/assets/{resolved_id}/ao.png" if has_ao else None,
            "ground_bin": _opt("ground.bin"),
            "depth_turbo": _opt("depth_turbo.png"),
            "elevation_metric": _opt("elevation_metric.png"),
            "metric_dsm_tif": _opt("metric_dsm.tif"),
            "relative_dsm_tif": _opt("relative_dsm.tif"),
            "error_map": f"/api/assets/{resolved_id}/error_map.png" if has_error_map else None,
        },
    }


@app.get("/api/jobs")
async def list_jobs() -> Dict[str, Any]:
    """
    Recent processed scenes, newest first.

    Only jobs that actually completed are listed - the viewer's "sample scenes"
    row is populated from this, so it can never advertise a scene the pipeline
    has not really produced.
    """
    out = []
    for p in sorted(JOBS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        mf = p / "manifest.json"
        if not (p.is_dir() and mf.exists()):
            continue
        try:
            with open(mf, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        src = next((c.name for c in p.iterdir()
                    if c.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                    and not c.name.startswith(("heightmap", "texture", "ao",
                                               "error_map", "depth_turbo",
                                               "elevation_metric"))), None)
        label = Path(src).stem if src else p.name.replace("job_", "")
        out.append({
            "job_id": p.name,
            "label": label[:18],
            "georeferenced": bool(m.get("is_georeferenced")),
            "size": f'{m.get("width")}x{m.get("height")}',
        })
        if len(out) >= 12:
            break
    return {"jobs": out}


@app.get("/api/benchmark")
async def get_benchmark() -> Dict[str, Any]:
    """Independent LiDAR benchmark results, produced by tools/eval_gamus.py."""
    bf = PROJECT_ROOT / "data" / "benchmark.json"
    if not bf.exists():
        raise HTTPException(status_code=404, detail="No benchmark results available.")
    with open(bf, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/assets/{job_id}/{filename}")
async def get_job_asset(job_id: str, filename: str):
    """Serve specific binary heightmap or texture asset."""
    job_dir = get_job_directory(job_id)
    file_path = job_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Asset '{filename}' not found for job '{job_id}'.")

    # Content type mapping
    content_type = "application/octet-stream"
    if filename.endswith(".png"):
        content_type = "image/png"
    elif filename.endswith(".json"):
        content_type = "application/json"
    elif filename.endswith(".bin"):
        content_type = "application/octet-stream"

    return FileResponse(file_path, media_type=content_type)


# Mount static web frontend to root
WEB_DIR = PROJECT_ROOT / "web"
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def create_initial_demo_job():
    """Create a default job from available data so the viewer opens with terrain immediately."""
    demo_job_dir = JOBS_DIR / "job_demo"
    if (demo_job_dir / "manifest.json").exists():
        return

    print("[INFO] Creating initial demo 3D terrain job...")
    demo_job_dir.mkdir(parents=True, exist_ok=True)

    # Use test1.png if available, else synthetic terrain
    test1_path = PROJECT_ROOT / "data" / "input" / "test1.png"
    if test1_path.exists():
        pil_img = Image.open(test1_path).convert("RGB")
        raw_npy = PROJECT_ROOT / "data" / "output" / "test1_raw.npy"
        if raw_npy.exists():
            elev_arr = np.load(raw_npy)
        else:
            estimator = get_estimator()
            res = estimator.predict(pil_img, robust=True)
            elev_arr = res.normalized_depth
        export_for_web(elev_arr, pil_img, demo_job_dir, max_dim=MESH_DIM, is_georeferenced=False)
    else:
        # Fallback synthetic
        from tests.test_stage1_synthetic import create_synthetic_terrain
        synth_elev = create_synthetic_terrain(256)
        synth_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        synth_rgb[:, :, 1] = np.clip(120 + synth_elev * 0.5, 0, 255).astype(np.uint8)
        synth_rgb[:, :, 0] = np.clip(80 + synth_elev * 0.3, 0, 255).astype(np.uint8)
        synth_rgb[120:165, 60:110] = [200, 180, 160]
        export_for_web(synth_elev, synth_rgb, demo_job_dir, max_dim=256, is_georeferenced=False)

    print("[INFO] Demo 3D terrain job created at data/web_jobs/job_demo")


if __name__ == "__main__":
    import uvicorn
    print("=" * 65)
    print("DepthWizard 3D Flythrough Server")
    print("Open viewer in your browser at: http://localhost:8000")
    print("=" * 65)
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
