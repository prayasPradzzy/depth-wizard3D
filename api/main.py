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

import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
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
from src.georef import align_to_reference, fetch_srtm, is_georeferenced, read_geotiff, to_metric_resolution
from src.calibrate import apply_calibration, calibrate_depth
from src.mesh_export import export_for_web

app = FastAPI(
    title="DepthWizard API",
    description="Single-View Height Estimation & 3D Flythrough (ISRO PS 26175)",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = PROJECT_ROOT / "data" / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Shared depth estimator instance
_estimator = None


def get_estimator() -> DepthEstimator:
    global _estimator
    if _estimator is None:
        _estimator = DepthEstimator(model_size="small", device="auto")
    return _estimator


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

    try:
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
            depth_res = estimator.predict(rgb_uint8, robust=True)
            rel_depth = depth_res.normalized_depth

            # Attempt SRTM fetch & calibration if key is present
            has_calibrated = False
            pred_dsm = rel_depth
            if "OPENTOPO_API_KEY" in os.environ:
                try:
                    srtm_path = fetch_srtm(bounds=bounds, crs=crs, out_dir="data/srtm")
                    ref_dem, srtm_valid = align_to_reference((height, width), transform, crs, srtm_path)
                    comb_mask = srtm_valid & (~nodata_mask) & np.isfinite(rel_depth)
                    calib_res = calibrate_depth(rel_depth, ref_dem, comb_mask, method="ransac")
                    pred_dsm = apply_calibration(rel_depth, calib_res.slope, calib_res.intercept)
                    has_calibrated = True
                except Exception as e:
                    print(f"[API WARNING] SRTM alignment/calibration skipped: {e}")

            gsd = to_metric_resolution(transform, crs)
            export_for_web(
                elevation_arr=pred_dsm,
                rgb_input=rgb_uint8,
                out_dir=job_dir,
                max_dim=512,
                is_georeferenced=has_calibrated,
                crs_str=crs.to_string(),
                gsd_m=gsd,
            )

        else:
            print(f"[API] Standard RGB image detected. Executing Stage 1 Relative Pipeline...")
            pil_img = Image.open(input_path).convert("RGB")
            depth_res = estimator.predict(pil_img, robust=True)
            rel_depth = depth_res.normalized_depth

            export_for_web(
                elevation_arr=rel_depth,
                rgb_input=pil_img,
                out_dir=job_dir,
                max_dim=512,
                is_georeferenced=False,
            )

        print(f"[API] Successfully generated web assets for job: {job_id}")
        return {"job_id": job_id, "status": "completed"}

    except Exception as e:
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
    return {
        "job_id": resolved_id,
        "manifest": manifest,
        "assets": {
            "heightmap": f"/api/assets/{resolved_id}/heightmap.png",
            "heightmap_bin": f"/api/assets/{resolved_id}/heightmap.bin",
            "texture": f"/api/assets/{resolved_id}/texture.png",
        },
    }


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
        export_for_web(elev_arr, pil_img, demo_job_dir, max_dim=512, is_georeferenced=False)
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


create_initial_demo_job()


if __name__ == "__main__":
    import uvicorn
    print("=" * 65)
    print("DepthWizard 3D Flythrough Server")
    print("Open viewer in your browser at: http://localhost:8000")
    print("=" * 65)
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
