# DepthWizard

> **Smart India Hackathon 2026** · **Problem Statement 26175** · **Organization: ISRO**  
> **Single-View Height Estimation and Real-Time 3D Browser Flythrough**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Frontend-Three.js-black.svg?logo=three.js&logoColor=white)](https://threejs.org)
[![Depth Anything V2](https://img.shields.io/badge/AI_Backbone-Depth_Anything_V2-blueviolet.svg)](https://github.com/DepthAnything/Depth-Anything-V2)
[![SRTM 30m](https://img.shields.io/badge/Datum-SRTM_30m_GL1-orange.svg)](https://opentopography.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is DepthWizard?

DepthWizard is a monocular terrain reconstruction pipeline that turns a **single RGB satellite or aerial image** into a navigable 3D representation, without requiring stereo imagery or LiDAR.

The system combines monocular depth estimation, satellite elevation data, and browser-based 3D rendering into a single workflow:

1. **Relative depth estimation**  
   Depth Anything V2 is used to estimate a relative Digital Surface Model (rDSM) from the input image. The depth output is normalized using robust 2nd and 98th percentile clipping to reduce the effect of extreme values.

2. **Metric elevation calibration**  
   For georeferenced GeoTIFF inputs, relative depth is converted into metric elevations using a stratified RANSAC regression model. The calibration is anchored against SRTM 30m elevation data retrieved through OpenTopography.

3. **Interactive 3D visualization**  
   The resulting terrain is rendered directly in the browser using Three.js. The viewer supports orbit controls, WASD flythrough, dynamic sun lighting, vertical exaggeration, and real-time elevation readouts.

4. **Browser-based workflow**  
   There is no requirement for Unity, a desktop GIS application, or a frontend build system. The viewer runs directly in a WebGL2-compatible browser, while processing is handled by the FastAPI backend.

---

## Quickstart

### 1. Start the server

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 2. Open the viewer

Open:

```text
http://localhost:8000
```

The browser interface provides the complete visualization and image-processing workflow.

---

## Demo Flow

A short demonstration can be structured around the following workflow:

| Time | Action | What to demonstrate |
|---|---|---|
| **0:00 - 0:15** | Open the viewer | Show a single RGB satellite image reconstructed as a 3D terrain mesh. Explain that the geometry is generated from monocular depth rather than stereo or LiDAR data. |
| **0:15 - 0:30** | Adjust Sun Azimuth and Vertical Exaggeration | Demonstrate the dynamic lighting and the ability to exaggerate terrain features for easier visual interpretation. |
| **0:30 - 0:45** | Enable Flythrough Mode | Use WASD controls to navigate through the generated environment. Hover over the terrain to demonstrate the real-time elevation readout. |
| **0:45 - 1:00** | Upload another image | Demonstrate the processing pipeline by dropping a new image into the viewer and showing the resulting 3D reconstruction. |

---

## Requirements vs. Current Implementation

The following table maps the current prototype against the main requirements of ISRO Problem Statement 26175 and the planned full implementation.

| Requirement | Current Prototype | Planned Full Build |
|---|---|---|
| **Single-View Height Estimation** | Depth Anything V2 monocular depth estimation with 2nd/98th percentile normalization. | Multi-scale tiled inference, CLAHE-based shadow enhancement, and LoRA fine-tuning on relevant datasets. |
| **Absolute Metric Elevation** | Stratified RANSAC regression calibrated against SRTM 30m data through OpenTopography. | Additional validation using NASA ICESat-2 ATL08 spaceborne LiDAR transects. |
| **Non-Georeferenced Input** | PNG/JPG inputs are processed into a unitless relative rDSM. | Automatic GSD estimation using image and building-footprint priors. |
| **Georeferenced GeoTIFF Input** | Rasterio-based ingestion, CRS validation, reprojection, and DSM GeoTIFF export. | Multispectral support and integration with additional satellite products and RPC-based rectification. |
| **3D Terrain Visualization** | Browser-based Three.js viewer with orbit controls, WASD navigation, dynamic lighting, and elevation HUD. | Quadtree-based Level of Detail for larger regional datasets. |
| **Zero Client Installation** | Frontend runs directly in a WebGL2-compatible browser using CDN imports. | PWA support, offline caching, and WebGPU acceleration. |

---

## CLI Pipeline

DepthWizard can also be used without the browser interface.

### Stage 1: Relative Depth Extraction

Process an individual image or a directory of images:

```powershell
.\.venv\Scripts\python.exe run_day1.py data/input/ --outdir data/output/
```

The pipeline generates:

```text
{stem}_raw.npy
{stem}_heightmap.png
{stem}_turbo.png
{stem}_compare.png
```

The heightmap is stored as a 16-bit PNG to preserve more depth information than a standard 8-bit representation.

### Stage 2: Metric Calibration

For georeferenced GeoTIFF processing, configure an OpenTopography API key:

```powershell
$env:OPENTOPO_API_KEY = "your_key_here"
```

Then run:

```powershell
.\.venv\Scripts\python.exe run_day2.py data/input/your_scene.tif --outdir data/output/
```

The pipeline generates:

```text
{stem}_dsm.tif
{stem}_dsm.npy
{stem}_error_map.png
{stem}_validation.json
```

The `_dsm.tif` output contains the calibrated metric elevation model.

---

## Automated Tests

The repository includes synthetic tests covering the main processing stages and mathematical components.

### Stage 1

Tests percentile-based normalization and 16-bit heightmap generation:

```powershell
.\.venv\Scripts\python.exe tests\test_stage1_synthetic.py
```

### Stage 2

Tests RANSAC parameter recovery, including slope and intercept estimation:

```powershell
.\.venv\Scripts\python.exe tests\test_stage2_synthetic.py
```

### Stage 3

Tests Web 3D asset generation, downsampling, and FastAPI endpoints:

```powershell
.\.venv\Scripts\python.exe tests\test_stage3_export.py
```

---

## Technical Documentation

Additional technical details are available in the `docs/` directory:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)  
  System architecture, processing pipeline, and design decisions, including the reasoning behind Depth Anything V2, RANSAC, and Three.js.

- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)  
  Known limitations and current system boundaries, including the distinction between DEM and DSM data and the limitations of the current validation approach.

- [`docs/RESULTS.md`](docs/RESULTS.md)  
  Validation results across different test images, including selected failure cases and error measurements.

---

## Project Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Depth Estimation | Depth Anything V2 |
| Raster Processing | Rasterio |
| Elevation Reference | SRTM 30m / OpenTopography |
| 3D Rendering | Three.js |
| Browser Graphics | WebGL2 |
| Language | Python / JavaScript |

---

## Authors & Acknowledgments

Built for **Smart India Hackathon 2026**, Problem Statement 26175, under **ISRO**.

The project uses and builds upon **Depth Anything V2, SRTMGL1, OpenTopography, Rasterio, FastAPI, and Three.js**.
