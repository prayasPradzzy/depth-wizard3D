# DepthWizard 🪄

> **Smart India Hackathon 2026** · **Problem Statement 26175** · **Organization: ISRO**  
> **Single-View Height Estimation and Real-Time 3D Browser Flythrough**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Frontend-Three.js-black.svg?logo=three.js&logoColor=white)](https://threejs.org)
[![Depth Anything V2](https://img.shields.io/badge/AI_Backbone-Depth_Anything_V2-blueviolet.svg)](https://github.com/DepthAnything/Depth-Anything-V2)
[![SRTM 30m](https://img.shields.io/badge/Datum-SRTM_30m_GL1-orange.svg)](https://opentopography.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 What is DepthWizard?

DepthWizard takes a **SINGLE optical RGB satellite or aerial drone image** (with zero stereo pairs and zero LiDAR) and:
1. **Estimates a relative Digital Surface Model (rDSM)** using Depth Anything V2 with robust 2nd/98th percentile outlier suppression.
2. **Calibrates relative depth into absolute metric elevations (metres)** for georeferenced GeoTIFFs via **Stratified RANSAC Regression** anchored to spaceborne **SRTM 30m** reference data.
3. **Renders an interactive, photorealistic 3D flythrough in the browser** using Three.js with dynamic sun shadows, real-time elevation readouts, and drag-and-drop processing.
4. **Requires Zero Installation for Judges**: No Unity, no heavy desktop software, no npm build step. Runs entirely on standard WebGL2.

---

## ⚡ 60-Second Quickstart

### 1. Launch Server (Single Command)
```powershell
# Activate environment and start the master server
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 2. Open Viewer in Browser
👉 Navigate to: **`http://localhost:8000`**

---

## 🎯 The 60-Second Jury Demo Script

*Follow these exact steps during your hackathon presentation to impress the judges:*

| Time | Action | What to Say / Point Out to Judges |
|---|---|---|
| **0:00 - 0:15** | Open `http://localhost:8000` | *"Judges, you are looking at a single monocular satellite photo of Manhattan transformed into a true 3D terrain mesh. The RGB photo is draped over a 16-bit displacement grid, with zero stepped terracing."* |
| **0:15 - 0:30** | Drag **Sun Azimuth** and **Vertical Exaggeration** sliders | *"Notice how the lighting responds dynamically. As I adjust the Sun Azimuth, shadows cast naturally off the building facades into the street canyons. The vertical exaggeration slider allows us to amplify subtle topographic features."* |
| **0:30 - 0:45** | Click **Flythrough Mode** (WASD flight) + Hover cursor | *"Instead of just an orbit view, we built a drone flythrough mode. We can fly between the skyscrapers using WASD. Notice the bottom-left HUD: our raycaster reads the exact elevation under the cursor in real time."* |
| **0:45 - 1:00** | Drag and drop any image onto the browser | *"The entire pipeline is live. I can drag and drop any satellite image or GeoTIFF onto the canvas. The FastAPI backend extracts relative depth, aligns it with spaceborne SRTM data, and updates the 3D world instantly."* |

---

## 📊 Pitch-Deck Gap Analysis (Requirements vs. Delivered)

*Use this table directly in your pitch deck to demonstrate engineering discipline and maturity:*

| Problem Statement Requirement (ISRO PS 26175) | Delivered in 4-Day Prototype | 15-Day Full Build Roadmap |
|---|---|---|
| **Single-View Height Estimation (Optical RGB)** | **Fully Delivered**: Depth Anything V2 monocular backbone with 2/98 percentile normalization. | Multi-scale tiled inference with CLAHE shadow enhancement & LoRA fine-tuning on SpaceNet. |
| **Absolute Metric Elevation (metres)** | **Fully Delivered**: Stratified RANSAC regression anchored to global SRTM 30m data via OpenTopography API. | Integration with NASA ICESat-2 (ATL08) spaceborne photon LiDAR transects for non-circular ground validation. |
| **Non-Georeferenced Input Mode** | **Fully Delivered**: Automatic detection routing PNG/JPG to unitless relative rDSM (0–1). | Automatic optical GSD estimation from building footprint priors. |
| **Georeferenced GeoTIFF Mode** | **Fully Delivered**: Native `rasterio` ingestion, CRS verification, spatial reprojection, and export to GeoTIFF DSM (`_dsm.tif`). | Multi-band multispectral support (Sentinel-2, Cartosat-3 RPC perspective rectification). |
| **Navigable 3D Terrain Visualization** | **Fully Delivered**: WebGL Three.js browser viewer with Orbit + WASD flythrough modes, dynamic shadows, and HUD. | Hierarchical Quadtree Level-of-Detail (LOD) for multi-gigabyte regional scale terrains. |
| **Zero-Client Installation** | **Fully Delivered**: Single-file frontend loaded via CDN importmap. Runs on any browser. | Progressive Web App (PWA) offline caching with WebGPU acceleration. |

---

## 🛠️ CLI Pipeline Tools

DepthWizard can also be run headlessly from the command line:

### Stage 1: Relative Depth Extraction (Day 1 CLI)
```powershell
# Process single image or folder of images
.\.venv\Scripts\python.exe run_day1.py data/input/ --outdir data/output/
```
*Outputs: `{stem}_raw.npy`, `{stem}_heightmap.png` (16-bit PNG), `{stem}_turbo.png`, `{stem}_compare.png`.*

### Stage 2: Georeferenced Metric Calibration (Day 2 CLI)
```powershell
# Set free OpenTopography API key
$env:OPENTOPO_API_KEY = "your_key_here"

# Process GeoTIFF into absolute metric DSM
.\.venv\Scripts\python.exe run_day2.py data/input/your_scene.tif --outdir data/output/
```
*Outputs: `{stem}_dsm.tif` (Metric GeoTIFF), `{stem}_dsm.npy`, `{stem}_error_map.png`, `{stem}_validation.json`.*

---

## 🔬 Automated Regression Test Suite

Verify all mathematical engines, outlier resilience, and WebGL exporters in seconds:

```powershell
# 1. Test Stage 1 Outlier Normalization & 16-bit PNG Precision
.\.venv\Scripts\python.exe tests\test_stage1_synthetic.py

# 2. Test Stage 2 RANSAC Mathematical Parameter Recovery (Slope/Intercept)
.\.venv\Scripts\python.exe tests\test_stage2_synthetic.py

# 3. Test Stage 3 Web 3D Asset Downsampling & FastAPI Endpoints
.\.venv\Scripts\python.exe tests\test_stage3_export.py
```

---

## 📚 Technical Documentation

- [**`docs/ARCHITECTURE.md`**](docs/ARCHITECTURE.md): Complete pipeline diagrams and deep-dive defenses of every design decision (Depth Anything V2 vs MiDaS, RANSAC vs OLS, Three.js vs Unity).
- [**`docs/LIMITATIONS.md`**](docs/LIMITATIONS.md): Transparent engineering disclosures of system boundaries (circular validation, DEM vs DSM) and our 15-day build solutions.
- [**`docs/RESULTS.md`**](docs/RESULTS.md): Master validation metrics table across diverse test imagery and intentional failure modes.

---

## 👥 Authors & Acknowledgments

Built for **Smart India Hackathon 2026** under **Problem Statement 26175 (ISRO)**.  
*Powered by Depth Anything V2, OpenTopography SRTMGL1, Rasterio, FastAPI, and Three.js.*
