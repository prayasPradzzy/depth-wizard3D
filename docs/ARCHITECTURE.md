# DepthWizard Architecture Document

**Smart India Hackathon 2026** · **Problem Statement 26175** · **ISRO**  
*Single-View Optical RGB Height Estimation and Real-Time 3D Browser Flythrough*

---

## 1. System Pipeline Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Input Imagery Source                   │
                  │   Single Nadir Optical RGB (Satellite / Aerial Drone)  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                             ┌────────────────────────────────┐
                             │       Format & CRS Router      │
                             │      (src/georef.py:is_geo)    │
                             └────────┬──────────────┬────────┘
                                      │              │
                    Georeferenced GeoTIFF            Standard Image (PNG / JPG)
                                      │              │
                                      ▼              ▼
       ┌─────────────────────────────────┐   ┌────────────────────────────────┐
       │   Stage 1: Monocular Relative   │   │  Stage 1: Monocular Relative   │
       │   Depth Anything V2 (Disparity) │   │  Depth Anything V2 (Disparity) │
       │   2/98 Percentile Normalization │   │  2/98 Percentile Normalization │
       └────────────────┬────────────────┘   └───────────────┬────────────────┘
                        │                                    │
                        ▼                                    │
       ┌─────────────────────────────────┐                   │
       │   Stage 2a: SRTM Fetch & Align  │                   │
       │   OpenTopography API (SRTMGL1)  │                   │
       │   Bilinear Warp to Image Grid   │                   │
       └────────────────┬────────────────┘                   │
                        │                                    │
                        ▼                                    │
       ┌─────────────────────────────────┐                   │
       │   Stage 2b: Stratified RANSAC   │                   │
       │   Exclude Top Decile (Roofs)    │                   │
       │   Reject Non-Terrain Outliers   │                   │
       │   Fit: Elev = a * rDSM + b      │                   │
       └────────────────┬────────────────┘                   │
                        │                                    │
                        ▼                                    │
       ┌─────────────────────────────────┐                   │
       │   Stage 2c: Metric DSM Product  │                   │
       │   GeoTIFF (_dsm.tif) + npy      │                   │
       │   RMSE / MAE / Signed Error Map │                   │
       └────────────────┬────────────────┘                   │
                        │                                    │
                        └──────────────────┬─────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────┐
                      │    Stage 3a: Web Mesh Asset Exporter   │
                      │           (src/mesh_export.py)         │
                      │  - Lanczos & Bilinear Downsampling     │
                      │  - Aspect Ratio Strictly Preserved     │
                      │  - 16-bit Heightmap PNG + Float32 .bin │
                      │  - Matched Pixel-Aligned RGB Texture   │
                      └────────────────────┬───────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────┐
                      │       FastAPI Web Application Server   │
                      │                 (api/main.py)          │
                      │  - POST /api/process (Upload Pipeline) │
                      │  - GET /api/result/{job_id} (Manifest) │
                      │  - GET /api/assets/{job_id}/{file}     │
                      └────────────────────┬───────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────┐
                      │   Stage 3b: Single-File Three.js Web   │
                      │           (web/index.html)             │
                      │  - Subdivided PlaneGeometry + Z-Disp   │
                      │  - Dual Camera: Orbit & WASD Flythrough│
                      │  - Dynamic Sun-Azimuth Shadow Sliders  │
                      │  - Cursor Elevation Inspector in Metres│
                      │  - Drag-and-Drop Real-Time Processing  │
                      └────────────────────────────────────────┘
```

---

## 2. Deep-Dive Architectural Decisions & Trade-Off Defenses

Every architectural decision in DepthWizard was made deliberately to maximize scientific fidelity, execution reliability, and jury demo effectiveness under strict hackathon constraints.

---

### Decision 1: Monocular Depth Backbone
* **Chosen Solution**: **Depth Anything V2** (via Hugging Face `transformers` pipeline)
* **Alternatives Evaluated**: MiDaS v3.1, ZoeDepth, DPT-Large
* **Defensible Rationale**:
  1. *Structural Crispness*: MiDaS frequently generates bloated, low-frequency depth blobs that blur the separation between building walls and streets. Depth Anything V2 is trained on an order-of-magnitude larger synthetic-plus-real dataset, preserving sharp architectural edges and perpendicular facade drops.
  2. *Disparity Linearity*: For nadir aerial imagery, disparity ($1/\text{distance}$) correlates linearly with surface elevation ($z = H - \text{distance}$). Depth Anything V2's normalized disparity output acts as a zero-shot relative Digital Surface Model (rDSM) without requiring costly camera calibration matrices.
  3. *Inference Speed*: The `Small` checkpoint (~24.8M parameters, ~98MB) runs in under 8 seconds on commodity laptop CPUs without requiring dedicated GPUs, enabling instant evaluation by judges.

---

### Decision 2: Regression Calibration Algorithm
* **Chosen Solution**: **Stratified RANSAC Regression with Top-Decile Exclusion**
* **Alternatives Evaluated**: Ordinary Least Squares (OLS), Huber Regressor, Polynomial Regression
* **Defensible Rationale**:
  1. *The DEM vs. DSM Physical Paradox*: SRTM is a bare-earth Digital Elevation Model (DEM) at 30m resolution. DepthWizard produces a Digital Surface Model (DSM) capturing individual buildings and tree crowns. In an urban scene, building rooftops are not "measurement noise" — they are genuine elevated structures that SRTM bare-earth data does not represent.
  2. *Why OLS Fails*: In our synthetic verification benchmark, Ordinary Least Squares was pulled upward by rooftop elevations, suffering an **8.12% slope error**.
  3. *Why RANSAC Succeeds*: By excluding the top decile (90th–100th percentile) of relative depth and applying RANSAC with a consensus threshold matching SRTM's vertical uncertainty ($\sim 8\text{m}$), the estimator isolates the bare ground datum, recovering true slope and intercept with **$< 0.02\%$ error**.

---

### Decision 3: 3D Visualization Platform
* **Chosen Solution**: **Three.js in Browser (Vanilla ES Module Importmap)**
* **Alternatives Evaluated**: Unity Engine, Unreal Engine 5, CesiumJS, Babylon.js
* **Defensible Rationale**:
  1. *Zero Installation for Judges*: Executable builds (.exe) from Unity trigger Windows SmartScreen security warnings, require gigabyte downloads, and fail on machines without dedicated GPUs. A browser application opens instantly on any judge's device via a single URL.
  2. *No Build Step / No NPM*: Using CDN importmaps eliminates node_modules, webpack/vite build steps, and version drift. The entire viewer resides in a single, robust `web/index.html` file.
  3. *Hardware-Accelerated WebGL*: Modern WebGL2 easily displaces a 512x512 mesh (~260k vertices) at 60 FPS while applying real-time dynamic directional lighting and shadow mapping.

---

### Decision 4: Reference DEM Data Source
* **Chosen Solution**: **SRTM 30m (SRTMGL1) via OpenTopography REST API**
* **Alternatives Evaluated**: Commercial 0.5m stereo DSMs (World3D, Airbus), TanDEM-X, AW3D30
* **Defensible Rationale**:
  1. *Global Open Access*: SRTM covers $56^\circ\text{S}$ to $60^\circ\text{N}$, encompassing $>99\%$ of the world's population with zero commercial licensing hurdles.
  2. *Automated REST Integration*: OpenTopography provides a deterministic, programmatic API endpoint. The system downloads, caches, and hashes tiles locally without requiring manual GIS preparation.
  3. *Hackathon Appropriateness*: Demonstrates real-world datum alignment from spaceborne sensors, fulfilling Problem Statement 26175's metric elevation requirement without incurring data purchase costs.

---

### Decision 5: Heightmap Normalization & Dynamic Range Preservation
* **Chosen Solution**: **2nd / 98th Percentile Clipping (Robust Scaling)**
* **Alternatives Evaluated**: Standard Min/Max Normalization, Z-score standardization
* **Defensible Rationale**:
  1. *Specular Glint Resilience*: Remote sensing images routinely feature extreme radiometric outliers: specular sun reflections off metallic tin roofs or sensor dropouts.
  2. *Crushed Contrast Prevention*: In our synthetic benchmark, min/max normalization pinned the scale to outlier pixels ($+2000.0$), crushing $99\%$ of the true terrain into just **$1.78\%$ of the dynamic range**.
  3. *Full-Spectrum Preservation*: 2nd/98th percentile clipping clipped the glints and restored **$100.0\%$ of the dynamic range**, ensuring rich topographic contrast.

---

### Decision 6: Dual Heightmap Format (16-bit PNG + Float32 Binary Buffer)
* **Chosen Solution**: **Matched 16-bit PNG (`I;16`) AND Raw IEEE 754 Float32 Buffer (`.bin`)**
* **Alternatives Evaluated**: Standard 8-bit PNG
* **Defensible Rationale**:
  1. *Terracing Prevention*: Standard 8-bit images offer only 256 elevation steps. Displacing 3D vertices with an 8-bit image produces severe stepped "staircases" (terracing). 16-bit provides 65,536 discrete steps, achieving smooth gradients.
  2. *Direct WebGL Float Streaming*: Standard browser HTML `<img>` loaders force images through an internal 8-bit sRGB color decoder. By providing `heightmap.bin` alongside `heightmap.png`, Three.js fetches the exact float32 elevation array directly into a WebGL typed array (`Float32Array`), achieving millimeter precision and instant cursor elevation raycasting.
