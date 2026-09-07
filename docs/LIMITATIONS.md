# DepthWizard: Engineering Limitations & 15-Day Full Build Roadmap

**Smart India Hackathon 2026** · **Problem Statement 26175** · **ISRO**

Scientific integrity requires candidly disclosing system boundaries, theoretical assumptions, and failure modes. This document records all known limitations of the 4-day prototype and specifies the exact engineering solutions scheduled for the 15-day round.

---

## Limitation Matrix & Roadmap Solutions

| # | Prototype Limitation | Root Cause & Failure Impact | 15-Day Full Build Engineering Solution |
|---|---|---|---|
| **1** | **Circular Validation** | The metric DSM is validated against the exact same SRTM 30m dataset used to calibrate its linear regression. This makes quantitative metrics (RMSE, MAE) systematically optimistic regarding ground slope errors. | **Independent Spaceborne LiDAR Integration**: Validate against NASA **ICESat-2 (ATL08)** and **GEDI** photon-counting LiDAR transects that are completely withheld from the calibration regression. |
| **2** | **DSM vs. DEM Datum Mismatch** | SRTM is a bare-earth Digital Elevation Model (DEM). DepthWizard produces a Digital Surface Model (DSM) that includes building rooftops and canopy tops. In built-up urban centers, our validation shows a systematic $+20\text{m}$ to $+50\text{m}$ positive bias against SRTM. | **Automated Ground/Canopy Segmentation**: Introduce a lightweight semantic segmentation head (e.g. SegFormer trained on LandCover.ai) to generate a binary ground mask. Calibrate the ground plane exclusively on bare ground pixels, and calculate structure heights as differential offsets ($\text{Height} = \text{DSM} - \text{DEM}$). |
| **3** | **Foundation Model Domain Gap** | Depth Anything V2 was trained primarily on egocentric, terrestrial photography (horizontal horizon, perspective vanishing points) rather than orthorectified nadir satellite views. On flat scenes lacking perspective cues, the model can output low variance ($\text{std} < 0.05$). | **Remote Sensing LoRA Fine-Tuning**: Apply Low-Rank Adaptation (LoRA) on the ViT backbone using aerial datasets with LiDAR ground truth (SpaceNet 7, DFC2019, Vaihingen/Potsdam ISPRS benchmarks) to learn Nadir shadow and texture disparity cues directly. |
| **4** | **Off-Nadir Oblique Angle Breakdown** | The system assumes that inverse depth (disparity, $1/\text{distance}$) maps monotonically to surface elevation ($z = H - \text{distance}$). This physics assumption holds for near-nadir viewing angles ($< 15^\circ$), but breaks down for high-obliquity satellite passes ($> 30^\circ$). | **Rational Polynomial Coefficient (RPC) Ray-Tracing**: For commercial satellite imagery (WorldView, Cartosat), ingest the embedded RPC metadata, compute the sensor incidence and azimuth vectors, and perform oblique-to-nadir perspective rectification prior to inference. |
| **5** | **Radiometric Edge Cases (Water, Canopy, Shadow)** | 1. Open water bodies exhibit specular glints and zero texture, causing noisy depth artifacts.<br>2. Uniform, featureless rainforest canopy lacks depth gradients.<br>3. Deep building shadows suffer radiometric clipping to near-zero luminance. | **Water Masking & CLAHE Shadow Boosting**: <br>1. Ingest OSM or Sentinel-2 NDWI water masks to clamp water bodies to zero relief.<br>2. Implement automated CLAHE (Contrast Limited Adaptive Histogram Equalization) on shadowed luminance regions (Debug Prompt A). |
| **6** | **Fixed 512x512 Mesh Resolution** | Downsampling high-resolution images ($> 2000 \times 2000$) to $512 \times 512$ limits spatial resolution to preserve browser frame rates, smoothing out small structural details like parapets and HVAC units. | **Hierarchical Quadtree Level-of-Detail (LOD)**: Implement tiled 3D terrain rendering using quadtrees (similar to Cesium / Google Earth). Stream high-resolution chunks dynamically based on camera distance and frustum culling. |
| **7** | **Single-Scene Benchmark Scope** | The prototype has been validated on synthetic benchmarks, urban Manhattan skyscrapers, and suburban scenes, but has not yet been stress-tested across desert dunes, snow-covered mountains, and agricultural terraces. | **ISRO Standard Multi-Terrain Benchmark Suite**: Build an automated benchmark harness testing 50 diverse tiles across all Indian physiographic zones (Himalayan ridges, Indo-Gangetic plains, Thar desert, Western Ghats, coastal deltas). |

---

## Summary for Evaluators

A system that claims $100\%$ accuracy from single-view optical imagery without LiDAR or stereo pairs is scientifically implausible. 

DepthWizard succeeds by:
1. Being mathematically honest about what monocular vision can extract (relative shape).
2. Using robust statistical filtering (Stratified RANSAC + Top-Decile exclusion) to anchor that shape to spaceborne reference datums (SRTM).
3. Explicitly logging alerts when variance or correlation fails, rather than silently generating plausible-looking hallucinations.
