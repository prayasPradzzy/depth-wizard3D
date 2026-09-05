# DepthWizard: Validation Metrics & Experimental Results

**Smart India Hackathon 2026** · **Problem Statement 26175** · **ISRO**

This document compiles quantitative benchmarks and empirical results across synthetic ground-truth tests, real-world satellite scenes, and deliberate failure/edge cases.

---

## 1. Master Benchmark Summary Table

| Scene / Test Identifier | Terrain Type | Pipeline Mode | Input Resolution | Metric Elevation Range | Variance ($\text{std}$) | $R^2$ / Corr ($r$) | Status & Gate Outcome |
|---|---|---|---|---|---|---|---|
| **Synthetic Ground Truth** | 2 Hills + Urban Block + 2 Outliers | Stage 1 (Relative) | $256 \times 256$ | N/A (Unitless 0–1) | $0.2619$ | N/A | **PASSED** (100% dynamic range preserved vs 1.78% in naive min/max) |
| **Synthetic Calibration** | Known Ground Truth ($\text{Slope}=145\text{m}, \text{Datum}=320\text{m}$) | Stage 2 (Metric) | $256 \times 256$ | $320.0\text{m} - 412.6\text{m}$ | $0.2741$ | $R^2 = 0.9916$ | **PASSED** (Slope recovered to $144.97\text{m}$, 0.02% error) |
| **Manhattan Skyscraper Scene** (`nyc_scene.jpg`) | Dense Urban Skyscraper Canyons | Stage 1 (Relative) | $1024 \times 693$ | N/A (Unitless 0–1) | $0.2603$ | N/A | **PASSED** ($\text{std} \ge 0.10$ gate met; roof-to-street gradient sharp) |
| **Manhattan Aerial Test 1** (`test1.png`) | High-Rise Financial District | Stage 1 (Relative) | $1259 \times 853$ | N/A (Unitless 0–1) | $0.2622$ | N/A | **PASSED** ($\text{std} \ge 0.10$ gate met; 3D extrusion confirmed) |
| **Simulated GeoTIFF** (`test_georef.tif`) | Georeferenced Extent (EPSG:4326) | Stage 2 (Metric) | $256 \times 256$ | $320.0\text{m} - 412.6\text{m}$ | $0.2741$ | $r = 0.9948$ | **PASSED** (Preserved CRS, affine transform, nodata) |
| **Featureless Water / Flat Desert** (Edge Case) | Calm Open Water / Salt Flat | Stage 1 (Relative) | $512 \times 512$ | N/A | $\mathbf{0.0210}$ | N/A | **CORRECTLY TRIPPED FAILURE ALERT** (`flat_warning: True`, $\text{std} < 0.05$) |
| **Unreferenced Image in Stage 2 CLI** | Standard JPG/PNG | Stage 2 Fallback | $1024 \times 693$ | N/A | $0.2603$ | N/A | **HANDLED** (Graceful fallback to Stage 1 rDSM; zero crash) |
| **GeoTIFF Missing SRTM Key** | GeoTIFF without API Key | Stage 2 Fallback | $256 \times 256$ | N/A | $0.2741$ | N/A | **HANDLED** (Actionable alert displayed; fallback to rDSM) |

---

## 2. Quantitative Deep-Dives

### A. Mathematical Calibration Parameter Recovery (Synthetic Benchmark)
- **Target True Formula**: $\text{Elevation} = 145.00 \times \text{rDSM} + 320.00\text{ m}$
- **Simulated Environmental Degradation**:
  - $\sigma = 2.0\text{m}$ Gaussian sensor noise on reference DEM
  - $9.3\%$ of scene contaminated by positive building outliers ($+25\text{m}$ to $+50\text{m}$)

| Metric | Ordinary Least Squares (OLS) | Stratified RANSAC (DepthWizard) | Benchmark Tolerance |
|---|---|---|---|
| **Recovered Slope** | $133.23\text{ m}$ (**$8.12\%$ error**) | **$144.97\text{ m}$ ($0.02\%$ error)** | $< 4.00\%$ |
| **Recovered Datum Intercept** | $321.91\text{ m}$ ($0.60\%$ error) | **$320.03\text{ m}$ ($0.01\%$ error)** | $< 3.00\%$ |
| **Inlier Fraction** | $100.0\%$ (Forced all outliers into fit) | **$95.3\%$** (Identified & isolated ground consensus) | $> 70\%$ |
| **Bare-Earth Terrain RMSE** | $4.87\text{ m}$ | **$2.01\text{ m}$** (Matched sensor noise bound) | $< 3.00\text{m}$ |
| **Building Pixels Mean Bias** | Skewed ground upward | **$+37.84\text{ m}$** (Accurately captured structural height) | Positive |

---

### B. Dynamic Range Preservation Under Extreme Radiometric Outliers
Tested with $+2000.0$ rooftop specular glint and $-1000.0$ sensor dropout:

| Normalization Technique | Dynamic Range of True Terrain | Standard Deviation | Verdict |
|---|---|---|---|
| **Naive Min/Max Scaling** | **$1.78\%$** of available levels | $\text{std} = 0.0041$ | **FAILED**: Topography completely crushed |
| **2nd/98th Percentile Clipping** | **$100.00\%$** of available levels | $\text{std} = 0.2618$ | **PASSED**: Full topographic contrast preserved |

---

### C. 16-Bit PNG vs. 8-Bit Quantization Precision
- **16-bit Max Absolute Round-Trip Error**: **$0.00000763$** (Exact theoretical limit $\frac{0.5}{65535}$).
- **8-bit Max Error Comparison**: $0.001961$ ($256\times$ coarser quantization).
- **Result in Three.js**: Zero terracing or stair-stepping artifacts during 3D vertex extrusion.

---

### D. System Latency & Hardware Profile
Tested on commodity hardware (Intel CPU, zero dedicated GPU):
- **Model Checkpoint**: `Depth-Anything-V2-Small-hf` (24.8M parameters)
- **Model Download Footprint**: ~98 MB
- **Inference Time (512x512)**: $\sim 7.0\text{ seconds}$
- **Inference Time (1024x693)**: $\sim 8.8\text{ seconds}$
- **Browser Frame Rate**: Stable **60 FPS** at $512 \times 346$ resolution (~177k vertices).
