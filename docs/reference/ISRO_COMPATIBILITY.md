# ISRO Data Compatibility

**Problem Statement 26175** states that final evaluation uses **ISRO RGB-band optical
satellite imagery**. This document records what the pipeline does to accept that data,
and — just as importantly — where it does not yet reach.

---

## 1. Sensor resolution

Monocular depth is **scale-dependent**. The backbone learned what a building edge, a
tree crown and a road width look like *in pixels*. Present it imagery at a very
different ground sample distance and those priors stop matching, which is a major
part of why models degrade across sensors.

`src/depth_extract.py` therefore resamples the input to the backbone's native GSD
(`NATIVE_GSD_M = 0.5 m`) before inference and maps the prediction back to the source
grid. Georeferenced inputs supply their true GSD from the affine transform, so this
happens automatically.

| ISRO sensor | GSD | Ratio to native | Handling | Realistic outcome |
|---|---|---|---|---|
| **Cartosat-3 PAN** | 0.25 m | 0.50 | downsample ×0.50 | **Good** — finer than training data |
| **Cartosat-2S PAN** | 0.65 m | 1.30 | within tolerance, unchanged | **Good** — closest match to training |
| **Cartosat-3 MX** | 1.14 m | 2.28 | upsample ×2.28 | Usable; large structures only |
| **Cartosat-1 PAN** | 2.50 m | 5.00 | upsample ×5.00 | Marginal; buildings are 2–4 px |
| **LISS-IV** | 5.8 m | 11.6 | upsample (edge-capped) | **Terrain relief only** — buildings sub-pixel |
| **LISS-III** | 23.5 m | 47 | upsample (edge-capped) | **Not suitable** for structure height |

**Be honest about the lower half of that table.** Resampling repositions the image into
a familiar scale; it cannot invent detail that the sensor never captured. Below roughly
2.5 m GSD a building occupies a handful of pixels and per-structure height estimation
is not physically recoverable. Broad terrain relief still works.

**Best fit: Cartosat-2S and Cartosat-3 PAN.** Those sit closest to the training regime.

## 2. Elevation reference

Default is **Copernicus DEM 30m (COP30)**, not SRTM.

SRTM was flown in 2000 by C-band radar and has well-documented **voids over steep
Himalayan terrain and persistent snow** — precisely the Indian topography where a
reference is hardest to obtain and most needed. COP30 derives from TanDEM-X, is
void-filled, and is the better choice for Indian scenes.

Selectable via `DW_DEM_TYPE` (`src/georef.py`):

| Value | Source | Notes |
|---|---|---|
| `COP30` *(default)* | Copernicus DEM 30m | Void-filled; best over Himalaya |
| `SRTMGL1` | SRTM 30m GL1 | Historical baseline; voids on steep/snow terrain |
| `NASADEM` | Reprocessed SRTM | Partially void-filled |
| `AW3D30` | JAXA ALOS World 3D | Optical stereo derived |

**CartoDEM** (ISRO's own Cartosat-1 derived DEM, 30 m and 10 m over India) would be the
most appropriate reference of all. It is distributed through Bhuvan and requires an
account, so it is not wired in. The DEM layer is a single pluggable function
(`fetch_srtm`), so adding it is a contained change once credentials exist.

## 3. Coordinate reference systems

Indian scenes are handled through the standard `rasterio` / `pyproj` stack, so UTM
zones **42N–47N (EPSG:32642–32647)** covering India work without special casing, as do
the WGS84 India zones. CRS and affine transform are preserved end to end and written
into the exported GeoTIFF.

Verified on a live Indian scene: `data/input/india_sikkim_flood.tif`, **EPSG:32645
(UTM 45N)**, Sikkim, 27.604°N 88.602°E — Maxar Open Data release for the October 2023
Teesta basin glacial-lake outburst flood.

## 4. Terrain coverage — the real gap

The fine-tuned model was trained on **GAMUS**, which covers three US cities:
Washington DC, New York and Philadelphia. All are **flat, temperate and urban**.

It contains **no** mountainous terrain, no desert, no agricultural landscape, and no
Indian scenes. The problem statement explicitly asks for stability across *urban,
sparse, hilly and forested* landscapes, and only the first is genuinely represented in
our training data.

Honest position: our benchmark demonstrates **6.22 m → 4.60 m RMSE on urban nadir
imagery at sub-metre resolution**. It does not demonstrate performance over the
Himalaya, the Thar, or the Western Ghats, and should not be claimed to.

Closing it needs paired image/height data over Indian physiography. Candidates:

- **CartoDEM + Cartosat imagery** — the natural pairing, via Bhuvan / NRSC Open Data.
- **ICESat-2 ATL08** — spaceborne LiDAR ground and canopy heights, global, free, and
  independent of anything used in calibration. Sparse transects rather than dense
  rasters, so useful for validation more than training.
- **DFC2019 / US3D** — adds varied terrain but remains US.

## 5. Disaster-management relevance

The problem statement's theme is Disaster Management, and the two demo scenes shipped
with the project are genuine disaster-response imagery:

| Scene | Event | Relevance |
|---|---|---|
| `india_sikkim_flood.tif` | Teesta basin GLOF, Oct 2023 | Indian territory, Himalayan terrain, flood response |
| `bangkok_quake_0p3m.tif` | Myanmar earthquake, Mar 2025 | Dense urban, structural height assessment |

Above-ground height is the operationally useful quantity in both cases — flood depth
modelling needs bare-earth terrain, while structural damage assessment needs building
heights. That is why the pipeline separates the two (`estimate_ground_surface`) rather
than emitting a single surface model.
