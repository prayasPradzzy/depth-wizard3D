# DepthWizard — what changed

Short version: the app looks and runs a lot better, and **we now have a real accuracy
number instead of made-up ones**. We also trained the model on satellite data and cut
the error by 26%.

Branch: `my-changes` · 7 commits · start at `e6cb8fc`

---

## The headline

We benchmarked against **GAMUS** — satellite images that ship with real LiDAR-measured
heights. 240 test tiles, 251 million pixels, 3 cities. None of it is used in our
calibration, so it's a genuinely independent test.

| | RMSE | MAE | Correlation |
|---|---|---|---|
| Off-the-shelf AI | 6.22 m | 4.04 m | 0.682 |
| **After we trained it** | **4.60 m** | **2.51 m** | **0.840** |
| | **26% better** | **38% better** | |

Every land-cover type improved (trees, buildings, roads, ground, water, vegetation) and
all three cities improved.

**The best fact:** the trained model outputs **real metres directly with no calibration
step** — 5.48 m. That still beats the old model *even though the old one was handed free
scale calibration*. That's the actual deliverable the problem statement asks for.

---

## ⚠️ Important — don't overclaim these

1. **The trained model is NOT running in the live app.** Uploads still use the
   off-the-shelf model. The 4.60 m is a measured benchmark result, not what the demo
   produces. Correct phrasing: *"we measured this on the held-out test set; wiring the
   weights into the pipeline is the next step."*
2. **No Indian or mountainous training data.** GAMUS is three flat US cities (DC, NYC,
   Philadelphia). The 4.60 m is valid for **urban, nadir, sub-metre imagery only**. Don't
   say "works on any satellite image" — it's the easiest thing to disprove with one
   awkward test image.
3. **Below ~2.5 m resolution buildings are sub-pixel**, so per-building height isn't
   physically recoverable. Fine for Cartosat-2S/3, not for LISS-III.

Being straight about these is a *strength*. "We measured our limits" beats numbers
nobody can verify.

---

## How to run it

### Option A — Docker (nothing to install but Docker)

```
docker compose up --build
```

Then open **http://localhost:8000**.

First build takes roughly 10-15 minutes and pulls about 2 GB: it installs PyTorch and
the geospatial stack, and bakes the depth model into the image so the container needs
no network at run time and the first upload is fast. Subsequent starts are seconds
(`docker compose up`).

Wait for `Model pre-warmed` in the log before demoing.

> **Not yet build-tested.** The Dockerfile was written against the verified dependency
> set but Docker is not installed on the machine it was authored on, so nobody has run
> `docker build` on it yet. If it fails, the error will almost certainly be a missing
> apt or pip package - add it to the Dockerfile and rebuild. Option B below *is*
> verified working.

Optional, for absolute metric elevation: get a free key at
[portal.opentopography.org](https://portal.opentopography.org) and put it in a `.env`
file beside `docker-compose.yml`:

```
OPENTOPO_API_KEY=your_key_here
```

### Option B — native Windows (verified working)

```
setup_laptop.bat     (first time only, ~10 min)
start_demo.bat
```

**Wait for `Model pre-warmed` in the console before demoing** (~15 s). After that an
upload takes ~6 s instead of the old 44 s.

Either way: open http://localhost:8000 and **hard-refresh with Ctrl+Shift+R** the first
time - the browser caches the old page aggressively.

---

## What's new in the app

**Dashboard rebuilt** — three columns: upload/processing/results on the left, 3D viewer
in the middle, analysis on the right.

Worth clicking during a demo:

- **Sharp Buildings** (viewer toolbar) — toggles the fix for buildings looking like
  slanted ramps. Great before/after moment.
- **Cross-sectional profile** (right panel) — slide through the scene, see the surface
  vs estimated ground. The shaded band is built structure.
- **Height / Slope / Wireframe** layers.
- **Ruler** — click two points to measure height difference.
- **Accuracy & Validation panel** — shows the benchmark with old → new comparison,
  per city and per land-cover type.

**Two demo scenes included:**

| File | What it is |
|---|---|
| `data/input/india_sikkim_flood.tif` | **Sikkim, India** — Himalayan, Oct 2023 Teesta flood |
| `data/input/bangkok_quake_0p3m.tif` | Dense urban Bangkok, 0.3 m/px |

Both are real georeferenced GeoTIFFs, so lat/lon, coordinate system and GeoTIFF export
all light up. `test 6.png` is a plain photo, so that half stays greyed out — that's
correct behaviour, not a bug.

---

## Performance fixes

- The viewer used to drop to **~19 FPS whenever you moved the mouse** (expensive
  raycasting). Now **60–130 FPS**.
- Sliders no longer lag.
- Resolution doubled (512 → 1024).
- Added ambient occlusion, proper sky, lighting, solid terrain base.

---

## ISRO-specific work

- **Indian scene** in an Indian coordinate zone (EPSG:32645, UTM 45N).
- **Copernicus DEM instead of SRTM** — SRTM has data voids over Himalayan terrain, which
  is exactly where Indian scenes need it. Good answer if a judge asks.
- **GSD normalisation** — resizes input to the scale the model was trained at, so one
  model can handle Cartosat-3 (0.25 m) and Cartosat-2S (0.65 m).
- See `docs/ISRO_COMPATIBILITY.md` for the full sensor matrix.

---

## Honesty fix worth knowing about

The old viewer had accuracy numbers **typed directly into the code** — it displayed
`2.01 m` RMSE, `0.992` correlation, `95.3%` inliers whenever real data was missing.
Those were in our screenshots and nothing had ever computed them.

All removed. Everything now shows a measured value or a dash.

---

## Where things live

| Path | What |
|---|---|
| `docs/RESULTS.md` | Full benchmark tables — **use these for slides** |
| `docs/ISRO_COMPATIBILITY.md` | Sensor matrix, DEM choice, stated limits |
| `tools/eval_gamus.py` | The benchmark script |
| `tools/train_gamus.py` | Fine-tuning script |
| `tools/fetch_scene.py` | Pulls georeferenced scenes from Maxar Open Data |
| `logs/overnight.log` | Training run log |
| `checkpoints/agl_vits.pt` | The trained model (95 MB) |
| `Dockerfile`, `docker-compose.yml` | Containerised run |

Running the benchmark or training scripts needs `h5py` as well (`pip install h5py`);
it is deliberately left out of the container, which only serves the app.

Training details: 180 train / 20 val tiles, 150 epochs, **51 minutes on a 6 GB RTX
3060 laptop GPU**. Only used 180 of ~5,000 available tiles — using more is the single
biggest remaining improvement.
