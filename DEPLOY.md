# Deploying DepthWizard

Two targets, because they serve different jobs.

| | What it is | Why |
|---|---|---|
| **GitHub Pages** | Static viewer, pre-processed scenes | Opens instantly, never sleeps, free forever. **This is the link to share.** |
| **Hugging Face Spaces** | Full app, upload your own image | Needs PyTorch + GDAL, so it sleeps between visits |

---

## 1. GitHub Pages — 2 minutes, no account needed

Everything is already committed to `main` under `docs/`.

1. Go to **github.com/prayasPradzzy/depth-wizard3D → Settings → Pages**
2. **Source:** Deploy from a branch
3. **Branch:** `main`, folder `/docs`
4. **Save**

Live in 1–2 minutes at:

```
https://prayaspradzzy.github.io/depth-wizard3D/
```

Page weight is **2.9 MB** on open. The GeoTIFF exports are larger but only transfer when someone clicks download.

### Rebuilding it after changes

```
python tools/build_static.py --out docs --full-app-url <hf-space-url>
git add docs && git commit -m "Rebuild static demo" && git push
```

It picks the newest job per source scene. Pin specific ones with `--jobs <id> <id>`.

---

## 2. Hugging Face Spaces — full app with upload

Free tier gives 2 vCPU and 16 GB RAM, which is enough. It sleeps after inactivity and takes ~30–60 s to wake, which is why the static build exists.

1. Create a free account at **huggingface.co**
2. **New Space** → SDK **Docker** → Hardware **CPU basic (free)** → name it `depthwizard`
3. Push:

```
git clone https://huggingface.co/spaces/<your-username>/depthwizard hf-space
cd hf-space
```

Copy in everything except the heavy extras:

```
api/  src/  web/  tools/  tests/  data/input/  data/benchmark.json
checkpoints/agl_vits.pt  requirements.txt  Dockerfile
```

Add a `README.md` at the root with this header — the Space will not start without it:

```yaml
---
title: DepthWizard
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
```

Then:

```
git add -A && git commit -m "DepthWizard" && git push
```

Hugging Face builds the Dockerfile automatically. First build is ~10–15 minutes.

> The checkpoint is 95 MB. If the push is rejected for file size, enable Git LFS
> (`git lfs track "*.pt"`) or omit it — the app falls back to the off-the-shelf model
> and simply reports relative heights instead of metres.

---

## 3. Running locally

```
docker compose up --build        # containerised
```

or

```
setup_laptop.bat                 # first time, ~10 min
start_demo.bat
```

Either way: **http://localhost:8000**, and wait for `Model pre-warmed` in the log.

Optional, for absolute sea-level elevation: a free key from
[portal.opentopography.org](https://portal.opentopography.org) in a `.env` file:

```
OPENTOPO_API_KEY=your_key_here
```

Without it, scenes are georeferenced but uncalibrated — heights above *ground* are
still reported in metres by the trained model, which is the number most analyses
actually want.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DW_USE_FINETUNED` | `true` | Serve the GAMUS-fine-tuned checkpoint |
| `DW_MESH_DIM` | `1024` | Web mesh and texture edge; `512` for faster loads |
| `DW_TILED` | `auto` | Tiled inference; `off` is faster for a live demo |
| `DW_PREWARM` | `true` | Load models at startup so the first upload is fast |
| `DW_DEM_TYPE` | `COP30` | Reference DEM: `COP30`, `SRTMGL1`, `NASADEM`, `AW3D30` |
| `DW_MAX_JOBS` | `25` | Processed scenes retained on disk |
| `OPENTOPO_API_KEY` | — | Enables absolute metric calibration |
