# Deploying DepthWizard

No containers. Frontend on **Vercel**, inference API on **Render**.

```
  Browser ──► Vercel (static frontend, free, global CDN)
                 │
                 └── POST /api/process ──► Render (FastAPI + PyTorch)
```

The frontend ships pre-processed scenes, so the 3D viewer, flood screening,
cross-section and benchmark panel all work **instantly and for free**, with no
backend involved. Render is only needed when someone uploads their own image.

---

## Read this before picking a Render plan

Measured peak resident memory of the API, with the depth model and the fine-tuned
checkpoint both loaded and a 768×768 inference running:

```
  ~835 MB
```

| Render plan | RAM | Works? |
|---|---|---|
| Free | 512 MB | **No** — killed on first inference |
| Starter ($7/mo) | 512 MB | **No** — same limit |
| **Standard ($25/mo)** | **2 GB** | **Yes**, with headroom |

Render's free and starter instances are both 512 MB, so neither can hold PyTorch.
This is not a configuration problem — it is what the model costs.

**You do not have to pay** - section 2 deploys the backend free on Modal. Render is kept below as a traditional always-on alternative. A third option is to deploy only the Vercel frontend: Everything except
upload works, which is most of the demo. Run the backend locally when you want to
show upload live.

Vercel cannot host the API at all: serverless functions cap at 250 MB deployment
size, and PyTorch alone exceeds that.

---

## 1. Frontend on Vercel — free, 3 minutes

The build is already committed in `docs/` (3.0 MB page load).

1. **vercel.com → Add New → Project → Import** your GitHub repo
2. Vercel reads `vercel.json`; leave the defaults:
   - Framework Preset: **Other**
   - Build Command: *(empty)*
   - Output Directory: **docs**
3. **Deploy**

Live at `https://<project>.vercel.app` in about a minute.

### Rebuilding after changes

```
python tools/build_static.py --out docs \
    --api https://<your-render-service>.onrender.com

git add docs && git commit -m "Rebuild frontend" && git push
```

Vercel redeploys on push. The `--api` value is baked into the page, so **rerun this
once you know your real Render URL** — the committed build currently points at the
predicted `https://depthwizard-api.onrender.com`.

Omit `--api` entirely to publish a read-only build with no upload.

---

## 2. Backend — free, on Modal

**This is the recommended option.** Modal's free credits cover a demo comfortably,
it scales to zero so idle costs nothing, and the image is declared in Python —
there is no Dockerfile and you never install Docker.

```
pip install modal
modal setup                       # one-time browser login
modal deploy deploy_modal.py
```

Modal prints a public `https://...modal.run` URL. First deploy takes ~10 minutes
while it builds the image and bakes in the model weights; later deploys reuse the
cached layers.

Then point the frontend at it and push:

```
python tools/build_static.py --out docs --api <modal-url> --full-app-url <modal-url>
git add docs && git commit -m "Point frontend at API" && git push
```

Vercel redeploys automatically. Upload now works end to end.

### What `deploy_modal.py` sets, and why

| Setting | Value | Reason |
|---|---|---|
| `memory` | 2048 MB | Measured peak is ~835 MB; this leaves real headroom |
| `max_containers` | 1 | A processed scene lives on the container's local disk and is fetched over the next few seconds. With several containers a follow-up request can hit one that never saw the job |
| `min_containers` | 0 | Scales to zero — an idle demo costs nothing |
| `scaledown_window` | 300 s | Stays warm 5 min after the last request |
| `DW_TILED` | `off` | Tiled inference costs ~20 s more per request |

Cold start after idle is roughly 20–40 s. Baked scenes still load instantly from
Vercel, so only upload pays that cost.

### Alternative: Hugging Face Spaces

Also genuinely free with 16 GB RAM, but the FastAPI path needs a Dockerfile.
Hugging Face builds it remotely — you never run Docker locally — so this is worth
considering if you prefer an always-on URL over scale-to-zero.

---

## 3. Backend on Render — paid alternative

`render.yaml` is a blueprint — Render reads it and configures everything.

1. **render.com → New → Blueprint**
2. Connect the repo and select it. Render finds `render.yaml`.
3. Confirm the plan is **Standard** (see the memory note above)
4. Set these two when prompted (both marked `sync: false`, so Render asks):
   - `DW_ALLOWED_ORIGINS` → your Vercel URL, e.g. `https://depthwizard.vercel.app`
   - `OPENTOPO_API_KEY` → optional; free from
     [portal.opentopography.org](https://portal.opentopography.org). Enables
     absolute sea-level calibration. Without it, heights above *ground* are still
     reported in metres by the trained model.
5. **Apply**

First build takes 10–15 minutes: it installs CPU PyTorch and prefetches the model
weights so the first request isn't slow.

Verify:

```
curl https://<your-service>.onrender.com/api/benchmark
```

Then rebuild the frontend with the real URL (step 1 above).

### What the blueprint sets, and why

| Setting | Value | Reason |
|---|---|---|
| `WEB_CONCURRENCY` | `1` | Each worker loads its own model copy; a second doubles memory for no gain on one CPU |
| `DW_TILED` | `off` | Tiled inference costs ~20 s more per request |
| `DW_PREWARM` | `true` | Models load at boot, not on the first user's request |
| `DW_MESH_DIM` | `768` | Smaller payload than 1024 over a metered connection |
| `DW_MAX_JOBS` | `12` | Render's disk is ephemeral; keep the working set small |
| `HF_HOME` | in-project | Weights cached during build, not downloaded at runtime |

Torch is installed from the CPU index explicitly. From PyPI it resolves to the CUDA
build — about 2.5 GB of wheels for GPU support no Render plan provides.

---

## 4. Running locally

```
setup_laptop.bat      # first time, ~10 min
start_demo.bat
```

→ **http://localhost:8000**, once the log says `Model pre-warmed`.

Or directly:

```
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DW_USE_FINETUNED` | `true` | Serve the GAMUS fine-tuned checkpoint |
| `DW_MESH_DIM` | `1024` | Mesh and texture edge; `512` loads fastest |
| `DW_TILED` | `auto` | Tiled inference; `off` is faster for a live demo |
| `DW_PREWARM` | `true` | Load models at startup |
| `DW_ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist |
| `DW_DEM_TYPE` | `COP30` | `COP30`, `SRTMGL1`, `NASADEM`, `AW3D30` |
| `DW_MAX_JOBS` | `25` | Processed scenes retained on disk |
| `PORT` | `8000` | Honoured automatically by Render |
| `OPENTOPO_API_KEY` | — | Enables absolute metric calibration |

---

## Troubleshooting

**Render build fails on torch** — the CUDA build was resolved instead of CPU. The
blueprint's `buildCommand` installs from the CPU index first; keep that ordering.

**Service restarts under load / "Ran out of memory"** — the instance is 512 MB.
Upgrade to Standard.

**Upload fails with a CORS error** — `DW_ALLOWED_ORIGINS` doesn't include your
Vercel domain. Preview deployments get their own subdomains, so either add them or
leave the value at `*`.

**Frontend loads but upload 404s** — the built page is pointing at the wrong
backend. Rebuild with `--api <real-url>` and push.

**First request after idle is slow** — Render spins services down on lower plans.
The baked scenes still load instantly because they come from Vercel.
