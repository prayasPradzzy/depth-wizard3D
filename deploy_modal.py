"""
deploy_modal.py - Serve the DepthWizard inference API on Modal, free tier, no Docker.

WHY MODAL
=========
The API needs ~835 MB resident with both models loaded. That rules out every
512 MB free tier (Render, Koyeb, Fly). The options that fit are Hugging Face
Spaces, which wants a Dockerfile, and Modal, where the image is declared in
Python. Modal also scales to zero, so an idle demo costs nothing.

DEPLOY
======
    pip install modal
    modal setup                 # one-time browser login
    modal deploy deploy_modal.py

Modal prints a public https URL. Point the frontend at it:

    python tools/build_static.py --out docs --api <that-url> --full-app-url <that-url>
    git add docs && git commit -m "Point frontend at API" && git push

NOTES
=====
max_containers=1 is deliberate. A processed scene is written to the container's
local disk and fetched by the browser over the following seconds. With several
containers a follow-up request can land on one that never saw the job and 404.
One container keeps job state coherent, which is the right trade for a demo;
scaling it would mean moving job output to a shared Volume or object store.
"""

import modal

APP_NAME = "depthwizard"

# Build the image declaratively. Each step is cached, so edits near the bottom do
# not re-run the expensive torch install at the top.
image = (
    modal.Image.debian_slim(python_version="3.12")
    # rasterio bundles its own GDAL but still links these; without libexpat the
    # import dies at container start with a bare "libexpat.so.1: cannot open".
    .apt_install("libexpat1", "libgomp1")
    # Torch from the CPU index explicitly. Resolved from PyPI it pulls the CUDA
    # build: ~2.5 GB of wheels for hardware this function does not request.
    .pip_install(
        "torch", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": "/opt/hf", "PYTHONUNBUFFERED": "1"})
    # Bake the backbone weights in at build time. Downloading them on the first
    # request would make a cold start far worse and tie the service to network
    # access it should not need.
    .run_commands(
        "python -c \"from transformers import pipeline; "
        "pipeline(task='depth-estimation', "
        "model='depth-anything/Depth-Anything-V2-Small-hf', device=-1)\""
    )
    # copy=True bakes these into the image layer rather than mounting at runtime,
    # so a cold container starts with everything already present.
    .add_local_dir("api", "/root/api", copy=True)
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("web", "/root/web", copy=True)
    .add_local_dir("tools", "/root/tools", copy=True)
    .add_local_dir("tests", "/root/tests", copy=True)
    .add_local_dir("data/input", "/root/data/input", copy=True)
    .add_local_file("data/benchmark.json", "/root/data/benchmark.json", copy=True)
    # The fine-tuned checkpoint is what makes heights metric rather than relative.
    # Without it the app still runs, on the stock backbone, reporting relative units.
    .add_local_file("checkpoints/agl_vits.pt", "/root/checkpoints/agl_vits.pt", copy=True)
)

app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=2,
    memory=2048,          # measured peak is ~835 MB; this leaves real headroom
    timeout=600,          # a tiled inference on a large GeoTIFF can take minutes
    max_containers=1,     # see the note in the module docstring
    min_containers=0,     # scale to zero: an idle demo should cost nothing
    scaledown_window=300, # stay warm 5 min after the last request
)
@modal.asgi_app()
def api():
    """Mount the existing FastAPI application unchanged."""
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    # Demo-tuned: single-pass inference and models loaded at import, so the first
    # upload after a cold start is not also paying for lazy initialisation.
    os.environ.setdefault("DW_TILED", "off")
    os.environ.setdefault("DW_PREWARM", "true")
    os.environ.setdefault("DW_MESH_DIM", "768")
    os.environ.setdefault("DW_MAX_JOBS", "12")
    os.environ.setdefault("DW_USE_FINETUNED", "true")
    # The frontend is served from a different origin, so cross-origin is required.
    os.environ.setdefault("DW_ALLOWED_ORIGINS", "*")

    from api.main import app as fastapi_app
    return fastapi_app
