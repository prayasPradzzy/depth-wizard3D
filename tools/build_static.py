"""
build_static.py - Produce a backend-free build of the viewer for GitHub Pages.

WHY THIS EXISTS
===============
The full application needs PyTorch and GDAL in a ~2.5 GB image, so free hosting for
it sleeps between visits and takes the better part of a minute to wake. A link
someone clicks once - from a CV, a post, a message - should open immediately.

Everything the viewer does *after* a scene is processed is client-side: the 3D
mesh, the flood screening, the cross-section, the benchmark panel. Only uploading a
new image needs the server. So this bakes already-processed scenes into flat files
and ships the viewer against those.

The viewer is not forked. It still calls the same /api/... paths; `api()` in
index.html rewrites them when window.__DW_STATIC__ is set. One copy to maintain.

USAGE
=====
    python tools/build_static.py --out docs --full-app-url https://...
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = PROJECT_ROOT / "data" / "web_jobs"

# Everything the viewer may request for a scene. Missing entries are skipped, so a
# scene processed before a given feature existed still builds.
ASSETS = [
    "heightmap.png", "heightmap.bin", "heightmap_sharp.bin", "texture.png",
    "ao.png", "ground.bin", "agl_model.bin", "depth_turbo.png",
    "elevation_metric.png", "error_map.png", "metric_dsm.tif", "relative_dsm.tif",
]


def scene_label(job_dir: Path, manifest: dict) -> str:
    src = next((c.name for c in job_dir.iterdir()
                if c.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                and not c.name.startswith(("heightmap", "texture", "ao", "error_map",
                                           "depth_turbo", "elevation_metric",
                                           "metric_dsm", "relative_dsm"))), None)
    return (Path(src).stem if src else job_dir.name.replace("job_", ""))[:22]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--jobs", nargs="*", default=None,
                    help="Job ids to include. Default: newest job per source scene.")
    ap.add_argument("--full-app-url", default="",
                    help="Where the uploadable version lives; shown in the dropzone.")
    args = ap.parse_args()

    out = PROJECT_ROOT / args.out
    static = out / "static"
    if out.exists():
        shutil.rmtree(out)
    static.mkdir(parents=True)

    # Pick scenes: newest job per distinct source image, so repeated reprocessing of
    # the same file does not ship three near-identical copies.
    candidates = sorted(
        (p for p in JOBS_DIR.iterdir() if p.is_dir() and (p / "manifest.json").exists()),
        key=lambda p: p.stat().st_mtime, reverse=True)
    chosen, seen = [], set()
    for p in candidates:
        if args.jobs and p.name not in args.jobs:
            continue
        try:
            m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        label = scene_label(p, m)
        if not args.jobs and label in seen:
            continue
        seen.add(label)
        chosen.append((p, m, label))
        if not args.jobs and len(chosen) >= 4:
            break
    if not chosen:
        print("No processed jobs found. Run the app and process a scene first.", file=sys.stderr)
        sys.exit(1)

    jobs_index, total = [], 0
    for job_dir, m, label in chosen:
        dest = static / job_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        assets = {}
        for name in ASSETS:
            src = job_dir / name
            if not src.exists():
                continue
            shutil.copy2(src, dest / name)
            total += src.stat().st_size
            assets[name] = f"/api/assets/{job_dir.name}/{name}"

        # Mirror the shape of GET /api/result/{id} exactly, so the viewer cannot tell
        # the difference between a served response and a baked one.
        (dest / "result.json").write_text(json.dumps({
            "job_id": job_dir.name,
            "manifest": m,
            "assets": {
                "heightmap": assets.get("heightmap.png"),
                "heightmap_bin": assets.get("heightmap.bin"),
                "heightmap_sharp_bin": assets.get("heightmap_sharp.bin", assets.get("heightmap.bin")),
                "texture": assets.get("texture.png"),
                "ao": assets.get("ao.png"),
                "ground_bin": assets.get("ground.bin"),
                "agl_model_bin": assets.get("agl_model.bin"),
                "depth_turbo": assets.get("depth_turbo.png"),
                "elevation_metric": assets.get("elevation_metric.png"),
                "error_map": assets.get("error_map.png"),
                "metric_dsm_tif": assets.get("metric_dsm.tif"),
                "relative_dsm_tif": assets.get("relative_dsm.tif"),
            },
        }, indent=1), encoding="utf-8")

        jobs_index.append({"job_id": job_dir.name, "label": label,
                           "georeferenced": bool(m.get("is_georeferenced")),
                           "size": f'{m.get("width")}x{m.get("height")}'})
        print(f"  {label:24s} {job_dir.name}  {len(assets)} assets")

    (static / "jobs.json").write_text(json.dumps({"jobs": jobs_index}, indent=1), encoding="utf-8")

    bm = PROJECT_ROOT / "data" / "benchmark.json"
    if bm.exists():
        shutil.copy2(bm, static / "benchmark.json")
        total += bm.stat().st_size

    # The newest scene also answers "latest", which is what the viewer asks for first.
    shutil.copytree(static / chosen[0][0].name, static / "latest", dirs_exist_ok=True)

    html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    inject = ("<script>window.__DW_STATIC__='static/';"
              f"window.__DW_FULL_APP__={json.dumps(args.full_app_url)};</script>\n")
    html = html.replace("<body>", "<body>\n" + inject, 1) if "<body>" in html else inject + html
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")   # keep Pages off _-prefixed paths

    print(f"\nbuilt {out}  ({len(chosen)} scenes, {total/1e6:.1f} MB)")
    print("GitHub Pages: Settings -> Pages -> Source: main branch, /docs folder")


if __name__ == "__main__":
    main()
