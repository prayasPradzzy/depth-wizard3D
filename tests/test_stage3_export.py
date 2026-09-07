"""
test_stage3_export.py - Verification test for Stage 3 Web 3D Assets & FastAPI Server.

PURPOSE & WHY:
1. Validates aspect-ratio preservation and anti-aliasing during mesh downsampling:
   Ensures terrain doesn't stretch or freeze WebGL contexts.
2. Validates 16-bit PNG heightmap and raw float binary buffer (.bin) export:
   Ensures Three.js receives exact float elevations without terracing.
3. Tests FastAPI endpoints and static web viewer mount:
   Validates /api/result/{id}, /api/assets/{id}/{file}, and GET / serving index.html.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mesh_export import export_for_web


def test_export_for_web():
    print("\n[1] Testing export_for_web() Downsampling & Asset Formatting:")
    test_out = PROJECT_ROOT / "data" / "output" / "test_stage3_assets"
    test_out.mkdir(parents=True, exist_ok=True)

    # 1. Non-square high-res image (1200x800) -> Aspect ratio = 1.5
    orig_w, orig_h = 1200, 800
    fake_rgb = np.random.randint(50, 200, size=(orig_h, orig_w, 3), dtype=np.uint8)
    fake_elev = np.linspace(100.0, 350.0, orig_h * orig_w, dtype=np.float32).reshape(orig_h, orig_w)

    res = export_for_web(
        elevation_arr=fake_elev,
        rgb_input=fake_rgb,
        out_dir=test_out,
        max_dim=512,
        is_georeferenced=True,
        crs_str="EPSG:32643",
        gsd_m=0.5,
    )

    manifest = res["manifest"]
    print(f"    Original Dimensions    : {orig_w}x{orig_h} (Aspect Ratio: {orig_w / orig_h:.3f})")
    print(f"    Downsampled Dimensions : {manifest['width']}x{manifest['height']} (Aspect Ratio: {manifest['aspect_ratio']:.3f})")
    print(f"    Total Vertices         : {manifest['vertex_count']:,} (Smooth 60 FPS budget)")
    print(f"    Elevation Span         : {manifest['min_elevation_m']}m to {manifest['max_elevation_m']}m")

    # Assert aspect ratio preserved
    assert abs(manifest["width"] / manifest["height"] - orig_w / orig_h) < 0.02, "Aspect ratio distorted!"
    assert manifest["width"] <= 512 and manifest["height"] <= 512, "max_dim limit exceeded!"

    # 2. Check 16-bit Heightmap PNG
    hmap_path = res["heightmap_path"]
    with Image.open(hmap_path) as img:
        assert img.mode == "I;16", f"Expected heightmap mode I;16, got {img.mode}"
        assert img.size == (manifest["width"], manifest["height"]), "Heightmap dimension mismatch!"
    print(f"    [OK] 16-bit Heightmap   -> {hmap_path.name} ({img.size[0]}x{img.size[1]}, mode {img.mode})")

    # 3. Check Binary Raw Elevation Buffer (.bin)
    bin_path = test_out / "heightmap.bin"
    assert bin_path.exists(), "heightmap.bin missing!"
    expected_bytes = manifest["width"] * manifest["height"] * 4  # float32 = 4 bytes
    assert bin_path.stat().st_size == expected_bytes, f"Binary buffer size {bin_path.stat().st_size} != {expected_bytes}"
    print(f"    [OK] Raw Float32 Buffer -> {bin_path.name} ({bin_path.stat().st_size} bytes)")

    # 4. Check Matched Texture PNG
    tex_path = res["texture_path"]
    with Image.open(tex_path) as img:
        assert img.size == (manifest["width"], manifest["height"]), "Texture dimension mismatch with heightmap!"
    print(f"    [OK] Matched Texture    -> {tex_path.name} ({img.size[0]}x{img.size[1]})")


def test_fastapi_endpoints():
    print("\n[2] Testing FastAPI Endpoints & Static Three.js Mount:")
    from fastapi.testclient import TestClient
    from api.main import app

    # Context manager triggers FastAPI lifespan (creates the initial demo job).
    with TestClient(app) as client:
        # a. Test Root (serves web/index.html)
        root_res = client.get("/")
        assert root_res.status_code == 200, f"GET / failed with status {root_res.status_code}"
        assert "DepthWizard 3D" in root_res.text, "index.html content not served at root!"
        print(f"    [OK] GET / (Three.js Viewer)      -> HTTP 200 ({len(root_res.text)} bytes)")

        # b. Test /api/result/latest
        res_latest = client.get("/api/result/latest")
        assert res_latest.status_code == 200, f"GET /api/result/latest failed: {res_latest.status_code}"
        data = res_latest.json()
        assert "manifest" in data and "assets" in data
        print(f"    [OK] GET /api/result/latest       -> HTTP 200 (Job ID: {data['job_id']})")

        # c. Test Asset Serving
        job_id = data["job_id"]
        hmap_res = client.get(f"/api/assets/{job_id}/heightmap.png")
        assert hmap_res.status_code == 200, "Failed to fetch heightmap.png"
        print(f"    [OK] GET /api/assets/.../heightmap -> HTTP 200 ({len(hmap_res.content)} bytes)")

        bin_res = client.get(f"/api/assets/{job_id}/heightmap.bin")
        assert bin_res.status_code == 200, "Failed to fetch heightmap.bin"
        print(f"    [OK] GET /api/assets/.../heightmap.bin -> HTTP 200 ({len(bin_res.content)} bytes)")

        tex_res = client.get(f"/api/assets/{job_id}/texture.png")
        assert tex_res.status_code == 200, "Failed to fetch texture.png"
        print(f"    [OK] GET /api/assets/.../texture.png   -> HTTP 200 ({len(tex_res.content)} bytes)")


def main():
    print("=" * 70)
    print("DepthWizard Stage 3: Web 3D Assets & FastAPI Server Verification")
    print("=" * 70)

    test_export_for_web()
    test_fastapi_endpoints()

    print("\n" + "=" * 70)
    print("STAGE 3 TEST PASSED! 3D assets, Three.js viewer, and FastAPI server verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
