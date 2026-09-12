"""
fetch_scene.py - Pull a georeferenced demo scene from Maxar Open Data.

WHY: the pipeline's georeferenced branch (metric DSM, lat/lon, GeoTIFF export)
cannot be demonstrated without imagery that actually carries a CRS. GAMUS tiles
are bare pixel arrays with no spatial metadata, so they cannot serve. Maxar Open
Data publishes sub-metre georeferenced imagery for disaster events with no
account required, which also matches this problem statement's theme.

Selection rejects cloud (bright + saturated) and empty fill, then ranks on edge
energy so the chosen window contains real structure or terrain relief rather
than blank water or haze.
"""
import argparse, os, sys
import numpy as np, requests, rasterio
from urllib.parse import urljoin
from rasterio.windows import Window
from rasterio.warp import transform_bounds
import scipy.ndimage as ndi

os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif'
ROOT = 'https://maxar-opendata.s3.amazonaws.com/events/'


def items_for(event, max_collections=8):
    base = ROOT + event + '/collection.json'
    kids = [urljoin(base, l['href'])
            for l in requests.get(base, timeout=30).json()['links'] if l.get('rel') == 'child']
    out = []
    for k in kids[:max_collections]:
        try:
            for l in requests.get(k, timeout=20).json()['links']:
                if l.get('rel') == 'item':
                    out.append(urljoin(k, l['href']))
        except Exception:
            pass
    return out


def pick_window(items, size, max_items, want_relief):
    best = None
    for iu in items[:max_items]:
        try:
            j = requests.get(iu, timeout=15).json()
            href = j['assets'].get('visual', {}).get('href')
            if not href:
                continue
            url = '/vsicurl/' + urljoin(iu, href)
            with rasterio.open(url) as src:
                T = 192
                th = src.read(1, out_shape=(T, T)).astype(np.float32)
                wt = max(2, int(round(size / src.width * T)))
                if wt >= T:
                    continue
                fill = np.cumsum(np.cumsum((th > 0).astype(np.float32), 0), 1)
                edge = np.hypot(ndi.sobel(th, 0), ndi.sobel(th, 1))
                ei = np.cumsum(np.cumsum(edge, 0), 1)
                bri = np.cumsum(np.cumsum(th, 0), 1)
                sat = np.cumsum(np.cumsum((th > 235).astype(np.float32), 0), 1)

                def box(I, r, c):
                    a = I[r + wt - 1, c + wt - 1]
                    if r: a -= I[r - 1, c + wt - 1]
                    if c: a -= I[r + wt - 1, c - 1]
                    if r and c: a += I[r - 1, c - 1]
                    return a

                # Cloud rejection. Brightness alone is not enough - thin haze sits in
                # the same range as bright ground. The reliable tell is that cloud has
                # almost no cast shadow, and cast shadow is the primary height cue in
                # nadir imagery. A measured comparison: a cloud-covered Nepal tile
                # scored edge energy 12 with 3.9% dark pixels, against 80-96 and
                # 19-33% for two usable scenes. So require real shadow and real edges.
                dark = np.cumsum(np.cumsum((th < 60).astype(np.float32), 0), 1)
                n = wt * wt
                for r in range(0, T - wt, 2):
                    for c in range(0, T - wt, 2):
                        if box(fill, r, c) / n < 0.995:
                            continue
                        mu, sf = box(bri, r, c) / n, box(sat, r, c) / n
                        if mu < 45 or mu > 160 or sf > 0.02:
                            continue
                        if box(dark, r, c) / n < 0.08:      # no shadows -> cloud/haze
                            continue
                        sc = box(ei, r, c) / n
                        if sc < 35:                          # too smooth to carry relief
                            continue
                        if best is None or sc > best[0]:
                            best = (sc, url, int(r / T * src.height), int(c / T * src.width),
                                    j['id'], src.width, src.height, mu)
        except Exception:
            continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=2048)
    ap.add_argument('--max-items', type=int, default=24)
    args = ap.parse_args()

    items = items_for(args.event)
    print(f'[{args.event}] items: {len(items)}', flush=True)
    best = pick_window(items, args.size, args.max_items, True)
    if not best:
        print('no clear window found', file=sys.stderr); sys.exit(1)
    sc, url, R, C, iid, W, H, mu = best
    R = min(max(0, R), H - args.size); C = min(max(0, C), W - args.size)
    print(f'chosen {iid} row={R} col={C} edges={sc:.1f} mean={mu:.0f}', flush=True)

    with rasterio.open(url) as src:
        win = Window(C, R, args.size, args.size)
        band = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(width=args.size, height=args.size, count=3, dtype='uint8',
                    transform=src.window_transform(win), compress='deflate',
                    tiled=True, blockxsize=512, blockysize=512)
        prof.pop('nodata', None); prof.pop('photometric', None)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with rasterio.open(args.out, 'w', **prof) as dst:
            dst.write(np.stack([band] * 3, axis=0))

    with rasterio.open(args.out) as t:
        b = transform_bounds(t.crs, 'EPSG:4326', *t.bounds)
        print(f'WROTE {args.out} ({os.path.getsize(args.out)//1024} KB)')
        print(f'  crs {t.crs} | {t.res[0]:.4f} m/px | {round(t.width*t.res[0])} m across')
        print(f'  centre {(b[1]+b[3])/2:.5f}, {(b[0]+b[2])/2:.5f}')


if __name__ == '__main__':
    main()
