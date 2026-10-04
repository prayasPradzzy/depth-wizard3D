"""
prefetch_models.py - Pull model weights at build time, not at first request.

A cold request that has to download ~100 MB of weights before it can answer is a
bad first impression, and it makes the service depend on outbound network access
at run time. Render runs this during the build, so the weights are already on disk
in the image layer by the time the service starts.

Safe to run repeatedly: the HuggingFace cache is content-addressed.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))


def main() -> int:
    from transformers import pipeline
    print(f"[prefetch] HF_HOME={os.environ['HF_HOME']}")
    print("[prefetch] Depth Anything V2 Small ...", flush=True)
    pipeline(task="depth-estimation",
             model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    print("[prefetch] base model cached", flush=True)

    ck = ROOT / "checkpoints" / "agl_vits.pt"
    if ck.exists():
        size_mb = ck.stat().st_size / 1e6
        print(f"[prefetch] fine-tuned checkpoint present ({size_mb:.0f} MB)")
    else:
        # Not fatal: the service falls back to the stock backbone and reports
        # relative heights instead of metres.
        print("[prefetch] WARNING: checkpoints/agl_vits.pt missing - "
              "heights will be relative, not metric", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
