"""Run every metric on two images without ComfyUI.

    python tests/smoke.py [image_a image_b]

With no arguments two synthetic images are drawn, so the run only proves the
models load and the code paths agree with the libraries. Models download into
tests/.out on first use.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw  # noqa: E402

import metrics  # noqa: E402


def synthetic(seed: int) -> Image.Image:
    image = Image.new("RGB", (384, 512), (240 - seed * 10, 235, 250))
    draw = ImageDraw.Draw(image)
    draw.ellipse((120 + seed * 15, 60, 260 + seed * 15, 200), fill=(250, 220, 200), outline=(90, 60, 50), width=4)
    draw.rectangle((140 + seed * 15, 200, 240 + seed * 15, 420), fill=(90, 120, 200), outline=(40, 40, 80), width=4)
    draw.ellipse((150 + seed * 15, 100, 175 + seed * 15, 125), fill=(30, 30, 30))
    draw.ellipse((205 + seed * 15, 100, 230 + seed * 15, 125), fill=(30, 30, 30))
    return image


def main() -> int:
    if len(sys.argv) == 3:
        a = Image.open(sys.argv[1]).convert("RGB")
        b = Image.open(sys.argv[2]).convert("RGB")
    else:
        a, b = synthetic(0), synthetic(2)

    metrics.configure(cache_root=os.path.join(os.path.dirname(__file__), ".out"), device=os.environ.get("DEVICE", "cpu"))
    started = time.time()
    results = metrics.compute_all(a, b)
    print(metrics.to_json(results))
    print(f"took {time.time() - started:.1f}s")

    failed = [key for key in metrics.METRIC_KEYS if "error" in results.get(key, {})]
    for key in failed:
        print(f"{key}: {results[key]['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
