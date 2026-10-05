"""Torch-free batch producer for multiprocessing workers.

DataLoader workers on Windows each re-import PyTorch and map its CUDA DLLs
(~1.5 GB committed memory apiece), which exhausts the page file. These workers
only need numpy/OpenCV/Pillow, so they cost ~150 MB each instead.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import random  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from render import sample_patch  # noqa: E402

_PATHS = _BROWSER = None
_P_BROWSER = 0.25


def init_worker(class_paths, browser, p_browser):
    global _PATHS, _BROWSER, _P_BROWSER
    cv2.setNumThreads(1)
    _PATHS, _BROWSER, _P_BROWSER = class_paths, browser, p_browser


def make_batch(job):
    """job = ("train", seed, n) -> random classes, browser mix-in
             ("val", start, n)  -> deterministic, class = index % C
    Returns (uint8 patches (n,64,128), int64 labels (n,))."""
    kind, base, n = job
    C = len(_PATHS)
    xs = np.empty((n, 64, 128), np.uint8)
    ys = np.empty(n, np.int64)
    rng = random.Random(base)
    for i in range(n):
        if kind == "val":
            rng = random.Random(1234 * 1_000_003 + base + i)
            label = (base + i) % C
            png = None
        else:
            label = rng.randrange(C)
            png = rng.choice(_BROWSER[label]) if _BROWSER.get(label) and rng.random() < _P_BROWSER else None
        patch = sample_patch(rng.choice(_PATHS[label]), rng, browser_png=png)
        xs[i] = np.clip(patch * 255 + 0.5, 0, 255).astype(np.uint8)
        ys[i] = label
    return xs, ys
