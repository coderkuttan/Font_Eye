"""Font class manifest (torch-free so data scripts don't need PyTorch)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_classes():
    data = json.loads((ROOT / "data/fonts/manifest.json").read_text(encoding="utf8"))["classes"]
    for c in data:
        c["paths"] = [p if Path(p).is_absolute() else str(ROOT / p) for p in c["files"]]
    return data


def safe_name(name):
    """Filesystem-safe folder name for a class (browser renders are keyed by name, not index)."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


SIG_TEXT = "Hamburgefonstiv Quick 123"
CLONE_MSE = 0.004   # Segoe UI vs its Latin clones ~0.001-0.003; distinct designs > 0.02


def glyph_signature(path):
    """Fixed-canvas render of a reference string, used to detect pixel-identical clones."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (520, 44), 0)
    try:
        ImageDraw.Draw(img).text((4, 4), SIG_TEXT, font=ImageFont.truetype(path, 30), fill=255)
    except Exception:  # noqa: BLE001
        return None
    a = np.asarray(img, np.float32).ravel() / 255.0
    return a if a.sum() > 20 else None


def clone_pairs(sigs, thresh=CLONE_MSE):
    """Indices (i, j), i < j, whose signatures are pixel-identical within `thresh`."""
    import numpy as np
    ok = [i for i, s in enumerate(sigs) if s is not None]
    X = np.stack([sigs[i] for i in ok])
    sq = (X * X).sum(1)
    mse = (sq[:, None] + sq[None, :] - 2 * X @ X.T) / X.shape[1]
    return [(ok[a], ok[b]) for a, b in zip(*np.where(np.triu(mse < thresh, 1)))]
