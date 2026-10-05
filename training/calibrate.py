"""Fit the confidence scale so the popup's % means what it says.

The popup shows softmax(scale * cosine). ArcFace trained with scale 30, but averaging
patch embeddings over a whole snip changes the cosine distribution, so 30 is not
calibrated. This fits `scale` on held-out real Edge renders (data/browser_renders/calib)
by minimising negative log-likelihood, then reports how well "X% confident" matches
"right X% of the time" (expected calibration error).

    python training/calibrate.py        -> models/calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fonteye.recognizer import MODEL_DIR, FontRecognizer  # noqa: E402
from fontset import ROOT, load_classes, safe_name  # noqa: E402


def group_logits(sims, scale, groups):
    """Softmax over twin *groups* (as the popup reports them), not raw index entries."""
    z = scale * sims
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p @ groups                      # (N, G)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return err


def main():
    rec = FontRecognizer()
    names = [e["name"] for e in rec.entries]
    gkey = [e.get("group") or e["name"] for e in rec.entries]
    gnames = sorted(set(gkey))
    G = np.zeros((len(names), len(gnames)), np.float32)
    for i, g in enumerate(gkey):
        G[i, gnames.index(g)] = 1
    entry_of = {n.split(" / ")[0]: i for i, n in enumerate(names)}

    sims, labels = [], []
    for c in load_classes():
        i = entry_of.get(c["display"].split(" / ")[0])
        d = ROOT / "data/browser_renders/calib" / safe_name(c["name"])
        if i is None or not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")):
            v = rec.embed(np.asarray(Image.open(p).convert("RGB")))
            if v is not None:
                sims.append(rec.index @ v)
                labels.append(gnames.index(gkey[i]))
    sims, labels = np.stack(sims), np.array(labels)
    print(f"{len(labels)} calibration images")

    best = None
    for scale in np.arange(5, 120.1, 0.5):
        P = group_logits(sims, scale, G)
        nll = -np.log(P[np.arange(len(labels)), labels] + 1e-12).mean()
        if best is None or nll < best[1]:
            best = (float(scale), float(nll))
    scale = best[0]
    for s in (30.0, scale):
        P = group_logits(sims, s, G)
        conf, pred = P.max(1), P.argmax(1)
        correct = (pred == labels).astype(float)
        print(f"scale {s:5.1f}: accuracy {correct.mean():.1%}  mean confidence {conf.mean():.1%}  "
              f"calibration error {ece(conf, correct):.3f}")
    # confidence buckets with the fitted scale, as the user will see them
    P = group_logits(sims, scale, G)
    conf, correct = P.max(1), (P.argmax(1) == labels)
    print("when the popup says ...")
    for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"   {lo:.0%}-{min(hi, 1):.0%}: {m.mean():5.1%} of snips, right {correct[m].mean():.1%} of the time")
    (MODEL_DIR / "calibration.json").write_text(json.dumps({"scale": scale}), encoding="utf8")
    print(f"wrote {MODEL_DIR / 'calibration.json'} (scale = {scale})")


if __name__ == "__main__":
    main()
