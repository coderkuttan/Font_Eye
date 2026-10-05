"""Benchmark the exported model + index on real (non-Pillow) images.

Uses exactly the app's inference path (fonteye.recognizer).

    python training/evaluate.py                      # browser-rendered test split
    python training/evaluate.py --dir data/real_test # your own screenshots:
                                                     #   data/real_test/<Font Name>/*.png
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fonteye.recognizer import FontRecognizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fontset import safe_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--model-dir", default=None, help="evaluate another model folder, e.g. models/v1")
    args = ap.parse_args()

    rec = FontRecognizer(Path(args.model_dir)) if args.model_dir else FontRecognizer()
    classes = json.loads((ROOT / "data/fonts/manifest.json").read_text(encoding="utf8"))["classes"]
    items = []
    if args.dir:
        for sub in Path(args.dir).iterdir():
            if sub.is_dir():
                items += [(p, sub.name) for p in sub.glob("*.png")]
    else:
        by_folder = {safe_name(c["name"]): c["display"] for c in classes}
        for sub in (ROOT / "data/browser_renders/test").iterdir():
            if sub.name in by_folder:
                items += [(p, by_folder[sub.name]) for p in sub.glob("*.png")]

    top1 = top5 = 0
    confs = []
    confusions = collections.Counter()
    for p, truth in items:
        res = rec.identify(np.asarray(Image.open(p).convert("RGB")), k=5)
        t = truth.lower().split(" / ")[0]
        # a twin group counts as correct if the truth is any member of it
        match = [t in [x.lower().split(" / ")[0] for x in [r["name"]] + r.get("twins", [])] for r in res]
        confs.append(res[0]["prob"] if res else 0.0)
        top1 += bool(match and match[0])
        top5 += any(match)
        if res and not match[0]:
            confusions[(truth, res[0]["name"])] += 1
    n = len(items)
    print(f"{n} images | top-1 {top1 / n:.1%} | top-5 {top5 / n:.1%}")
    confs = np.array(confs)
    print(f"mean confidence of the top answer: {confs.mean():.1%}  |  snips at >=90% confidence: {(confs >= 0.9).mean():.1%}")
    print("most common confusions (truth -> predicted):")
    for (t, pr), c in confusions.most_common(15):
        print(f"  {c:3d}  {t}  ->  {pr}")


if __name__ == "__main__":
    main()
