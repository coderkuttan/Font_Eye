"""Learn twin groups from the model's own confusions on the calibration split.

Many Google families share Latin letters without being pixel clones: script
variants of one family (Baloo 2 / Baloo Paaji 2 / Baloo Tamma 2, Anek *, Mukta *)
and Indic/Khmer/Arabic fonts that borrow their Latin from another family
(Khula = Open Sans Latin, Mada = Source Sans, Lilex = IBM Plex Mono ...). For a
Latin snip they are indistinguishable, so results should report them together.

Pairs the current model confuses on >= MIN_CONFUSED of their 8 calibration images
become twins. Only the calib split is used, so the test benchmark stays unbiased.

    python training/find_twins.py      -> data/fonts/confusion_twins.json
    (then rebuild: build_index.py, calibrate.py)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fonteye.recognizer import FontRecognizer  # noqa: E402
from fontset import ROOT, load_classes, safe_name  # noqa: E402

MIN_CONFUSED = 3
# Pairs that ARE visibly different despite being confused (keep them apart).
EXCLUDE = {frozenset({"Nunito", "Nunito Sans"})}
OUT = ROOT / "data/fonts/confusion_twins.json"


def main():
    rec = FontRecognizer()
    conf = collections.Counter()
    for c in load_classes():
        d = ROOT / "data/browser_renders/calib" / safe_name(c["name"])
        truth = c["display"].split(" / ")[0]
        for p in sorted(d.glob("*.png")):
            res = rec.identify(np.asarray(Image.open(p).convert("RGB")), k=1)
            if res and truth not in [res[0]["name"].split(" / ")[0]] + res[0].get("twins", []):
                conf[tuple(sorted((truth, res[0]["name"].split(" / ")[0])))] += 1
    pairs = [list(p) for p, n in conf.most_common()
             if n >= MIN_CONFUSED and frozenset(p) not in EXCLUDE]
    OUT.write_text(json.dumps({"min_confused": MIN_CONFUSED, "pairs": pairs}, indent=1), encoding="utf8")
    print(f"{len(pairs)} twin pairs -> {OUT}")
    for a, b in pairs:
        print(f"   {a}  <->  {b}")


if __name__ == "__main__":
    main()
