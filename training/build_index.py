"""Build the font fingerprint index the app searches.

Each font gets a reference embedding = mean of many rendered samples. Because
matching is nearest-neighbour, fonts the model never trained on can be added here
(--system-fonts adds every installed Regular font) without retraining.

    python training/build_index.py [--system-fonts] [--extra-dir D:/myfonts]
"""
import argparse
import re
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fontset import ROOT, load_classes  # noqa: E402
from render import sample_patch  # noqa: E402


# Design-level twins (same letterforms at Regular weight). Pixel-identical clones,
# e.g. Windows script fonts that reuse Segoe UI's Latin glyphs, are found automatically.
TWINS = [
    ["Lexend", "Lexend Deca"],
    ["Cascadia Code", "Cascadia Mono"],
    ["Cormorant", "Cormorant Garamond", "Cormorant Infant"],
    ["Atkinson Hyperlegible", "Atkinson Hyperlegible Next"],
    ["Fira Code", "Fira Mono"],
    ["Palatino Linotype", "Book Antiqua"],
    ["Google Sans", "Google Sans Flex"],
    ["Roboto", "Roboto Flex"],
    ["Noto Sans", "Noto Sans Display"],
    ["Red Hat Display", "Red Hat Text"],
    ["Radio Canada", "Radio Canada Big"],
    ["Signika", "Signika Negative"],
    ["SimSun", "SimSun-ExtB", "SimSun-ExtG"],           # same Latin glyphs, spacing differs slightly
    ["Noto Sans", "Sans Serif Collection"],             # Windows SSC reuses Noto Sans Latin
]
# Script-companion families: every member shares one Latin design by construction.
FAMILY_RULES = [
    r"^Anek ",
    r"^Playpen Sans",
    r"^Baloo (.+ )?2$",
    r"^Mukta",
    r"^IBM Plex Sans( (Arabic|Thai|Thai Looped|Hebrew|Devanagari|KR|JP))?$",
    r"^Noto Sans(?! (Mono|Display)$)( .+)?$",       # Noto Sans <script> reuse Noto Sans Latin
    r"^Noto Serif(?! Display$)( .+)?$",
]

# When a group forms, report it under the name people actually know.
PREFERRED = {"Segoe UI", "Palatino Linotype", "Cascadia Code", "Verdana", "Roboto", "Noto Sans", "Lexend", "SimSun",
             "Cormorant Garamond", "Fira Code", "Google Sans", "Atkinson Hyperlegible", "Red Hat Display"}
SIG_TEXT = "Hamburgefonstiv Quick 123"
CLONE_MSE = 0.004   # Segoe UI vs its clones ~0.001-0.003; distinct designs > 0.02


def glyph_signature(path):
    """Fixed-canvas render of a reference string, used to detect pixel clones."""
    img = Image.new("L", (520, 44), 0)
    try:
        ImageDraw.Draw(img).text((4, 4), SIG_TEXT, font=ImageFont.truetype(path, 30), fill=255)
    except Exception:  # noqa: BLE001
        return None
    return np.asarray(img, np.float32).ravel() / 255.0


def assign_groups(entries):
    """Union manual twins + pixel clones; each group reports under one primary name."""
    n = len(entries)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    base = {e["name"].split(" / ")[0]: i for i, e in enumerate(entries)}
    learned = ROOT / "data/fonts/confusion_twins.json"      # from find_twins.py (calib split)
    extra = json.loads(learned.read_text(encoding="utf8"))["pairs"] if learned.exists() else []
    for rule in FAMILY_RULES:
        members = [n for n in base if re.match(rule, n)]
        extra = extra + [members] if len(members) > 1 else extra
    for twins in TWINS + extra:
        idx = [base[t] for t in twins if t in base]
        for i in idx[1:]:
            union(idx[0], i)

    sigs = [glyph_signature(e["path"]) for e in entries]
    ok = [i for i, s in enumerate(sigs) if s is not None and s.sum() > 20]
    X = np.stack([sigs[i] for i in ok])
    sq = (X * X).sum(1)
    mse = (sq[:, None] + sq[None, :] - 2 * X @ X.T) / X.shape[1]
    for a, b in zip(*np.where(np.triu(mse < CLONE_MSE, 1))):
        union(ok[a], ok[b])

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        # primary = a well-known name if present, then trained fonts, then first listed
        members.sort(key=lambda i: (entries[i]["name"] not in PREFERRED, not entries[i]["trained"], i))
        primary = entries[members[0]]["name"]
        names = [entries[i]["name"] for i in members]
        for i in members:
            entries[i]["group"] = primary
            entries[i]["twins"] = [nm for nm in names if nm != primary]
    # Pixel clones merged into one training class (fetch_fonts --all-system) are twins too.
    for e in entries:
        merged = e.pop("merged", [])
        if merged:
            e.setdefault("group", e["name"])
            e["twins"] = e.get("twins", []) + [m for m in merged if m not in e.get("twins", [])]
    print(f"twin groups: {sum(len(m) > 1 for m in groups.values())}")
    for m in groups.values():
        if len(m) > 1:
            print("   ", " = ".join(entries[i]["name"] for i in m))


# Icon/dingbat fonts map pictures onto Latin codepoints, so they pass a glyph check.
SYMBOL_FONTS = re.compile(r"Symbol|Dings|Webdings|Marlett|MDL2|Fluent Icons|Emoji|MT Extra|Bookshelf|"
                          r"Holo|Assets|Math|Wingding|Outlook|Specialty", re.I)


def portable_path(p):
    """Store font paths without machine-specific parts: repo-relative when inside the
    project, '~/...' under the user's home folder, else as is (e.g. C:/Windows/Fonts)."""
    p = Path(p).resolve()
    for base, prefix in ((ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + p.relative_to(base).as_posix()
        except ValueError:
            pass
    return p.as_posix()


def scan_dir(d):
    out = {}
    for f in Path(d).rglob("*"):
        if f.suffix.lower() not in (".ttf", ".otf", ".ttc"):
            continue
        try:
            family, style = ImageFont.truetype(str(f), 20).getname()
        except Exception:  # noqa: BLE001
            continue
        if SYMBOL_FONTS.search(family):
            continue
        if style.lower() in ("regular", "normal", "book", "roman") and family not in out:
            try:   # skip symbol / non-Latin fonts and files FreeType can't render
                if ImageFont.truetype(str(f), 20).getbbox("Hamburg")[2] > 10:
                    out[family] = str(f)
            except Exception:  # noqa: BLE001
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--system-fonts", action="store_true", help="also index all installed Regular fonts")
    ap.add_argument("--extra-dir", action="append", default=[], help="folder of extra fonts to index")
    args = ap.parse_args()

    entries = [{"name": c["display"], "path": c["paths"][0], "trained": True, "merged": c.get("twins", [])}
               for c in load_classes()]
    known = {e["name"].split(" / ")[0].lower() for e in entries}
    extra = {}
    if args.system_fonts:
        extra.update(scan_dir("C:/Windows/Fonts"))
        extra.update(scan_dir(Path.home() / "AppData/Local/Microsoft/Windows/Fonts"))
    for d in args.extra_dir:
        extra.update(scan_dir(d))
    for fam, path in sorted(extra.items()):
        if fam.lower() not in known:
            entries.append({"name": fam, "path": path, "trained": False})
            known.add(fam.lower())

    assign_groups(entries)
    sess = ort.InferenceSession(str(ROOT / "models/fonteye.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    embs = []
    for i, e in enumerate(entries):
        rng = random.Random(i)
        batch = np.stack([sample_patch(e["path"], rng, aug=0.5) for _ in range(args.samples)])[:, None]
        v = sess.run(None, {"patches": batch.astype(np.float32)})[0].mean(0)
        embs.append(v / np.linalg.norm(v))
        if i % 50 == 0:
            print(f"  {i}/{len(entries)} {e['name']}")
    np.save(ROOT / "models/index.npy", np.stack(embs).astype(np.float32))
    for e in entries:
        e["path"] = portable_path(e["path"])
    (ROOT / "models/index.json").write_text(json.dumps(entries, indent=1), encoding="utf8")
    n_extra = sum(not e["trained"] for e in entries)
    print(f"indexed {len(entries)} fonts ({n_extra} zero-shot extras)")


if __name__ == "__main__":
    main()
