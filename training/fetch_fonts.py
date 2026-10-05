"""Download the training font set.

Sources (all freely licensed):
  * Google Fonts: the N most popular Latin families (--top 0 = the whole catalog), Regular weight
  * URW base-35 (Ghostscript): free clones of Helvetica, Times, Courier, Avant Garde, Palatino ...
  * DejaVu: Linux default UI fonts (macOS Menlo derives from DejaVu Sans Mono)
  * installed Windows fonts (common whitelist, or every text font with --all-system)
Writes data/fonts/manifest.json, which every other script reads.

    python training/fetch_fonts.py --top 0 --all-system --extras
"""
import io
import zipfile
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fontset import clone_pairs, glyph_signature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "data" / "fonts"
MANIFEST = FONT_DIR / "manifest.json"

# Families that are icon/emoji fonts or render no Latin glyphs.
SKIP_PATTERNS = re.compile(r"Material|Icons|Symbols|Emoji|Barcode|Redacted|Flow |Blank|Wavefont|Linefont", re.I)

# Windows fonts people actually see on screen every day.
WINDOWS_FONTS = [
    "Arial", "Times New Roman", "Calibri", "Cambria", "Segoe UI", "Verdana", "Tahoma",
    "Georgia", "Trebuchet MS", "Courier New", "Consolas", "Comic Sans MS", "Impact",
    "Candara", "Constantia", "Corbel", "Franklin Gothic Medium", "Lucida Console",
    "Palatino Linotype", "Segoe Print", "Segoe Script", "Gabriola", "Cascadia Code",
    "Cascadia Mono", "Sitka Text", "Bahnschrift", "Ink Free", "Lucida Sans Unicode",
    "Sylfaen", "MV Boli", "Microsoft Sans Serif", "Century Gothic", "Garamond",
    "Book Antiqua", "Bookman Old Style", "Century Schoolbook", "Aptos", "Gill Sans MT",
    "Rockwell", "Tw Cen MT", "Arial Narrow", "Arial Black", "Javanese Text", "Leelawadee UI",
]

# Metric-compatible clones that are visually near-identical: merge into one class.
ALIASES = {
    "Arimo": "Arial",
    "Tinos": "Times New Roman",
    "Cousine": "Courier New",
    "Carlito": "Calibri",
    "Caladea": "Cambria",
}


def fetch_metadata():
    text = requests.get("https://fonts.google.com/metadata/fonts", timeout=60).text
    return json.loads(text[text.index("{"):])["familyMetadataList"]


CJK_SCRIPTS = {"Hani", "Hans", "Hant", "Jpan", "Kore", "Hira", "Kana", "Hang"}


def pick_google_families(meta, top):
    """[(family, weight)] most popular first. top <= 0 -> the whole Latin catalog."""
    out = []
    for fam in sorted(meta, key=lambda f: f.get("popularity") or 1e9):
        name, script = fam["family"], fam.get("primaryScript")
        if SKIP_PATTERNS.search(name) or "latin" not in fam.get("subsets", []):
            continue
        # CJK fonts are huge and Latin is incidental; Noto per-script fonts reuse Noto Sans Latin.
        # Other non-Latin-primary families (Poppins, Kanit, Heebo...) have their own Latin design.
        if script in CJK_SCRIPTS or (script and name.startswith("Noto ")):
            continue
        weights = [int(k) for k in fam.get("fonts", {}) if k.isdigit()]
        if not weights:
            continue
        out.append((name, min(weights, key=lambda w: abs(w - 400))))
        if 0 < top <= len(out):
            break
    return out


def download_family(item):
    name, weight = item
    dest = FONT_DIR / "google" / (re.sub(r"[^A-Za-z0-9]+", "_", name) + ".ttf")
    if dest.exists() and dest.stat().st_size > 1000:
        return name, dest
    url = "https://fonts.googleapis.com/css2?family=" + name.replace(" ", "+") + f":wght@{weight}"
    try:
        css = requests.get(url, timeout=30).text  # no browser UA -> server returns TTF
        m = re.search(r"url\((https://[^)]+\.ttf)\)", css)
        if not m:
            return name, None
        data = requests.get(m.group(1), timeout=60).content
        dest.write_bytes(data)
        ImageFont.truetype(str(dest), 20)        # sanity: loadable
        return name, dest
    except Exception as e:  # noqa: BLE001
        print(f"  ! {name}: {e}", file=sys.stderr)
        return name, None


# Icon/dingbat fonts map pictures onto Latin codepoints, so they pass a glyph check.
SYMBOL_FONTS = re.compile(r"Symbol|Dings|Webdings|Marlett|MDL2|Fluent Icons|Emoji|MT Extra|Bookshelf|"
                          r"Holo|Assets|Math|Wingding|Outlook|Specialty", re.I)

# When pixel clones merge into one class, the class is named after the font people know.
PREFERRED = {"Segoe UI", "Palatino Linotype", "Verdana", "Arial", "Times New Roman", "Courier New",
             "Calibri", "Cambria"}


# Free clones of classic commercial fonts. Class name = the font people know.
URW = "https://raw.githubusercontent.com/ArtifexSoftware/urw-base35-fonts/master/fonts/"
EXTRAS = {
    "Helvetica / Nimbus Sans": URW + "NimbusSans-Regular.otf",
    "Helvetica Narrow / Nimbus Sans Narrow": URW + "NimbusSansNarrow-Regular.otf",
    "Times / Nimbus Roman": URW + "NimbusRoman-Regular.otf",
    "Courier / Nimbus Mono PS": URW + "NimbusMonoPS-Regular.otf",
    "ITC Avant Garde Gothic / URW Gothic": URW + "URWGothic-Book.otf",
    "ITC Bookman / URW Bookman": URW + "URWBookman-Light.otf",
    "New Century Schoolbook / C059": URW + "C059-Roman.otf",
    "Palatino / P052": URW + "P052-Roman.otf",
    "ITC Zapf Chancery / Z003": URW + "Z003-MediumItalic.otf",
}
DEJAVU_ZIP = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
DEJAVU = {"DejaVu Sans": "DejaVuSans.ttf", "DejaVu Serif": "DejaVuSerif.ttf", "DejaVu Sans Mono": "DejaVuSansMono.ttf"}


def fetch_extras():
    """Download the URW + DejaVu fonts -> {class display name: path}."""
    d = FONT_DIR / "extras"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in EXTRAS.items():
        dest = d / url.rsplit("/", 1)[1]
        if not dest.exists():
            dest.write_bytes(requests.get(url, timeout=60).content)
        out[name] = dest
    if not all((d / f).exists() for f in DEJAVU.values()):
        z = zipfile.ZipFile(io.BytesIO(requests.get(DEJAVU_ZIP, timeout=120).content))
        for member in z.namelist():
            if member.rsplit("/", 1)[-1] in DEJAVU.values():
                (d / member.rsplit("/", 1)[-1]).write_bytes(z.read(member))
    for name, f in DEJAVU.items():
        out[name] = d / f
    for name, p in out.items():
        ImageFont.truetype(str(p), 20)   # sanity: loadable
    return out


def find_windows_fonts(all_system=False):
    """Map family name -> Regular font file among installed fonts.
    Default: the WINDOWS_FONTS whitelist. all_system: every installed Latin text font."""
    dirs = [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts"]
    found = {}
    wanted = {w.lower(): w for w in WINDOWS_FONTS}
    for d in dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            try:
                family, style = ImageFont.truetype(str(f), 20).getname()
            except Exception:  # noqa: BLE001
                continue
            if style.lower() not in ("regular", "normal", "book", "roman"):
                continue
            key = family.lower()
            if key in wanted and wanted[key] not in found:
                found[wanted[key]] = f
            elif all_system and family not in found and not SYMBOL_FONTS.search(family):
                try:
                    if ImageFont.truetype(str(f), 20).getbbox("Hamburg")[2] > 10:
                        found[family] = f
                except Exception:  # noqa: BLE001
                    pass
    return found


def renders_ok(path):
    from PIL import Image, ImageDraw
    p = path if Path(path).is_absolute() else str(ROOT / path)
    try:
        for size in (11, 33, 96):
            f = ImageFont.truetype(p, size)
            b = f.getbbox("Hamburgefonstiv 0123")
            ImageDraw.Draw(Image.new("L", (b[2] + 4, b[3] + 4))).text((2, 2), "Hamburgefonstiv 0123", font=f, fill=255)
        return True
    except Exception:  # noqa: BLE001
        return False


def merge_clones(classes):
    """Merge classes whose Latin glyphs are pixel-identical (no model can separate them).
    Returns {kept class: [merged class names]}."""
    names = list(classes)
    sigs = [glyph_signature(p if Path(p).is_absolute() else str(ROOT / p)) for p in (classes[n][0] for n in names)]
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in clone_pairs(sigs):
        parent[find(a)] = find(b)
    groups = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(names[i])
    merged = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda n: (n not in PREFERRED, n.lower() not in {w.lower() for w in WINDOWS_FONTS}, n))
        keep = members[0]
        for n in members[1:]:
            classes[keep] += classes.pop(n)
        merged[keep] = members[1:]
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300, help="number of Google Fonts families")
    ap.add_argument("--user-dir", action="append", default=[],
                    help="folder of fonts you downloaded yourself (e.g. from dafont.com); repeatable")
    ap.add_argument("--extras", action="store_true", help="add URW base-35 + DejaVu free clones")
    ap.add_argument("--all-system", action="store_true",
                    help="train on every installed Latin text font, not just the common whitelist")
    args = ap.parse_args()

    (FONT_DIR / "google").mkdir(parents=True, exist_ok=True)
    print("Fetching Google Fonts metadata ...")
    families = pick_google_families(fetch_metadata(), args.top)
    print(f"Downloading {len(families)} families ...")
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(download_family, families))

    print(f"downloaded {sum(p is not None for _, p in results)}/{len(results)}; failed: "
          f"{[n for n, p in results if p is None][:20]}")
    # Popularity rankings drift daily: keep every family downloaded before so the
    # trained font set only grows (never silently drops fonts the model knew).
    have = {p for _, p in results if p is not None}
    for f in sorted((FONT_DIR / "google").glob("*.ttf")):
        if f not in have:
            try:
                ImageFont.truetype(str(f), 20)              # still loadable?
                results.append((f.stem.replace("_", " "), f))   # file was named from the API family
            except Exception:  # noqa: BLE001
                pass

    classes = {}  # class name -> list of files
    for name, path in results:
        if path is None:
            continue
        cls = ALIASES.get(name, name)
        classes.setdefault(cls, []).append(str(path.relative_to(ROOT)))

    if args.extras:
        for name, path in fetch_extras().items():
            classes.setdefault(name, []).append(str(path.relative_to(ROOT)))

    for d in args.user_dir:
        n0 = len(classes)
        for f in sorted(Path(d).rglob("*")):
            if f.suffix.lower() not in (".ttf", ".otf"):
                continue
            try:
                family, style = ImageFont.truetype(str(f), 20).getname()
                if style.lower() in ("regular", "normal", "book", "roman", "medium") and                         ImageFont.truetype(str(f), 20).getbbox("Hamburg")[2] > 10 and family not in classes:
                    classes[family] = [str(f.resolve())]
            except Exception:  # noqa: BLE001
                pass
        print(f"added {len(classes) - n0} fonts from {d}")

    win = find_windows_fonts(args.all_system)
    print(f"Found {len(win)} Windows system fonts")
    for name, path in win.items():
        classes.setdefault(name, []).append(str(path))

    # Drop fonts FreeType can't render (e.g. hinting bytecode errors) before they reach training.
    broken = [c for c in classes if not renders_ok(classes[c][0])]
    for c in broken:
        del classes[c]
    print(f"dropped {len(broken)} unrenderable fonts: {broken}")

    merged = merge_clones(classes)
    for keep, others in sorted(merged.items()):
        print(f"  merged pixel clones: {keep} = {' = '.join(others)}")

    # Display name shows metric-compatible aliases, e.g. "Arial / Arimo"; pixel clones are
    # listed separately ("twins") so the popup can say "same letters as ...".
    entries = []
    for cls in sorted(classes, key=str.lower):
        clones = [k for k, v in ALIASES.items() if v == cls and any(k.replace(" ", "_") in p for p in classes[cls])]
        entries.append({"name": cls, "display": " / ".join([cls] + clones), "files": classes[cls],
                        "twins": merged.get(cls, []),
                        "source": "windows" if any(p.startswith("C:") for p in classes[cls])
                        else "extra" if any("extras" in p for p in classes[cls])
                        else "google" if any("google" in p for p in classes[cls]) else "user"})

    MANIFEST.write_text(json.dumps({"classes": entries}, indent=1), encoding="utf8")
    print(f"Wrote {MANIFEST} with {len(entries)} classes")


if __name__ == "__main__":
    main()
