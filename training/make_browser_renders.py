"""Render real browser text (Microsoft Edge via Playwright) for every font class.

Real browser rasterisation (hinting, anti-aliasing, DPI scaling) differs from
Pillow's, so these images are used two ways:
  * data/browser_renders/train  -> mixed into training
  * data/browser_renders/test   -> held-out "real screenshot" benchmark (evaluate.py)
  * data/browser_renders/calib  -> held-out set for fitting the confidence scale (calibrate.py)

    python training/make_browser_renders.py --train 24 --test 6
"""
import argparse
import io
import html
import random
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fontset import ROOT, load_classes, safe_name  # noqa: E402
from render import random_text  # noqa: E402

OUT = ROOT / "data" / "browser_renders"
HOST = "http://fonteye.local"

SENTENCES = [
    "Sign in to continue", "Privacy and security settings", "The quick brown fox jumps",
    "Download the latest version", "Add to cart", "Read more about our story", "Terms of Service",
    "Breaking news this morning", "Quarterly Report 2026", "Welcome back, Alex", "Subscribe for updates",
    "Frequently Asked Questions", "Chapter 3: Reaction Kinetics", "Open in new window", "Pricing plans",
    "Contact our support team", "Free shipping over $50", "Upcoming Events", "Search results for fonts",
]


def page_html(font_url, samples):
    divs = []
    for k, s in enumerate(samples):
        style = (f"font-size:{s['size']}px;color:{s['fg']};background:{s['bg']};"
                 f"padding:{s['pad']}px {s['pad'] + 3}px;display:inline-block;white-space:nowrap;"
                 f"{'font-weight:normal;'}{'letter-spacing:0.02em;' if s['track'] else ''}")
        divs.append(f'<div id="s{k}" style="{style}">{html.escape(s["text"])}</div><br>')
    return f"""<!doctype html><html><head><style>
@font-face {{ font-family: 'Target'; src: url('{font_url}'); }}
body {{ margin: 8px; background: #fff; font-family: 'Target'; }}
</style></head><body>{''.join(divs)}</body></html>"""


def random_sample(rng):
    dark_on_light = rng.random() < 0.72
    l_bg = rng.randint(215, 255) if dark_on_light else rng.randint(0, 50)
    l_fg = rng.randint(0, 80) if dark_on_light else rng.randint(190, 255)
    tint = lambda l: "#%02x%02x%02x" % tuple(max(0, min(255, l + rng.randint(-25, 25))) for _ in range(3))
    text = rng.choice(SENTENCES) if rng.random() < 0.4 else random_text(rng)
    return {"text": text, "size": rng.choice([11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48]),
            "fg": tint(l_fg), "bg": tint(l_bg), "pad": rng.randint(3, 12), "track": rng.random() < 0.05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=24, help="renders per class for training")
    ap.add_argument("--test", type=int, default=6, help="renders per class for the benchmark")
    ap.add_argument("--calib", type=int, default=0, help="renders per class for confidence calibration")
    ap.add_argument("--shard", default="0/1", help="i/n: render every n-th class starting at i (run in parallel)")
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))

    classes = load_classes()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        contexts = {dsf: browser.new_context(device_scale_factor=dsf) for dsf in (1.0, 1.25, 1.5)}
        state = {}

        def handler(route):
            url = route.request.url
            if url.endswith("/page.html"):
                route.fulfill(body=state["html"], content_type="text/html")
            elif "/font/" in url:
                route.fulfill(body=Path(state["font"]).read_bytes(), content_type="font/ttf")
            else:
                route.abort()

        pages = {}
        for dsf, ctx in contexts.items():
            p = ctx.new_page()
            p.route(f"{HOST}/**", handler)
            pages[dsf] = p

        for ci, c in enumerate(classes):
            if ci % shard_n != shard_i:
                continue
            for split, n, seed in (("train", args.train, 1), ("test", args.test, 2), ("calib", args.calib, 3)):
                d = OUT / split / safe_name(c["name"])
                if n == 0 or (d.exists() and len(list(d.glob("*.png"))) >= n):
                    continue
                d.mkdir(parents=True, exist_ok=True)
                rng = random.Random(seed * 100_000 + ci)
                samples = [random_sample(rng) for _ in range(n)]
                k = 0
                for dsf, page in pages.items():
                    chunk = samples[k::len(pages)]
                    if not chunk:
                        continue
                    state["font"] = rng.choice(c["paths"])
                    state["html"] = page_html(f"{HOST}/font/{ci}-{rng.randrange(1 << 30)}.ttf", chunk)
                    page.goto(f"{HOST}/page.html")
                    ok = page.evaluate("async () => { await document.fonts.ready;"
                                       " return [...document.fonts].some(f => f.status === 'loaded'); }")
                    if not ok:   # never save fallback-font renders under this label
                        print(f"  ! font failed to load in browser: {c['name']}")
                        k += 1
                        continue
                    # One full-page capture, then crop each sample (much faster than per-element shots)
                    boxes = page.evaluate(f"""() => [...Array({len(chunk)}).keys()].map(j => {{
                        const r = document.getElementById('s' + j).getBoundingClientRect();
                        return [r.left, r.top, r.right, r.bottom]; }})""")
                    shot = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
                    for j, (x0, y0, x1, y1) in enumerate(boxes):
                        box = tuple(int(round(v * dsf)) for v in (x0, y0, x1, y1))
                        shot.crop(box).save(d / f"{dsf}_{j}.png")
                    k += 1
            if ci % 25 == 0:
                print(f"  {ci}/{len(classes)} {c['name']}", flush=True)
        browser.close()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
