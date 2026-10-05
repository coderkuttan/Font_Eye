"""Synthetic 'screenshot of text' generator used for training and indexing."""
import io
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fonteye.preprocess import PATCH_W, image_to_lines  # noqa: E402

WORDS = """the be to of and a in that have it for not on with he as you do at this but his by from they we
say her she or an will my one all would there their what so up out if about who get which go me when make can
like time no just him know take people into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even new want because any these give
day most us great world small large every found still between never under might while last next always
quick brown fox jumps lazy dog sphinx black quartz judge vow pack box five dozen liquor jugs waltz bad nymph
vex design typography modern classic elegant bold light regular italic serif sans mono display heading
system window screen button menu settings profile account search download upload share open close save cancel
privacy policy terms service contact about home news blog shop cart checkout price free premium welcome
monday january february march april june july august september october november december weekend
chemistry physics history science research project report summary analysis result chapter section page
coffee garden mountain river ocean forest island village city street market music movie travel photo
Hamburgefonstiv Quixotic Zephyr Jazz Wizard Rhythm Galaxy Vortex Kingdom Puzzle Oxygen Aquarium""".split()


def random_text(rng):
    r = rng.random()
    if r < 0.08:   # random letters stress glyph shapes rather than word familiarity
        alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(rng.choice(alpha) for _ in range(rng.randint(5, 12)))
    if r < 0.12:
        return "".join(rng.choice("0123456789$%.,:-/ ") for _ in range(rng.randint(5, 10)))
    words = [rng.choice(WORDS) for _ in range(rng.randint(1, 4))]
    text = " ".join(words)
    case = rng.random()
    if case < 0.30:
        text = text.title()
    elif case < 0.42:
        text = text.upper()
    elif case < 0.55:
        text = text.capitalize()
    if rng.random() < 0.15:
        text += rng.choice([".", ",", "!", "?", " 2026", " 42", ":", " &"])
    if len(text) < 5:
        text += " " + rng.choice(WORDS)
    return text


@lru_cache(maxsize=1024)
def load_font(path, size):
    return ImageFont.truetype(path, size)


def _text_mask(path, size, text):
    font = load_font(path, size)
    x0, y0, x1, y1 = font.getbbox(text)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    m = Image.new("L", (w + 4, h + 4), 0)
    ImageDraw.Draw(m).text((2 - x0, 2 - y0), text, font=font, fill=255)
    return np.asarray(m, dtype=np.float32) / 255.0


def _subpixel_coverage(path, size, text, rng):
    """Approximate ClearType: render 3x, keep horizontal subpixels as RGB."""
    big = _text_mask(path, size * 3, text)
    h3, w3 = big.shape
    h, w = max(1, h3 // 3), max(1, w3 // 3)
    m = cv2.resize(big, (w * 3, h), interpolation=cv2.INTER_AREA)
    m = cv2.filter2D(m, -1, np.array([[1, 2, 3, 2, 1]], np.float32) / 9.0)
    cov = m.reshape(h, w, 3)
    return cov[..., ::-1] if rng.random() < 0.15 else cov   # BGR panels occasionally


def _colors(rng):
    if rng.random() < 0.72:   # dark text on light background
        bg_l, fg_l = rng.uniform(0.80, 1.0), rng.uniform(0.0, 0.35)
    else:
        bg_l, fg_l = rng.uniform(0.0, 0.25), rng.uniform(0.70, 1.0)

    def tint(l, amount):
        c = np.array([l] * 3) + np.array([rng.uniform(-amount, amount) for _ in range(3)])
        return np.clip(c, 0, 1)
    return tint(fg_l, 0.25), tint(bg_l, 0.12)


def _background(h, w, bg, rng):
    img = np.ones((h, w, 3), np.float32) * bg
    r = rng.random()
    if r < 0.18:   # soft gradient
        bg2 = np.clip(bg + rng.uniform(-0.12, 0.12), 0, 1)
        t = np.linspace(0, 1, w if rng.random() < 0.5 else h, dtype=np.float32)
        t = t[None, :, None] if t.shape[0] == w else t[:, None, None]
        img = img * (1 - t) + bg2 * t
    elif r < 0.28:  # paper / texture noise
        n = cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.05, (h, w)).astype(np.float32), (0, 0), 2)
        img = img + n[..., None]
    return np.clip(img, 0, 1)


def augment(img, rng, strength=1.0):
    """Degradations seen in real screenshots. img: uint8 RGB."""
    h, w = img.shape[:2]
    if rng.random() < 0.15 * strength:
        ang = rng.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
    if rng.random() < 0.30 * strength:
        f = rng.uniform(0.55, 0.9)
        small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if rng.random() < 0.20 * strength:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.8))
    if rng.random() < 0.20 * strength:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 7), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.30 * strength:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=rng.randint(45, 92))
        img = np.asarray(Image.open(buf).convert("RGB"))
    return img


def render_screenshot(path, rng, text=None, size=None):
    """Render one text line as if it were a crop of a screen. Returns uint8 RGB."""
    text = text or random_text(rng)
    size = size or int(round(math.exp(rng.uniform(math.log(11), math.log(72)))))
    fg, bg = _colors(rng)
    if rng.random() < 0.35 and size <= 40:
        cov = _subpixel_coverage(path, size, text, rng)
    else:
        cov = _text_mask(path, size, text)[..., None]
    h, w = cov.shape[:2]
    pt, pb, pl, pr = (rng.randint(2, 4 + size // 3) for _ in range(4))
    canvas = _background(h + pt + pb, w + pl + pr, bg, rng)
    region = canvas[pt:pt + h, pl:pl + w]
    canvas[pt:pt + h, pl:pl + w] = region * (1 - cov) + fg * cov
    return (canvas * 255).astype(np.uint8)


def longest_line(img):
    lines = image_to_lines(img)
    return max(lines, key=lambda l: l.shape[1]) if lines else None


def random_patch(line, rng):
    from fonteye.preprocess import fit_patch
    w = line.shape[1]
    x0 = rng.randint(0, w - PATCH_W) if w > PATCH_W else 0
    return fit_patch(line, x0)


def sample_patch(path, rng, browser_png=None, aug=1.0):
    """One training patch (LINE_H x PATCH_W float32) for the given font file."""
    for _ in range(6):
        try:
            if browser_png is not None:
                img = np.asarray(Image.open(browser_png).convert("RGB"))
                img = augment(img, rng, 0.6 * aug)
            else:
                img = augment(render_screenshot(path, rng), rng, aug)
        except OSError:
            # some fonts' hinting bytecode fails at certain sizes in FreeType
            # ("execution context too long"); retry with another size/text
            browser_png = None
            continue
        line = longest_line(img)
        if line is not None and line.shape[1] >= 24:
            return random_patch(line, rng)
        browser_png = None   # fall back to synthetic if a browser render was unusable
    return np.ones((64, PATCH_W), np.float32)


if __name__ == "__main__":
    # Quick visual check: python training/render.py <font.ttf>  -> writes preview grid
    import json
    root = Path(__file__).resolve().parents[1]
    classes = json.loads((root / "data/fonts/manifest.json").read_text(encoding="utf8"))["classes"]
    rng = random.Random(0)
    rows = []
    for c in rng.sample(classes, 8):
        p = c["files"][0]
        p = p if Path(p).is_absolute() else str(root / p)
        rows.append(np.concatenate([sample_patch(p, rng) for _ in range(6)], axis=1))
    grid = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    out = root / "data" / "preview_patches.png"
    Image.fromarray(grid).save(out)
    print("wrote", out)
