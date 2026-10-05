"""Image normalisation shared by training and the desktop app.

Any screen snip -> normalised text-line images (dark ink on white, fixed height)
-> fixed-size patches the model consumes. Keeping this in one place guarantees the
model sees identical inputs at train and inference time.
"""
import cv2
import numpy as np

LINE_H = 64          # every text line is rescaled to this height
PATCH_W = 128        # model input is LINE_H x PATCH_W
INNER_H = 56         # ink height inside the line (4 px margin top/bottom)


def to_gray01(img):
    """uint8 RGB/RGBA/gray -> float32 gray in [0,1]."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    return a.astype(np.float32) / 255.0


def normalise_polarity(g):
    """Make background white (1.0) and ink dark (0.0), stretching contrast."""
    # Background = most common intensity in a 3px frame: robust to a stray 1px page edge,
    # and unlike a whole-image mode it isn't fooled by big bold glyphs filling the snip.
    f = min(3, max(1, min(g.shape) // 4))
    frame = np.concatenate([g[:f].ravel(), g[-f:].ravel(), g[:, :f].ravel(), g[:, -f:].ravel()])
    hist, edges = np.histogram(frame, bins=32, range=(0.0, 1.0))
    b = int(np.argmax(hist))
    sel = frame[(frame >= edges[b]) & (frame <= edges[b + 1])]
    bg = float(np.median(sel))
    if bg < 0.5:                       # light text on dark background
        g = 1.0 - g
        bg = 1.0 - bg
    ink = float(np.percentile(g, 1))
    if bg - ink < 0.08:                # no contrast -> no text
        return None
    return np.clip((g - ink) / (bg - ink), 0.0, 1.0)


def split_lines(g, thresh=0.5, min_h=6):
    """Split a normalised image into tight per-line crops using row projection."""
    ink = g < thresh
    rows = ink.any(axis=1)
    bands, start = [], None
    for y, r in enumerate(list(rows) + [False]):
        if r and start is None:
            start = y
        elif not r and start is not None:
            bands.append([start, y])
            start = None
    # Merge small bands (i/j dots, accents, underscores) into a close neighbour.
    merged = True
    while merged and len(bands) > 1:
        merged = False
        for i, (a, b) in enumerate(bands):
            h = b - a
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(bands)]
            big = max(bands[j][1] - bands[j][0] for j in nbrs)
            if h < 0.5 * big:
                j = min(nbrs, key=lambda j: abs(bands[j][0] - a))
                gap = max(bands[j][0] - b, a - bands[j][1])
                if gap < 0.4 * big:
                    bands[j] = [min(a, bands[j][0]), max(b, bands[j][1])]
                    del bands[i]
                    merged = True
                    break
    lines = [(a, b) for a, b in bands if b - a >= min_h]
    out = []
    for y0, y1 in lines:
        band = ink[y0:y1]
        cols = np.where(band.any(axis=0))[0]
        if len(cols) == 0:
            continue
        out.append(g[y0:y1, cols[0]:cols[-1] + 1])
    return out


def rescale_line(line):
    """Tight line crop -> LINE_H tall image, white margins."""
    h, w = line.shape
    scale = INNER_H / h
    nw = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    r = np.clip(cv2.resize(line, (nw, INNER_H), interpolation=interp), 0, 1)
    pad = (LINE_H - INNER_H) // 2
    return cv2.copyMakeBorder(r, pad, LINE_H - INNER_H - pad, 4, 4, cv2.BORDER_CONSTANT, value=1.0)


def image_to_lines(img):
    """Full pipeline: snip -> list of normalised LINE_H-tall line images."""
    g = to_gray01(img)
    g = normalise_polarity(g)
    if g is None:
        return []
    return [rescale_line(l) for l in split_lines(g)]


def fit_patch(line, x0=0):
    """Cut one PATCH_W window starting at x0, padding with white if short."""
    patch = line[:, x0:x0 + PATCH_W]
    if patch.shape[1] < PATCH_W:
        extra = PATCH_W - patch.shape[1]
        patch = cv2.copyMakeBorder(patch, 0, 0, extra // 2, extra - extra // 2,
                                   cv2.BORDER_CONSTANT, value=1.0)
    return patch


def line_patches(line, stride=48, max_patches=24):
    """Sliding windows across a line for inference."""
    w = line.shape[1]
    if w <= PATCH_W:
        return [fit_patch(line)]
    xs = list(range(0, w - PATCH_W + 1, stride))
    if xs[-1] != w - PATCH_W:
        xs.append(w - PATCH_W)
    if len(xs) > max_patches:
        xs = [xs[i] for i in np.linspace(0, len(xs) - 1, max_patches).astype(int)]
    return [fit_patch(line, x) for x in xs]


def snip_to_batch(img, max_lines=6):
    """Snip -> (N,1,LINE_H,PATCH_W) float32 batch, or None if no text found."""
    lines = image_to_lines(img)
    if not lines:
        return None
    # Prefer the longest lines: they carry the most glyph evidence.
    lines = sorted(lines, key=lambda l: l.shape[1], reverse=True)[:max_lines]
    patches = [p for l in lines for p in line_patches(l)]
    return np.stack(patches)[:, None].astype(np.float32)
