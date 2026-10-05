"""End-to-end test on the real screen.

Opens real windows with known fonts (Notepad -> Consolas, an Edge page -> Georgia +
Montserrat), drives FontEye's snip overlay with synthetic mouse events over the
text, and saves a screenshot of each result popup to tests/out/.

    python tests/e2e_screen_test.py
"""
import base64
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ctypes.windll.shcore.SetProcessDpiAwareness(2)

import mss  # noqa: E402
import uiautomation as auto  # noqa: E402
from PIL import Image  # noqa: E402

from fonteye.app import FontEyeApp  # noqa: E402

OUT = ROOT / "tests" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def grab(x, y, w, h):
    with getattr(mss, "MSS", mss.mss)() as s:
        g = s.grab({"left": x, "top": y, "width": w, "height": h})
    return Image.frombytes("RGB", g.size, g.bgra, "raw", "BGRX")


def open_notepad():
    f = Path(tempfile.gettempdir()) / "fonteye_e2e.txt"
    f.write_text("Typography test: the quick brown fox jumps over the lazy dog 0123456789\n", encoding="utf8")
    proc = subprocess.Popen(["notepad.exe", str(f)])
    time.sleep(2.5)
    win = auto.WindowControl(searchDepth=1, SubName="fonteye_e2e")
    win.SetActive()
    win.Maximize()
    time.sleep(1.0)
    doc = win.DocumentControl(searchDepth=12)
    if not doc.Exists(1):
        doc = win.EditControl(searchDepth=12)
    r = doc.GetPattern(auto.PatternId.TextPattern).DocumentRange.GetBoundingRectangles()[0]
    return win, (r.left, r.top - 4, r.right, r.bottom + 4), "Consolas"


def open_edge():
    b64 = lambda rel: base64.b64encode((ROOT / rel).read_bytes()).decode()
    mont, pop = b64("data/fonts/google/Montserrat.ttf"), b64("data/fonts/google/Poppins.ttf")
    helv = b64("data/fonts/extras/NimbusSans-Regular.otf")
    html = f"""<!doctype html><html><head><title>fonteye_e2e_page</title><style>
@font-face {{ font-family: M; src: url(data:font/ttf;base64,{mont}); }}
@font-face {{ font-family: P; src: url(data:font/ttf;base64,{pop}); }}
@font-face {{ font-family: H; src: url(data:font/otf;base64,{helv}); }}
body {{ margin: 60px; background: #fafafa; }}
h1 {{ font: 44px M; color: #222; margin: 0 0 40px; }}
p {{ font: 30px Georgia; color: #333; }}
.pop {{ font: 28px P; color: #1a1a2e; }}
.helv {{ font: 26px H; color: #111; }}
</style></head><body><h1 id="a">Explore our new collection</h1>
<p id="b">Quietly elegant serif reading text for long articles</p>
<p class="pop">Follow us for daily updates and stories</p>
<p class="helv">Swiss design built on clean neutral type</p></body></html>"""
    f = Path(tempfile.gettempdir()) / "fonteye_e2e_page.html"
    f.write_text(html, encoding="utf8")
    subprocess.Popen(["cmd", "/c", "start", "msedge", "--new-window", f.as_uri()])
    time.sleep(5)
    win = auto.WindowControl(searchDepth=1, SubName="fonteye_e2e_page")
    win.SetActive()
    win.Maximize()
    time.sleep(2)
    targets = []
    for name, font in (("Explore our new collection", "Montserrat"),
                       ("Quietly elegant serif reading text for long articles", "Georgia"),
                       ("Follow us for daily updates and stories", "Poppins"),
                       ("Swiss design built on clean neutral type", "Helvetica")):
        el = win.TextControl(searchDepth=30, Name=name)
        r = el.BoundingRectangle
        targets.append(((r.left - 6, r.top - 6, r.right + 6, r.bottom + 6), font))
    return win, targets


CURRENT = {}   # results of the case in progress, filled by the patched popup methods


def patch_popup():
    """Record what the popup is shown (patched once; one Tk root for every case)."""
    from fonteye.ui import ResultWindow
    show_visual, show_uia = ResultWindow.show_visual, ResultWindow.show_uia

    def visual(self, res, error=None, known=()):
        CURRENT["visual"], CURRENT["popup"] = res, self
        show_visual(self, res, error, known)

    def uia(self, info):
        CURRENT["uia"] = info
        show_uia(self, info)
    ResultWindow.show_visual, ResultWindow.show_uia = visual, uia


def run_case(app, label, rect, expected):
    from fonteye.snip import SnipOverlay
    CURRENT.clear()
    before = set(app.root.winfo_children())
    ov = SnipOverlay(app.root, lambda crop, r: FontEyeApp.on_snip(app, crop, r))
    left, top = ov.left, ov.top
    x0, y0, x1, y1 = rect

    def drive():
        c = ov.c
        c.event_generate("<ButtonPress-1>", x=x0 - left, y=y0 - top)
        c.event_generate("<B1-Motion>", x=(x0 + x1) // 2 - left, y=(y0 + y1) // 2 - top)
        c.event_generate("<ButtonRelease-1>", x=x1 - left, y=y1 - top)

    def finish():
        if "visual" not in CURRENT or "uia" not in CURRENT:
            app.root.after(200, finish)
            return
        app.root.update()
        new = [t for t in app.root.winfo_children() if t not in before and t.winfo_class() == "Toplevel"]
        w = new[-1]
        grab(w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()).save(OUT / f"{label}_popup.png")
        w.destroy()
        app.root.quit()

    app.root.after(600, drive)
    app.root.after(1500, finish)
    app.root.mainloop()

    top3 = [r["name"] for r in (CURRENT.get("visual") or [])[:3]]
    uia = CURRENT.get("uia")
    ok_uia = bool(uia) and uia["name"].lower().startswith(expected.lower()[:5])
    flagged = getattr(CURRENT.get("popup"), "_reconciled", False)
    ok_vis = bool(top3) and expected.lower() in top3[0].lower()
    print(f"[{label}] expected {expected!r}  ->  exact: {uia['name'] if uia else None} {'PASS' if ok_uia else ('flagged as not matching' if flagged else 'WRONG, NOT FLAGGED')}"
          f"  |  visual #1: {top3[0] if top3 else None} {'PASS' if ok_vis else 'FAIL'}  (top-3 {top3})", flush=True)
    return ok_vis


def main():
    patch_popup()
    app = FontEyeApp()
    for _ in range(200):   # wait for the model to load
        if app.recognizer or app.load_error:
            break
        time.sleep(0.05)
    app._pump()
    results = []

    win, rect, font = open_notepad()
    results.append(run_case(app, "notepad", rect, font))
    win.GetWindowPattern().Close()

    win, targets = open_edge()
    for rect, font in targets:
        results.append(run_case(app, f"edge_{font.lower()}", rect, font))
    win.GetWindowPattern().Close()
    app.root.destroy()
    print(f"visual model: {sum(results)}/{len(results)} correct", flush=True)


if __name__ == "__main__":
    main()
