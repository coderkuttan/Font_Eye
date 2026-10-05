"""Result popup shown next to the snipped area."""
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from .lookalikes import hint_for

BG, PANEL, FG, MUTED, ACCENT = "#1e1f22", "#2b2d31", "#f2f3f5", "#9aa0a6", "#4cc2ff"
WARN = "#f0b232"
SAMPLE = "The quick brown fox 123"


def font_sample(path, text=SAMPLE, size=22, width=330, bg=PANEL):
    img = Image.new("RGB", (width, size + 14), bg)
    try:
        f = ImageFont.truetype(path, size)
        ImageDraw.Draw(img).text((4, 4), text, font=f, fill=FG)
    except Exception:  # noqa: BLE001
        ImageDraw.Draw(img).text((4, 4), "(preview unavailable)", fill=MUTED)
    return ImageTk.PhotoImage(img)


class ResultWindow:
    def __init__(self, root, snip, rect):
        self.root = root
        self.win = tk.Toplevel(root, bg=BG)
        self.win.title("FontEye")
        self.win.attributes("-topmost", True)
        self.win.attributes("-toolwindow", True)
        self.win.bind("<Escape>", lambda e: self.win.destroy())
        self._imgs = []

        thumb = Image.fromarray(snip)
        thumb.thumbnail((380, 90))
        self._imgs.append(ImageTk.PhotoImage(thumb))
        tk.Label(self.win, image=self._imgs[-1], bg=BG).pack(padx=12, pady=(12, 6), anchor="w")

        self._uia = self._vis = None
        self.uia_title = self._section("REPORTED BY THE APP  ·  exact")
        self.uia_frame = tk.Frame(self.win, bg=PANEL)
        self.uia_frame.pack(fill="x", padx=12)
        self.uia_label = tk.Label(self.uia_frame, text="checking…", bg=PANEL, fg=MUTED,
                                  font=("Segoe UI", 10), anchor="w", justify="left")
        self.uia_label.pack(fill="x", padx=10, pady=8)

        self._section("VISUAL MATCH  ·  AI model")
        self.vis_frame = tk.Frame(self.win, bg=BG)
        self.vis_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.vis_status = tk.Label(self.vis_frame, text="analysing…", bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.vis_status.pack(anchor="w")

        self.toast = tk.Label(self.win, text="Click a name to copy it  ·  Esc to close",
                              bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.toast.pack(pady=(4, 10))
        self.rect = rect
        self._place()
        self.win.focus_force()

    def _section(self, title):
        lbl = tk.Label(self.win, text=title, bg=BG, fg=MUTED, font=("Segoe UI Semibold", 8))
        lbl.pack(anchor="w", padx=12, pady=(10, 3))
        return lbl

    def _place(self):
        """Size to content and keep the popup beside the snip, fully on screen."""
        rect = self.rect
        self.win.update_idletasks()
        w, h = 420, self.win.winfo_reqheight()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        x = rect[2] + 12 if rect[2] + 12 + w < sw else max(0, rect[0] - w - 12)
        y = min(max(0, rect[1]), max(0, sh - h - 60))
        self.win.geometry(f"{w}x{h}+{x}+{y}")

    def _copy(self, text):
        self.win.clipboard_clear()
        self.win.clipboard_append(text)
        self.toast.config(text=f"Copied “{text}”", fg=ACCENT)

    def _name_label(self, parent, text, size=13):
        lbl = tk.Label(parent, text=text, bg=parent["bg"], fg=FG, font=("Segoe UI Semibold", size),
                       cursor="hand2", anchor="w")
        lbl.bind("<Button-1>", lambda e: self._copy(text.split(" / ")[0]))
        return lbl

    # --- results -------------------------------------------------------
    def show_uia(self, info):
        if not self.win.winfo_exists():
            return
        if not info:
            self.uia_label.config(text="This app doesn't expose font info here (image, game, PDF "
                                       "raster…) — use the visual match below.", wraplength=370)
            return
        self._uia = info
        self.uia_label.destroy()
        self._name_label(self.uia_frame, info["name"], 15).pack(fill="x", padx=10, pady=(8, 0))
        bits = [f"{info['size']:g} pt" if info.get("size") else None, info.get("weight"),
                "Italic" if info.get("italic") else None]
        detail = " · ".join(b for b in bits if b)
        if info.get("app"):
            detail += f"   —   {info['app'][:40]}"
        tk.Label(self.uia_frame, text=detail, bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", padx=10, pady=(0, 8))
        self._reconcile()
        self._place()

    def _reconcile(self):
        """Browsers report their default font (e.g. Times New Roman) for web-font body text
        through UI Automation. If the app's answer is a font the model knows, yet nothing in
        the visual top-5 resembles it while the visual match is confident, don't call it exact."""
        if not self._uia or not self._vis or getattr(self, "_reconciled", False):
            return
        results, known = self._vis
        name = self._uia["name"].lower()
        seen = {n.lower().split(" / ")[0] for r in results for n in [r["name"]] + r.get("twins", [])}
        if name in known and name not in seen and results[0]["prob"] >= 0.5:
            self._reconciled = True
            self.uia_title.config(text="REPORTED BY THE APP  ·  doesn't match the pixels", fg=WARN)
            tk.Label(self.uia_frame, text="The text on screen doesn't look like this font. Browsers often "
                     "report their default font for web fonts, so trust the visual match below.",
                     bg=PANEL, fg=WARN, font=("Segoe UI", 9), anchor="w", justify="left",
                     wraplength=370).pack(fill="x", padx=10, pady=(0, 8))

    def show_visual(self, results, error=None, known=()):
        if not self.win.winfo_exists():
            return
        if results:
            self._vis = (results, set(known))
        if error or not results:
            self.vis_status.config(text=error or "No text found in the selection — try snipping a "
                                                  "whole word or line.", wraplength=370)
            return
        self.vis_status.destroy()
        # Full rows only for real contenders; near-zero candidates go on one muted line.
        strong = [r for i, r in enumerate(results) if i == 0 or r["prob"] >= 0.01]
        weak = results[len(strong):]
        for i, r in enumerate(strong):
            row = tk.Frame(self.vis_frame, bg=PANEL if i == 0 else BG)
            row.pack(fill="x", pady=2)
            top = tk.Frame(row, bg=row["bg"])
            top.pack(fill="x", padx=8, pady=(6, 0))
            self._name_label(top, r["name"], 13 if i == 0 else 11).pack(side="left")
            tk.Label(top, text=f"{r['prob'] * 100:.0f}%", bg=row["bg"], fg=ACCENT if i == 0 else MUTED,
                     font=("Segoe UI", 10)).pack(side="right")
            if not r.get("trained", True):
                tk.Label(top, text="installed", bg=row["bg"], fg=MUTED, font=("Segoe UI", 8)).pack(side="right", padx=6)
            if hint_for(r["name"]):
                tk.Label(row, text=f"looks similar to {hint_for(r['name'])} (proprietary, not in the model)",
                         bg=row["bg"], fg=MUTED, font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=8)
            if r.get("twins"):
                tk.Label(row, text="near-identical to: " + ", ".join(r["twins"][:4]), bg=row["bg"], fg=MUTED,
                         font=("Segoe UI", 8), anchor="w", wraplength=380, justify="left").pack(fill="x", padx=8)
            if i < 3:
                self._imgs.append(font_sample(r["path"], size=20 if i == 0 else 16, bg=row["bg"]))
                tk.Label(row, image=self._imgs[-1], bg=row["bg"]).pack(anchor="w", padx=6, pady=(0, 6))
        if weak:
            tk.Label(self.vis_frame, text="Next closest: " + "  ·  ".join(r["name"] for r in weak), bg=BG,
                     fg=MUTED, font=("Segoe UI", 9), anchor="w", wraplength=380, justify="left").pack(
                fill="x", padx=8, pady=(4, 0))
        self._reconcile()
        self._place()
