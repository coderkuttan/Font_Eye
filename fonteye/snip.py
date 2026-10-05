"""Full-screen freeze-frame overlay: drag a box over text to capture it."""
import tkinter as tk

import mss
import numpy as np
from PIL import Image, ImageEnhance, ImageTk


class SnipOverlay:
    def __init__(self, root, on_done, on_cancel=None):
        self.on_done, self.on_cancel = on_done, on_cancel
        with getattr(mss, "MSS", mss.mss)() as s:
            mon = s.monitors[0]                      # whole virtual desktop
            shot = s.grab(mon)
        self.full = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        self.left, self.top = mon["left"], mon["top"]
        w, h = self.full.size

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.geometry(f"{w}x{h}+{self.left}+{self.top}")
        self.c = tk.Canvas(self.win, width=w, height=h, highlightthickness=0, cursor="crosshair")
        self.c.pack()
        self._dim = ImageTk.PhotoImage(ImageEnhance.Brightness(self.full).enhance(0.5))
        self.c.create_image(0, 0, anchor="nw", image=self._dim)

        # hint on the primary monitor
        px, py = -self.left, -self.top
        self.c.create_rectangle(px + 20, py + 20, px + 470, py + 58, fill="#111", outline="#555")
        self.c.create_text(px + 35, py + 39, anchor="w", fill="#eee", font=("Segoe UI", 11),
                           text="Drag over some text to identify its font   ·   Esc to cancel")

        self.start = None
        self.rect = self.lit = self._lit_img = None
        self.c.bind("<ButtonPress-1>", self._press)
        self.c.bind("<B1-Motion>", self._drag)
        self.c.bind("<ButtonRelease-1>", self._release)
        self.win.bind("<Escape>", lambda e: self._cancel())
        self.win.bind("<ButtonPress-3>", lambda e: self._cancel())
        self.win.focus_force()
        self.win.grab_set()

    def _box(self, e):
        x0, y0 = self.start
        return min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y)

    def _press(self, e):
        self.start = (e.x, e.y)

    def _drag(self, e):
        if not self.start:
            return
        x0, y0, x1, y1 = self._box(e)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        self._lit_img = ImageTk.PhotoImage(self.full.crop((x0, y0, x1, y1)))
        if self.lit is None:
            self.lit = self.c.create_image(x0, y0, anchor="nw", image=self._lit_img)
            self.rect = self.c.create_rectangle(x0, y0, x1, y1, outline="#4cc2ff", width=2)
        else:
            self.c.itemconfig(self.lit, image=self._lit_img)
            self.c.coords(self.lit, x0, y0)
            self.c.coords(self.rect, x0, y0, x1, y1)

    def _release(self, e):
        if not self.start:
            return
        x0, y0, x1, y1 = self._box(e)
        self.win.destroy()
        if x1 - x0 < 8 or y1 - y0 < 6:
            if self.on_cancel:
                self.on_cancel()
            return
        crop = np.asarray(self.full.crop((x0, y0, x1, y1)))
        rect = (self.left + x0, self.top + y0, self.left + x1, self.top + y1)
        self.on_done(crop, rect)

    def _cancel(self):
        self.win.destroy()
        if self.on_cancel:
            self.on_cancel()
