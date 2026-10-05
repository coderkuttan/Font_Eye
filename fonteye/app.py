"""FontEye desktop app: tray icon + global hotkey -> snip -> font answer.

    python -m fonteye                 # run in tray (Ctrl+Shift+F to snip)
    python -m fonteye --image x.png   # identify an image file from the command line
"""
import argparse
import ctypes
import queue
import threading
import time
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont

HOTKEY = "<ctrl>+<shift>+f"
HOTKEY_LABEL = "Ctrl+Shift+F"


def make_dpi_aware():
    # Must run before any window exists so Tk, mss and UIA all use physical pixels.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        ctypes.windll.user32.SetProcessDPIAware()


def tray_image():
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((2, 2, 62, 62), 14, fill=(30, 31, 34, 255))
    try:
        f = ImageFont.truetype("georgiab.ttf", 34)
    except OSError:
        f = ImageFont.load_default()
    d.text((32, 33), "Aa", font=f, fill=(76, 194, 255, 255), anchor="mm")
    return img


class FontEyeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.q = queue.Queue()
        self.recognizer = None
        self.load_error = None
        self.busy = False
        threading.Thread(target=self._load_model, daemon=True).start()

    # --- threading helpers: Tk must only be touched from the main thread ---
    def call_main(self, fn, *a):
        self.q.put((fn, a))

    def _pump(self):
        try:
            while True:
                fn, a = self.q.get_nowait()
                fn(*a)
        except queue.Empty:
            pass
        self.root.after(30, self._pump)

    def _load_model(self):
        try:
            from .recognizer import FontRecognizer
            self.recognizer = FontRecognizer()
        except Exception as e:  # noqa: BLE001
            self.load_error = f"Model not available ({e}). Train it first — see README."

    # --- flow ---
    def trigger(self):
        if self.busy:
            return
        self.busy = True
        from .snip import SnipOverlay
        SnipOverlay(self.root, self.on_snip, on_cancel=self._idle)

    def _idle(self):
        self.busy = False

    def on_snip(self, crop, rect):
        from .ui import ResultWindow
        self.busy = False
        win = ResultWindow(self.root, crop, rect)
        cx, cy = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2

        def uia_job():
            from .uia_probe import font_at
            time.sleep(0.12)                      # let the overlay vanish so UIA hits the real app
            self.call_main(win.show_uia, font_at(cx, cy))

        def visual_job():
            for _ in range(200):                  # wait for model load (first launch)
                if self.recognizer or self.load_error:
                    break
                time.sleep(0.05)
            if self.recognizer is None:
                self.call_main(win.show_visual, [], self.load_error or "Model still loading…")
                return
            try:
                res = self.recognizer.identify(crop, k=5)
                self.call_main(win.show_visual, res, None, self.recognizer.known_names)
            except Exception as e:  # noqa: BLE001
                self.call_main(win.show_visual, [], f"Recognition failed: {e}")

        threading.Thread(target=uia_job, daemon=True).start()
        threading.Thread(target=visual_job, daemon=True).start()

    def quit(self):
        try:
            self.hotkeys.stop()
            self.icon.stop()
        finally:
            self.root.quit()

    def run(self, snip_now=False):
        import pystray
        from pynput import keyboard

        self.hotkeys = keyboard.GlobalHotKeys({HOTKEY: lambda: self.call_main(self.trigger)})
        self.hotkeys.start()
        self.icon = pystray.Icon("FontEye", tray_image(), f"FontEye — {HOTKEY_LABEL} to identify a font",
                                 menu=pystray.Menu(
                                     pystray.MenuItem(f"Identify font ({HOTKEY_LABEL})",
                                                      lambda: self.call_main(self.trigger), default=True),
                                     pystray.MenuItem("Quit", lambda: self.call_main(self.quit))))
        self.icon.run_detached()
        if snip_now:
            self.root.after(300, self.trigger)
        self._pump()
        print(f"FontEye running in the tray. Press {HOTKEY_LABEL} to snip text.")
        self.root.mainloop()


def identify_file(path):
    import numpy as np
    from .recognizer import FontRecognizer
    res = FontRecognizer().identify(np.asarray(Image.open(path).convert("RGB")), k=5)
    if not res:
        print("No text found.")
    for i, r in enumerate(res, 1):
        print(f"{i}. {r['name']:<35} {r['prob'] * 100:5.1f}%   (cos {r['score']:.3f})")


def main():
    ap = argparse.ArgumentParser(prog="fonteye")
    ap.add_argument("--image", help="identify the font in an image file and exit")
    ap.add_argument("--snip", action="store_true", help="open the snip overlay immediately")
    args = ap.parse_args()
    if args.image:
        identify_file(args.image)
        return
    make_dpi_aware()
    FontEyeApp().run(snip_now=args.snip)


if __name__ == "__main__":
    main()
