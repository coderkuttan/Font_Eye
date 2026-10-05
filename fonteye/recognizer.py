"""Visual font recognition: snip -> top-k fonts, using the ONNX model + index."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .preprocess import snip_to_batch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class FontRecognizer:
    def __init__(self, model_dir=MODEL_DIR):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "fonteye.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.index = np.load(model_dir / "index.npy")
        self.entries = json.loads((model_dir / "index.json").read_text(encoding="utf8"))
        for e in self.entries:   # index stores portable paths (see build_index.portable_path)
            p = e["path"]
            e["path"] = str(Path(p).expanduser() if p.startswith("~") else
                            Path(p) if Path(p).is_absolute() else model_dir.parent / p)
        # confidence = softmax(scale * cosine); scale fitted on held-out renders by calibrate.py
        cal = model_dir / "calibration.json"
        self.scale = json.loads(cal.read_text())["scale"] if cal.exists() else 30.0
        # every name the index can recognise (lower case), incl. clone/twin names
        self.known_names = {n.lower().split(" / ")[0] for e in self.entries
                            for n in [e["name"]] + e.get("twins", [])}

    def embed(self, img):
        batch = snip_to_batch(img)
        if batch is None:
            return None
        embs = self.sess.run(None, {"patches": batch})[0]
        v = embs.mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    # Fonts indexed zero-shot (never trained on) get less well-separated fingerprints;
    # a small handicap stops them stealing matches from trained fonts (tuned in evaluate.py).
    UNTRAINED_PENALTY = 0.02

    def identify(self, img, k=5, penalty=None):
        """img: uint8 RGB array. Returns [{name, path, score, trained}] best first."""
        v = self.embed(img)
        if v is None:
            return []
        sims = self.index @ v
        if not hasattr(self, "_untrained"):
            self._untrained = np.array([not e.get("trained", True) for e in self.entries])
        sims = sims - self._untrained * (self.UNTRAINED_PENALTY if penalty is None else penalty)
        p = np.exp(self.scale * (sims - sims.max()))
        p /= p.sum()
        # Merge twin fonts (see build_index.TWINS) into one result line.
        out, by_group = [], {}
        for i in np.argsort(-sims):
            e = self.entries[i]
            g = e.get("group")
            if g and g in by_group:
                by_group[g]["prob"] += float(p[i])
                continue
            r = dict(e, score=float(sims[i]), prob=float(p[i]))
            if g:
                # report the group under its primary font; e["twins"] lists the rest
                prim = next((x for x in self.entries if x["name"] == g), e)
                twins = [t for t in e.get("twins", []) if t != g]
                if e["name"] != g:
                    twins = [e["name"]] + [t for t in twins if t != e["name"]]
                r = dict(prim, score=float(sims[i]), prob=float(p[i]), twins=twins)
                by_group[g] = r
            out.append(r)
            if len(out) >= k and float(p[i]) < 1e-4:
                break
        return out[:k]
