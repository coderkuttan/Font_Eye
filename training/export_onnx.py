"""Export the trained embedder to ONNX so the app runs without PyTorch.

    python training/export_onnx.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ROOT, FontEmbedder  # noqa: E402


def main():
    ck = torch.load(ROOT / "models/fonteye.pt", map_location="cpu")
    model = FontEmbedder(ck["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    out = ROOT / "models/fonteye.onnx"
    dummy = torch.rand(4, 1, 64, 128)
    torch.onnx.export(model, dummy, str(out), input_names=["patches"], output_names=["embedding"],
                      dynamic_axes={"patches": {0: "n"}, "embedding": {0: "n"}}, opset_version=17,
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ref = model(dummy).detach().numpy()
    got = sess.run(None, {"patches": dummy.numpy()})[0]
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), max abs diff vs torch = {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
