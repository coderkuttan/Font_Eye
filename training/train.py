"""Train the font embedder.

    python training/train.py --epochs 30

Samples are rendered on the fly by a pool of CPU workers (datagen.py), so no
dataset is stored on disk. If data/browser_renders/train exists (see make_browser_renders.py), real
Chromium renders are mixed in to close the synthetic-vs-real gap.
"""
import argparse
import collections
import itertools
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

# NOTE: no torch imports at module level. Windows "spawn" workers re-import this
# file, and torch would map ~1.5 GB of CUDA DLLs into every one of them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen import init_worker, make_batch  # noqa: E402
from fontset import ROOT, load_classes, safe_name  # noqa: E402

BROWSER_DIR = ROOT / "data" / "browser_renders"


def browser_files(split, classes):
    """class index -> list of PNGs rendered by a real browser (folders named by class)."""
    out = {}
    for i, c in enumerate(classes):
        d = BROWSER_DIR / split / safe_name(c["name"])
        if d.is_dir():
            out[i] = [str(p) for p in sorted(d.glob("*.png"))]
    return out


def bounded_imap(pool, jobs, window=24):
    """Ordered pool.imap with at most `window` batches in flight (bounded RAM)."""
    it = iter(jobs)
    q = collections.deque(pool.apply_async(make_batch, (j,)) for j in itertools.islice(it, window))
    while q:
        res = q.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            q.append(pool.apply_async(make_batch, (nxt,)))
        yield res


def evaluate(model, head, val_x, val_y, batch=512):
    import torch
    model.eval()
    correct1 = correct5 = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(val_y), batch):
            x = val_x[i:i + batch].float().div_(255).unsqueeze(1)
            y = val_y[i:i + batch]
            top5 = head(model(x)).topk(5, dim=1).indices
            correct1 += (top5[:, 0] == y).sum().item()
            correct5 += (top5 == y[:, None]).any(1).sum().item()
    model.train()
    return correct1 / len(val_y), correct5 / len(val_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", help="warm-start the backbone from an earlier checkpoint (new ArcFace head)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from model import ArcFace, FontEmbedder

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    classes = load_classes()
    print(f"{len(classes)} font classes")

    br_train = browser_files("train", classes)
    if br_train:
        print(f"mixing in browser renders for {len(br_train)} classes")
    pool = mp.get_context("spawn").Pool(args.workers, initializer=init_worker,
                                        initargs=([c["paths"] for c in classes], br_train, 0.25))
    n_val = len(classes) * 12
    val = list(bounded_imap(pool, [("val", i, min(256, n_val - i)) for i in range(0, n_val, 256)]))
    val_x = torch.from_numpy(np.concatenate([v[0] for v in val])).to(device)
    val_y = torch.from_numpy(np.concatenate([v[1] for v in val])).to(device)
    steps_per_epoch = args.samples_per_epoch // args.batch
    print(f"validation set: {n_val} patches")

    model = FontEmbedder(args.backbone).to(device)  # NCHW: channels_last is 3x slower for 1-channel input here
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck0["model"])
        print(f"warm-started backbone from {args.init_from} (epoch {ck0['epoch']}, {len(ck0['classes'])} classes)")
    head = ArcFace(len(classes)).to(device)
    params = [{"params": model.parameters()}, {"params": head.parameters(), "lr": args.lr * 5}]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 5], total_steps=steps,
                                                pct_start=0.05, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()

    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "fonteye.pt"
    start_epoch, best = 0, 0.0
    last_path = out_dir / "fonteye_last.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start_epoch}, best top1 {best:.3f}")

    log = open(out_dir / "train_log.jsonl", "a", encoding="utf8")
    for epoch in range(start_epoch, args.epochs):
        t0, tot, seen, hit = time.time(), 0.0, 0, 0
        jobs = [("train", random.randrange(1 << 62), args.batch) for _ in range(steps_per_epoch)]
        for step, (xb, yb) in enumerate(bounded_imap(pool, jobs)):
            x = torch.from_numpy(xb).pin_memory().to(device, non_blocking=True)
            x = x.float().div_(255).unsqueeze(1)
            y = torch.from_numpy(yb).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                emb = model(x)
                logits = head(emb.float(), y)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            seen += len(y)
            with torch.no_grad():
                hit += (head(emb.float()).argmax(1) == y).sum().item()
            if step % 100 == 0:
                print(f"  ep {epoch} step {step}/{steps_per_epoch} loss {tot / seen:.3f} "
                      f"train-acc {hit / seen:.3f} {seen / (time.time() - t0):.0f} img/s", flush=True)
        top1, top5 = evaluate(model, head, val_x, val_y)
        rec = {"epoch": epoch, "loss": tot / seen, "train_acc": hit / seen, "val_top1": top1,
               "val_top5": top5, "minutes": (time.time() - t0) / 60}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if top1 >= best:
            best = top1
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "epoch": epoch,
                        "val_top1": top1, "backbone": args.backbone,
                        "classes": [c["name"] for c in classes]}, ckpt_path)
            print(f"  saved best ({top1:.3f}) -> {ckpt_path}")
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best": best}, last_path)
    pool.terminate()
    print(f"done. best synthetic val top1 = {best:.3f}")


if __name__ == "__main__":
    main()
