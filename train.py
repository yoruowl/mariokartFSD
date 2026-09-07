import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

DATA = Path("dataset")
IMG_SIZE = (66, 200)   # H, W  — PilotNet-ish, wide
STACK = 1              # set to 4 later if you want
BATCH = 512
EPOCHS = 40
LR = 1e-3
VAL_RUNS = {"run_014", "run_015"}  # hold out two full races
WORKERS = max(8, os.cpu_count() or 8)

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. This venv has a CPU-only PyTorch build, or the NVIDIA driver is missing.\n"
        "RTX 50-series needs the CUDA 13.0 wheels (plain `pip install torch` from PyPI is CPU-only):\n"
        "  pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 "
        "--index-url https://download.pytorch.org/whl/cu130"
    )
device = torch.device("cuda")
print("device", device, torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
use_bf16 = torch.cuda.is_bf16_supported()
print("amp", "bf16" if use_bf16 else "fp32", "batch", BATCH, "workers", WORKERS)

tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # 0..1, CHW
])


def collect_items(runs):
    items = []
    for run in runs:
        rows = list(csv.DictReader((run / "labels.csv").open()))
        frames = run / "frames"
        for r in rows:
            p = frames / f"{int(r['index']):06d}.png"
            if p.exists():
                items.append((p, float(r["steer"])))
    return items


def load_frame(item):
    path, steer = item
    with Image.open(path) as im:
        x = tfm(im.convert("RGB"))
    return x, steer


def load_tensors(items, name):
    n = len(items)
    print(f"loading {name} ({n} frames, {WORKERS} threads)...")
    t0 = time.perf_counter()
    xs = [None] * n
    ys = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(load_frame, item): i for i, item in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            x, steer = fut.result()
            xs[i] = x
            ys[i] = steer
            done += 1
            if done % 2000 == 0 or done == n:
                print(f"  {done}/{n}")
    x = torch.stack(xs).to(device, non_blocking=True)
    y = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
    print(f"  {name} {tuple(x.shape)} on {device}  {time.perf_counter() - t0:.1f}s")
    return x, y


def iterate(x, y, batch, shuffle):
    n = x.size(0)
    if shuffle:
        idx = torch.randperm(n, device=x.device)
        n = n - (n % batch)
        if n == 0:
            return
        idx = idx[:n]
        x, y = x[idx], y[idx]
    for i in range(0, n, batch):
        yield x[i:i + batch], y[i:i + batch]


class Pilot(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),  # may need a print if this errors
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# discover flatten size once if the Linear line crashes:
# x = torch.zeros(1, 3, *IMG_SIZE)
# print(Pilot().net[:-4](x).shape)

if __name__ == "__main__":
    all_runs = sorted(
        p
        for p in DATA.iterdir()
        if p.is_dir() and (p.name.startswith("run_") or p.name.startswith("dagger_"))
    )
    # DAgger corrections are always training data. Validation remains limited to
    # complete human-driven runs so sparse interventions cannot leak into it.
    train_runs = [
        p for p in all_runs
        if p.name.startswith("dagger_") or p.name not in VAL_RUNS
    ]
    val_runs = [p for p in all_runs if p.name.startswith("run_") and p.name in VAL_RUNS]
    print("train", [p.name for p in train_runs])
    print("val", [p.name for p in val_runs])

    x_train, y_train = load_tensors(collect_items(train_runs), "train")
    x_val, y_val = load_tensors(collect_items(val_runs), "val")
    print("n train", x_train.size(0), "n val", x_val.size(0))

    model = Pilot().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best = 1e9
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        tr = 0.0
        n_seen = 0
        for x, y in iterate(x_train, y_train, BATCH, shuffle=True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr += loss.item() * x.size(0)
            n_seen += x.size(0)
        tr /= max(n_seen, 1)

        model.eval()
        va = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for x, y in iterate(x_val, y_val, BATCH, shuffle=False):
                va += loss_fn(model(x), y).item() * x.size(0)
        va /= max(x_val.size(0), 1)
        dt = time.perf_counter() - t0
        print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}  {dt:.2f}s")
        if va < best:
            best = va
            torch.save(model.state_dict(), "pilot_best.pt")
            print("  saved pilot_best.pt")
