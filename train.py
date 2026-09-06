import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

DATA = Path("dataset")
IMG_SIZE = (66, 200)   # H, W  — PilotNet-ish, wide
STACK = 1              # set to 4 later if you want
BATCH = 64
EPOCHS = 15
LR = 1e-3
VAL_RUNS = {"run_014", "run_015"}  # hold out two full races

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # 0..1, CHW
])

class Runs(Dataset):
    def __init__(self, runs):
        self.items = []
        for run in runs:
            rows = list(csv.DictReader((run / "labels.csv").open()))
            frames = run / "frames"
            for r in rows:
                p = frames / f"{int(r['index']):06d}.png"
                if p.exists():
                    self.items.append((p, float(r["steer"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, steer = self.items[i]
        x = tfm(Image.open(path).convert("RGB"))
        y = torch.tensor([steer], dtype=torch.float32)
        return x, y

all_runs = sorted(p for p in DATA.iterdir() if p.is_dir() and p.name.startswith("run_"))
train_runs = [p for p in all_runs if p.name not in VAL_RUNS]
val_runs = [p for p in all_runs if p.name in VAL_RUNS]
print("train", [p.name for p in train_runs])
print("val", [p.name for p in val_runs])

train_ds, val_ds = Runs(train_runs), Runs(val_runs)
print("n train", len(train_ds), "n val", len(val_ds))

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

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

model = Pilot().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def weighted_mse(pred, y):
    w = 1.0 + 4.0 * y.abs()
    return (w * (pred - y).pow(2)).mean()

best = 1e9
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = weighted_mse(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tr += loss.item() * x.size(0)
    tr /= len(train_ds)

    model.eval()
    va = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            va += weighted_mse(model(x), y).item() * x.size(0)
    va /= max(len(val_ds), 1)
    print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}")
    if va < best:
        best = va
        torch.save(model.state_dict(), "pilot_best.pt")
        print("  saved pilot_best.pt")