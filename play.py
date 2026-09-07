import csv
import math
import time
from pathlib import Path

import mss
import pygame
import pygetwindow as gw
import torch
import torch.nn as nn
import vgamepad as vg
from PIL import Image
from torchvision import transforms

WINDOW_TITLE = "Eden | v0.2.1 | Clang 22.1.4 | Mario Kart 8 Deluxe (64-bit) | 3.0.5 | Nvidia"

# Eden must start after the virtual controller or it may show the pad in the
# controls screen without sending its input to the running game.
pad = vg.VX360Gamepad()
pad.reset()
pad.update()
print("Virtual Xbox 360 Controller connected (neutral). Start Eden and launch Mario Kart.")

wins = []
while not wins:
    wins = gw.getWindowsWithTitle(WINDOW_TITLE)
    time.sleep(0.25)

win = wins[0]
MONITOR = {"left": win.left, "top": win.top, "width": win.width, "height": win.height}
print("Capture region:", MONITOR)

CROP = (400, 80, MONITOR["width"] - 400, MONITOR["height"] - 170)  # left, top, right, bottom in the grab
IMG_SIZE = (66, 200)
HZ = 15
WEIGHTS = Path("pilot_best.pt")
ARM_BTN = 7  # left stick click (L3); same as recording.py
TAKEOVER_BTN = 10  # right bumper (R); hold to override the model
OUT_ROOT = Path("dataset")


def next_dagger_dir() -> Path:
    existing = []
    if OUT_ROOT.exists():
        for path in OUT_ROOT.iterdir():
            if path.is_dir() and path.name.startswith("dagger_"):
                try:
                    existing.append(int(path.name.split("_", 1)[1]))
                except ValueError:
                    pass
    number = max(existing, default=0) + 1
    out = OUT_ROOT / f"dagger_{number:03d}"
    (out / "frames").mkdir(parents=True, exist_ok=False)
    return out

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
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),  # same size as train.py
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

def is_virtual_pad_name(name: str) -> bool:
    n = name.lower()
    return "xbox 360" in n or "vigem" in n or "vgamepad" in n


def find_physical_joystick():
    pygame.joystick.quit()
    pygame.joystick.init()
    for i in range(pygame.joystick.get_count()):
        j = pygame.joystick.Joystick(i)
        j.init()
        if not is_virtual_pad_name(j.get_name()):
            return j
    return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Pilot().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

tfm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

pygame.init()
pygame.joystick.init()
joy = find_physical_joystick()
if joy is None:
    raise SystemExit("No physical controller found for L3 arm. Pair the Pro Controller first.")
print("Arm controller:", joy.get_name())

sct = mss.mss()
dt = 1.0 / HZ

print()
print("In Eden: Controls → Player 1 → Input Device = 'Xbox 360 Controller'")
print("  (leave your Pro Controller unselected for Player 1)")
print("Then focus the race and click L3 on the Pro Controller to start AI.")
print("While AI drives, hold R and use the left stick to take over steering.")
print("Tip: turn off Pause emulation when in background.")
while True:
    pygame.event.pump()
    if joy.get_button(ARM_BTN):
        break
    time.sleep(0.02)
while joy.get_button(ARM_BTN):
    pygame.event.pump()
    time.sleep(0.02)

try:
    win.activate()
except Exception:
    pass

out = next_dagger_dir()
labels_file = (out / "labels.csv").open("w", newline="")
labels = csv.writer(labels_file)
labels.writerow(["index", "t", "steer", "model_steer", "throttle", "drift", "intervention"])
correction_index = 0
started_at = time.time()

print(f"AI driving. Corrections will be saved to {out}. Ctrl+C to stop.")

try:
    next_t = time.perf_counter()
    while True:
        pygame.event.pump()
        raw = sct.grab(MONITOR)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        img = img.crop(CROP)
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            model_steer = float(model(x).item())
        if not math.isfinite(model_steer):
            model_steer = 0.0
        model_steer = max(-1.0, min(1.0, model_steer))

        intervening = bool(joy.get_button(TAKEOVER_BTN))
        expert_steer = max(-1.0, min(1.0, float(joy.get_axis(0))))
        steer = expert_steer if intervening else model_steer

        # Xbox B → Switch A (throttle). Release Xbox A so Switch B (brake) stays up.
        pad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        pad.update()

        if intervening:
            img.save(out / "frames" / f"{correction_index:06d}.png", optimize=False)
            labels.writerow([
                correction_index,
                f"{time.time() - started_at:.4f}",
                f"{expert_steer:.4f}",
                f"{model_steer:.4f}",
                1,
                0,
                1,
            ])
            correction_index += 1
            if correction_index % 15 == 0:
                labels_file.flush()

        driver = "HUMAN" if intervening else "AI   "
        print(
            f"\r{driver} steer {steer:+.2f}  corrections {correction_index}",
            end="",
            flush=True,
        )

        next_t += dt
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.perf_counter()
except KeyboardInterrupt:
    print(f"\nstopped; saved {correction_index} corrections to {out}")
finally:
    pad.reset()
    pad.update()
    labels_file.close()
