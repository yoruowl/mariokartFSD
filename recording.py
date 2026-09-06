import csv, time
from pathlib import Path

import mss
import pygame
import pygetwindow as gw
from PIL import Image

OUT_ROOT = Path("dataset")

WINDOW_TITLE = "Eden | v0.2.1 | Clang 22.1.4 | Mario Kart 8 Deluxe (64-bit) | 3.0.5 | Nvidia"
wins = gw.getWindowsWithTitle(WINDOW_TITLE)
if not wins:
    raise SystemExit(f"Window not found: {WINDOW_TITLE!r}")
win = wins[0]
MONITOR = {"left": win.left, "top": win.top, "width": win.width, "height": win.height}
print("Capture region:", MONITOR)

# Optional tighter "windshield" crop inside that window (pixels)
CROP = (400, 80, MONITOR["width"] - 400, MONITOR["height"] - 170)  # left, top, right, bottom in the grab

HZ = 15
DT = 1.0 / HZ

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() < 1:
    raise SystemExit("No controller. Pair the Pro Controller first.")
joy = pygame.joystick.Joystick(0)
joy.init()
print("Using", joy.get_name())

ARM_BTN = 7  # left stick click (L3); right stick was 8
sct = mss.mss()


def next_out_dir() -> Path:
    existing = []
    if OUT_ROOT.exists():
        for p in OUT_ROOT.iterdir():
            if p.is_dir() and p.name.startswith("run_"):
                try:
                    existing.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    pass
    n = (max(existing) + 1) if existing else 1
    out = OUT_ROOT / f"run_{n:03d}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    return out


def wait_for_arm():
    print("Hold gas on track, then click the left stick to start. Ctrl+C to quit.")
    while True:
        pygame.event.pump()
        if joy.get_button(ARM_BTN):
            break
        time.sleep(0.02)
    while joy.get_button(ARM_BTN):  # wait for release so start doesn't also stop
        pygame.event.pump()
        time.sleep(0.02)
    time.sleep(0.1)


def record_run(out: Path) -> int:
    idx = 0
    t0 = time.time()
    with (out / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "t", "steer", "throttle", "drift"])
        print("Recording", out.name, "— click left stick to stop this run.")
        next_t = time.perf_counter()
        while True:
            pygame.event.pump()
            if joy.get_button(ARM_BTN):
                break
            steer = joy.get_axis(0)          # left stick X
            throttle = 1 if joy.get_button(0) else 0  # A
            # axis 4 = LT, axis 5 = RT; Switch triggers are digital — treat as pressed/not
            drift = 1 if joy.get_axis(5) > 0 else 0

            raw = sct.grab(MONITOR)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            img = img.crop(CROP)
            img.save(out / "frames" / f"{idx:06d}.png", optimize=False)

            w.writerow([idx, f"{time.time()-t0:.4f}", f"{steer:.4f}", throttle, drift])
            if idx % 50 == 0:
                f.flush()
                print(idx, f"steer={steer:+.2f} drift={drift}")
            idx += 1

            next_t += DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    while joy.get_button(ARM_BTN):  # wait for release before next arm
        pygame.event.pump()
        time.sleep(0.02)
    return idx


try:
    while True:
        wait_for_arm()
        out = next_out_dir()
        print("Writing to", out)
        n = record_run(out)
        print("saved", n, "frames to", out)
except KeyboardInterrupt:
    print("\nExiting.")
