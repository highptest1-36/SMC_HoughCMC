"""
tune_perception.py — data-driven Hough/HSV tuning against the REAL sim.

Problem: on the real map the yellow centre line is thin/dashed/curved, so the
detector's HSV range + Hough thresholds must be tuned to THIS map. Guessing blind
is slow; this tool measures.

What it does:
  1. Drives the car for a few seconds with a very-sensitive fallback detector,
     collecting real camera frames (keeps moving so we see on-track frames).
  2. Replays those frames through a set of candidate (HSV, ROI, Hough) configs and
     reports detection-rate + mean |error| + yellow-pixel-count for each.
  3. Saves a few sample frames (raw / yellow-mask / overlay) to
     results/perception_debug/ so they can be inspected visually.

Run (sim must be open):
    python tune_perception.py
    python tune_perception.py --seconds 20 --speed 25
Then read the printed table (or send the PNGs in results/perception_debug/).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from simulator import SimClient

DEBUG_DIR = os.path.join(C.RESULTS_DIR, "perception_debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

# Candidate configs to compare. (name, hsv_lo, hsv_hi, roi_top_frac, thr, minlen, gap)
CANDIDATES = [
    ("tight/full/20-5-40",   (20, 100, 100), (30, 255, 255), 0.0, 20, 5, 40),
    ("tight/half/20-5-40",   (20, 100, 100), (30, 255, 255), 0.5, 20, 5, 40),
    ("conf40/full/40-40-40", (20, 100, 100), (30, 255, 255), 0.0, 40, 40, 40),
    ("wide/full/15-5-50",    (18, 60, 80),   (35, 255, 255), 0.0, 15, 5, 50),
    ("wide/half/15-5-50",    (18, 60, 80),   (35, 255, 255), 0.5, 15, 5, 50),
    ("verywide/full/10-5-50",(15, 40, 60),   (40, 255, 255), 0.0, 10, 5, 50),
    ("verywide/lowroi/10-5-50",(15, 40, 60), (40, 255, 255), 0.35, 10, 5, 50),
]

# Sensitive detector used ONLY to keep the car moving while collecting frames.
COLLECT = ("verywide/full/10-5-50", (15, 40, 60), (40, 255, 255), 0.0, 10, 5, 50)


def yellow_mask(img, lo, hi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.GaussianBlur(m, (5, 5), 0)


def detect(img, cfg, draw=False):
    _, lo, hi, roi_frac, thr, minlen, gap = cfg
    h, w = img.shape[:2]
    mask = yellow_mask(img, lo, hi)
    npix = int((mask > 0).sum())
    roi_top = int(h * roi_frac)
    roi = mask[roi_top:, :]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=thr,
                            minLineLength=minlen, maxLineGap=gap)
    mids = []
    viz = img.copy() if draw else None
    if lines is not None:
        for ln in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = int(ln[0]), int(ln[1]), int(ln[2]), int(ln[3])
            mids.append((x1 + x2) / 2.0)
            if draw:
                cv2.line(viz, (x1, y1 + roi_top), (x2, y2 + roi_top), (0, 255, 255), 2)
    detected = len(mids) > 0
    center = w / 2.0
    mid_x = float(np.mean(mids)) if detected else center
    error = center - mid_x
    if draw:
        cv2.line(viz, (int(mid_x), 0), (int(mid_x), h), (0, 255, 0), 2)
        cv2.line(viz, (int(center), 0), (int(center), h), (0, 0, 255), 2)
    return error, detected, len(mids), npix, mask, viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--speed", type=int, default=25)
    ap.add_argument("--host", default=C.HOST)
    ap.add_argument("--port", type=int, default=C.PORT)
    args = ap.parse_args()

    client = SimClient(host=args.host, port=args.port).connect()
    print(f"[tune] connected. Collecting ~{args.seconds}s of frames "
          f"(car drives with a sensitive detector)...")
    frames = []
    import time
    last_angle, last_speed = 0, 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            speed_kmh, _, img = client.step(last_angle, last_speed)
            frames.append(img)
            err, det, *_ = detect(img, COLLECT)
            last_angle = int(np.clip(0.05 * err, -20, 20))     # gentle proportional steer
            last_speed = args.speed
    finally:
        client.close()

    n = len(frames)
    print(f"[tune] collected {n} frames. Evaluating {len(CANDIDATES)} configs:\n")
    header = f"{'config':28s} {'det_rate':>8s} {'mean|err|':>9s} {'med_lines':>9s} {'med_yellowpx':>12s}"
    print(header)
    print("-" * len(header))
    results = []
    for cfg in CANDIDATES:
        dets, errs, nlines, npixs = [], [], [], []
        for img in frames:
            err, det, nl, npx, _, _ = detect(img, cfg)
            dets.append(det)
            npixs.append(npx)
            nlines.append(nl)
            if det:
                errs.append(abs(err))
        dr = float(np.mean(dets)) if dets else 0.0
        me = float(np.mean(errs)) if errs else float("nan")
        results.append((cfg[0], dr, me))
        print(f"{cfg[0]:28s} {dr:8.2f} {me:9.1f} {int(np.median(nlines)):9d} {int(np.median(npixs)):12d}")

    best = max(results, key=lambda r: (r[1], -(r[2] if r[2] == r[2] else 1e9)))
    print(f"\n[tune] BEST by detection-rate: '{best[0]}'  (det_rate={best[1]:.2f}, mean|err|={best[2]:.1f})")

    # save sample debug frames for the best + conference config
    idxs = np.linspace(0, n - 1, min(5, n)).astype(int)
    for cfg in CANDIDATES:
        if cfg[0] != best[0]:
            continue
        for j, i in enumerate(idxs):
            _, _, _, _, mask, viz = detect(frames[i], cfg, draw=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"sample{j}_raw.png"), frames[i])
            cv2.imwrite(os.path.join(DEBUG_DIR, f"sample{j}_mask.png"), mask)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"sample{j}_overlay.png"), viz)
    print(f"[tune] saved sample raw/mask/overlay PNGs to {DEBUG_DIR}")
    print("\nApply the winner by editing config.py: HSV_LOWER/UPPER_YELLOW, ROI_TOP_FRAC,\n"
          "HOUGH_THRESHOLD, HOUGH_MIN_LEN, HOUGH_MAX_GAP  (or report this table).")


if __name__ == "__main__":
    main()
