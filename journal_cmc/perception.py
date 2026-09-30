"""
perception.py — Hough-based yellow-lane detection + visual perturbation injection.

ONE canonical perception is used by every controller so that method comparisons
isolate the CONTROL layer (the paper's contribution). The lane detector is the
bottom-half-ROI yellow+Hough pipeline from the conference SMC_no_Hough.py.

It also:
  * estimates lane width (px) so deviation can be reported as % of lane width, and
  * injects controlled visual perturbations (noise / shadow / occlusion / blur)
    used for the robustness experiments — the sim only has ONE physical track,
    so robustness stress comes from perturbing the received camera image.
"""
import cv2
import numpy as np

import config as C


# --------------------------------------------------------------------------- #
# Visual perturbations (applied to the RGB frame BEFORE lane detection)
# --------------------------------------------------------------------------- #
def apply_perturbation(image, kind, rng):
    """Return a perturbed copy of `image`. `rng` is a numpy Generator (seeded)."""
    if kind is None or kind == "clean":
        return image
    img = image.copy()
    h, w = img.shape[:2]

    if kind == "noise":
        # additive Gaussian sensor noise
        noise = rng.normal(0, 25, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif kind == "shadow":
        # a dark semi-transparent polygon over part of the frame (cast shadow)
        overlay = img.copy()
        x0 = int(rng.uniform(0, w * 0.5))
        x1 = int(rng.uniform(w * 0.5, w))
        pts = np.array([[x0, 0], [x1, 0], [x1, h], [x0, h]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 0))
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    elif kind == "occlusion":
        # an opaque black rectangle occluding a chunk of the road
        bw, bh = int(w * 0.25), int(h * 0.30)
        x = int(rng.uniform(0, max(1, w - bw)))
        y = int(rng.uniform(h * 0.4, max(h * 0.4 + 1, h - bh)))
        img[y:y + bh, x:x + bw] = 0

    elif kind == "blur":
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)

    else:
        raise ValueError(f"unknown perturbation: {kind}")
    return img


# --------------------------------------------------------------------------- #
# Lane detection
# --------------------------------------------------------------------------- #
def detect_lane(image, draw=False):
    """
    Detect the yellow lane in the bottom-half ROI and return:
        error       : center_x - mid_x  (px). +ve => lane center is LEFT of image
                      center => steer should correct accordingly. Same sign
                      convention as the conference code.
        lane_width  : estimated lane width in px (spread of detected segments),
                      np.nan if it cannot be estimated this frame.
        detected    : bool, whether any Hough segment was found.
        viz         : BGR visualization image (only if draw=True, else None).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(C.HSV_LOWER_YELLOW, np.uint8),
                       np.array(C.HSV_UPPER_YELLOW, np.uint8))
    mask = cv2.GaussianBlur(mask, C.GAUSS_KERNEL, 0)

    roi_top = int(h * C.ROI_TOP_FRAC)
    roi = mask[roi_top:, :]

    lines = cv2.HoughLinesP(roi, C.HOUGH_RHO, np.pi / C.HOUGH_THETA_DIV,
                            threshold=C.HOUGH_THRESHOLD,
                            minLineLength=C.HOUGH_MIN_LEN,
                            maxLineGap=C.HOUGH_MAX_GAP)

    mid_points = []
    xs = []
    viz = None
    if draw:
        viz = image.copy()

    if lines is not None:
        # HoughLinesP returns (N,1,4) on some OpenCV builds and (N,4) on others;
        # reshape makes this version-agnostic.
        for ln in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = int(ln[0]), int(ln[1]), int(ln[2]), int(ln[3])
            mid_points.append((x1 + x2) / 2.0)
            xs.extend([x1, x2])
            if draw:
                cv2.line(viz, (x1, y1 + roi_top), (x2, y2 + roi_top),
                         (0, 255, 255), 2)

    center_x = w / 2.0
    detected = len(mid_points) > 0
    mid_x = float(np.mean(mid_points)) if detected else center_x
    error = center_x - mid_x

    # Lane width estimate: horizontal spread of detected endpoints. Rough but the
    # only signal available (the sim sends no ground-truth pose). Used ONLY to
    # express deviation as % of lane width; declared as an approximation.
    lane_width = float(np.max(xs) - np.min(xs)) if len(xs) >= 2 else np.nan

    if draw:
        cv2.line(viz, (int(mid_x), roi_top), (int(mid_x), h), (0, 255, 0), 2)
        cv2.line(viz, (int(center_x), roi_top), (int(center_x), h), (0, 0, 255), 2)

    return error, lane_width, detected, viz
