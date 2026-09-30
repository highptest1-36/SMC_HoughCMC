"""
metrics.py — compute all paper metrics from a single episode trajectory log.

A `log` is a dict of equal-length lists produced by simulator.run_episode:
    t, dt, lane_error, lane_width, detected,
    angle, speed_cmd, speed, speed_err,
    s_angle, ds_angle, s_speed, ds_speed

`compute_metrics(log)` returns a flat dict of scalars grouped as in the paper:
  Group 1  Lane keeping accuracy
  Group 2  Steering smoothness / chattering
  Group 3  Speed and recovery
  Group 4  Safety / control constraints  (+ Lyapunov)

All "lower is better" except detection_rate. NaN-safe for short/empty logs.
"""
import numpy as np

import config as C


def _arr(log, key):
    return np.asarray(log.get(key, []), dtype=float)


def compute_metrics(log):
    e_signed = _arr(log, "lane_error")
    e_abs = np.abs(e_signed)
    angle = _arr(log, "angle")
    speed = _arr(log, "speed")
    speed_err = _arr(log, "speed_err")
    speed_cmd = _arr(log, "speed_cmd")
    lane_w = _arr(log, "lane_width")
    detected = _arr(log, "detected")
    s_a = _arr(log, "s_angle")
    ds_a = _arr(log, "ds_angle")
    t = _arr(log, "t")

    n = len(e_signed)
    m = {"n_frames": int(n)}
    if n < 2:
        # Not enough data — mark as failure-ish so optimizer avoids it.
        m.update(dict(
            lane_rmse_px=np.nan, lane_mean_abs_px=np.nan, lane_max_px=np.nan,
            lane_rmse_detected=np.nan, lane_dev_pct=np.nan, lane_departures=0,
            ssi=np.nan, steer_var=np.nan, steer_jerk=np.nan,
            chattering_index=np.nan, chatter_tv=np.nan,
            speed_rmse=np.nan, speed_mean_err=np.nan, speed_overshoot=np.nan,
            settling_time=np.nan, recovery_events=0, recovery_time=np.nan,
            steer_sat_ratio=np.nan, speed_sat_ratio=np.nan, sat_ratio=np.nan,
            lyap_violation_rate=np.nan, lyap_penalty=np.nan,
            control_effort=np.nan, detection_rate=float(np.mean(detected)) if len(detected) else 0.0,
            duration=float(t[-1] - t[0]) if len(t) >= 2 else 0.0, fps=np.nan,
        ))
        return m

    duration = float(t[-1] - t[0]) if len(t) >= 2 else float(np.sum(_arr(log, "dt")))
    duration = max(duration, 1e-6)

    # ---- Group 1: lane keeping accuracy ------------------------------------ #
    # IMPORTANT: perception forces error=0 on frames where the lane is lost
    # (no Hough segment). Scoring those as perfect tracking would REWARD lane loss
    # and deflate RMSE exactly under the occlusion/blur stressors. So we impute
    # lost-lane frames with a departure-level error (last-known drift direction,
    # magnitude >= LANE_DEPARTURE_PX). Accuracy is computed on this imputed signal;
    # `lane_rmse_detected` reports the honest detected-only value for transparency.
    det = detected.astype(bool) if len(detected) == n else np.ones(n, bool)
    e_imp = _impute_lost(e_signed, det, C.LANE_DEPARTURE_PX)
    e_imp_abs = np.abs(e_imp)

    m["lane_rmse_px"] = float(np.sqrt(np.mean(e_imp ** 2)))
    m["lane_mean_abs_px"] = float(np.mean(e_imp_abs))
    m["lane_max_px"] = float(np.max(e_imp_abs))
    det_mask = det & np.isfinite(e_signed)
    m["lane_rmse_detected"] = (float(np.sqrt(np.mean(e_signed[det_mask] ** 2)))
                               if det_mask.any() else np.nan)
    valid_w = lane_w[np.isfinite(lane_w) & (lane_w > 1)]
    lane_width_med = float(np.median(valid_w)) if len(valid_w) else np.nan
    m["lane_width_px"] = lane_width_med
    m["lane_dev_pct"] = (float(np.mean(e_imp_abs) / lane_width_med * 100.0)
                         if np.isfinite(lane_width_med) else np.nan)
    m["lane_departures"] = int(_count_rising(e_imp_abs, C.LANE_DEPARTURE_PX))

    # ---- Group 2: steering smoothness / chattering ------------------------- #
    d_angle = np.diff(angle)
    m["ssi"] = float(np.mean(np.abs(d_angle)))              # conference SSI
    m["steer_var"] = float(np.var(angle))
    m["steer_jerk"] = float(np.mean(np.abs(np.diff(d_angle)))) if len(d_angle) >= 2 else 0.0
    m["chatter_tv"] = float(np.sum(np.abs(d_angle)) / duration)   # total variation / s
    m["chattering_index"] = float(_reversals(d_angle) / duration) # direction reversals / s

    # ---- Group 3: speed and recovery --------------------------------------- #
    m["speed_rmse"] = float(np.sqrt(np.mean(speed_err ** 2)))
    m["speed_mean_err"] = float(np.mean(np.abs(speed_err)))
    m["speed_overshoot"] = (float((np.max(speed) - C.SPEED_TARGET) / C.SPEED_TARGET * 100.0)
                            if np.max(speed) > C.SPEED_TARGET else 0.0)
    m["settling_time"] = _settling_time(t, speed)
    rec_events, rec_time = _recovery(t, e_abs)
    m["recovery_events"] = int(rec_events)
    m["recovery_time"] = float(rec_time)

    # ---- Group 4: safety / control constraints + Lyapunov ------------------ #
    m["steer_sat_ratio"] = float(np.mean(np.abs(angle) >= C.STEER_CLIP * (1 - 1e-3)))
    hi = speed_cmd >= C.SPEED_CLIP[1] * (1 - 1e-3)
    lo = speed_cmd <= C.SPEED_CLIP[0] + 1e-3
    m["speed_sat_ratio"] = float(np.mean(hi | lo))
    m["sat_ratio"] = m["steer_sat_ratio"]                    # steering = safety-critical
    prod = s_a * ds_a                                        # s * ds  (reaching condition)
    m["lyap_violation_rate"] = float(np.mean(prod > 0))      # V increasing => leaving surface
    m["lyap_penalty"] = float(np.mean(np.maximum(0.0, prod)))
    m["control_effort"] = float(np.mean(angle ** 2))

    m["detection_rate"] = float(np.mean(detected)) if len(detected) else 0.0
    m["duration"] = duration
    m["fps"] = float(n / duration)
    return m


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _impute_lost(e_signed, det, departure_px):
    """Replace lost-lane frames (det==False) with a departure-level error in the
    last-known drift direction (magnitude >= departure_px). Keeps detected frames
    (incl. genuine near-zero errors) untouched. A lane loss can no longer read as
    'centered'."""
    e = np.array(e_signed, dtype=float)
    last = 0.0
    for i in range(len(e)):
        if det[i] and np.isfinite(e[i]):
            last = e[i]
        else:
            sign = 1.0 if last >= 0 else -1.0
            e[i] = sign * max(abs(last), departure_px)
    return e


def _count_rising(x, thr):
    """Count rising-edge crossings above threshold (one per excursion)."""
    above = x > thr
    return int(np.sum(above[1:] & ~above[:-1])) + (1 if len(above) and above[0] else 0)


def _reversals(d):
    """Number of sign changes in the steering increment sequence."""
    sd = np.sign(d)
    sd = sd[sd != 0]
    if len(sd) < 2:
        return 0
    return int(np.sum(sd[1:] != sd[:-1]))


def _settling_time(t, speed):
    """Time after which speed stays within +/-band of target for the rest of the run."""
    band = C.SPEED_SETTLE_BAND * C.SPEED_TARGET
    outside = np.abs(speed - C.SPEED_TARGET) > band
    idx = np.where(outside)[0]
    if len(idx) == 0:
        return 0.0
    last = idx[-1]
    if last >= len(t) - 1:
        return float(t[-1] - t[0])       # never settled
    return float(t[last] - t[0])


def _recovery(t, e_abs):
    """Conference recovery logic on |lane error|. Returns (events, mean_time)."""
    last = None
    recs = []
    for now, e in zip(t, e_abs):
        if e > C.DISTURBANCE_THRESHOLD and last is None:
            last = now
        elif last is not None and e <= C.RESPONSE_TARGET:
            recs.append(now - last)
            last = None
    return len(recs), (float(np.mean(recs)) if recs else 0.0)
