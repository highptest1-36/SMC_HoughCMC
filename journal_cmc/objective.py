"""
objective.py — the black-box objective J(theta) the optimizers minimise.

    J(theta) = sum_i  w_i * ( metric_i(theta) / reference_i )

Six terms (all lower-is-better):
    lane_rmse_px   tracking accuracy
    ssi            steering smoothness
    steer_jerk     control jerk
    speed_rmse     speed regulation
    sat_ratio      actuator saturation
    lyap_penalty   Lyapunov-inspired reaching-condition penalty  (mean(max(0, s*ds)))

Terms are normalised by per-metric reference scales taken from a Manual-SMC
baseline run (so weights are comparable across metrics). If no baseline has been
recorded yet, config.REFERENCE_FALLBACK is used.

`Evaluator` wraps: theta -> build SMC -> run_episode(s) -> metrics -> J, while
recording every evaluation (theta, metrics, J, log file) for later tables/figures.
"""
import json
import os
import time

import numpy as np

import config as C
import controller as ctrl
import metrics as M
from simulator import run_episode

OBJ_KEYS = list(C.OBJECTIVE_WEIGHTS.keys())


# --------------------------------------------------------------------------- #
# reference scales
# --------------------------------------------------------------------------- #
def _floor_ref(k, v):
    """Consistent floor: never let a reference fall below a fraction of its fallback,
    so a near-zero baseline metric cannot make its normalised term explode."""
    floor = C.REFERENCE_FALLBACK[k] * C.REFERENCE_FLOOR_FRAC
    if v is None or not np.isfinite(v) or v <= 0:
        return float(C.REFERENCE_FALLBACK[k])
    return float(max(v, floor))


def load_reference_scales():
    if os.path.exists(C.REFERENCE_SCALES_FILE):
        with open(C.REFERENCE_SCALES_FILE) as f:
            refs = json.load(f)
        return {k: _floor_ref(k, refs.get(k)) for k in OBJ_KEYS}
    return dict(C.REFERENCE_FALLBACK)


def save_reference_scales(metrics):
    refs = {k: _floor_ref(k, metrics.get(k, np.nan)) for k in OBJ_KEYS}
    with open(C.REFERENCE_SCALES_FILE, "w") as f:
        json.dump(refs, f, indent=2)
    return refs


def objective_from_metrics(metrics, weights=None, refs=None):
    """Weighted normalised sum. Returns FAILURE_OBJECTIVE if any term is unusable."""
    weights = weights or C.OBJECTIVE_WEIGHTS
    refs = refs or load_reference_scales()
    total, breakdown = 0.0, {}
    for k in OBJ_KEYS:
        v = metrics.get(k, np.nan)
        if v is None or not np.isfinite(v):
            return C.FAILURE_OBJECTIVE, {"_failed": k}
        ref = refs.get(k, C.REFERENCE_FALLBACK[k]) or C.REFERENCE_FALLBACK[k]
        ratio = float(v) / float(ref)
        # clamp so no single metric dominates (see config OBJECTIVE_TERM_FLOOR/CAP)
        ratio = min(max(ratio, C.OBJECTIVE_TERM_FLOOR), C.OBJECTIVE_TERM_CAP)
        term = weights[k] * ratio
        breakdown[k] = term
        total += term
    return float(total), breakdown


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class Evaluator:
    """
    Callable theta -> J. Records every evaluation. Use ONE Evaluator per optimizer
    run so its `.history` holds the full trajectory (for Table 6 / Fig 2).

    n_seeds > 1 averages J over several perturbation seeds per theta (more robust,
    proportionally slower). perturb fixes the visual condition during tuning.
    """

    def __init__(self, client, t_settle, t_measure, switch="sat",
                 perturb="clean", n_seeds=1, base_seed=0, weights=None,
                 refs=None, tag="opt", save_logs=False,
                 param_names=None, low=None, high=None, build_fn=None, kind="smc"):
        self.client = client
        self.t_settle = t_settle
        self.t_measure = t_measure
        self.switch = switch
        self.perturb = perturb
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.weights = weights or C.OBJECTIVE_WEIGHTS
        self.refs = refs or load_reference_scales()
        self.tag = tag
        self.save_logs = save_logs
        # search-space abstraction (defaults to the SMC space -> backward compatible)
        self.kind = kind
        self.param_names = param_names or C.PARAM_NAMES
        self.low = list(low) if low is not None else list(C.BOUNDS_LOW)
        self.high = list(high) if high is not None else list(C.BOUNDS_HIGH)
        if build_fn is None:
            build_fn = lambda th: ctrl.SlidingModeController(
                ctrl.vector_to_gains(th), switch=self.switch)
        self.build_fn = build_fn
        self.history = []       # list of dicts: {theta, gains, J, metrics, breakdown}
        self.n_calls = 0

    def __call__(self, theta):
        theta = [float(x) for x in theta]
        gains = dict(zip(self.param_names, theta))      # generic (SMC or PID)
        con = self.build_fn(theta)

        seed_metrics = []
        for k in range(self.n_seeds):
            seed = self.base_seed + k
            log = run_episode(self.client, con, self.t_settle, self.t_measure,
                              perturb=self.perturb, seed=seed)
            met = M.compute_metrics(log)
            seed_metrics.append(met)
            if self.save_logs:
                _dump_log(log, f"{self.tag}_call{self.n_calls:03d}_seed{seed}")

        metrics = _avg_metrics(seed_metrics)
        J, breakdown = objective_from_metrics(metrics, self.weights, self.refs)
        self.history.append(dict(call=self.n_calls, theta=theta, gains=gains,
                                 J=J, metrics=metrics, breakdown=breakdown,
                                 perturb=self.perturb, kind=self.kind))
        self.n_calls += 1
        return J

    def best(self):
        if not self.history:
            return None
        return min(self.history, key=lambda h: h["J"])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _avg_metrics(list_of_metrics):
    if len(list_of_metrics) == 1:
        return list_of_metrics[0]
    keys = list_of_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in list_of_metrics
                if isinstance(m.get(k), (int, float)) and np.isfinite(m.get(k))]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def _dump_log(log, name):
    path = os.path.join(C.LOG_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump({k: list(map(float, v)) for k, v in log.items()}, f)
    return path
