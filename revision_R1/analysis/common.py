"""
common.py — shared paths, loaders and statistics for the Round-1 revision analysis.

Everything here is OFFLINE: it only reads the JSON files already produced by the
journal_cmc experiments. No simulator connection is made and no experiment file
is modified. All outputs go to revision_R1/output/.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

try:                                   # Windows console defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REV_DIR = os.path.dirname(HERE)
PROJECT = os.path.dirname(REV_DIR)
RESULTS = os.path.join(PROJECT, "journal_cmc", "results")
OPTIM = os.path.join(RESULTS, "optim")
LOGS = os.path.join(RESULTS, "logs")

OUT = os.path.join(REV_DIR, "output")
OUT_TAB = os.path.join(OUT, "tables")
OUT_FIG = os.path.join(OUT, "figures")
for _d in (OUT, OUT_TAB, OUT_FIG):
    os.makedirs(_d, exist_ok=True)

RNG_SEED = 20260930          # fixed seed for every bootstrap -> reproducible numbers
N_BOOT = 10000

# Objective configuration used in the submitted study (journal_cmc/config.py)
WEIGHTS = {"lane_rmse_px": 3.0, "speed_rmse": 1.0, "ssi": 0.4,
           "steer_jerk": 0.2, "sat_ratio": 0.2, "lyap_penalty": 0.4}
CLIP_LO, CLIP_HI = 0.34, 3.0
BOUNDS = {"lambda_a": (0.01, 1.0), "eta_a": (0.5, 20.0), "phi_a": (1.0, 80.0),
          "lambda_v": (1.0, 150.0), "eta_v": (0.5, 30.0), "phi_v": (1.0, 50.0)}
PARAMS = list(BOUNDS)

METHODS = ["PID", "TunedPID", "ManualSMC", "GridSMC", "PSO_SMC", "BO_SMC"]
LABEL = {"PID": "PID", "TunedPID": "Tuned PID", "ManualSMC": "Manual SMC",
         "GridSMC": "Grid-search SMC", "PSO_SMC": "PSO SMC", "BO_SMC": "Proposed BO-SMC"}

RESPAWN_SPEED_RMSE = 7.0     # km/h; runs above this contain a simulator respawn (gap 4.3 -> 10.7)
STALL_DETECTION = 0.1        # detection rate below this = stalled simulator (no driving)


def load(name):
    with open(os.path.join(OPTIM, name)) as f:
        return json.load(f)


def load_log(name):
    with open(os.path.join(LOGS, name)) as f:
        return json.load(f)


def arr(runs, key):
    return np.array([float(r[key]) for r in runs])


# --------------------------------------------------------------------------- #
# objective (exact port of journal_cmc/objective.py)
# --------------------------------------------------------------------------- #
def objective(metrics, refs, weights=None, lo=CLIP_LO, hi=CLIP_HI):
    weights = weights or WEIGHTS
    total = 0.0
    for k, w in weights.items():
        v = metrics.get(k)
        if v is None or not np.isfinite(v):
            return 50.0
        ratio = float(v) / float(refs[k])
        ratio = min(max(ratio, lo), hi)
        total += w * ratio
    return total


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilcoxon_exact(d):
    """Exact two-sided Wilcoxon signed-rank p-value by enumerating all sign flips
    (valid with ties; zeros dropped). Returns (p, W_plus, n)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0.0, 0
    ranks = stats.rankdata(np.abs(d))
    w_obs = ranks[d > 0].sum()
    mu = ranks.sum() / 2.0
    if n <= 20:
        signs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
        w_all = signs @ ranks
        p = np.mean(np.abs(w_all - mu) >= abs(w_obs - mu) - 1e-9)
    else:
        p = stats.wilcoxon(d).pvalue
    return float(min(1.0, p)), float(w_obs), n


def mwu_exact(x, y):
    return float(stats.mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def rank_biserial_paired(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return float((gt - lt) / (len(x) * len(y)))


def boot_ci_mean(x, paired_diff=True):
    """BCa 95% CI of the mean of x (x = paired differences or a single sample)."""
    x = np.asarray(x, float)
    if len(x) < 3 or np.allclose(x, x[0]):
        return float(np.mean(x)), float(np.mean(x))
    rng = np.random.default_rng(RNG_SEED)
    try:
        res = stats.bootstrap((x,), np.mean, n_resamples=N_BOOT, method="BCa",
                              random_state=rng)
    except Exception:
        res = stats.bootstrap((x,), np.mean, n_resamples=N_BOOT, method="percentile",
                              random_state=rng)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def boot_ci_diff_unpaired(x, y):
    """Percentile 95% CI of mean(x) - mean(y) with independent resampling."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(RNG_SEED)
    bx = rng.choice(x, (N_BOOT, len(x))).mean(1)
    by = rng.choice(y, (N_BOOT, len(y))).mean(1)
    lo, hi = np.percentile(bx - by, [2.5, 97.5])
    return float(lo), float(hi)


def mean_sd(x):
    x = np.asarray(x, float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_csv(name, header, rows):
    import csv
    path = os.path.join(OUT_TAB, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_text(name, text, folder=OUT_TAB):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
