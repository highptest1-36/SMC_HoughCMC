"""
config.py — Central configuration for the CMC journal experiments.

Everything tunable lives here so experiments are reproducible and the search
space / weights / episode lengths can be changed in ONE place.

Journal direction: "Lyapunov-Penalized Bayesian Optimization of Boundary-Layer
Sliding Mode Control for Real-Time Lane Keeping" (CMC EIDT2026).
See ../journal_cmc/README_RUN.md for the full run guide.
"""
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")       # raw per-episode trajectories
OPTIM_DIR   = os.path.join(RESULTS_DIR, "optim")      # optimizer histories + best params
TABLE_DIR   = os.path.join(RESULTS_DIR, "tables")     # assembled CSV / LaTeX tables
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")    # PNG figures
for _d in (RESULTS_DIR, LOG_DIR, OPTIM_DIR, TABLE_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Simulator (Unity Map_demo_v3.exe TCP server)
# --------------------------------------------------------------------------- #
HOST = "127.0.0.1"
PORT = 54321
RECV_BUF = 100000

# --------------------------------------------------------------------------- #
# Control targets / actuator limits (matched to the conference code)
# --------------------------------------------------------------------------- #
SPEED_TARGET = 50          # km/h  (conference 2006_hough.py used 50)
STEER_CLIP   = 20.0        # deg   (conference clipped steering to [-20, 20])
SPEED_CLIP   = (0.0, 150.0)
EPSILON      = 1e-3        # softening constant for the 'conference' switch

# --------------------------------------------------------------------------- #
# Episode timing (Option A harness: one exe, sequential trials on a looping track)
# --------------------------------------------------------------------------- #
# Wall-clock seconds. The Unity sim runs in REAL TIME, so total runtime of any
# experiment ~= (number of episodes) x (T_SETTLE + T_MEASURE). GPU cannot shorten
# this — see README section "GPU & runtime reality".
T_SETTLE_FULL  = 6.0       # drive-but-don't-log warm-up so each trial starts past transients
T_MEASURE_FULL = 45.0      # logged measurement window
T_SETTLE_MED   = 4.0
T_MEASURE_MED  = 20.0
T_SETTLE_QUICK = 2.0
T_MEASURE_QUICK = 8.0

# Disturbance / recovery detection (ported from conference code)
DISTURBANCE_THRESHOLD = 10.0   # px lane error that counts as a disturbance onset
RESPONSE_TARGET       = 5.0    # px lane error considered "recovered"
LANE_DEPARTURE_PX     = 60.0   # |lane error| above this counts as a lane-departure event
SPEED_SETTLE_BAND     = 0.05   # +/-5% of SPEED_TARGET band for settling-time

# Sim-health guard: a WORKING map detects the yellow lane ~100% of frames. If many
# consecutive episodes see almost no lane, the sim has crashed/frozen and is returning
# blank frames WITHOUT closing the socket (so auto-reconnect can't catch it). Abort with
# a clear message after this many dead episodes, instead of wasting hours on garbage data.
SIM_HEALTH_MIN_DETECTION     = 0.1
SIM_HEALTH_MAX_DEAD_EPISODES = 8
SIM_HEALTH_RECOVER_AT        = (3, 5)   # dead-streak counts at which to force a reconnect
                                        # (attempt to make the sim reset a stuck/flipped car)

# --------------------------------------------------------------------------- #
# Perception (Hough) — ONE canonical config used by ALL controllers so the
# comparison isolates the CONTROL layer (fair comparison).
# Based on SMC_no_Hough.py YellowLaneHough (bottom-half ROI).
# --------------------------------------------------------------------------- #
# HSV tuned empirically for THIS map (see tune_perception.py + results/perception_debug).
# Key insight: the yellow centre line is a PALE yellow (LOW saturation S~8-90, high value
# V>110), while grass is a vivid green (HIGH S~100-130) and the road is grey (S~5). So an
# S BAND [8, 90] keeps the line, rejects grass (upper S) AND road (lower S). This isolates
# the yellow line cleanly (grass leakage ~0), unlike a naive wide range that floods on grass.
HSV_LOWER_YELLOW = (22, 8, 110)
HSV_UPPER_YELLOW = (32, 90, 255)
GAUSS_KERNEL     = (5, 5)
HOUGH_RHO        = 1
HOUGH_THETA_DIV  = 180          # np.pi / 180
HOUGH_THRESHOLD  = 15
HOUGH_MIN_LEN    = 5
HOUGH_MAX_GAP    = 50
ROI_TOP_FRAC     = 0.0          # 0.0 = full frame (the clean mask has no grass to exclude)

# --------------------------------------------------------------------------- #
# Sliding-Mode search space  theta = [lambda_a, eta_a, phi_a, lambda_v, eta_v, phi_v]
# (matches the 6-parameter vector in the journal plan)
#   lambda : sliding-surface gain   s = lambda*e + de
#   eta    : reaching-law gain       u += eta * switch(s)
#   phi    : boundary-layer width    sat(s/phi)
# Bounds chosen around the conference values but wide enough to matter.
# --------------------------------------------------------------------------- #
PARAM_NAMES = ["lambda_a", "eta_a", "phi_a", "lambda_v", "eta_v", "phi_v"]
SEARCH_SPACE = {
    "lambda_a": (0.01, 1.0),
    "eta_a":    (0.5, 20.0),
    "phi_a":    (1.0, 80.0),
    "lambda_v": (1.0, 150.0),
    "eta_v":    (0.5, 30.0),
    "phi_v":    (1.0, 50.0),
}
BOUNDS_LOW  = [SEARCH_SPACE[n][0] for n in PARAM_NAMES]
BOUNDS_HIGH = [SEARCH_SPACE[n][1] for n in PARAM_NAMES]

# --------------------------------------------------------------------------- #
# Baseline controller configurations
# --------------------------------------------------------------------------- #
# "Manual SMC" == the conference controller, faithfully reproduced:
#   angle switch = s/(|s|+eps)  (the original soft-sign),  eta_a tiny.
# Represented here as switch='conference' with the conference gains.
MANUAL_SMC_GAINS = {
    "lambda_a": 0.1, "eta_a": 0.001, "phi_a": 1.0,     # phi unused for 'conference' switch
    "lambda_v": 70.0, "eta_v": 0.001, "phi_v": 1.0,
}
MANUAL_SMC_SWITCH = "conference"

# PID baseline == the conference PID (2106_PID_hough.py) gains, exactly.
PID_GAINS = {
    "Kp_angle": 0.2, "Ki_angle": 0.001, "Kd_angle": 0.01,
    "Kp_speed": 30.0, "Ki_speed": 0.001, "Kd_speed": 0.01,
}

# Tuned-PID baseline: the SAME PID structure, but its gains are optimized by the
# SAME Bayesian Optimization + objective as the SMC. This isolates the controller
# STRUCTURE contribution from the TUNING contribution (answers the reviewer:
# "does the proposed method beat PID because it is SMC, or just because it is tuned?").
PID_PARAM_NAMES = ["Kp_angle", "Ki_angle", "Kd_angle", "Kp_speed", "Ki_speed", "Kd_speed"]
PID_SEARCH_SPACE = {
    "Kp_angle": (0.02, 2.0),
    "Ki_angle": (0.0, 0.05),
    "Kd_angle": (0.0, 0.5),
    "Kp_speed": (1.0, 80.0),
    "Ki_speed": (0.0, 0.05),
    "Kd_speed": (0.0, 0.5),
}
PID_BOUNDS_LOW  = [PID_SEARCH_SPACE[n][0] for n in PID_PARAM_NAMES]
PID_BOUNDS_HIGH = [PID_SEARCH_SPACE[n][1] for n in PID_PARAM_NAMES]

# --------------------------------------------------------------------------- #
# Objective  J(theta) = sum_i  w_i * (metric_i / reference_i)
# Lower is better for every term. References come from a Manual-SMC baseline run
# (results/optim/reference_scales.json); fall back to REFERENCE_FALLBACK if absent.
# --------------------------------------------------------------------------- #
# Weighting philosophy (stated a priori): this is a LANE-KEEPING task, so the PRIMARY
# control objectives are tracking accuracy and speed regulation; steering smoothness,
# actuator effort and Lyapunov stability are SECONDARY regularisers. Primary weights
# therefore dominate (sum 4.0) over secondary (sum 1.2). Without this, the optimizer
# exploits an ultra-wide boundary layer -> very smooth but loose tracking (worst lane
# error), which is the opposite of what a lane-keeping controller should do.
OBJECTIVE_WEIGHTS = {
    "lane_rmse_px":   3.0,   # PRIMARY: lane-tracking accuracy
    "speed_rmse":     1.0,   # PRIMARY: speed regulation
    "ssi":            0.4,   # secondary: steering smoothness
    "steer_jerk":     0.2,   # secondary: control jerk
    "sat_ratio":      0.2,   # secondary: actuator saturation
    "lyap_penalty":   0.4,   # secondary: Lyapunov-inspired stability penalty
}
# Sensible fallback scales (order-of-magnitude of each metric) if no baseline yet.
REFERENCE_FALLBACK = {
    "lane_rmse_px":  40.0,
    "ssi":           2.0,
    "steer_jerk":    2.0,
    "speed_rmse":    10.0,
    "sat_ratio":     0.1,
    "lyap_penalty":  50.0,
}
REFERENCE_SCALES_FILE = os.path.join(OPTIM_DIR, "reference_scales.json")

# Per-term clamp on the normalised ratio metric/ref (multi-objective balancing).
# FLOOR = diminishing returns: once a metric is >~3x better than the Manual-SMC
# baseline (ratio < FLOOR) it stops reducing J, so no single metric with a huge
# dynamic range (e.g. speed) can be over-optimised at the expense of lane accuracy.
# CAP = a catastrophic metric can't dwarf every other term.
OBJECTIVE_TERM_FLOOR = 0.34
OBJECTIVE_TERM_CAP   = 3.0

# Penalty returned if an episode fails (metrics uncomputable) so optimizers avoid
# that region. Kept close to the realistic worst-case finite J (~O(10)) rather than
# a huge sentinel: a 1e3 outlier would dominate the GP's output standardisation and
# cripple the surrogate. BO also drops failed points from the GP fit (optimizers.py).
FAILURE_OBJECTIVE = 50.0

# Reference-scale floor: a near-zero baseline reference (e.g. Manual-SMC never
# saturates -> sat_ratio ref ~ 0) would let its term explode. Floor every reference
# at REFERENCE_FLOOR_FRAC of its fallback, consistently in save and load.
REFERENCE_FLOOR_FRAC = 0.2

# Robustness / methodology knobs (from adversarial review):
OPT_EVAL_SEEDS       = 1     # episodes averaged per candidate during tuning (>=2 = more
                             # robust to sim/timing noise, proportionally slower)
N_REFERENCE_EPISODES = 3     # Manual-SMC episodes averaged for the normalisation scales
SEED_JITTER_SETTLE   = 3.0   # max extra seed-dependent warm-up (s) so different seeds
                             # start at different track positions (genuine replication)

# --------------------------------------------------------------------------- #
# Optimizer budgets   (FULL = paper quality, QUICK = smoke test)
# --------------------------------------------------------------------------- #
BUDGETS = {
    "full": {
        "bo_n_calls": 60, "bo_n_init": 12,
        "pso_particles": 10, "pso_iters": 8,     # 80 evals
        "random_n": 40,
        "grid_levels": 3,                         # 3 levels on the 3 angle params = 27 evals
        "compare_seeds": 15,
        "robustness_seeds": 10,
        "ablation_seeds": 10,
    },
    "medium": {
        "bo_n_calls": 30, "bo_n_init": 8,
        "pso_particles": 6, "pso_iters": 5,       # 30 evals
        "random_n": 20,
        "grid_levels": 3,                         # 27 evals
        "compare_seeds": 6,
        "robustness_seeds": 4,
        "ablation_seeds": 4,
    },
    "quick": {
        "bo_n_calls": 8, "bo_n_init": 4,
        "pso_particles": 4, "pso_iters": 2,       # 8 evals
        "random_n": 6,
        "grid_levels": 2,                         # 8 evals
        "compare_seeds": 2,
        "robustness_seeds": 2,
        "ablation_seeds": 2,
    },
}

# --------------------------------------------------------------------------- #
# Visual-perturbation conditions (injected in Python; the sim has ONE track).
# --------------------------------------------------------------------------- #
PERTURBATIONS = ["clean", "noise", "shadow", "occlusion", "blur"]

# The methods compared in the paper (Table 3). TunedPID = BO-tuned PID baseline.
METHODS = ["PID", "TunedPID", "ManualSMC", "GridSMC", "PSO_SMC", "BO_SMC"]

# --------------------------------------------------------------------------- #
# Device (GPU used for the BoTorch surrogate only; see README).
# --------------------------------------------------------------------------- #
def torch_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def profile(name):
    """Return (T_settle, T_measure, budget_dict) for the chosen profile.
    Accepts a name ('quick'|'medium'|'full') or a bool (True=quick, False=full)
    for backward compatibility."""
    if name is True:
        name = "quick"
    elif name is False or name is None:
        name = "full"
    if name == "quick":
        return T_SETTLE_QUICK, T_MEASURE_QUICK, BUDGETS["quick"]
    if name == "medium":
        return T_SETTLE_MED, T_MEASURE_MED, BUDGETS["medium"]
    return T_SETTLE_FULL, T_MEASURE_FULL, BUDGETS["full"]
