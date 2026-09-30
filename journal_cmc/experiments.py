"""
experiments.py — the experiment drivers that produce the paper's raw results.

Each driver saves JSON under results/ so everything is reusable and the tables /
figures (analysis.py) are built OFFLINE from these files — you never have to
re-run the simulator to regenerate a table or plot.

Drivers
  set_reference   run Manual-SMC once, store per-metric normalisation scales
  run_optimize    tune SMC gains with one optimizer (bo/pso/grid/random)
  run_all_opt     run all four optimizers (Table 6 + Table 2 + Fig 2)
  run_compare     5 methods x N seeds, clean condition (Table 3 + Fig 3/5)
  run_robustness  {PID,ManualSMC,BO_SMC} x perturbations x seeds (Table 4 + Fig 4)
  run_ablation    5 SMC variants (+/-boundary, +/-Lyapunov) (Table 5)
"""
import copy
import json
import os
import time

import numpy as np

import config as C
import controller as ctrl
import metrics as M
import objective as OBJ
import optimizers as OPT
from simulator import run_episode


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #
def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)
    return path


def _clean(o):
    """Make numpy/bool JSON-serialisable."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _dump_log(log, name):
    path = os.path.join(C.LOG_DIR, name + ".json")
    _save_json({k: list(v) for k, v in log.items()}, path)
    return path


def _load_prior(path, resume):
    """Load a prior checkpoint file for resume, or None. Corrupt/partial files are
    ignored (start fresh)."""
    if resume and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[resume] could not read {os.path.basename(path)} ({e}); starting fresh")
    return None


def load_optimized_gains(method_key):
    """method_key in {'grid','pso','bo'} -> gains dict from its saved result."""
    path = os.path.join(C.OPTIM_DIR, f"{method_key}_result.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No optimizer result at {path}. Run `python run.py optimize --which {method_key}` "
            f"(or `python run.py optimize`) first.")
    with open(path) as f:
        res = json.load(f)
    return res["best_gains"]


def controller_for(method):
    """Build the controller for a named comparison method."""
    if method == "PID":
        return ctrl.build_controller("PID")                 # conference, untuned
    if method == "ManualSMC":
        return ctrl.build_controller("ManualSMC")           # conference, untuned
    if method == "TunedPID":
        return ctrl.PIDController(load_optimized_gains("pid"))   # BO-tuned PID
    key = {"GridSMC": "grid", "PSO_SMC": "pso", "BO_SMC": "bo"}[method]
    gains = load_optimized_gains(key)
    return ctrl.build_controller(method, gains=gains, switch="sat")


# --------------------------------------------------------------------------- #
# reference scales (Manual-SMC baseline)
# --------------------------------------------------------------------------- #
def set_reference(client, profile="full"):
    t_settle, t_measure, _ = C.profile(profile)
    con = ctrl.build_controller("ManualSMC")
    n_ref = 2 if profile == "quick" else C.N_REFERENCE_EPISODES
    per = []
    for k in range(n_ref):                       # average over several episodes (#15)
        log = run_episode(client, con, t_settle, t_measure, perturb="clean", seed=k)
        per.append(M.compute_metrics(log))
        if k == 0:
            _dump_log(log, "reference_manualSMC")
    met = OBJ._avg_metrics(per)
    refs = OBJ.save_reference_scales(met)
    print(f"[reference] Manual-SMC baseline over {n_ref} episodes. Normalisation scales:")
    for kk, v in refs.items():
        print(f"    {kk:14s} = {v:.4f}")
    return refs, met


# --------------------------------------------------------------------------- #
# optimization
# --------------------------------------------------------------------------- #
def run_optimize(client, which, profile="full", seed=0, perturb="clean",
                 n_seeds=None, save_logs=False):
    t_settle, t_measure, budget = C.profile(profile)
    if n_seeds is None:
        n_seeds = C.OPT_EVAL_SEEDS          # #10: >=2 makes J robust to sim noise
    refs = OBJ.load_reference_scales()

    if which == "pid":
        # Tuned-PID baseline: same BO + same objective, but over the PID gain space.
        ev = OBJ.Evaluator(client, t_settle, t_measure, perturb=perturb,
                           n_seeds=n_seeds, base_seed=seed * 1000, refs=refs,
                           tag="pid", save_logs=save_logs,
                           param_names=C.PID_PARAM_NAMES, low=C.PID_BOUNDS_LOW,
                           high=C.PID_BOUNDS_HIGH, kind="pid",
                           build_fn=lambda th: ctrl.PIDController(ctrl.pid_vector_to_gains(th)))
    else:
        ev = OBJ.Evaluator(client, t_settle, t_measure, switch="sat", perturb=perturb,
                           n_seeds=n_seeds, base_seed=seed * 1000, refs=refs,
                           tag=which, save_logs=save_logs)

    print(f"[optimize:{which}] starting (profile={profile}) ...")
    if which in ("bo", "pid"):
        res = OPT.bayes_opt(ev, n_calls=budget["bo_n_calls"],
                            n_init=budget["bo_n_init"], seed=seed, backend="auto")
    elif which == "pso":
        res = OPT.pso(ev, n_particles=budget["pso_particles"],
                      iters=budget["pso_iters"], seed=seed)
    elif which == "grid":
        res = OPT.grid_search(ev, levels=budget["grid_levels"])
    elif which == "random":
        res = OPT.random_search(ev, n=budget["random_n"], seed=seed)
    else:
        raise ValueError(f"unknown optimizer '{which}'")

    _save_json(res, os.path.join(C.OPTIM_DIR, f"{which}_result.json"))
    # full per-eval history with metrics (for deeper analysis)
    _save_json(ev.history, os.path.join(C.OPTIM_DIR, f"{which}_history.json"))
    print(f"[optimize:{which}] done in {res['wall_time']:.1f}s | "
          f"{res['n_evals']} evals | best J={res['best_J']:.4f} "
          f"| backend={res['backend']} device={res['device']}")
    print(f"    best gains: {res['best_gains']}")
    return res


def run_all_opt(client, profile="full", seed=0):
    results = {}
    # SMC tuning (random/grid/pso/bo) for Table 2/6 + Fig 2 ...
    for which in ("random", "grid", "pso", "bo"):
        results[which] = run_optimize(client, which, profile=profile, seed=seed)
    # ... plus the BO-tuned PID baseline (TunedPID method in Table 3/7).
    results["pid"] = run_optimize(client, "pid", profile=profile, seed=seed)
    _save_json(results, os.path.join(C.OPTIM_DIR, "all_optimizers_summary.json"))
    return results


# --------------------------------------------------------------------------- #
# overall comparison (Table 3, Fig 3/5)
# --------------------------------------------------------------------------- #
def run_compare(client, profile="full", seed=0, methods=None, resume=True):
    t_settle, t_measure, budget = C.profile(profile)
    methods = methods or C.METHODS
    n_seeds = budget["compare_seeds"]
    path = os.path.join(C.OPTIM_DIR, "compare_results.json")

    out = {"methods": {m: [] for m in methods}, "profile": profile,
           "n_seeds": n_seeds, "perturb": "clean", "log_paths": {},
           "design": "seed-outer/method-inner (interleaved, counterbalanced); "
                     "same seed => paired conditions"}
    done = set()                                    # (method, seed) already computed
    prior = _load_prior(path, resume)
    if prior and prior.get("n_seeds") == n_seeds and set(prior.get("methods", {})) == set(methods):
        out = prior
        out.setdefault("log_paths", {})
        for m in methods:
            out["methods"].setdefault(m, [])
            for rec in out["methods"][m]:
                done.add((m, rec["seed"]))
        print(f"[compare] RESUME: {len(done)} episodes already done — skipping them")

    # Build controllers once; INTERLEAVE method order within each seed (#6) so every
    # method is sampled across the same track positions and same-seed samples are
    # genuinely paired for Table 7 (#13).
    cons = {m: controller_for(m) for m in methods}
    for k in range(n_seeds):
        sd = seed * 100 + k
        order = list(methods)                       # counterbalance order per seed
        order = order[k % len(order):] + order[:k % len(order)]
        pending = [m for m in order if (m, sd) not in done]
        if not pending:
            continue
        print(f"[compare] seed {sd} | pending={pending}")
        for method in pending:
            log = run_episode(client, cons[method], t_settle, t_measure,
                              perturb="clean", seed=sd)
            met = M.compute_metrics(log)
            met["seed"] = sd
            out["methods"][method].append(met)
            done.add((method, sd))
            if k == 0 and method not in out["log_paths"]:   # rep trajectory for Fig 3/5
                out["log_paths"][method] = _dump_log(log, f"compare_{method}_seed{sd}")
            _save_json(out, path)                   # checkpoint after every episode
    _save_json(out, path)
    print("[compare] saved results/optim/compare_results.json")
    return out


# --------------------------------------------------------------------------- #
# robustness under perturbations (Table 4, Fig 4)
# --------------------------------------------------------------------------- #
def run_robustness(client, profile="full", seed=0,
                   methods=("PID", "TunedPID", "ManualSMC", "BO_SMC"),
                   perturbations=None, resume=True):
    t_settle, t_measure, budget = C.profile(profile)
    perturbations = perturbations or C.PERTURBATIONS
    n_seeds = budget["robustness_seeds"]
    path = os.path.join(C.OPTIM_DIR, "robustness_results.json")

    out = {"methods": list(methods), "perturbations": list(perturbations),
           "n_seeds": n_seeds, "data": {m: {p: [] for p in perturbations} for m in methods},
           "log_paths": {},
           "design": "seed-outer, methods interleaved within each (seed, perturbation)"}
    done = set()                                    # (method, pert, seed) already computed
    prior = _load_prior(path, resume)
    if (prior and prior.get("n_seeds") == n_seeds
            and set(prior.get("methods", [])) == set(methods)
            and set(prior.get("perturbations", [])) == set(perturbations)):
        out = prior
        out.setdefault("log_paths", {})
        for m in methods:
            out["data"].setdefault(m, {})
            for p in perturbations:
                out["data"][m].setdefault(p, [])
                for rec in out["data"][m][p]:
                    done.add((m, p, rec["seed"]))
        print(f"[robust] RESUME: {len(done)} episodes already done — skipping them")

    cons = {m: controller_for(m) for m in methods}
    # INTERLEAVE (#6): for each seed and perturbation, run all methods back-to-back so
    # they share nearly the same track position -> fair + paired.
    for k in range(n_seeds):
        sd = seed * 100 + k
        for pert in perturbations:
            pending = [m for m in methods if (m, pert, sd) not in done]
            if not pending:
                continue
            print(f"[robust] seed {sd} / {pert} : pending={pending}")
            for method in pending:
                log = run_episode(client, cons[method], t_settle, t_measure,
                                  perturb=pert, seed=sd)
                met = M.compute_metrics(log)
                met["seed"] = sd
                out["data"][method][pert].append(met)
                done.add((method, pert, sd))
                if k == 0 and f"{method}|{pert}" not in out["log_paths"]:
                    out["log_paths"][f"{method}|{pert}"] = _dump_log(
                        log, f"robust_{method}_{pert}_seed{sd}")
                _save_json(out, path)               # checkpoint after every episode
    _save_json(out, path)
    print("[robust] saved results/optim/robustness_results.json")
    return out


# --------------------------------------------------------------------------- #
# ablation (Table 5) — each variant is re-optimized with its own switch/weights
# --------------------------------------------------------------------------- #
ABLATION_VARIANTS = [
    # name              switch   use_lyap  optimize
    ("ManualSMC",       "conference", False, False),
    ("BO_noBL_noLyap",  "sign",       False, True),
    ("BO_BL_noLyap",    "sat",        False, True),
    ("BO_noBL_Lyap",    "sign",       True,  True),
    ("BO_full",         "sat",        True,  True),
]


def run_ablation(client, profile="full", seed=0, resume=True):
    t_settle, t_measure, budget = C.profile(profile)
    refs = OBJ.load_reference_scales()
    n_seeds = budget["ablation_seeds"]
    path = os.path.join(C.OPTIM_DIR, "ablation_results.json")

    out = {"variants": {}, "n_seeds": n_seeds}
    prior = _load_prior(path, resume)
    if prior and prior.get("n_seeds") == n_seeds:
        out = prior
        out.setdefault("variants", {})
        if out["variants"]:
            print(f"[ablation] RESUME: variants already done = {list(out['variants'])}")

    for name, switch, use_lyap, optimize in ABLATION_VARIANTS:
        if name in out["variants"]:
            print(f"[ablation] {name}: already done — skipping")
            continue
        print(f"[ablation] {name} (switch={switch}, lyap={use_lyap}, optimize={optimize})")
        if optimize:
            var_path = os.path.join(C.OPTIM_DIR, f"ablation_{name}_result.json")
            if resume and os.path.exists(var_path):
                # BO for this variant already finished — reuse its gains, skip the ~50min re-optimize
                gains = json.load(open(var_path))["best_gains"]
                print(f"[ablation] {name}: reusing saved BO gains (skip re-optimize)")
            else:
                weights = copy.deepcopy(C.OBJECTIVE_WEIGHTS)
                if not use_lyap:
                    weights["lyap_penalty"] = 0.0
                ev = OBJ.Evaluator(client, t_settle, t_measure, switch=switch,
                                   perturb="clean", n_seeds=1, base_seed=seed * 2000,
                                   weights=weights, refs=refs, tag=f"abl_{name}")
                res = OPT.bayes_opt(ev, n_calls=budget["bo_n_calls"],
                                    n_init=budget["bo_n_init"], seed=seed, backend="auto")
                gains = res["best_gains"]
                _save_json(res, var_path)
            con = ctrl.SlidingModeController(gains, switch=switch)
        else:
            gains = C.MANUAL_SMC_GAINS
            con = ctrl.build_controller("ManualSMC")

        per_seed = []
        for k in range(n_seeds):
            sd = seed * 100 + k
            log = run_episode(client, con, t_settle, t_measure, perturb="clean", seed=sd)
            met = M.compute_metrics(log)
            met["seed"] = sd
            per_seed.append(met)
        out["variants"][name] = {"switch": switch, "use_lyap": use_lyap,
                                 "gains": gains, "metrics": per_seed}
        _save_json(out, path)                       # checkpoint after every variant
    _save_json(out, path)
    print("[ablation] saved results/optim/ablation_results.json")
    return out
