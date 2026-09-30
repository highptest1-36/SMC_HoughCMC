"""
fair_compare.py — REVIEWER-DRIVEN fair optimizer comparison (Table 6 / Fig 2 redo).

Motivation (reviewer, major point #1): the original Table 6 compared the optimizers
at their DEFAULT budgets (BO 60, PSO 80, Random 40, Grid 27) and Grid searched only
3 of the 6 gains. That is not a controlled comparison. This driver removes those
confounds:

  * SAME search space   : the full 6-D boundary-layer SMC gain vector (switch='sat')
  * SAME budget         : exactly N_EVALS closed-loop trials for every optimizer
  * SAME objective      : the identical J and reference scales used everywhere else
  * >= N_SEEDS repeats  : each optimizer is run from scratch N_SEEDS times with
                          different random seeds, so best-J is reported as mean +/- SD
                          and BO-vs-PSO / BO-vs-Random are tested for significance.

Stochastic optimizers compared on equal footing: random, pso, bo.
Grid search is deterministic and 6-D-infeasible (levels**6); it is therefore run
ONCE, separately, at a matched *evaluation count* on the 3 steering gains, and is
reported as a dimension-reduced reference only (never averaged into the seeded set).

Everything is checkpointed after each (optimizer, seed): if the sim freezes, just
re-run the SAME command and it resumes, skipping finished runs.

Usage (from inside journal_cmc/, with the Unity map or mock_server.py running):

    python run.py faircompare                       # full: 60 evals x 10 seeds x {random,pso,bo}
    python run.py faircompare --evals 60 --seeds 10 # explicit
    python run.py faircompare --evals 40 --seeds 8  # cheaper but still fair
    python run.py faircompare --profile medium      # shorter episodes (20s) to cut wall time
    python run.py faircompare --with-grid           # also run the 3-D grid reference once
    python run.py faircompare --fresh               # ignore checkpoint, start over

Runtime reality: every trial is a REAL-TIME episode (T_settle + T_measure). At the
full profile that is ~51 s/eval, so one 60-eval run is ~51 min and the full
{random,pso,bo} x 10-seed grid is ~25 h. Use --profile medium and/or fewer
evals/seeds to shorten it; it is checkpointed so it can be spread over sessions.
"""
import json
import os
import time

import numpy as np

import config as C
import objective as OBJ
import optimizers as OPT

RESULT_PATH = os.path.join(C.OPTIM_DIR, "fair_compare_results.json")
SUMMARY_PATH = os.path.join(C.OPTIM_DIR, "fair_compare_summary.json")
CONV_CSV = os.path.join(C.TABLE_DIR, "fair_convergence.csv")

SEEDED_OPTIMIZERS = ("random", "pso", "bo")


def _save_json(obj, path):
    from experiments import _clean
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)
    return path


def _make_evaluator(client, t_settle, t_measure, seed, refs):
    """A fresh 6-D SMC evaluator (switch='sat') for one optimizer run.

    base_seed = seed*1000 gives each outer seed a distinct per-candidate episode
    seed stream, so the outer seeds are genuine independent replications."""
    return OBJ.Evaluator(client, t_settle, t_measure, switch="sat",
                         perturb="clean", n_seeds=C.OPT_EVAL_SEEDS,
                         base_seed=seed * 1000, refs=refs, tag=f"fair_s{seed}")


def _run_one(which, evaluator, n_evals, seed):
    """Run one optimizer for exactly (as close as the method allows to) n_evals
    evaluations on the shared evaluator, all from random seed `seed`."""
    if which == "random":
        return OPT.random_search(evaluator, n=n_evals, seed=seed)
    if which == "bo":
        # keep the same init fraction as the paper (12/60 = 0.2), floored at 4
        n_init = max(4, min(n_evals - 1, int(round(0.2 * n_evals))))
        return OPT.bayes_opt(evaluator, n_calls=n_evals, n_init=n_init,
                             seed=seed, backend="auto")
    if which == "pso":
        # match the eval count: n_particles * iters == n_evals (10 x 6 = 60 by default)
        n_particles = 10
        iters = max(1, int(round(n_evals / n_particles)))
        return OPT.pso(evaluator, n_particles=n_particles, iters=iters, seed=seed)
    raise ValueError(f"unknown optimizer '{which}'")


def _compact(res, seed):
    """Keep only what the tables/figures need (drop the heavy per-eval history)."""
    return {
        "seed": seed,
        "best_J": res["best_J"],
        "n_evals": res["n_evals"],
        "wall_time": res["wall_time"],
        "backend": res.get("backend"),
        "device": res.get("device"),
        "running_best": res["running_best"],
        "best_gains": res["best_gains"],
        "best_metrics": {k: res["best_metrics"].get(k) for k in
                         ("lane_rmse_px", "ssi", "steer_jerk", "speed_rmse",
                          "sat_ratio", "lyap_penalty")}
        if res.get("best_metrics") else None,
    }


def run_fair_compare(client, profile="full", n_evals=60, n_seeds=10,
                     with_grid=False, resume=True):
    t_settle, t_measure, _ = C.profile(profile)
    refs = OBJ.load_reference_scales()

    out = {"n_evals": int(n_evals), "n_seeds": int(n_seeds), "dims": 6,
           "space": "boundary-layer SMC, 6-D (switch=sat)", "profile": profile,
           "optimizers": list(SEEDED_OPTIMIZERS),
           "data": {o: [] for o in SEEDED_OPTIMIZERS}, "grid": None,
           "design": "equal search space + equal budget + N_SEEDS independent repeats; "
                     "each (optimizer, seed) run from scratch on the identical objective"}

    done = set()
    if resume and os.path.exists(RESULT_PATH):
        try:
            prior = json.load(open(RESULT_PATH))
        except Exception as e:
            print(f"[fair] could not read checkpoint ({e}); starting fresh")
            prior = None
        if (prior and prior.get("n_evals") == n_evals
                and prior.get("n_seeds") == n_seeds):
            out = prior
            out.setdefault("data", {o: [] for o in SEEDED_OPTIMIZERS})
            for o in SEEDED_OPTIMIZERS:
                out["data"].setdefault(o, [])
                for rec in out["data"][o]:
                    done.add((o, rec["seed"]))
            print(f"[fair] RESUME: {len(done)} (optimizer,seed) runs already done")

    total = len(SEEDED_OPTIMIZERS) * n_seeds
    t_start = time.time()
    for which in SEEDED_OPTIMIZERS:
        for seed in range(n_seeds):
            if (which, seed) in done:
                continue
            idx = len(done) + 1
            print(f"\n[fair] ({idx}/{total}) optimizer={which} seed={seed} "
                  f"| {n_evals} evals x {C.OPT_EVAL_SEEDS} episode(s) each")
            ev = _make_evaluator(client, t_settle, t_measure, seed, refs)
            res = _run_one(which, ev, n_evals, seed)
            rec = _compact(res, seed)
            out["data"][which].append(rec)
            done.add((which, seed))
            _save_json(out, RESULT_PATH)                # checkpoint after every run
            print(f"[fair]   done: best J={rec['best_J']:.4f} "
                  f"in {rec['n_evals']} evals ({rec['wall_time']/60:.1f} min) "
                  f"| backend={rec['backend']}")

    if with_grid and out.get("grid") is None:
        print("\n[fair] grid reference (deterministic, 3 steering gains only)")
        # pick levels so levels**3 is close to n_evals (4**3 = 64 ~ 60)
        levels = max(2, int(round(n_evals ** (1.0 / 3.0))))
        ev = _make_evaluator(client, t_settle, t_measure, 0, refs)
        gres = OPT.grid_search(ev, levels=levels)
        out["grid"] = {"levels": levels, "n_evals": gres["n_evals"],
                       "best_J": gres["best_J"], "best_gains": gres["best_gains"],
                       "dims_searched": gres.get("dims_searched"),
                       "note": "dimension-reduced reference; NOT part of the seeded set"}
        _save_json(out, RESULT_PATH)

    _save_json(out, RESULT_PATH)
    summarize(out)
    print(f"\n[fair] complete in {(time.time()-t_start)/60:.1f} min "
          f"(this session). Results: {RESULT_PATH}")
    return out


# --------------------------------------------------------------------------- #
# summary: mean +/- SD, paired significance, convergence CSV, paste-ready lines
# --------------------------------------------------------------------------- #
def summarize(out=None):
    if out is None:
        out = json.load(open(RESULT_PATH))
    data = out["data"]
    n_evals = out["n_evals"]
    summary = {"n_evals": n_evals, "n_seeds": out["n_seeds"], "per_optimizer": {}}

    print("\n" + "=" * 66)
    print(f"FAIR OPTIMIZER COMPARISON  ({n_evals} evals, "
          f"6-D SMC space, {out['n_seeds']} seeds requested)")
    print("=" * 66)
    print(f"{'optimizer':<14}{'runs':>5}{'best J mean':>14}{'SD':>9}"
          f"{'min':>9}{'wall min':>10}")
    for o in out["optimizers"]:
        recs = data.get(o, [])
        Js = [r["best_J"] for r in recs if r.get("best_J") is not None]
        walls = [r["wall_time"] for r in recs if r.get("wall_time") is not None]
        if not Js:
            print(f"{o:<14}{0:>5}{'--':>14}")
            continue
        mean, sd = float(np.mean(Js)), float(np.std(Js, ddof=1) if len(Js) > 1 else 0.0)
        summary["per_optimizer"][o] = {
            "runs": len(Js), "best_J_mean": mean, "best_J_sd": sd,
            "best_J_min": float(np.min(Js)),
            "wall_mean_s": float(np.mean(walls)) if walls else None,
            "best_J_all": Js}
        print(f"{o:<14}{len(Js):>5}{mean:>14.4f}{sd:>9.4f}"
              f"{np.min(Js):>9.4f}{(np.mean(walls)/60 if walls else 0):>10.1f}")

    # paired significance vs BO (only over the common set of completed seeds)
    print("\nPaired Wilcoxon signed-rank vs Bayesian optimization "
          "(same seeds only):")
    bo = {r["seed"]: r["best_J"] for r in data.get("bo", [])}
    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except Exception:
        have_scipy = False
        print("  (scipy not available — skipping significance test)")
    summary["significance"] = {}
    for o in ("pso", "random"):
        other = {r["seed"]: r["best_J"] for r in data.get(o, [])}
        common = sorted(set(bo) & set(other))
        if len(common) < 2:
            continue
        a = [bo[s] for s in common]
        b = [other[s] for s in common]
        diff_mean = float(np.mean(np.array(b) - np.array(a)))
        entry = {"n_pairs": len(common), "mean_J_gap_vs_bo": diff_mean}
        line = (f"  BO vs {o:<7} n={len(common):2d}  "
                f"mean(J_{o} - J_BO)={diff_mean:+.4f}")
        if have_scipy and len(common) >= 3 and any(x != y for x, y in zip(a, b)):
            try:
                stat, p = wilcoxon(a, b)
                entry["p_value"] = float(p)
                line += f"  p={p:.4f}"
            except Exception as e:
                line += f"  (wilcoxon failed: {e})"
        summary["significance"][f"bo_vs_{o}"] = entry
        print(line)

    _write_convergence_csv(out)
    _save_json(summary, SUMMARY_PATH)
    print(f"\n[fair] summary -> {SUMMARY_PATH}")
    print(f"[fair] convergence (mean+/-SD per eval) -> {CONV_CSV}")
    _print_paste_block(summary, out)
    return summary


def _write_convergence_csv(out):
    """Mean and SD of the running-best objective at each evaluation index, per
    optimizer, aligned on the common eval axis (for a fair Fig 2)."""
    n_evals = out["n_evals"]
    rows = [["eval"] + [f"{o}_mean" for o in out["optimizers"]]
            + [f"{o}_sd" for o in out["optimizers"]]]
    curves = {}
    for o in out["optimizers"]:
        seqs = []
        for r in out["data"].get(o, []):
            rb = r.get("running_best") or []
            if len(rb) >= n_evals:
                seqs.append(rb[:n_evals])
            elif rb:
                seqs.append(rb + [rb[-1]] * (n_evals - len(rb)))   # pad with last
        if seqs:
            arr = np.array(seqs, dtype=float)
            curves[o] = (arr.mean(axis=0), arr.std(axis=0, ddof=1) if arr.shape[0] > 1
                         else np.zeros(arr.shape[1]))
    for i in range(n_evals):
        row = [i + 1]
        for o in out["optimizers"]:
            row.append(f"{curves[o][0][i]:.5f}" if o in curves else "")
        for o in out["optimizers"]:
            row.append(f"{curves[o][1][i]:.5f}" if o in curves else "")
        rows.append(row)
    with open(CONV_CSV, "w") as f:
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _print_paste_block(summary, out):
    po = summary["per_optimizer"]
    label = {"random": "Random search", "pso": "Particle swarm optimization",
             "bo": "Proposed Bayesian optimization"}
    print("\n" + "-" * 66)
    print("PASTE-READY  (Table 6, equal-budget fair comparison)")
    print("-" * 66)
    print(f"All optimizers: {out['n_evals']} evaluations, 6-D space, "
          f"{out['n_seeds']} seeds. Best objective, mean +/- SD:")
    for o in ("random", "pso", "bo"):
        if o in po:
            s = po[o]
            print(f"  {label[o]:<32} & {out['n_evals']} & 6 & "
                  f"{s['best_J_mean']:.2f}$\\pm${s['best_J_sd']:.2f} \\\\")
    if out.get("grid"):
        g = out["grid"]
        print(f"  {'Grid search (3-D reference)':<32} & {g['n_evals']} & 3 & "
              f"{g['best_J']:.2f} \\\\   % deterministic, not seeded")
    print("-" * 66)
