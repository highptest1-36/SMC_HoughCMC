"""
analysis.py — build the paper's tables (CSV + LaTeX) and figures from saved JSON.

Runs OFFLINE: it only reads results/optim/*.json and results/logs/*.json, so you
can regenerate every table/figure without touching the simulator. Missing inputs
are skipped with a message (so you can build tables incrementally as experiments
finish).

Tables: 2 (params), 3 (overall), 4 (robustness), 5 (ablation),
        6 (optimizer efficiency), 7 (statistical significance).
Figures: 2 (convergence), 3 (steering-over-time), 4 (lateral error under
         occlusion), 5 (accuracy-smoothness trade-off).
Table 1 (conference vs journal) is descriptive and written by hand in the paper.
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless; save PNGs
import matplotlib.pyplot as plt

import config as C


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _optim(name):
    return _load(os.path.join(C.OPTIM_DIR, name))


def _mean_std(per_seed, key):
    vals = [d.get(key) for d in per_seed
            if isinstance(d.get(key), (int, float)) and np.isfinite(d.get(key))]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def _save_table(df, stem, floatfmt="%.3f"):
    csv = os.path.join(C.TABLE_DIR, stem + ".csv")
    df.to_csv(csv, index=False)
    try:
        tex = os.path.join(C.TABLE_DIR, stem + ".tex")
        with open(tex, "w") as f:
            f.write(df.to_latex(index=False, float_format=lambda x: floatfmt % x
                                if isinstance(x, float) else x))
    except Exception as e:
        print(f"    (LaTeX export skipped for {stem}: {e})")
    print(f"    wrote {csv}")
    return df


# --------------------------------------------------------------------------- #
# Table 2 — optimized parameters
# --------------------------------------------------------------------------- #
def make_table2():
    rows = []
    manual = {"Method": "Manual SMC", **C.MANUAL_SMC_GAINS}
    rows.append(manual)
    for key, label in [("grid", "Grid-SMC"), ("pso", "PSO-SMC"), ("bo", "BO-SMC (proposed)")]:
        res = _optim(f"{key}_result.json")
        if res and res.get("best_gains"):
            rows.append({"Method": label, **res["best_gains"]})
    if len(rows) <= 1:
        print("[table2] no optimizer results yet — skipped.")
        return None
    df = pd.DataFrame(rows)[["Method"] + C.PARAM_NAMES]
    print("[table2] optimized parameters")
    return _save_table(df, "table2_parameters", floatfmt="%.4f")


# --------------------------------------------------------------------------- #
# Table 3 — overall performance comparison
# --------------------------------------------------------------------------- #
T3_METRICS = [("lane_rmse_px", "LaneRMSE(px)"), ("lane_dev_pct", "Dev(%lane)"),
              ("lane_max_px", "MaxDev(px)"), ("ssi", "SSI"),
              ("steer_jerk", "Jerk"), ("speed_rmse", "SpeedRMSE"),
              ("recovery_time", "Recovery(s)"), ("sat_ratio", "SatRatio")]


def make_table3():
    data = _optim("compare_results.json")
    if not data:
        print("[table3] no compare_results.json — skipped.")
        return None
    rows = []
    label = {"PID": "PID (untuned)", "TunedPID": "Tuned PID (BO)",
             "ManualSMC": "Manual SMC", "GridSMC": "Grid-SMC",
             "PSO_SMC": "PSO-SMC", "BO_SMC": "BO-SMC (proposed)"}
    for method, per_seed in data["methods"].items():
        row = {"Method": label.get(method, method)}
        for key, disp in T3_METRICS:
            mu, sd = _mean_std(per_seed, key)
            row[disp] = f"{mu:.2f}±{sd:.2f}" if np.isfinite(mu) else "n/a"
        rows.append(row)
    df = pd.DataFrame(rows)
    print("[table3] overall performance")
    return _save_table(df, "table3_overall")


# --------------------------------------------------------------------------- #
# Table 4 — robustness under visual perturbations
# --------------------------------------------------------------------------- #
T4_METRICS = [("lane_rmse_px", "LaneRMSE(px)"), ("ssi", "SSI"),
              ("recovery_time", "Recovery(s)"), ("lane_departures", "Departures"),
              ("detection_rate", "DetRate")]


def make_table4():
    data = _optim("robustness_results.json")
    if not data:
        print("[table4] no robustness_results.json — skipped.")
        return None
    rows = []
    for pert in data["perturbations"]:
        for method in data["methods"]:
            per_seed = data["data"][method][pert]
            row = {"Scenario": pert, "Method": method}
            for key, disp in T4_METRICS:
                mu, sd = _mean_std(per_seed, key)
                row[disp] = f"{mu:.2f}±{sd:.2f}" if np.isfinite(mu) else "n/a"
            rows.append(row)
    df = pd.DataFrame(rows)
    print("[table4] robustness under perturbations")
    return _save_table(df, "table4_robustness")


# --------------------------------------------------------------------------- #
# Table 5 — ablation
# --------------------------------------------------------------------------- #
T5_METRICS = [("lane_rmse_px", "LaneRMSE(px)"), ("ssi", "SSI"),
              ("chattering_index", "Chatter/s"), ("lyap_violation_rate", "LyapViol")]


def make_table5():
    data = _optim("ablation_results.json")
    if not data:
        print("[table5] no ablation_results.json — skipped.")
        return None
    rows = []
    for name, info in data["variants"].items():
        row = {"Variant": name,
               "Boundary": "Yes" if info["switch"] == "sat" else "No",
               "Lyapunov": "Yes" if info["use_lyap"] else "No"}
        for key, disp in T5_METRICS:
            mu, sd = _mean_std(info["metrics"], key)
            row[disp] = f"{mu:.3f}±{sd:.3f}" if np.isfinite(mu) else "n/a"
        rows.append(row)
    df = pd.DataFrame(rows)
    print("[table5] ablation study")
    return _save_table(df, "table5_ablation")


# --------------------------------------------------------------------------- #
# Table 6 — optimization efficiency
# --------------------------------------------------------------------------- #
def make_table6():
    rows = []
    label = {"random": "Random Search", "grid": "Grid Search",
             "pso": "PSO", "bo": "Bayesian Optimization"}
    for key in ("random", "grid", "pso", "bo"):
        res = _optim(f"{key}_result.json")
        if not res:
            continue
        allJ = [h.get("J", np.nan) for h in res.get("history", [])]
        Js = [j for j in allJ if np.isfinite(j) and j < C.FAILURE_OBJECTIVE * 0.9]
        n_fail = sum(1 for j in allJ if np.isfinite(j) and j >= C.FAILURE_OBJECTIVE * 0.9)
        rows.append({
            "Optimizer": label[key],
            "Trials": res.get("n_evals"),
            "Fails": n_fail,                  # failed probes excluded from Mean/Std (#14)
            "Dims": res.get("n_dims", len(C.PARAM_NAMES)),   # grid searches fewer (#9/#16)
            "BestJ": round(res.get("best_J", np.nan), 4),
            "MeanJ": round(float(np.mean(Js)), 4) if Js else np.nan,
            "StdJ": round(float(np.std(Js)), 4) if Js else np.nan,
            "WallTime(s)": round(res.get("wall_time", np.nan), 1),
            "Backend": res.get("backend"),
            "Device": res.get("device"),
        })
    if not rows:
        print("[table6] no optimizer results — skipped.")
        return None
    df = pd.DataFrame(rows)
    print("[table6] optimization efficiency")
    return _save_table(df, "table6_optimizer_efficiency", floatfmt="%.4f")


# --------------------------------------------------------------------------- #
# Table 7 — statistical significance (paired tests vs proposed BO-SMC)
# --------------------------------------------------------------------------- #
T7_TESTS = [("BO_SMC", "ManualSMC", "ssi"),
            ("BO_SMC", "ManualSMC", "lane_rmse_px"),
            ("BO_SMC", "PSO_SMC", "chattering_index"),
            # the key structure-vs-tuning tests: both are BO-tuned, so a win here is
            # attributable to the SMC STRUCTURE, not just to being optimized.
            ("BO_SMC", "TunedPID", "lane_rmse_px"),
            ("BO_SMC", "TunedPID", "ssi")]


def make_table7():
    from scipy import stats
    data = _optim("compare_results.json")
    if not data:
        print("[table7] no compare_results.json — skipped.")
        return None

    def paired(method, key):
        rows = sorted(data["methods"][method], key=lambda d: d.get("seed", 0))
        return np.array([r.get(key, np.nan) for r in rows], float)

    # A-priori design (no p-hacking, #7/#13): ONE primary test = Wilcoxon signed-rank
    # (small n, non-normal, genuinely paired by seed via interleaved runs). Holm-
    # Bonferroni family-wise correction across the pre-registered comparison set.
    # The t-test p is reported for reference only; significance uses the corrected
    # Wilcoxon p.
    recs = []
    for a, b, key in T7_TESTS:
        if a not in data["methods"] or b not in data["methods"]:
            continue
        xa, xb = paired(a, key), paired(b, key)
        mask = np.isfinite(xa) & np.isfinite(xb)
        xa, xb = xa[mask], xb[mask]
        if len(xa) < 2:
            continue
        improve = float(np.mean(xb) - np.mean(xa))     # positive => proposed lower(better)
        try:
            _, p_t = stats.ttest_rel(xa, xb)
        except Exception:
            p_t = np.nan
        try:
            p_w = 1.0 if np.allclose(xa, xb) else float(stats.wilcoxon(xa, xb)[1])
        except Exception:
            p_w = np.nan
        recs.append({"a": a, "b": b, "key": key, "improve": improve,
                     "p_t": p_t, "p_w": p_w, "n": len(xa)})
    if not recs:
        print("[table7] not enough paired data — skipped.")
        return None

    p_holm = _holm([r["p_w"] for r in recs])
    rows = []
    for r, ph in zip(recs, p_holm):
        rows.append({
            "Comparison": f"{r['a']} vs {r['b']}",
            "Metric": r["key"],
            "n": r["n"],
            "MeanImprovement": round(r["improve"], 4),
            "p(Wilcoxon)": round(float(r["p_w"]), 4) if np.isfinite(r["p_w"]) else "n/a",
            "p(Holm-adj)": round(float(ph), 4) if np.isfinite(ph) else "n/a",
            "p(t-test,ref)": round(float(r["p_t"]), 4) if np.isfinite(r["p_t"]) else "n/a",
            "Significant(Holm<0.05)": "Yes" if (np.isfinite(ph) and ph < 0.05) else "No",
        })
    df = pd.DataFrame(rows)
    print("[table7] statistical significance (Wilcoxon + Holm correction)")
    return _save_table(df, "table7_significance", floatfmt="%.4f")


def _holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (nan-safe)."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = [np.nan] * len(pvals)
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)              # enforce monotonicity
        out[i] = prev
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _load_log(path):
    d = _load(path)
    return d


def fig2_convergence():
    plt.figure(figsize=(7, 4))
    any_data = False
    for key, lbl in [("random", "Random"), ("grid", "Grid"),
                     ("pso", "PSO"), ("bo", "Bayesian Opt")]:
        res = _optim(f"{key}_result.json")
        if res and res.get("running_best"):
            rb = res["running_best"]
            plt.plot(range(1, len(rb) + 1), rb, marker=".", label=lbl)
            any_data = True
    if not any_data:
        print("[fig2] no optimizer results — skipped.")
        return
    plt.xlabel("Evaluation #")
    plt.ylabel("Best objective J (lower=better)")
    plt.title("Optimizer convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig2_convergence.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"    wrote {p}")


def fig3_steering():
    data = _optim("compare_results.json")
    if not data:
        print("[fig3] no compare_results.json — skipped.")
        return
    plt.figure(figsize=(9, 4))
    plotted = False
    for method in ("PID", "ManualSMC", "BO_SMC"):
        path = data.get("log_paths", {}).get(method)
        log = _load_log(path) if path else None
        if log and log.get("t"):
            plt.plot(log["t"], log["angle"], label=method, alpha=0.8)
            plotted = True
    if not plotted:
        print("[fig3] no logs — skipped.")
        plt.close()
        return
    plt.xlabel("Time (s)")
    plt.ylabel("Steering command (deg)")
    plt.title("Steering command over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig3_steering.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"    wrote {p}")


def fig4_lateral(perturb="occlusion"):
    data = _optim("robustness_results.json")
    if not data:
        print("[fig4] no robustness_results.json — skipped.")
        return
    plt.figure(figsize=(9, 4))
    plotted = False
    for method in data["methods"]:
        path = data.get("log_paths", {}).get(f"{method}|{perturb}")
        log = _load_log(path) if path else None
        if log and log.get("t"):
            plt.plot(log["t"], np.abs(log["lane_error"]), label=method, alpha=0.8)
            plotted = True
    if not plotted:
        print("[fig4] no logs — skipped.")
        plt.close()
        return
    plt.xlabel("Time (s)")
    plt.ylabel("|Lateral error| (px)")
    plt.title(f"Lateral error under '{perturb}' perturbation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(C.FIG_DIR, f"fig4_lateral_{perturb}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"    wrote {p}")


def fig5_tradeoff():
    data = _optim("compare_results.json")
    if not data:
        print("[fig5] no compare_results.json — skipped.")
        return
    plt.figure(figsize=(6, 5))
    for method, per_seed in data["methods"].items():
        x, _ = _mean_std(per_seed, "lane_rmse_px")
        y, _ = _mean_std(per_seed, "ssi")
        if np.isfinite(x) and np.isfinite(y):
            plt.scatter(x, y, s=80)
            plt.annotate(method, (x, y), textcoords="offset points", xytext=(6, 4))
    plt.xlabel("Lane RMSE (px)  — accuracy →")
    plt.ylabel("SSI — smoothness (lower=smoother)")
    plt.title("Accuracy vs smoothness trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(C.FIG_DIR, "fig5_tradeoff.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"    wrote {p}")


# --------------------------------------------------------------------------- #
def make_all():
    print("=== TABLES ===")
    for fn in (make_table2, make_table3, make_table4, make_table5, make_table6, make_table7):
        try:
            fn()
        except Exception as e:
            print(f"    ({fn.__name__} failed: {type(e).__name__}: {e})")
    print("=== FIGURES ===")
    for fn in (fig2_convergence, fig3_steering, lambda: fig4_lateral("occlusion"), fig5_tradeoff):
        try:
            fn()
        except Exception as e:
            print(f"    (figure failed: {type(e).__name__}: {e})")
    print(f"\nAll artefacts in: {C.TABLE_DIR}  and  {C.FIG_DIR}")
