"""
s1_optimizer.py — matched-budget optimizer study (revised Table 2 + Fig 2).
Answers: Editor 3, Reviewer 1 (grid dagger), Reviewer 2 comments 2, 3 (wall time), 4.
Source: journal_cmc/results/optim/fair_compare_results.json
"""
import numpy as np
from scipy import stats

import common as K

OPTS = [("random", "Random search"), ("pso", "Particle swarm optimization"),
        ("bo", "Proposed Bayesian optimization")]


def run():
    fc = K.load("fair_compare_results.json")
    data = {o: sorted(fc["data"][o], key=lambda r: r["seed"]) for o, _ in OPTS}
    J = {o: np.array([r["best_J"] for r in data[o]]) for o, _ in OPTS}
    rb = {o: np.array([r["running_best"] for r in data[o]]) for o, _ in OPTS}
    wall = {o: np.array([r["wall_time"] for r in data[o]]) for o, _ in OPTS}
    n_evals = {o: sorted({r["n_evals"] for r in data[o]}) for o, _ in OPTS}

    md = ["## S1. Matched-budget optimizer study (Table 2, Fig. 2)",
          f"Budget per run: {n_evals['bo']} evaluations; seeds per optimizer: {len(J['bo'])}; "
          "search space: 6-D (identical bounds). Grid search was not part of this study.\n"]

    # --- descriptive table --------------------------------------------------
    rows, csv_rows = [], []
    for o, name in OPTS:
        m, s = K.mean_sd(J[o])
        lo, hi = K.boot_ci_mean(J[o])
        q1, med, q3 = np.percentile(J[o], [25, 50, 75])
        rows.append([name, f"{m:.2f} ({s:.2f})", f"[{lo:.2f}, {hi:.2f}]", f"{med:.2f} [{q1:.2f}, {q3:.2f}]",
                     f"{J[o].min():.2f}–{J[o].max():.2f}", f"{wall[o].mean():.0f}"])
        csv_rows.append([name, m, s, lo, hi, med, q1, q3, J[o].min(), J[o].max(), wall[o].mean()])
    md.append(K.md_table(["Optimizer", "Best J mean (SD)", "95% CI (mean)", "Median [IQR]",
                          "Range (best–worst seed)", "Wall time/run (s)"], rows))
    K.write_csv("table2_optimizers.csv",
                ["optimizer", "mean", "sd", "ci_lo", "ci_hi", "median", "q1", "q3", "min", "max",
                 "wall_s"], csv_rows)

    # --- tests --------------------------------------------------------------
    comps = [("bo", "pso"), ("bo", "random"), ("pso", "random")]
    trows, p_w, p_u = [], [], []
    for a, b in comps:
        d = J[a] - J[b]
        pw, _, _ = K.wilcoxon_exact(d)
        pu = K.mwu_exact(J[a], J[b])
        p_w.append(pw)
        p_u.append(pu)
    hw = K.holm(p_w[:2])            # confirmatory family: BO vs PSO, BO vs random
    hu = K.holm(p_u[:2])
    for i, (a, b) in enumerate(comps):
        lo, hi = K.boot_ci_diff_unpaired(J[a], J[b])
        delta = K.cliffs_delta(J[a], J[b])
        wins = int((J[a] < J[b]).sum())
        trows.append([f"{a.upper()} vs {b.upper()}", f"{(J[a]-J[b]).mean():+.2f} [{lo:+.2f}, {hi:+.2f}]",
                      f"{delta:+.2f}", f"{wins}/10",
                      K.fmt_p(p_w[i]), K.fmt_p(hw[i]) if i < 2 else "—",
                      K.fmt_p(p_u[i]), K.fmt_p(hu[i]) if i < 2 else "—"])
    md.append("\n**Tests** (difference = first − second; negative favours the first optimizer). "
              "Wilcoxon = paired by seed index, as in the submitted manuscript. "
              "Mann–Whitney = unpaired; appropriate because optimizers ran sequentially "
              "(all random, then PSO, then BO) and the simulator noise is not seeded, so the seed "
              "index does not create a real pairing. Holm over the 2 confirmatory comparisons.\n")
    md.append(K.md_table(["Comparison", "Mean diff [95% CI]", "Cliff's δ", "Seed-wise wins",
                          "Wilcoxon p", "Holm", "Mann–Whitney p", "Holm"], trows))
    K.write_csv("table2_tests.csv", ["comparison", "mean_diff_ci", "cliffs_delta", "wins",
                                     "wilcoxon_p", "wilcoxon_holm", "mwu_p", "mwu_holm"], trows)

    # --- dispersion -----------------------------------------------------------
    bf_r = stats.levene(J["bo"], J["random"], center="median").pvalue
    bf_p = stats.levene(J["bo"], J["pso"], center="median").pvalue
    rng = np.random.default_rng(K.RNG_SEED)
    ratios = []
    for _ in range(K.N_BOOT):
        a = rng.choice(J["random"], 10)
        b = rng.choice(J["bo"], 10)
        if np.std(b, ddof=1) > 0:
            ratios.append(np.std(a, ddof=1) / np.std(b, ddof=1))
    rlo, rhi = np.percentile(ratios, [2.5, 97.5])
    sd = {o: np.std(J[o], ddof=1) for o, _ in OPTS}
    md.append(f"\n**Dispersion.** SD: random {sd['random']:.3f}, PSO {sd['pso']:.3f}, BO {sd['bo']:.3f}. "
              f"SD ratio random/BO = {sd['random']/sd['bo']:.2f} (bootstrap 95% CI {rlo:.2f}–{rhi:.2f}); "
              f"Brown–Forsythe p = {bf_r:.3f} (random vs BO), {bf_p:.3f} (PSO vs BO). "
              f"Worst seed: BO {J['bo'].max():.2f}, random {J['random'].max():.2f}, PSO {J['pso'].max():.2f}. "
              f"Best single run of the study: random seed {int(np.argmin(J['random']))} (J = {J['random'].min():.2f}).")

    # --- convergence ------------------------------------------------------------
    mean_rb = {o: rb[o].mean(0) for o, _ in OPTS}
    sd_rb = {o: rb[o].std(0, ddof=1) for o, _ in OPTS}
    first_below = {}
    for o in ("random", "pso"):
        target = mean_rb[o][-1]
        idx = np.where(mean_rb["bo"] <= target)[0]
        first_below[o] = int(idx[0]) + 1 if len(idx) else None
    at = lambda o, e: f"{mean_rb[o][e-1]:.2f} ({sd_rb[o][e-1]:.2f})"
    md.append("\n**Convergence: mean (SD) of the running best.** "
              + "; ".join(f"{o.upper()}: eval 12 {at(o,12)}, eval 30 {at(o,30)}, eval 60 {at(o,60)}"
                          for o, _ in OPTS)
              + f". BO mean first reaches the final (eval-60) mean of PSO at evaluation {first_below['pso']} "
              f"and of random search at evaluation {first_below['random']}.")
    found_at = [int(np.argmax(np.array(r["running_best"]) <= r["best_J"] + 1e-12)) + 1
                for r in data["bo"]]
    md.append(f"BO found its final best at evaluations {found_at} (still improving near 60 → not converged).")

    # --- wall-time composition -------------------------------------------------
    over = (wall["bo"] - wall["random"]) / 48.0
    tot_h = sum(wall[o].sum() for o, _ in OPTS) / 3600
    md.append(f"\n**Wall time.** Per evaluation ≈ 6 s settle + seed-dependent extra settle U(0,3) s "
              f"(constant within a run) + 45 s measurement. Mean per 60-evaluation run: random "
              f"{wall['random'].mean():.0f} s, PSO {wall['pso'].mean():.0f} s, BO {wall['bo'].mean():.0f} s. "
              f"BO overhead (GP fit + LogEI optimisation, same seeds as random) = "
              f"{over.mean():.2f} s (SD {over.std(ddof=1):.2f} s) per BO iteration (48 iterations). "
              f"Total closed-loop time of the study = {tot_h:.1f} h.")

    # --- BO per-seed gains ----------------------------------------------------
    grows = []
    for r in data["bo"]:
        g = r["best_gains"]
        lam, eta, phi = g["lambda_a"], g["eta_a"], g["phi_a"]
        grows.append([r["seed"], f"{r['best_J']:.3f}", f"{lam:.3f}", f"{eta:.3f}", f"{phi:.1f}",
                      f"{eta/phi:.4f}", f"{lam*(1+eta/phi):.3f}"])
    md.append("\n**BO best gains per seed** (steering channel):\n")
    md.append(K.md_table(["Seed", "Best J", "λθ", "ηθ", "φθ", "ηθ/φθ", "K_P = λθ(1+ηθ/φθ)"], grows))
    K.write_csv("bo_seeds_gains.csv", ["seed", "best_J", "lambda", "eta", "phi", "eta_over_phi", "K_P"], grows)

    # --- LaTeX table 2 -----------------------------------------------------------
    tex = [r"\begin{tabularx}{\textwidth}{lCCCC}", r"\toprule",
           r"\textbf{Optimizer} & \textbf{Best objective, mean $\pm$ SD} & \textbf{95\% CI} & "
           r"\textbf{Median [IQR]} & \textbf{Worst seed}\\", r"\midrule"]
    for o, name in OPTS:
        m, s = K.mean_sd(J[o]); lo, hi = K.boot_ci_mean(J[o])
        q1, med, q3 = np.percentile(J[o], [25, 50, 75])
        tex.append(f"{name} & {m:.2f}$\\pm${s:.2f} & [{lo:.2f}, {hi:.2f}] & {med:.2f} [{q1:.2f}, {q3:.2f}] & {J[o].max():.2f}\\\\")
    tex += [r"\bottomrule", r"\end{tabularx}"]
    K.write_text("table2_optimizers.tex", "\n".join(tex))

    _figure(rb, J)
    return "\n".join(md)


def _figure(rb, J):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"random": "#7f7f7f", "pso": "#9467bd", "bo": "#2ca02c"}
    name = {"random": "Random search", "pso": "PSO", "bo": "Proposed BO"}
    fig, a1 = plt.subplots(1, 1, figsize=(6.4, 3.6))
    x = np.arange(1, rb["bo"].shape[1] + 1)
    for o in ("random", "pso", "bo"):
        for row in rb[o]:
            a1.plot(x, row, color=col[o], lw=0.5, alpha=0.35)
        m, s = rb[o].mean(0), rb[o].std(0, ddof=1)
        a1.plot(x, m, color=col[o], lw=2.2, label=name[o])
        a1.fill_between(x, m - s, m + s, color=col[o], alpha=0.12)
    a1.set_xlabel("Evaluation")
    a1.set_ylabel("Best objective so far, J")
    a1.set_xlim(1, x[-1])
    a1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(K.os.path.join(K.OUT_FIG, "fig2_convergence_rev.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    print(run())
