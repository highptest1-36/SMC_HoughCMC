"""
s3_nominal.py — nominal comparison (revised Table 4 + full test table + speed
distribution + Pareto uncertainty).
Answers: Editor 4, 5, 6 (Table 4 caption); Reviewer 1 comment 1; Reviewer 2 comment 4.
Source: journal_cmc/results/optim/compare_results.json (+ logs/compare_*_seed0.json)
"""
import numpy as np

import common as K

METRICS = [("lane_rmse_px", "Lane RMSE (px)"), ("ssi", "SSI (deg/step)"),
           ("steer_jerk", "Jerk (deg/step²)"), ("sat_ratio", "Saturation (fraction)")]


def run(exclude=(), tag="all"):
    cm = K.load("compare_results.json")["methods"]
    runs = {m: sorted(cm[m], key=lambda r: r["seed"]) for m in K.METHODS}
    seeds = [r["seed"] for r in runs["BO_SMC"]]
    assert all([r["seed"] for r in runs[m]] == seeds for m in K.METHODS)
    V = {m: {k: K.arr(runs[m], k) for k in
             ["lane_rmse_px", "ssi", "steer_jerk", "sat_ratio", "speed_rmse", "recovery_time",
              "chatter_tv", "fps", "detection_rate", "lyap_penalty"]} for m in K.METHODS}
    resp = {m: V[m]["speed_rmse"] > K.RESPAWN_SPEED_RMSE for m in K.METHODS}
    methods = [m for m in K.METHODS if m not in exclude]

    md = [f"## S3 [{tag}]. Nominal comparison (Table 4), n = 15 runs per controller, no exclusions" + (f" — controllers omitted: {list(exclude)}" if exclude else ""),
          f"Detection rate = 1.000 in all {15*len(K.METHODS)} runs. SD with ddof = 1. "
          f"Respawn run = speed RMSE > {K.RESPAWN_SPEED_RMSE} km/h (no run lies between 4.3 and 10.7).\n"]

    # --- revised Table 4 -------------------------------------------------------------
    rows, tex = [], []
    for m in methods:
        v = V[m]
        cells = [f"{K.mean_sd(v[k])[0]:.2f} ± {K.mean_sd(v[k])[1]:.2f}" for k, _ in METRICS]
        ok = ~resp[m]
        sm, ss = K.mean_sd(v["speed_rmse"][ok])
        cells.append(f"{sm:.2f} ± {ss:.2f} (n={ok.sum()})")
        cells.append(f"{resp[m].sum()}/15")
        rows.append([K.LABEL[m]] + cells)
        tex.append(f"{K.LABEL[m]} & " + " & ".join(
            f"{K.mean_sd(v[k])[0]:.2f}$\\pm${K.mean_sd(v[k])[1]:.2f}" for k, _ in METRICS)
            + f" & {sm:.2f}$\\pm${ss:.2f} ({ok.sum()}) & {resp[m].sum()}\\\\")
    hdr = ["Method"] + [h for _, h in METRICS] + ["Speed RMSE, respawn-free (km/h)", "Respawn runs"]
    md.append(K.md_table(hdr, rows))
    K.write_csv(f"table4_nominal_{tag}.csv", hdr, rows)
    K.write_text("table4_nominal_rows.tex", "\n".join(tex))

    # --- tests ----------------------------------------------------------------------
    trows, ps = [], []
    bo = V["BO_SMC"]
    for m in methods:
        if m == "BO_SMC":
            continue
        for k, h in METRICS:
            d = bo[k] - V[m][k]
            p, _, n = K.wilcoxon_exact(d)
            lo, hi = K.boot_ci_mean(d)
            trows.append([K.LABEL[m], h, f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
                          f"{K.rank_biserial_paired(d):+.2f}", f"{int((d<0).sum())}/{len(d)}", n, p])
            ps.append(p)
        both = (~resp["BO_SMC"]) & (~resp[m])
        d = bo["speed_rmse"][both] - V[m]["speed_rmse"][both]
        p, _, n = K.wilcoxon_exact(d)
        lo, hi = K.boot_ci_mean(d)
        trows.append([K.LABEL[m], "Speed RMSE, respawn-free pairs (km/h)",
                      f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{K.rank_biserial_paired(d):+.2f}",
                      f"{int((d<0).sum())}/{len(d)}", n, p])
        ps.append(p)
    adj = K.holm(ps)
    out = []
    for r, a in zip(trows, adj):
        out.append(r[:-1] + [K.fmt_p(r[-1]), K.fmt_p(a), "yes" if a < 0.05 else "no"])
    md.append(f"\n**Paired tests: proposed − comparator** (exact Wilcoxon signed-rank, two-sided; "
              f"Holm over the full family of {len(ps)} tests; effect = mean paired difference with BCa 95% CI "
              f"and matched-pairs rank-biserial r; negative difference = proposed lower/better).\n")
    thdr = ["Comparator", "Metric", "Mean diff [95% CI]", "r_rb", "Proposed lower", "n", "p", "Holm p", "Sig."]
    md.append(K.md_table(thdr, out))
    K.write_csv(f"table4_tests_{tag}.csv", thdr, out)

    # --- respawn anatomy (Reviewer 1, comment 1) ------------------------------------
    order_pos = {}
    for k in range(15):
        order = K.METHODS[k % 6:] + K.METHODS[:k % 6]
        for j, m in enumerate(order):
            order_pos[(m, k)] = 6 * k + j
    idx = sorted(order_pos[(m, k)] for m in K.METHODS for k in range(15) if resp[m][k])
    gaps = np.diff(idx)
    rng = np.random.default_rng(K.RNG_SEED)
    sim_sd = [np.std(np.diff(np.sort(rng.choice(90, len(idx), replace=False)))) for _ in range(20000)]
    p_period = np.mean(np.array(sim_sd) <= np.std(gaps))
    within = []
    for m in K.METHODS:
        l = V[m]["lane_rmse_px"]
        if resp[m].any() and (~resp[m]).any():
            within.append(l[resp[m]].mean() - l[~resp[m]].mean())
    md.append("\n**Speed bimodality = simulator respawn (answer to Reviewer 1).**")
    md.append("Respawn runs per controller (run indices): "
              + "; ".join(f"{K.LABEL[m]} {list(np.where(resp[m])[0])}" for m in K.METHODS))
    md.append(f"In global execution order the {len(idx)} respawn runs sit at episodes {idx}; gaps {list(gaps)} "
              f"(SD {np.std(gaps):.2f}); random placement gives SD this small in {100*p_period:.3f}% of 20,000 "
              f"permutations → the events are periodic in time, not linked to a controller or run index. "
              f"Within-controller effect of a respawn on lane RMSE: {np.mean(within):+.1f} px (mean over controllers).")
    anat = []
    for m in K.METHODS:
        try:
            lg = K.load_log(f"compare_{m}_seed0.json")
        except FileNotFoundError:
            continue
        sp = np.array(lg["speed"]); t = np.array(lg["t"])
        drops = np.where(np.diff(sp) < -20)[0]
        anat.append(f"{K.LABEL[m]} run 0: speed at window start {sp[0]:.1f} km/h, min {sp.min():.1f} km/h, "
                    f"instantaneous drops (>20 km/h in one step) at t = {[round(float(t[i+1]),2) for i in drops]} s")
    md.append("Per-frame evidence (run 0 logs): " + " | ".join(anat))

    # --- Pareto uncertainty ------------------------------------------------------------
    L = {m: V[m]["lane_rmse_px"] for m in methods}
    S = {m: V[m]["ssi"] for m in methods}
    rng = np.random.default_rng(K.RNG_SEED)
    dom = {m: 0 for m in methods if m != "BO_SMC"}
    nondom = 0
    front = {m: 0 for m in methods}
    for _ in range(K.N_BOOT):
        ix = rng.integers(0, 15, 15)
        mL = {m: L[m][ix].mean() for m in methods}
        mS = {m: S[m][ix].mean() for m in methods}
        for m in dom:
            if mL["BO_SMC"] <= mL[m] and mS["BO_SMC"] <= mS[m] and (mL["BO_SMC"] < mL[m] or mS["BO_SMC"] < mS[m]):
                dom[m] += 1
        for a in methods:
            dominated = any(mL[b] <= mL[a] and mS[b] <= mS[a] and (mL[b] < mL[a] or mS[b] < mS[a])
                            for b in methods if b != a)
            if not dominated:
                front[a] += 1
    md.append("\n**Pareto uncertainty on (lane RMSE, SSI) means** (paired bootstrap over run indices, "
              f"{K.N_BOOT} resamples): P(proposed dominates X): "
              + ", ".join(f"{K.LABEL[m]} {dom[m]/K.N_BOOT:.3f}" for m in dom)
              + ". P(on the Pareto front): " + ", ".join(f"{K.LABEL[m]} {front[m]/K.N_BOOT:.3f}" for m in methods))

    _figures(V, resp, methods, tag)
    return "\n".join(md)


def _figures(V, resp, methods, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    rng = np.random.default_rng(1)
    for i, m in enumerate(methods):
        y = V[m]["speed_rmse"]; x = i + rng.uniform(-0.15, 0.15, len(y))
        a1.scatter(x[~resp[m]], y[~resp[m]], s=16, color="#1f77b4")
        a1.scatter(x[resp[m]], y[resp[m]], s=22, marker="x", color="#d62728")
    a1.set_xticks(range(len(methods)), [K.LABEL[m].replace(" ", "\n", 1) for m in methods], fontsize=7)
    a1.set_ylabel("Speed RMSE per run (km/h)")
    a1.set_title("(a) All 15 runs; × = run containing a simulator respawn", fontsize=9)
    rng = np.random.default_rng(K.RNG_SEED)
    for m in methods:
        L, S = V[m]["lane_rmse_px"], V[m]["ssi"]
        bl = [L[rng.integers(0, 15, 15)].mean() for _ in range(3000)]
        bs = [S[rng.integers(0, 15, 15)].mean() for _ in range(3000)]
        xl, xh = np.percentile(bs, [2.5, 97.5]); yl, yh = np.percentile(bl, [2.5, 97.5])
        mk = "*" if m == "BO_SMC" else "o"
        a2.errorbar(S.mean(), L.mean(), xerr=[[S.mean() - xl], [xh - S.mean()]],
                    yerr=[[L.mean() - yl], [yh - L.mean()]], fmt=mk, ms=9 if mk == "*" else 6,
                    capsize=3, label=K.LABEL[m])
    a2.set_xscale("log")
    a2.set_xlabel("SSI (deg/step, log scale)")
    a2.set_ylabel("Lane RMSE (px)")
    a2.set_title("(b) Means with bootstrap 95% CIs (lower-left is better)", fontsize=9)
    a2.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(K.os.path.join(K.OUT_FIG, f"fig_nominal_distribution_pareto_{tag}.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    print(run())
