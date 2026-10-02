"""
s4_robustness.py — robustness under visual perturbations (revised Table 5).
Answers: Editor 4 (effective n, excluded runs, uncertainty), Editor 5, Reviewer 1 comment 2.
Source: journal_cmc/results/optim/robustness_results.json
Design facts (from experiments.py): run index k uses seed k; for each k the 5 conditions
run in the order clean, noise, shadow, occlusion, blur and, within each, the 4 controllers
run in the FIXED order PID, Tuned PID, Manual SMC, proposed (not counterbalanced).
"""
import numpy as np

import common as K

M4 = ["PID", "TunedPID", "ManualSMC", "BO_SMC"]
COND = ["clean", "noise", "shadow", "occlusion", "blur"]
CNAME = {"clean": "Nominal", "noise": "Noise", "shadow": "Shadow", "occlusion": "Occlusion", "blur": "Blur"}


def run():
    rb = K.load("robustness_results.json")["data"]
    D = {m: {c: sorted(rb[m][c], key=lambda r: r["seed"]) for c in COND} for m in M4}
    det = {m: {c: K.arr(D[m][c], "detection_rate") for c in COND} for m in M4}
    stalled = {m: {c: det[m][c] < K.STALL_DETECTION for c in COND} for m in M4}

    md = ["## S4. Robustness (Table 5): 4 controllers × 5 conditions × 10 runs",
          f"Stalled run = detection rate < {K.STALL_DETECTION} (simulator returned frames without a lane on every "
          "frame: lane RMSE = 60 px imputed on all frames, SSI = 0, speed RMSE ≈ 90–100 km/h). "
          "Run order within each run index is FIXED (PID → Tuned PID → Manual SMC → proposed).\n"]

    # excluded runs
    ex = []
    for c in COND:
        for m in M4:
            for i in np.where(stalled[m][c])[0]:
                r = D[m][c][i]
                ex.append([CNAME[c], K.LABEL[m], r["seed"], f"{r['detection_rate']:.3f}",
                           f"{r['lane_rmse_px']:.1f}", f"{r['ssi']:.2f}", f"{r['speed_rmse']:.1f}", f"{r['fps']:.0f}"])
    md.append("**Stalled (excluded) runs:**\n")
    md.append(K.md_table(["Condition", "Controller", "Run", "Detection", "Lane RMSE", "SSI", "Speed RMSE", "fps"], ex))
    partial = []
    for c in COND:
        for m in M4:
            for r in D[m][c]:
                if K.STALL_DETECTION <= r["detection_rate"] < 0.9:
                    partial.append(f"{CNAME[c]}/{K.LABEL[m]} run {r['seed']} (detection {r['detection_rate']:.3f}, "
                                   f"lane {r['lane_rmse_px']:.1f})")
    md.append("\nPartially stalled runs kept in all variants: " + ("; ".join(partial) if partial else "none"))

    # three exclusion variants
    def cell(m, c, variant):
        keep = np.ones(10, bool)
        if variant == "A":                    # as in the submitted manuscript
            keep = ~stalled[m][c]
        elif variant == "C":                  # balanced: drop the run index for every controller
            keep = ~np.any([stalled[x][c] for x in M4], axis=0)
        return keep

    for variant, title in [("A", "as submitted (per-cell exclusion)"), ("B", "no exclusion"),
                           ("C", "balanced exclusion (same run indices removed for all controllers)")]:
        rows = []
        for metric, mname in [("ssi", "SSI (deg/step)"), ("lane_rmse_px", "Lane RMSE (px)"),
                              ("chatter_tv", "Steering variation per second (deg/s)")]:
            for c in COND:
                row = [mname, CNAME[c]]
                for m in M4:
                    keep = cell(m, c, variant)
                    x = K.arr(D[m][c], metric)[keep]
                    mu, sd = K.mean_sd(x)
                    row.append(f"{mu:.2f} ({sd:.2f}), n={keep.sum()}")
                rows.append(row)
        md.append(f"\n**Variant {variant}: {title}**\n")
        md.append(K.md_table(["Metric", "Condition"] + [K.LABEL[m] for m in M4], rows))
        K.write_csv(f"table5_variant{variant}.csv", ["metric", "condition"] + M4, rows)

    # loop rate per condition
    fps_rows = [[CNAME[c]] + [f"{K.arr(D[m][c], 'fps')[~stalled[m][c]].mean():.0f}" for m in M4] for c in COND]
    md.append("\n**Mean control-loop rate per condition (Hz, non-stalled runs).** Per-step SSI is not comparable "
              "across conditions when the loop rate differs; the per-second variation above is.\n")
    md.append(K.md_table(["Condition"] + [K.LABEL[m] for m in M4], fps_rows))

    # tests (variant A, complete-case pairs)
    trows, ps = [], []
    for comp in ("PID", "TunedPID"):
        for metric, mname in [("ssi", "SSI"), ("lane_rmse_px", "Lane RMSE")]:
            for c in COND:
                keep = (~stalled["BO_SMC"][c]) & (~stalled[comp][c])
                a = K.arr(D["BO_SMC"][c], metric)[keep]
                b = K.arr(D[comp][c], metric)[keep]
                d = a - b
                p, _, n = K.wilcoxon_exact(d)
                lo, hi = K.boot_ci_mean(d)
                trows.append([K.LABEL[comp], mname, CNAME[c], n, f"{a.mean():.2f} vs {b.mean():.2f}",
                              f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]", p])
                ps.append(p)
    adj = K.holm(ps)
    out = [r[:-1] + [K.fmt_p(r[-1]), K.fmt_p(a), "yes" if a < 0.05 else "no"] for r, a in zip(trows, adj)]
    md.append(f"\n**Paired tests, proposed vs PID and vs Tuned PID** (complete-case pairs, exact Wilcoxon, "
              f"Holm over {len(ps)} tests). Smallest attainable exact p: n=8 → 0.0078, n=9 → 0.0039, n=10 → 0.0020.\n")
    thdr = ["Comparator", "Metric", "Condition", "n pairs", "Means (proposed vs comparator)",
            "Mean diff [95% CI]", "p", "Holm p", "Sig."]
    md.append(K.md_table(thdr, out))
    K.write_csv("table5_tests.csv", thdr, out)

    oc = lambda m, k: K.arr(D[m]["occlusion"], k)[~stalled[m]["occlusion"]].mean()
    md.append(f"\nOcclusion check: lane RMSE proposed {oc('BO_SMC','lane_rmse_px'):.2f} vs PID {oc('PID','lane_rmse_px'):.2f} "
              f"(ratio {oc('BO_SMC','lane_rmse_px')/oc('PID','lane_rmse_px'):.3f}); SSI {oc('BO_SMC','ssi'):.2f} vs "
              f"{oc('PID','ssi'):.2f} ({oc('PID','ssi')/oc('BO_SMC','ssi'):.2f}-fold). "
              f"Tuned PID under occlusion: lane {oc('TunedPID','lane_rmse_px'):.2f}, SSI {oc('TunedPID','ssi'):.2f}.")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
