"""
s8_jerk_ssi.py — relationship between the jerk m3 and the SSI m2 (editorial follow-up).
Question: in Table 6 the jerk-to-SSI ratio is 1.94-1.95 in every variant; why?
Identity: with d_k = delta_k - delta_{k-1}, |d_k - d_{k-1}| = |d_k| + |d_{k-1}| whenever d_k and d_{k-1}
have opposite signs or one of them is zero, so
    sum_k |d_k - d_{k-1}| = 2 sum_k |d_k| - 2 sum_{same-sign pairs} min(|d_k|, |d_{k-1}|)   (+ end terms),
i.e. m3 = 2 m2 except for consecutive steering increments of equal sign.
Sources: journal_cmc/results/optim/{ablation,compare,robustness}_results.json (per run, all runs) and
journal_cmc/results/logs/*_seed0.json (per step, run index 0 only).
"""
import glob
import os

import numpy as np

import common as K

ABL = [("BO_noBL_noLyap", "Sign, no"), ("BO_BL_noLyap", "Boundary layer, no"),
       ("BO_noBL_Lyap", "Sign, yes"), ("BO_full", "Boundary layer, yes")]
COND = ["clean", "noise", "shadow", "occlusion", "blur"]


def _ratio_rows(groups):
    rows = []
    for name, s, j in groups:
        r = j / s
        rows.append([name, len(s), f"{j.mean() / s.mean():.3f}", f"{j.std(ddof=1) / s.std(ddof=1):.3f}",
                     f"{r.min():.3f}", f"{r.max():.3f}", f"{np.corrcoef(s, j)[0, 1]:.4f}"])
    return rows


def _step_log(path):
    d = K.load_log(os.path.basename(path))
    a, e = np.array(d["angle"], float), np.array(d["lane_error"], float)
    da = np.diff(a)
    x, y = da[:-1], da[1:]
    same = x * y > 0
    s_sum, j_sum = np.abs(da).sum(), np.abs(np.diff(da)).sum()
    deficit = (2 * np.minimum(np.abs(x), np.abs(y)))[same].sum()
    new = np.r_[True, np.diff(e) != 0]
    gap = np.diff(np.where(new)[0])
    ratio = (j_sum / (len(da) - 1)) / (s_sum / len(da))
    return ratio, gap.mean(), 100 * same.mean(), 100 * deficit / (2 * s_sum)


def run():
    hdr = ["Group", "n runs", "jerk/SSI (means)", "jerk/SSI (SDs)", "per-run min", "per-run max", "corr(SSI, jerk)"]
    md = ["## S8. Jerk versus SSI (editorial follow-up)",
          "m3 = 2·m2 exactly, except for consecutive steering increments of equal sign (see module docstring).\n"]

    ab = K.load("ablation_results.json")["variants"]
    g6 = [(name, K.arr(ab[v]["metrics"], "ssi"), K.arr(ab[v]["metrics"], "steer_jerk")) for v, name in ABL]
    rows6 = _ratio_rows(g6)
    md.append("**Table 6 variants (10 runs each)**\n")
    md.append(K.md_table(hdr, rows6))
    K.write_csv("jerk_ssi_table6.csv", hdr, rows6)

    cm = K.load("compare_results.json")["methods"]
    g4 = [(K.LABEL[m], K.arr(cm[m], "ssi"), K.arr(cm[m], "steer_jerk")) for m in K.METHODS if m != "PSO_SMC"]
    rows4 = _ratio_rows(g4)
    md.append("\n**Nominal comparison, Table 4 (15 runs each)**\n")
    md.append(K.md_table(hdr, rows4))
    K.write_csv("jerk_ssi_table4.csv", hdr, rows4)

    rb = K.load("robustness_results.json")["data"]
    rows5 = []
    for c in COND:
        runs = [r for m in ("PID", "TunedPID", "ManualSMC", "BO_SMC") for r in rb[m][c]
                if r["detection_rate"] >= K.STALL_DETECTION]
        s, j = K.arr(runs, "ssi"), K.arr(runs, "steer_jerk")
        rows5 += _ratio_rows([(f"{c} (4 controllers, non-stalled)", s, j)])
    md.append("\n**Robustness conditions, Table 5 (all controllers pooled, stalled runs excluded)**\n")
    md.append(K.md_table(hdr, rows5))
    K.write_csv("jerk_ssi_table5.csv", hdr, rows5)

    lrows = []
    for p in sorted(glob.glob(os.path.join(K.LOGS, "compare_*_seed0.json")) +
                    glob.glob(os.path.join(K.LOGS, "robust_*_seed0.json"))):
        ratio, gap, same, deficit = _step_log(p)
        lrows.append([os.path.basename(p)[:-11], f"{ratio:.3f}", f"{gap:.2f}", f"{same:.1f}", f"{deficit:.1f}"])
    lhdr = ["Log (run index 0)", "jerk/SSI", "steps between new measurements", "same-sign pairs (% of steps)",
            "same-sign deficit (% of 2·SSI)"]
    md.append("\n**Per-step logs: mechanism.** A new measurement is a step in which the lateral error changes. "
              "In nominal runs a new camera frame arrives every 2.5 to 2.9 steps and the command is held "
              "in between, so consecutive increments are rarely of equal sign and m3/m2 is close to 2. Under "
              "noise, shadow, and occlusion the perturbation changes the measurement at every step, and the "
              "ratio falls (for differenced white noise the expected ratio is √3 ≈ 1.73).\n")
    md.append(K.md_table(lhdr, lrows))
    K.write_csv("jerk_ssi_steplogs.csv", lhdr, lrows)
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
