"""
s5_ablation.py — component analysis (revised Table 6).
Answers: Editor 4 (n, uncertainty, units, non-comparable metrics, "isolate" wording),
Editor 6 ("sample frequency" = violation rate).
Source: journal_cmc/results/optim/ablation_results.json
Design facts: each variant = ONE independent 60-evaluation BO run (seed 0) followed by
10 evaluation runs; variants ran hours apart, so runs are NOT paired across variants
(unpaired Mann-Whitney tests are used).
"""
import numpy as np

import common as K

VAR = [("BO_noBL_noLyap", "Sign switching, no stability term", "sign", False),
       ("BO_BL_noLyap", "Boundary layer, no stability term", "sat", False),
       ("BO_noBL_Lyap", "Sign switching + stability term", "sign", True),
       ("BO_full", "Boundary layer + stability term (full)", "sat", True)]
MET = [("lane_rmse_px", "Lane RMSE (px)"), ("ssi", "SSI (deg/step)"), ("steer_jerk", "Jerk (deg/step²)"),
       ("lyap_penalty", "m6 (px²)"), ("lyap_violation_rate", "Violation rate (fraction of steps)")]


def run():
    ab = K.load("ablation_results.json")["variants"]
    V = {v: {k: K.arr(ab[v]["metrics"], k) for k, _ in MET + [("speed_rmse", ""), ("detection_rate", "")]}
         for v, *_ in VAR}
    md = ["## S5. Component analysis / ablation (Table 6), 10 runs per variant, no exclusions",
          "Violation rate (the manuscript's 'sample frequency') = fraction of logged control steps with "
          "s_k·Δs_k > 0 on the steering channel. m6 = mean max(0, s_k·Δs_k), in px². Speed RMSE is driven by "
          "the number of simulator respawns per variant and is NOT comparable across rows.\n"]
    rows = []
    for v, name, sw, ly in VAR:
        r = [name, f"{min(V[v]['detection_rate']):.3f}"]
        for k, _ in MET:
            mu, sd = K.mean_sd(V[v][k])
            r.append(f"{mu:.3f} ({sd:.3f})" if k == "lyap_violation_rate" else f"{mu:.2f} ({sd:.2f})")
        sp = V[v]["speed_rmse"]
        r.append(f"{np.median(sp):.2f} ({int((sp>K.RESPAWN_SPEED_RMSE).sum())})")
        g = ab[v]["gains"]
        r.append(f"λ={g['lambda_a']:.3f}, η={g['eta_a']:.2f}, φ={'inert' if sw=='sign' else round(g['phi_a'],1)}")
        rows.append(r)
    hdr = ["Variant", "Min detection"] + [h for _, h in MET] + ["Speed RMSE median (respawn runs)",
                                                                "Steering gains found by BO"]
    md.append(K.md_table(hdr, rows))
    K.write_csv("table6_ablation.csv", hdr, rows)

    tests = [("BO_noBL_noLyap", "BO_BL_noLyap", "boundary layer (no stability term)"),
             ("BO_noBL_Lyap", "BO_full", "boundary layer (with stability term)"),
             ("BO_noBL_noLyap", "BO_noBL_Lyap", "stability term (sign switching)"),
             ("BO_BL_noLyap", "BO_full", "stability term (boundary layer)")]
    trows, ps = [], []
    for a, b, eff in tests:
        for k, h in MET[:4]:
            x, y = V[a][k], V[b][k]
            p = K.mwu_exact(y, x)
            lo, hi = K.boot_ci_diff_unpaired(y, x)
            trows.append([eff, h, f"{x.mean():.2f} → {y.mean():.2f}", f"{(y.mean()-x.mean()):+.2f} [{lo:+.2f}, {hi:+.2f}]",
                          f"{K.cliffs_delta(y, x):+.2f}", p])
            ps.append(p)
    adj = K.holm(ps)
    out = [r[:-1] + [K.fmt_p(r[-1]), K.fmt_p(a), "yes" if a < 0.05 else "no"] for r, a in zip(trows, adj)]
    md.append(f"\n**Effect of adding a component** (unpaired exact Mann–Whitney, Holm over {len(ps)} tests; "
              "each comparison contrasts two separately optimised controllers, so it reflects the component "
              "AND the outcome of one optimisation run).\n")
    thdr = ["Component added", "Metric", "Mean before → after", "Diff [95% CI]", "Cliff's δ", "p", "Holm p", "Sig."]
    md.append(K.md_table(thdr, out))
    K.write_csv("table6_tests.csv", thdr, out)
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
