"""
s7_text_stats.py — statistics quoted in the text of the revised manuscript that are not
part of a table (loop and frame rates, respawn effects, single-episode objective noise,
switching-share proxy, robustness exclusion variants).
Sources: compare_results.json, robustness_results.json, reference_scales.json, logs/*.json
"""
import glob
import os

import numpy as np

import common as K

REPORTED = ["PID", "TunedPID", "ManualSMC", "GridSMC", "BO_SMC"]     # PSO-SMC withdrawn


def run():
    md = ["## S7. Statistics quoted in the text (reported controllers only)"]
    cm = K.load("compare_results.json")["methods"]
    refs = K.load("reference_scales.json")

    fps = np.concatenate([K.arr(cm[m], "fps") for m in REPORTED])
    md.append(f"- Control loop over the {len(fps)} reported nominal episodes: mean {fps.mean():.1f} Hz, "
              f"range {fps.min():.0f}–{fps.max():.0f} Hz.")
    rates = []
    for m in K.METHODS:
        p = os.path.join(K.LOGS, f"compare_{m}_seed0.json")
        if os.path.exists(p):
            lg = K.load_log(f"compare_{m}_seed0.json")
            e, t = np.array(lg["lane_error"]), np.array(lg["t"])
            rep = np.mean(np.diff(e) == 0)
            rates.append((m, rep, len(e) / t[-1] * (1 - rep)))
    md.append("- New camera frames (run 0 logs, loop rate × fraction of steps with a changed lane error): "
              + "; ".join(f"{K.LABEL[m]} {100*r:.0f}% repeated, {hz:.0f} Hz" for m, r, hz in rates))

    eff, lo_hi, hi_lo = [], [], []
    for m in REPORTED:
        l, s = K.arr(cm[m], "lane_rmse_px"), K.arr(cm[m], "speed_rmse")
        r = s > K.RESPAWN_SPEED_RMSE
        eff.append(l[r].mean() - l[~r].mean())
        lo_hi.append((s[~r].min(), s[~r].max()))
        hi_lo.append((s[r].min(), s[r].max()))
    md.append(f"- Respawn effect on lateral error (within controller, mean over {len(REPORTED)} reported controllers): "
              f"{np.mean(eff):+.1f} px. Speed clusters: {min(a for a, _ in lo_hi):.2f}–{max(b for _, b in lo_hi):.2f} km/h "
              f"and {min(a for a, _ in hi_lo):.2f}–{max(b for _, b in hi_lo):.2f} km/h.")

    J = {m: np.array([K.objective(r, refs) for r in cm[m]]) for m in REPORTED}
    md.append("- Single-episode objective J at fixed gains (15 nominal episodes): "
              + "; ".join(f"{K.LABEL[m]} SD {J[m].std(ddof=1):.2f}" for m in REPORTED)
              + f". Proposed mean J {J['BO_SMC'].mean():.2f} (tuning J {K.load('bo_result.json')['best_J']:.2f}); "
              f"manual mean J {J['ManualSMC'].mean():.2f}, median {np.median(J['ManualSMC']):.2f}.")

    drops = []
    for f in sorted(glob.glob(os.path.join(K.LOGS, "*.json"))):
        lg = K.load_log(os.path.basename(f))
        sp, det = np.array(lg["speed"]), np.array(lg["detected"])
        for i in np.where(np.diff(sp) < -40)[0]:
            if det[max(0, i - 5):i + 5].mean() > 0.95:
                drops.append(f"{os.path.basename(f)}: {sp[i]:.1f} -> {sp[i+1]:.1f} km/h")
    md.append("- Instantaneous respawn drops inside logged windows (lane still detected): " + "; ".join(drops[:4]))

    g = K.load("bo_result.json")["best_gains"]
    lg = K.load_log("compare_BO_SMC_seed0.json")
    e, s = np.array(lg["lane_error"]), np.array(lg["s_angle"])
    p_term = np.abs(g["lambda_a"] * e).mean()
    share = {phi: 100 * np.abs(g["eta_a"] * np.clip(s / phi, -1, 1)).mean() / p_term for phi in (80.0, 41.3, 20.1, 14.8)}
    md.append("- Switching share of the command, using the deployed trajectory as a proxy: "
              + ", ".join(f"phi={k:g}: {v:.1f}%" for k, v in share.items()))

    md.append(_robustness_variants())
    return "\n".join(md)


def _robustness_variants():
    rb = K.load("robustness_results.json")["data"]
    M4, COND = ["PID", "TunedPID", "ManualSMC", "BO_SMC"], ["clean", "noise", "shadow", "occlusion", "blur"]
    D = {m: {c: sorted(rb[m][c], key=lambda r: r["seed"]) for c in COND} for m in M4}
    st = {m: {c: K.arr(D[m][c], "detection_rate") < K.STALL_DETECTION for c in COND} for m in M4}
    out = []
    for variant in ("B (no exclusion)", "C (balanced exclusion)"):
        tests, ps = [], []
        for comp in ("PID", "TunedPID"):
            for metric in ("ssi", "lane_rmse_px"):
                for c in COND:
                    if variant.startswith("B"):
                        keep = np.ones(10, bool)
                    else:
                        keep = ~np.any([st[x][c] for x in M4], axis=0)
                    d = K.arr(D["BO_SMC"][c], metric)[keep] - K.arr(D[comp][c], metric)[keep]
                    p, _, _ = K.wilcoxon_exact(d)
                    tests.append((comp, metric, c, keep.sum()))
                    ps.append(p)
        adj = K.holm(ps)
        sig = [f"{c}/{m}/{c2}" for (c, m, c2, n), a in zip(tests, adj) if a < 0.05]
        blur = [a for (c, m, c2, n), a in zip(tests, adj) if c == "PID" and m == "ssi" and c2 == "blur"][0]
        out.append(f"- Robustness variant {variant}: significant after Holm (20 tests): {sig}; "
                   f"blur SSI vs PID Holm p = {blur:.3f}")
        if variant.startswith("C"):
            keep = ~np.any([st[x]["clean"] for x in M4], axis=0)
            out.append(f"  nominal lane RMSE with balanced exclusion: PID {K.arr(D['PID']['clean'], 'lane_rmse_px')[keep].mean():.2f}, "
                       f"proposed {K.arr(D['BO_SMC']['clean'], 'lane_rmse_px')[keep].mean():.2f} px")
    return "\n".join(out)


if __name__ == "__main__":
    print(run())
