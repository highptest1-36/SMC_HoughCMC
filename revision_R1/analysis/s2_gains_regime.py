"""
s2_gains_regime.py — tuning runs, gains, control-law numbers, reference scales,
operating regime and OFFLINE weight sensitivity.
Answers: Editor 2, 3; Reviewer 2 comments 1, 2, 3, 5.
Sources: *_history.json, *_result.json, fair_compare_results.json, ablation_*,
reference_scales.json, compare_results.json, logs/compare_BO_SMC_seed0.json
"""
import numpy as np

import common as K

VALID_DET = 0.5      # an evaluation episode with detection < 0.5 ran on a stalled simulator


def _history(name):
    h = K.load(name)
    return h if isinstance(h, list) else h["history"]


def run():
    md = ["## S2. Tuning runs, gains, control law, reference scales, regime, sensitivity"]
    refs = K.load("reference_scales.json")

    # --- 1. validity of the deployed tuning runs ------------------------------
    rows = []
    for name, fn in [("BO (deployed)", "bo_history.json"), ("PSO (deployed)", "pso_history.json"),
                     ("Random (not deployed)", "random_history.json"), ("Grid (deployed)", "grid_history.json"),
                     ("Tuned PID (BO over PID gains)", "pid_history.json")]:
        h = _history(fn)
        det = np.array([e["metrics"].get("detection_rate", 1.0) for e in h])
        Js = np.array([e["J"] for e in h])
        valid = det >= VALID_DET
        first_bad = int(np.argmax(~valid)) if (~valid).any() else None
        best_all = int(np.argmin(Js))
        rows.append([name, len(h), int(valid.sum()), "—" if first_bad is None else first_bad,
                     f"{Js.min():.3f} (call {best_all})", "yes" if valid[best_all] else "NO"])
    md.append("\n### 2.1 Validity of the deployed tuning runs (detection rate per evaluation)\n")
    md.append(K.md_table(["Run", "Evaluations", "Valid (detection ≥ 0.5)", "First invalid call",
                          "Best J (call)", "Best point valid?"], rows))
    K.write_csv("tuning_runs_validity.csv", ["run", "evals", "valid", "first_invalid", "best", "best_valid"], rows)
    pso = _history("pso_history.json")
    pso_bad = [e for e in pso if e["metrics"].get("detection_rate", 1) < VALID_DET]
    if pso_bad:
        jb = sorted({round(e["J"], 4) for e in pso_bad})
        md.append(f"\nPSO invalid evaluations: {len(pso_bad)}; their J values: {jb[:5]} "
                  f"(stalled simulator: detection ≈ 0, speed RMSE ≈ "
                  f"{np.mean([e['metrics']['speed_rmse'] for e in pso_bad]):.0f} km/h, "
                  f"fps ≈ {np.mean([e['metrics']['fps'] for e in pso_bad]):.0f}).")

    # --- 2. gains table ----------------------------------------------------------
    grid = _history("grid_history.json")
    th = np.array([e["theta"] for e in grid])
    levels = {K.PARAMS[i]: sorted(set(np.round(th[:, i], 4))) for i in range(6)}
    md.append("\n### 2.2 Grid-search levels (from grid_history.json)\n")
    md.append("; ".join(f"{p}: {v}" for p, v in levels.items()))
    md.append("\nλv, ηv, φv are constant over all 27 grid evaluations → not searched; copied from the manual "
              "SMC configuration (λv = 70, ηv = 0.001, φv = 1). With ηv = 0.001 the longitudinal switching term "
              "is ≤ 0.001 (inert), so the grid controller's speed loop is effectively the manual one.")

    grows = []
    ab = K.load("ablation_results.json")["variants"]
    for name, g, note in [
        ("Manual SMC [20]", ab["ManualSMC"]["gains"], "soft-sign s/(|s|+1e-3); φ unused"),
        ("Grid-search SMC", K.load("grid_result.json")["best_gains"], "λv, ηv, φv fixed (not searched)"),
        ("PSO SMC", K.load("pso_result.json")["best_gains"], "best of 13 valid evaluations"),
        ("Proposed BO-SMC", K.load("bo_result.json")["best_gains"], "60 valid evaluations"),
    ]:
        grows.append([name] + [f"{g[p]:.4g}" for p in K.PARAMS] + [note])
    md.append("\n### 2.3 Deployed gains\n")
    md.append(K.md_table(["Method"] + K.PARAMS + ["Note"], grows))
    pid = K.load("pid_result.json")["best_gains"]
    md.append("\nTuned PID gains (BO, 60 evaluations over the PID space; bounds Kp_a [0.02,2], Ki_a [0,0.05], "
              "Kd_a [0,0.5], Kp_v [1,80], Ki_v [0,0.05], Kd_v [0,0.5]): "
              + ", ".join(f"{k} = {v:.4g}" for k, v in pid.items()))
    K.write_csv("gains_deployed.csv", ["method"] + K.PARAMS + ["note"], grows)

    # --- 3. control-law numbers (Reviewer 2, comment 1) -----------------------------
    g = K.load("bo_result.json")["best_gains"]
    lam, eta, phi = g["lambda_a"], g["eta_a"], g["phi_a"]
    kp, kd = lam * (1 + eta / phi), eta / phi
    log = K.load_log("compare_BO_SMC_seed0.json")
    e, s, u = np.array(log["lane_error"]), np.array(log["s_angle"]), np.array(log["angle"])
    sw = eta * np.clip(s / phi, -1, 1)
    frac_out = np.mean(np.abs(s) > phi)
    ls = -np.sum(u * e) / np.sum(e * e)
    rep = np.mean(np.diff(e) == 0)
    md.append("\n### 2.4 Control-law numbers (deployed BO-SMC)\n")
    md.append(f"λθ = {lam:.5f}, ηθ = {eta:.3f}, φθ = {phi:.2f} → inside the layer "
              f"u = −[K_P e + K_D Δe] with K_P = λθ(1+ηθ/φθ) = {kp:.4f} deg/px and K_D = ηθ/φθ = {kd:.5f} deg/px per step. "
              f"Least-squares fit of logged steering on e (run 0): {ls:.4f}. "
              f"Run 0 only (per-frame logs exist for run 0): |s| > φ on {100*frac_out:.2f}% of steps; "
              f"mean |switching term| = {np.abs(sw).mean():.4f}° vs mean |λθ e| = {np.abs(lam*e).mean():.3f}° "
              f"({100*np.abs(sw).mean()/np.abs(lam*e).mean():.2f}%). "
              f"Speed channel: λv = {g['lambda_v']:.3f}, ηv = {g['eta_v']:.3f}, φv = {g['phi_v']:.3f} → "
              f"K_P,v = {g['lambda_v']*(1+g['eta_v']/g['phi_v']):.1f}. "
              f"Limits: φ→0 gives u = −λe − η sign(s) (discontinuous); φ→∞ gives u = −λe (pure P, NOT PD).")
    dt_pid = np.mean([1.0 / r["fps"] for r in K.load("compare_results.json")["methods"]["TunedPID"]])
    md.append(f"Tuned PID: Kp_angle = {pid['Kp_angle']:.4f}, Ki_angle = {pid['Ki_angle']:.3f} (upper bound), "
              f"Kd_angle = {pid['Kd_angle']:.3f}; windowed integral (10 samples) adds ≈ 10·Ki·dt = "
              f"{10*pid['Ki_angle']*dt_pid:.4f} → effective ≈ {pid['Kp_angle']+10*pid['Ki_angle']*dt_pid:.4f} "
              f"({100*abs(pid['Kp_angle']+10*pid['Ki_angle']*dt_pid-kp)/kp:.1f}% from BO-SMC K_P).")

    # --- 4. loop rate / frame repetition ----------------------------------------------
    cmp_ = K.load("compare_results.json")["methods"]
    fps = np.concatenate([K.arr(cmp_[m], "fps") for m in K.METHODS])
    md.append(f"\n**Control loop (Table 4 runs, n = {len(fps)} episodes).** fps mean {fps.mean():.1f}, "
              f"median {np.median(fps):.1f}, range {fps.min():.0f}–{fps.max():.0f} Hz. "
              f"In run 0 of the proposed controller, {100*rep:.0f}% of consecutive control steps reuse the "
              f"same lane error (repeated camera frame) → new images at ≈ {np.median(fps)*(1-rep):.0f} Hz.")

    # --- 5. reference scales -----------------------------------------------------------
    man = cmp_["ManualSMC"]
    Jman = np.array([K.objective(r, refs) for r in man])
    md.append("\n### 2.5 Reference scales r_i (reference_scales.json; 3 Manual-SMC episodes, seeds 0–2, clean)\n")
    md.append(K.md_table(["Measure", "r_i", "Origin"], [
        ["m1 lane RMSE", f"{refs['lane_rmse_px']:.3f} px", "manual-SMC mean"],
        ["m2 SSI", f"{refs['ssi']:.4f} deg/step", "manual-SMC mean"],
        ["m3 jerk", f"{refs['steer_jerk']:.4f} deg/step²", "manual-SMC mean"],
        ["m4 speed RMSE", f"{refs['speed_rmse']:.1f} km/h", "floor = 0.2 × fallback 10 (raw value was lower)"],
        ["m5 saturation", f"{refs['sat_ratio']:.2f}", "fallback (manual SMC never saturates, raw = 0)"],
        ["m6 Lyapunov penalty", f"{refs['lyap_penalty']:.2f} px²", "manual-SMC mean"]]))
    md.append(f"\nManual SMC scores J = {Jman.mean():.2f} (mean) / {np.median(Jman):.2f} (median) over its 15 "
              f"Table-4 episodes (Σw = 5.2 only if every normalised term equals 1).")
    K.write_csv("reference_scales.csv", ["measure", "value"], [[k, v] for k, v in refs.items()])

    # --- 6. operating regime across ALL boundary-layer BO runs --------------------------
    runs = [("deployed BO", K.load("bo_result.json")["best_gains"], "baseline")]
    for r in sorted(K.load("fair_compare_results.json")["data"]["bo"], key=lambda r: r["seed"]):
        runs.append((f"fair BO seed {r['seed']}", r["best_gains"], "baseline"))
    runs.append(("ablation BO_full", ab["BO_full"]["gains"], "baseline (replicate)"))
    runs.append(("ablation BO_BL_noLyap", ab["BO_BL_noLyap"]["gains"], "w6 = 0"))
    rrows = []
    for name, gg, obj in runs:
        e_, p_ = gg["eta_a"], gg["phi_a"]
        rrows.append([name, obj, f"{e_:.3f}", f"{p_:.1f}", f"{e_/p_:.4f}", f"{gg['lambda_a']*(1+e_/p_):.3f}"])
    etas = np.array([gg["eta_a"] for _, gg, _ in runs]); phis = np.array([gg["phi_a"] for _, gg, _ in runs])
    sign_eta = [ab["BO_noBL_noLyap"]["gains"]["eta_a"], ab["BO_noBL_Lyap"]["gains"]["eta_a"]]
    md.append("\n### 2.6 Operating regime over all boundary-layer BO runs\n")
    md.append(K.md_table(["Run", "Objective", "ηθ", "φθ", "ηθ/φθ", "K_P"], rrows))
    md.append(f"\nηθ ≤ 0.6 in {int((etas<=0.6).sum())}/{len(etas)} boundary-layer runs (+ ηθ = {sign_eta} in the 2 "
              f"sign-switching ablation runs → {int((etas<=0.6).sum())+sum(e<=0.6 for e in sign_eta)}/"
              f"{len(etas)+2} BO runs); φθ ≥ 75 in {int((phis>=75).sum())}/{len(phis)} "
              f"(range {phis.min():.1f}–{phis.max():.1f}); ηθ/φθ ≤ {max(etas/phis):.3f} in all.")
    K.write_csv("regime_all_bo_runs.csv", ["run", "objective", "eta", "phi", "eta_over_phi", "K_P"], rrows)
    fc = K.load("fair_compare_results.json")["data"]
    for o in ("random", "pso"):
        ratios = [r["best_gains"]["eta_a"] / r["best_gains"]["phi_a"] for r in fc[o]]
        md.append(f"{o}: median ηθ/φθ of the 10 best points = {np.median(ratios):.3f} "
                  f"(uniform-prior median over the search box ≈ 0.25).")

    md.append(_sensitivity(refs))
    return "\n".join(md)


def _sensitivity(refs):
    """Offline re-ranking of all VALID logged evaluations under perturbed objectives."""
    pool = []
    for fn in ("bo_history.json", "pso_history.json", "random_history.json", "grid_history.json"):
        for e in _history(fn):
            m = e["metrics"]
            if m.get("detection_rate", 1) >= 0.9:
                pool.append((fn.split("_")[0], e["theta"], m, e["J"]))
    err = max(abs(K.objective(m, refs) - J) for _, _, m, J in pool)
    th = np.array([p[1] for p in pool])
    eta, phi = th[:, 1], th[:, 2]

    def summarise(Js, k=10):
        idx = np.argsort(Js)[:k]
        return (np.median(eta[idx]), np.median(phi[idx]), np.median(eta[idx] / phi[idx]),
                eta[idx[0]], phi[idx[0]])

    base = dict(K.WEIGHTS)
    scen = [("baseline", base, K.CLIP_LO, K.CLIP_HI)]
    for k in base:
        for f in (0.5, 2.0):
            w = dict(base); w[k] *= f
            scen.append((f"{k} x{f}", w, K.CLIP_LO, K.CLIP_HI))
    scen += [("equal weights", {k: 1.0 for k in base}, K.CLIP_LO, K.CLIP_HI),
             ("w6 = 0", {**base, "lyap_penalty": 0.0}, K.CLIP_LO, K.CLIP_HI),
             ("no clipping", base, 0.0, 1e9),
             ("clip (0.2, 5)", base, 0.2, 5.0)]
    rows = []
    for name, w, lo, hi in scen:
        Js = np.array([K.objective(m, refs, w, lo, hi) for _, _, m, _ in pool])
        me, mp, mr, ae, ap = summarise(Js)
        rows.append([name, f"{ae:.2f}", f"{ap:.1f}", f"{me:.2f}", f"{mp:.1f}", f"{mr:.4f}"])
    # Dirichlet draws around the baseline
    rng = np.random.default_rng(K.RNG_SEED)
    wb = np.array(list(base.values())); alpha = 20 * wb / wb.sum()
    lin, low_eta = 0, 0
    nd = 200
    for _ in range(nd):
        wv = rng.dirichlet(alpha) * wb.sum()
        w = dict(zip(base, wv))
        Js = np.array([K.objective(m, refs, w) for _, _, m, _ in pool])
        i = int(np.argmin(Js))
        low_eta += eta[i] <= 0.6
        lin += eta[i] / phi[i] <= 0.05
    # sweeps: where does the regime break?
    sweep = []
    for f in (1.0, 0.5, 0.25, 0.1, 0.0):
        w = {**base, "ssi": base["ssi"] * f, "steer_jerk": base["steer_jerk"] * f}
        Js = np.array([K.objective(m, refs, w) for _, _, m, _ in pool]); i = int(np.argmin(Js))
        sweep.append([f"ssi & jerk weights x{f}", f"{eta[i]:.2f}", f"{phi[i]:.1f}", f"{eta[i]/phi[i]:.4f}"])
    for wl in (0.5, 1.5, 6.0, 9.0, 12.0):
        w = {**base, "lane_rmse_px": wl}
        Js = np.array([K.objective(m, refs, w) for _, _, m, _ in pool]); i = int(np.argmin(Js))
        sweep.append([f"lane weight = {wl}", f"{eta[i]:.2f}", f"{phi[i]:.1f}", f"{eta[i]/phi[i]:.4f}"])
    txt_sweep = K.md_table(["Sweep", "argmin ηθ", "argmin φθ", "argmin ηθ/φθ"], sweep)
    txt = ["\n### 2.7 Offline weight/clip sensitivity (re-ranking of logged evaluations; no new simulation)\n",
           f"Pool: {len(pool)} valid evaluations with raw metrics (BO/PSO/random/grid deployed runs). "
           f"Re-computed J reproduces logged J to max |Δ| = {err:.2e}. "
           "Top-10 = the 10 lowest-J evaluations under each scenario.\n",
           K.md_table(["Scenario", "argmin ηθ", "argmin φθ", "top-10 median ηθ", "top-10 median φθ",
                       "top-10 median ηθ/φθ"], rows),
           f"\nRandom Dirichlet weights around the baseline ({nd} draws): argmin has ηθ ≤ 0.6 in "
           f"{100*low_eta/nd:.0f}% and ηθ/φθ ≤ 0.05 in {100*lin/nd:.0f}% of draws.\n",
           "Break-point sweeps (argmin under each setting):\n", txt_sweep, "",
           "Caveat: re-ranking is not re-optimisation; the pool is dominated by the 60 BO evaluations "
           "of one run, each a single noisy episode (fixed-gain SD of J ≈ 1)."]
    K.write_csv("sensitivity_offline.csv", ["scenario", "argmin_eta", "argmin_phi", "top10_eta",
                                            "top10_phi", "top10_ratio"], rows)
    return "\n".join(txt)


if __name__ == "__main__":
    print(run())
