# Revision R1 — analysis report (CMC ID 92446)
Generated: 2026-09-30 18:40:50 · Python 3.10.6 · numpy 1.26.4 · scipy 1.15.3 · matplotlib 3.10.9
Data folder: D:\Xetuhanh\UTE_Car_2025\Map_demo____1\Window\cdoe\SMC_Hough_CMC\journal_cmc\results


## S1. Matched-budget optimizer study (Table 2, Fig. 2)
Budget per run: [60] evaluations; seeds per optimizer: 10; search space: 6-D (identical bounds). Grid search was not part of this study.

| Optimizer | Best J mean ± SD | 95% CI (mean) | Median [IQR] | Range (best–worst seed) | Wall time/run (s) |
|---|---|---|---|---|---|
| Random search | 4.63 ± 0.64 | [4.08, 4.91] | 4.80 [4.70, 4.99] | 3.09–5.24 | 3138 |
| Particle swarm optimization | 4.77 ± 0.41 | [4.55, 5.04] | 4.69 [4.40, 5.07] | 4.31–5.49 | 3138 |
| Proposed Bayesian optimization | 4.17 ± 0.21 | [4.03, 4.28] | 4.18 [4.11, 4.29] | 3.78–4.49 | 3300 |

**Tests** (difference = first − second; negative favours the first optimizer). Wilcoxon = paired by seed index, as in the submitted manuscript. Mann–Whitney = unpaired; appropriate because optimizers ran sequentially (all random, then PSO, then BO) and the simulator noise is not seeded, so the seed index does not create a real pairing. Holm over the 2 confirmatory comparisons.

| Comparison | Mean diff [95% CI] | Cliff's δ | Seed-wise wins | Wilcoxon p | Holm | Mann–Whitney p | Holm |
|---|---|---|---|---|---|---|---|
| BO vs PSO | -0.60 [-0.88, -0.34] | -0.90 | 9/10 | 0.004 | 0.008 | <0.001 | <0.001 |
| BO vs RANDOM | -0.46 [-0.81, -0.03] | -0.64 | 8/10 | 0.084 | 0.084 | 0.015 | 0.015 |
| PSO vs RANDOM | +0.14 [-0.27, +0.61] | -0.02 | 4/10 | 0.922 | — | 0.971 | — |

**Dispersion.** SD: random 0.644, PSO 0.413, BO 0.205. SD ratio random/BO = 3.14 (bootstrap 95% CI 0.71–7.13); Brown–Forsythe p = 0.195 (random vs BO), 0.033 (PSO vs BO). Worst seed: BO 4.49, random 5.24, PSO 5.49. Best single run of the study: random seed 1 (J = 3.09).

**Convergence (mean ± SD of running best).** RANDOM: eval 12 5.26 ± 0.82, eval 30 5.10 ± 0.64, eval 60 4.63 ± 0.64; PSO: eval 12 5.34 ± 0.69, eval 30 4.93 ± 0.40, eval 60 4.77 ± 0.41; BO: eval 12 5.53 ± 0.67, eval 30 4.37 ± 0.35, eval 60 4.17 ± 0.21. BO mean first reaches the final (eval-60) mean of PSO at evaluation 20 and of random search at evaluation 24.
BO found its final best at evaluations [60, 50, 41, 24, 43, 50, 57, 54, 29, 32] (still improving near 60 → not converged).

**Wall time.** Per evaluation ≈ 6 s settle + seed-dependent extra settle U(0,3) s (constant within a run) + 45 s measurement. Mean per 60-evaluation run: random 3138 s, PSO 3138 s, BO 3300 s. BO overhead (GP fit + LogEI optimisation, same seeds as random) = 3.37 ± 0.62 s per BO iteration (48 iterations). Total closed-loop time of the study = 26.6 h.

**BO best gains per seed** (steering channel):

| Seed | Best J | λθ | ηθ | φθ | ηθ/φθ | K_P = λθ(1+ηθ/φθ) |
|---|---|---|---|---|---|---|
| 0 | 4.273 | 0.193 | 0.500 | 73.7 | 0.0068 | 0.194 |
| 1 | 4.168 | 0.155 | 0.500 | 42.6 | 0.0117 | 0.157 |
| 2 | 4.180 | 0.218 | 0.500 | 62.7 | 0.0080 | 0.220 |
| 3 | 4.173 | 0.166 | 0.500 | 80.0 | 0.0063 | 0.167 |
| 4 | 3.920 | 0.128 | 0.500 | 64.6 | 0.0077 | 0.129 |
| 5 | 4.334 | 0.135 | 0.500 | 58.4 | 0.0086 | 0.136 |
| 6 | 4.493 | 0.175 | 0.500 | 26.5 | 0.0189 | 0.179 |
| 7 | 4.094 | 0.166 | 0.500 | 14.8 | 0.0339 | 0.171 |
| 8 | 4.297 | 0.137 | 0.500 | 51.5 | 0.0097 | 0.138 |
| 9 | 3.780 | 0.149 | 0.500 | 68.2 | 0.0073 | 0.150 |

---

## S2. Tuning runs, gains, control law, reference scales, regime, sensitivity

### 2.1 Validity of the deployed tuning runs (detection rate per evaluation)

| Run | Evaluations | Valid (detection ≥ 0.5) | First invalid call | Best J (call) | Best point valid? |
|---|---|---|---|---|---|
| BO (deployed) | 60 | 60 | — | 4.279 (call 20) | yes |
| PSO (deployed) | 80 | 13 | 13 | 4.989 (call 12) | yes |
| Random (not deployed) | 40 | 40 | — | 5.043 (call 20) | yes |
| Grid (deployed) | 27 | 27 | — | 6.361 (call 9) | yes |
| Tuned PID (BO over PID gains) | 60 | 60 | — | 4.203 (call 58) | yes |

PSO invalid evaluations: 67; their J values: [7.7436, 13.505] (stalled simulator: detection ≈ 0, speed RMSE ≈ 99 km/h, fps ≈ 434).

### 2.2 Grid-search levels (from grid_history.json)

lambda_a: [0.01, 0.505, 1.0]; eta_a: [0.5, 10.25, 20.0]; phi_a: [1.0, 40.5, 80.0]; lambda_v: [70.0]; eta_v: [0.001]; phi_v: [1.0]

λv, ηv, φv are constant over all 27 grid evaluations → not searched; copied from the manual SMC configuration (λv = 70, ηv = 0.001, φv = 1). With ηv = 0.001 the longitudinal switching term is ≤ 0.001 (inert), so the grid controller's speed loop is effectively the manual one.

### 2.3 Deployed gains

| Method | lambda_a | eta_a | phi_a | lambda_v | eta_v | phi_v | Note |
|---|---|---|---|---|---|---|---|
| Manual SMC [20] | 0.1 | 0.001 | 1 | 70 | 0.001 | 1 | soft-sign s/(|s|+1e-3); φ unused |
| Grid-search SMC | 0.505 | 0.5 | 1 | 70 | 0.001 | 1 | λv, ηv, φv fixed (not searched) |
| PSO SMC | 0.2745 | 6.978 | 46.54 | 89.38 | 17.6 | 33.18 | best of 13 valid evaluations |
| Proposed BO-SMC | 0.1752 | 0.5 | 80 | 143.2 | 0.5 | 6.458 | 60 valid evaluations |

Tuned PID gains (BO, 60 evaluations over the PID space; bounds Kp_a [0.02,2], Ki_a [0,0.05], Kd_a [0,0.5], Kp_v [1,80], Ki_v [0,0.05], Kd_v [0,0.5]): Kp_angle = 0.1818, Ki_angle = 0.05, Kd_angle = 0, Kp_speed = 66.87, Ki_speed = 0.05, Kd_speed = 0.11

### 2.4 Control-law numbers (deployed BO-SMC)

λθ = 0.17519, ηθ = 0.500, φθ = 80.00 → inside the layer u = −[K_P e + K_D Δe] with K_P = λθ(1+ηθ/φθ) = 0.1763 deg/px and K_D = ηθ/φθ = 0.00625 deg/px per step. Least-squares fit of logged steering on e (run 0): 0.1758. Run 0 only (per-frame logs exist for run 0): |s| > φ on 0.13% of steps; mean |switching term| = 0.0564° vs mean |λθ e| = 4.288° (1.31%). Speed channel: λv = 143.158, ηv = 0.500, φv = 6.458 → K_P,v = 154.2. Limits: φ→0 gives u = −λe − η sign(s) (discontinuous); φ→∞ gives u = −λe (pure P, NOT PD).
Tuned PID: Kp_angle = 0.1818, Ki_angle = 0.050 (upper bound), Kd_angle = 0.000; windowed integral (10 samples) adds ≈ 10·Ki·dt = 0.0037 → effective ≈ 0.1855 (5.2% from BO-SMC K_P).

**Control loop (Table 4 runs, n = 90 episodes).** fps mean 140.9, median 142.3, range 103–179 Hz. In run 0 of the proposed controller, 64% of consecutive control steps reuse the same lane error (repeated camera frame) → new images at ≈ 52 Hz.

### 2.5 Reference scales r_i (reference_scales.json; 3 Manual-SMC episodes, seeds 0–2, clean)

| Measure | r_i | Origin |
|---|---|---|
| m1 lane RMSE | 41.516 px | manual-SMC mean |
| m2 SSI | 0.5764 deg/step | manual-SMC mean |
| m3 jerk | 1.1205 deg/step² | manual-SMC mean |
| m4 speed RMSE | 2.0 km/h | floor = 0.2 × fallback 10 (raw value was lower) |
| m5 saturation | 0.10 | fallback (manual SMC never saturates, raw = 0) |
| m6 Lyapunov penalty | 186.26 px² | manual-SMC mean |

Manual SMC scores J = 5.21 (mean) / 5.07 (median) over its 15 Table-4 episodes (Σw = 5.2 only if every normalised term equals 1).

### 2.6 Operating regime over all boundary-layer BO runs

| Run | Objective | ηθ | φθ | ηθ/φθ | K_P |
|---|---|---|---|---|---|
| deployed BO | baseline | 0.500 | 80.0 | 0.0063 | 0.176 |
| fair BO seed 0 | baseline | 0.500 | 73.7 | 0.0068 | 0.194 |
| fair BO seed 1 | baseline | 0.500 | 42.6 | 0.0117 | 0.157 |
| fair BO seed 2 | baseline | 0.500 | 62.7 | 0.0080 | 0.220 |
| fair BO seed 3 | baseline | 0.500 | 80.0 | 0.0063 | 0.167 |
| fair BO seed 4 | baseline | 0.500 | 64.6 | 0.0077 | 0.129 |
| fair BO seed 5 | baseline | 0.500 | 58.4 | 0.0086 | 0.136 |
| fair BO seed 6 | baseline | 0.500 | 26.5 | 0.0189 | 0.179 |
| fair BO seed 7 | baseline | 0.500 | 14.8 | 0.0339 | 0.171 |
| fair BO seed 8 | baseline | 0.500 | 51.5 | 0.0097 | 0.138 |
| fair BO seed 9 | baseline | 0.500 | 68.2 | 0.0073 | 0.150 |
| ablation BO_full | baseline (replicate) | 0.500 | 20.1 | 0.0248 | 0.171 |
| ablation BO_BL_noLyap | w6 = 0 | 0.500 | 41.3 | 0.0121 | 0.108 |

ηθ ≤ 0.6 in 13/13 boundary-layer runs (+ ηθ = [0.5, 0.5] in the 2 sign-switching ablation runs → 15/15 BO runs); φθ ≥ 75 in 2/13 (range 14.8–80.0); ηθ/φθ ≤ 0.034 in all.
random: median ηθ/φθ of the 10 best points = 0.129 (uniform-prior median over the search box ≈ 0.25).
pso: median ηθ/φθ of the 10 best points = 0.150 (uniform-prior median over the search box ≈ 0.25).

### 2.7 Offline weight/clip sensitivity (re-ranking of logged evaluations; no new simulation)

Pool: 140 valid evaluations with raw metrics (BO/PSO/random/grid deployed runs). Re-computed J reproduces logged J to max |Δ| = 0.00e+00. Top-10 = the 10 lowest-J evaluations under each scenario.

| Scenario | argmin ηθ | argmin φθ | top-10 median ηθ | top-10 median φθ | top-10 median ηθ/φθ |
|---|---|---|---|---|---|
| baseline | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| lane_rmse_px x0.5 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| lane_rmse_px x2.0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| speed_rmse x0.5 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| speed_rmse x2.0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| ssi x0.5 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| ssi x2.0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| steer_jerk x0.5 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| steer_jerk x2.0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| sat_ratio x0.5 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| sat_ratio x2.0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| lyap_penalty x0.5 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| lyap_penalty x2.0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| equal weights | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| w6 = 0 | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| no clipping | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |
| clip (0.2, 5) | 0.50 | 80.0 | 0.50 | 80.0 | 0.0063 |

Random Dirichlet weights around the baseline (200 draws): argmin has ηθ ≤ 0.6 in 86% and ηθ/φθ ≤ 0.05 in 86% of draws.

Break-point sweeps (argmin under each setting):

| Sweep | argmin ηθ | argmin φθ | argmin ηθ/φθ |
|---|---|---|---|
| ssi & jerk weights x1.0 | 0.50 | 80.0 | 0.0063 |
| ssi & jerk weights x0.5 | 0.50 | 80.0 | 0.0063 |
| ssi & jerk weights x0.25 | 17.19 | 72.0 | 0.2387 |
| ssi & jerk weights x0.1 | 17.19 | 72.0 | 0.2387 |
| ssi & jerk weights x0.0 | 17.19 | 72.0 | 0.2387 |
| lane weight = 0.5 | 0.50 | 80.0 | 0.0063 |
| lane weight = 1.5 | 0.50 | 80.0 | 0.0063 |
| lane weight = 6.0 | 0.50 | 80.0 | 0.0063 |
| lane weight = 9.0 | 0.50 | 80.0 | 0.0063 |
| lane weight = 12.0 | 20.00 | 80.0 | 0.2500 |

Caveat: re-ranking is not re-optimisation; the pool is dominated by the 60 BO evaluations of one run, each a single noisy episode (fixed-gain SD of J ≈ 1).

---

## S3 [all]. Nominal comparison (Table 4), n = 15 runs per controller, no exclusions
Detection rate = 1.000 in all 90 runs. SD with ddof = 1. Respawn run = speed RMSE > 7.0 km/h (no run lies between 4.3 and 10.7).

| Method | Lane RMSE (px) | SSI (deg/step) | Jerk (deg/step²) | Saturation (fraction) | Speed RMSE, respawn-free (km/h) | Respawn runs |
|---|---|---|---|---|---|---|
| PID | 33.30 ± 3.44 | 10.14 ± 0.34 | 18.32 ± 0.49 | 0.18 ± 0.02 | 3.73 ± 0.22 (n=10) | 5/15 |
| Tuned PID | 30.02 ± 3.32 | 1.09 ± 0.10 | 2.12 ± 0.18 | 0.00 ± 0.00 | 1.30 ± 0.01 (n=11) | 4/15 |
| Manual SMC | 40.85 ± 8.07 | 0.54 ± 0.06 | 1.06 ± 0.11 | 0.00 ± 0.01 | 1.26 ± 0.35 (n=11) | 4/15 |
| Grid-search SMC | 39.72 ± 7.59 | 2.60 ± 0.20 | 5.03 ± 0.37 | 0.26 ± 0.05 | 1.83 ± 0.04 (n=11) | 4/15 |
| PSO SMC | 32.17 ± 6.31 | 3.37 ± 0.24 | 6.29 ± 0.39 | 0.06 ± 0.03 | 1.00 ± 0.02 (n=10) | 5/15 |
| Proposed BO-SMC | 31.22 ± 4.59 | 1.06 ± 0.10 | 2.06 ± 0.19 | 0.01 ± 0.01 | 0.61 ± 0.01 (n=10) | 5/15 |

**Paired tests: proposed − comparator** (exact Wilcoxon signed-rank, two-sided; Holm over the full family of 25 tests; effect = mean paired difference with BCa 95% CI and matched-pairs rank-biserial r; negative difference = proposed lower/better).

| Comparator | Metric | Mean diff [95% CI] | r_rb | Proposed lower | n | p | Holm p | Sig. |
|---|---|---|---|---|---|---|---|---|
| PID | Lane RMSE (px) | -2.079 [-5.652, +1.169] | -0.27 | 9/15 | 15 | 0.389 | 1.000 | no |
| PID | SSI (deg/step) | -9.076 [-9.241, -8.856] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| PID | Jerk (deg/step²) | -16.269 [-16.542, -15.963] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| PID | Saturation (fraction) | -0.174 [-0.182, -0.163] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| PID | Speed RMSE, respawn-free pairs (km/h) | -3.139 [-3.549, -2.977] | -1.00 | 5/5 | 5 | 0.062 | 0.500 | no |
| Tuned PID | Lane RMSE (px) | +1.200 [-2.828, +4.341] | +0.17 | 4/15 | 15 | 0.599 | 1.000 | no |
| Tuned PID | SSI (deg/step) | -0.035 [-0.108, +0.022] | -0.18 | 7/15 | 15 | 0.561 | 1.000 | no |
| Tuned PID | Jerk (deg/step²) | -0.061 [-0.198, +0.045] | -0.17 | 7/15 | 15 | 0.599 | 1.000 | no |
| Tuned PID | Saturation (fraction) | +0.001 [-0.002, +0.006] | +0.07 | 9/15 | 15 | 0.847 | 1.000 | no |
| Tuned PID | Speed RMSE, respawn-free pairs (km/h) | -0.680 [-0.684, -0.677] | -1.00 | 6/6 | 6 | 0.031 | 0.312 | no |
| Manual SMC | Lane RMSE (px) | -9.627 [-15.235, -6.694] | -0.98 | 14/15 | 15 | <0.001 | 0.002 | yes |
| Manual SMC | SSI (deg/step) | +0.517 [+0.481, +0.553] | +1.00 | 0/15 | 15 | <0.001 | 0.002 | yes |
| Manual SMC | Jerk (deg/step²) | +0.997 [+0.929, +1.062] | +1.00 | 0/15 | 15 | <0.001 | 0.002 | yes |
| Manual SMC | Saturation (fraction) | +0.004 [-0.001, +0.008] | +0.72 | 1/15 | 12 | 0.027 | 0.295 | no |
| Manual SMC | Speed RMSE, respawn-free pairs (km/h) | -0.543 [-0.548, -0.541] | -1.00 | 9/9 | 9 | 0.004 | 0.047 | yes |
| Grid-search SMC | Lane RMSE (px) | -8.503 [-13.186, -4.414] | -0.82 | 12/15 | 15 | 0.003 | 0.044 | yes |
| Grid-search SMC | SSI (deg/step) | -1.540 [-1.640, -1.435] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| Grid-search SMC | Jerk (deg/step²) | -2.977 [-3.162, -2.777] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| Grid-search SMC | Saturation (fraction) | -0.258 [-0.281, -0.234] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| Grid-search SMC | Speed RMSE, respawn-free pairs (km/h) | -1.227 [-1.254, -1.199] | -1.00 | 6/6 | 6 | 0.031 | 0.312 | no |
| PSO SMC | Lane RMSE (px) | -0.947 [-4.956, +2.786] | -0.15 | 7/15 | 15 | 0.639 | 1.000 | no |
| PSO SMC | SSI (deg/step) | -2.314 [-2.433, -2.204] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| PSO SMC | Jerk (deg/step²) | -4.240 [-4.443, -4.055] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| PSO SMC | Saturation (fraction) | -0.054 [-0.070, -0.039] | -1.00 | 15/15 | 15 | <0.001 | 0.002 | yes |
| PSO SMC | Speed RMSE, respawn-free pairs (km/h) | -0.378 [-0.408, -0.359] | -1.00 | 5/5 | 5 | 0.062 | 0.500 | no |

**Speed bimodality = simulator respawn (answer to Reviewer 1).**
Respawn runs per controller (run indices): PID [2, 5, 7, 12, 14]; Tuned PID [1, 3, 8, 10]; Manual SMC [2, 4, 11, 13]; Grid-search SMC [0, 5, 7, 14]; PSO SMC [1, 3, 8, 10, 12]; Proposed BO-SMC [4, 6, 9, 11, 13]
In global execution order the 27 respawn runs sit at episodes [3, 6, 9, 12, 16, 19, 22, 25, 28, 31, 34, 41, 44, 47, 50, 53, 56, 60, 63, 66, 69, 72, 76, 79, 82, 85, 88]; gaps [3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 7, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3] (SD 0.81); random placement gives SD this small in 0.000% of 20,000 permutations → the events are periodic in time, not linked to a controller or run index. Within-controller effect of a respawn on lane RMSE: -7.1 px (mean over controllers).
Per-frame evidence (run 0 logs): PID run 0: speed at window start 36.4 km/h, min 36.4 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Tuned PID run 0: speed at window start 48.5 km/h, min 48.1 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Manual SMC run 0: speed at window start 48.9 km/h, min 48.6 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Grid-search SMC run 0: speed at window start 7.9 km/h, min 7.9 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | PSO SMC run 0: speed at window start 49.1 km/h, min 48.4 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Proposed BO-SMC run 0: speed at window start 49.3 km/h, min 49.0 km/h, instantaneous drops (>20 km/h in one step) at t = [] s

**Pareto uncertainty on (lane RMSE, SSI) means** (paired bootstrap over run indices, 10000 resamples): P(proposed dominates X): PID 0.876, Tuned PID 0.237, Manual SMC 0.000, Grid-search SMC 1.000, PSO SMC 0.674. P(on the Pareto front): PID 0.004, Tuned PID 0.762, Manual SMC 1.000, Grid-search SMC 0.000, PSO SMC 0.046, Proposed BO-SMC 0.863

---

## S3 [paper]. Nominal comparison (Table 4), n = 15 runs per controller, no exclusions — controllers omitted: ['PSO_SMC']
Detection rate = 1.000 in all 90 runs. SD with ddof = 1. Respawn run = speed RMSE > 7.0 km/h (no run lies between 4.3 and 10.7).

| Method | Lane RMSE (px) | SSI (deg/step) | Jerk (deg/step²) | Saturation (fraction) | Speed RMSE, respawn-free (km/h) | Respawn runs |
|---|---|---|---|---|---|---|
| PID | 33.30 ± 3.44 | 10.14 ± 0.34 | 18.32 ± 0.49 | 0.18 ± 0.02 | 3.73 ± 0.22 (n=10) | 5/15 |
| Tuned PID | 30.02 ± 3.32 | 1.09 ± 0.10 | 2.12 ± 0.18 | 0.00 ± 0.00 | 1.30 ± 0.01 (n=11) | 4/15 |
| Manual SMC | 40.85 ± 8.07 | 0.54 ± 0.06 | 1.06 ± 0.11 | 0.00 ± 0.01 | 1.26 ± 0.35 (n=11) | 4/15 |
| Grid-search SMC | 39.72 ± 7.59 | 2.60 ± 0.20 | 5.03 ± 0.37 | 0.26 ± 0.05 | 1.83 ± 0.04 (n=11) | 4/15 |
| Proposed BO-SMC | 31.22 ± 4.59 | 1.06 ± 0.10 | 2.06 ± 0.19 | 0.01 ± 0.01 | 0.61 ± 0.01 (n=10) | 5/15 |

**Paired tests: proposed − comparator** (exact Wilcoxon signed-rank, two-sided; Holm over the full family of 20 tests; effect = mean paired difference with BCa 95% CI and matched-pairs rank-biserial r; negative difference = proposed lower/better).

| Comparator | Metric | Mean diff [95% CI] | r_rb | Proposed lower | n | p | Holm p | Sig. |
|---|---|---|---|---|---|---|---|---|
| PID | Lane RMSE (px) | -2.079 [-5.652, +1.169] | -0.27 | 9/15 | 15 | 0.389 | 1.000 | no |
| PID | SSI (deg/step) | -9.076 [-9.241, -8.856] | -1.00 | 15/15 | 15 | <0.001 | 0.001 | yes |
| PID | Jerk (deg/step²) | -16.269 [-16.542, -15.963] | -1.00 | 15/15 | 15 | <0.001 | 0.001 | yes |
| PID | Saturation (fraction) | -0.174 [-0.182, -0.163] | -1.00 | 15/15 | 15 | <0.001 | 0.001 | yes |
| PID | Speed RMSE, respawn-free pairs (km/h) | -3.139 [-3.549, -2.977] | -1.00 | 5/5 | 5 | 0.062 | 0.375 | no |
| Tuned PID | Lane RMSE (px) | +1.200 [-2.828, +4.341] | +0.17 | 4/15 | 15 | 0.599 | 1.000 | no |
| Tuned PID | SSI (deg/step) | -0.035 [-0.108, +0.022] | -0.18 | 7/15 | 15 | 0.561 | 1.000 | no |
| Tuned PID | Jerk (deg/step²) | -0.061 [-0.198, +0.045] | -0.17 | 7/15 | 15 | 0.599 | 1.000 | no |
| Tuned PID | Saturation (fraction) | +0.001 [-0.002, +0.006] | +0.07 | 9/15 | 15 | 0.847 | 1.000 | no |
| Tuned PID | Speed RMSE, respawn-free pairs (km/h) | -0.680 [-0.684, -0.677] | -1.00 | 6/6 | 6 | 0.031 | 0.250 | no |
| Manual SMC | Lane RMSE (px) | -9.627 [-15.235, -6.694] | -0.98 | 14/15 | 15 | <0.001 | 0.001 | yes |
| Manual SMC | SSI (deg/step) | +0.517 [+0.481, +0.553] | +1.00 | 0/15 | 15 | <0.001 | 0.001 | yes |
| Manual SMC | Jerk (deg/step²) | +0.997 [+0.929, +1.062] | +1.00 | 0/15 | 15 | <0.001 | 0.001 | yes |
| Manual SMC | Saturation (fraction) | +0.004 [-0.001, +0.008] | +0.72 | 1/15 | 12 | 0.027 | 0.242 | no |
| Manual SMC | Speed RMSE, respawn-free pairs (km/h) | -0.543 [-0.548, -0.541] | -1.00 | 9/9 | 9 | 0.004 | 0.039 | yes |
| Grid-search SMC | Lane RMSE (px) | -8.503 [-13.186, -4.414] | -0.82 | 12/15 | 15 | 0.003 | 0.037 | yes |
| Grid-search SMC | SSI (deg/step) | -1.540 [-1.640, -1.435] | -1.00 | 15/15 | 15 | <0.001 | 0.001 | yes |
| Grid-search SMC | Jerk (deg/step²) | -2.977 [-3.162, -2.777] | -1.00 | 15/15 | 15 | <0.001 | 0.001 | yes |
| Grid-search SMC | Saturation (fraction) | -0.258 [-0.281, -0.234] | -1.00 | 15/15 | 15 | <0.001 | 0.001 | yes |
| Grid-search SMC | Speed RMSE, respawn-free pairs (km/h) | -1.227 [-1.254, -1.199] | -1.00 | 6/6 | 6 | 0.031 | 0.250 | no |

**Speed bimodality = simulator respawn (answer to Reviewer 1).**
Respawn runs per controller (run indices): PID [2, 5, 7, 12, 14]; Tuned PID [1, 3, 8, 10]; Manual SMC [2, 4, 11, 13]; Grid-search SMC [0, 5, 7, 14]; PSO SMC [1, 3, 8, 10, 12]; Proposed BO-SMC [4, 6, 9, 11, 13]
In global execution order the 27 respawn runs sit at episodes [3, 6, 9, 12, 16, 19, 22, 25, 28, 31, 34, 41, 44, 47, 50, 53, 56, 60, 63, 66, 69, 72, 76, 79, 82, 85, 88]; gaps [3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 7, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3] (SD 0.81); random placement gives SD this small in 0.000% of 20,000 permutations → the events are periodic in time, not linked to a controller or run index. Within-controller effect of a respawn on lane RMSE: -7.1 px (mean over controllers).
Per-frame evidence (run 0 logs): PID run 0: speed at window start 36.4 km/h, min 36.4 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Tuned PID run 0: speed at window start 48.5 km/h, min 48.1 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Manual SMC run 0: speed at window start 48.9 km/h, min 48.6 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Grid-search SMC run 0: speed at window start 7.9 km/h, min 7.9 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | PSO SMC run 0: speed at window start 49.1 km/h, min 48.4 km/h, instantaneous drops (>20 km/h in one step) at t = [] s | Proposed BO-SMC run 0: speed at window start 49.3 km/h, min 49.0 km/h, instantaneous drops (>20 km/h in one step) at t = [] s

**Pareto uncertainty on (lane RMSE, SSI) means** (paired bootstrap over run indices, 10000 resamples): P(proposed dominates X): PID 0.876, Tuned PID 0.237, Manual SMC 0.000, Grid-search SMC 1.000. P(on the Pareto front): PID 0.004, Tuned PID 0.762, Manual SMC 1.000, Grid-search SMC 0.000, Proposed BO-SMC 0.863

---

## S4. Robustness (Table 5): 4 controllers × 5 conditions × 10 runs
Stalled run = detection rate < 0.1 (simulator returned frames without a lane on every frame: lane RMSE = 60 px imputed on all frames, SSI = 0, speed RMSE ≈ 90–100 km/h). Run order within each run index is FIXED (PID → Tuned PID → Manual SMC → proposed).

**Stalled (excluded) runs:**

| Condition | Controller | Run | Detection | Lane RMSE | SSI | Speed RMSE | fps |
|---|---|---|---|---|---|---|---|
| Nominal | PID | 6 | 0.000 | 60.0 | 0.00 | 100.0 | 364 |
| Nominal | PID | 7 | 0.000 | 60.0 | 0.00 | 97.4 | 395 |
| Nominal | Tuned PID | 6 | 0.000 | 60.0 | 0.00 | 100.0 | 368 |
| Nominal | Manual SMC | 6 | 0.000 | 60.0 | 0.00 | 100.0 | 366 |
| Nominal | Proposed BO-SMC | 6 | 0.000 | 60.0 | 0.00 | 100.0 | 365 |
| Blur | Proposed BO-SMC | 5 | 0.000 | 60.0 | 0.00 | 88.9 | 246 |

Partially stalled runs kept in all variants: Shadow/PID run 1 (detection 0.896, lane 163.0); Shadow/PID run 8 (detection 0.894, lane 157.1); Blur/Manual SMC run 5 (detection 0.736, lane 139.5); Blur/Proposed BO-SMC run 6 (detection 0.228, lane 134.0)

**Variant A: as submitted (per-cell exclusion)**

| Metric | Condition | PID | Tuned PID | Manual SMC | Proposed BO-SMC |
|---|---|---|---|---|---|
| SSI (deg/step) | Nominal | 10.07 ± 0.29 (n=8) | 1.07 ± 0.10 (n=9) | 0.59 ± 0.06 (n=9) | 1.10 ± 0.08 (n=9) |
| SSI (deg/step) | Noise | 5.08 ± 0.45 (n=10) | 1.82 ± 0.05 (n=10) | 1.04 ± 0.05 (n=10) | 1.92 ± 0.05 (n=10) |
| SSI (deg/step) | Shadow | 24.63 ± 0.25 (n=10) | 12.51 ± 0.55 (n=10) | 11.01 ± 0.18 (n=10) | 12.84 ± 0.39 (n=10) |
| SSI (deg/step) | Occlusion | 21.50 ± 0.43 (n=10) | 3.80 ± 0.13 (n=10) | 2.08 ± 0.05 (n=10) | 3.72 ± 0.17 (n=10) |
| SSI (deg/step) | Blur | 8.85 ± 0.19 (n=10) | 0.91 ± 0.07 (n=10) | 0.48 ± 0.05 (n=10) | 0.87 ± 0.21 (n=9) |
| Lane RMSE (px) | Nominal | 34.02 ± 2.39 (n=8) | 32.84 ± 6.30 (n=9) | 42.98 ± 7.11 (n=9) | 33.74 ± 3.16 (n=9) |
| Lane RMSE (px) | Noise | 16.12 ± 2.70 (n=10) | 16.42 ± 2.79 (n=10) | 14.83 ± 2.27 (n=10) | 16.16 ± 0.45 (n=10) |
| Lane RMSE (px) | Shadow | 163.76 ± 4.13 (n=10) | 147.31 ± 2.88 (n=10) | 142.42 ± 2.01 (n=10) | 147.03 ± 3.86 (n=10) |
| Lane RMSE (px) | Occlusion | 73.87 ± 8.80 (n=10) | 36.39 ± 1.80 (n=10) | 47.15 ± 3.77 (n=10) | 36.72 ± 7.00 (n=10) |
| Lane RMSE (px) | Blur | 32.29 ± 2.18 (n=10) | 33.98 ± 6.12 (n=10) | 49.50 ± 35.15 (n=10) | 41.94 ± 34.61 (n=9) |
| Steering variation per second (deg/s) | Nominal | 1468.52 ± 175.34 (n=8) | 148.07 ± 20.30 (n=9) | 86.14 ± 15.94 (n=9) | 157.36 ± 17.81 (n=9) |
| Steering variation per second (deg/s) | Noise | 96.04 ± 18.33 (n=10) | 34.70 ± 4.14 (n=10) | 19.62 ± 2.38 (n=10) | 37.26 ± 2.27 (n=10) |
| Steering variation per second (deg/s) | Shadow | 4642.79 ± 530.72 (n=10) | 2257.79 ± 323.69 (n=10) | 2012.24 ± 145.66 (n=10) | 2367.96 ± 203.88 (n=10) |
| Steering variation per second (deg/s) | Occlusion | 3426.08 ± 231.90 (n=10) | 555.23 ± 58.01 (n=10) | 317.11 ± 28.13 (n=10) | 573.29 ± 80.78 (n=10) |
| Steering variation per second (deg/s) | Blur | 931.59 ± 109.99 (n=10) | 103.48 ± 8.06 (n=10) | 51.78 ± 6.10 (n=10) | 97.20 ± 12.01 (n=9) |

**Variant B: no exclusion**

| Metric | Condition | PID | Tuned PID | Manual SMC | Proposed BO-SMC |
|---|---|---|---|---|---|
| SSI (deg/step) | Nominal | 8.06 ± 4.25 (n=10) | 0.96 ± 0.35 (n=10) | 0.53 ± 0.20 (n=10) | 0.99 ± 0.35 (n=10) |
| SSI (deg/step) | Noise | 5.08 ± 0.45 (n=10) | 1.82 ± 0.05 (n=10) | 1.04 ± 0.05 (n=10) | 1.92 ± 0.05 (n=10) |
| SSI (deg/step) | Shadow | 24.63 ± 0.25 (n=10) | 12.51 ± 0.55 (n=10) | 11.01 ± 0.18 (n=10) | 12.84 ± 0.39 (n=10) |
| SSI (deg/step) | Occlusion | 21.50 ± 0.43 (n=10) | 3.80 ± 0.13 (n=10) | 2.08 ± 0.05 (n=10) | 3.72 ± 0.17 (n=10) |
| SSI (deg/step) | Blur | 8.85 ± 0.19 (n=10) | 0.91 ± 0.07 (n=10) | 0.48 ± 0.05 (n=10) | 0.78 ± 0.34 (n=10) |
| Lane RMSE (px) | Nominal | 39.21 ± 11.16 (n=10) | 35.56 ± 10.44 (n=10) | 44.69 ± 8.60 (n=10) | 36.37 ± 8.82 (n=10) |
| Lane RMSE (px) | Noise | 16.12 ± 2.70 (n=10) | 16.42 ± 2.79 (n=10) | 14.83 ± 2.27 (n=10) | 16.16 ± 0.45 (n=10) |
| Lane RMSE (px) | Shadow | 163.76 ± 4.13 (n=10) | 147.31 ± 2.88 (n=10) | 142.42 ± 2.01 (n=10) | 147.03 ± 3.86 (n=10) |
| Lane RMSE (px) | Occlusion | 73.87 ± 8.80 (n=10) | 36.39 ± 1.80 (n=10) | 47.15 ± 3.77 (n=10) | 36.72 ± 7.00 (n=10) |
| Lane RMSE (px) | Blur | 32.29 ± 2.18 (n=10) | 33.98 ± 6.12 (n=10) | 49.50 ± 35.15 (n=10) | 43.75 ± 33.12 (n=10) |
| Steering variation per second (deg/s) | Nominal | 1174.82 ± 638.20 (n=10) | 133.26 ± 50.58 (n=10) | 77.53 ± 31.11 (n=10) | 141.62 ± 52.52 (n=10) |
| Steering variation per second (deg/s) | Noise | 96.04 ± 18.33 (n=10) | 34.70 ± 4.14 (n=10) | 19.62 ± 2.38 (n=10) | 37.26 ± 2.27 (n=10) |
| Steering variation per second (deg/s) | Shadow | 4642.79 ± 530.72 (n=10) | 2257.79 ± 323.69 (n=10) | 2012.24 ± 145.66 (n=10) | 2367.96 ± 203.88 (n=10) |
| Steering variation per second (deg/s) | Occlusion | 3426.08 ± 231.90 (n=10) | 555.23 ± 58.01 (n=10) | 317.11 ± 28.13 (n=10) | 573.29 ± 80.78 (n=10) |
| Steering variation per second (deg/s) | Blur | 931.59 ± 109.99 (n=10) | 103.48 ± 8.06 (n=10) | 51.78 ± 6.10 (n=10) | 87.48 ± 32.76 (n=10) |

**Variant C: balanced exclusion (same run indices removed for all controllers)**

| Metric | Condition | PID | Tuned PID | Manual SMC | Proposed BO-SMC |
|---|---|---|---|---|---|
| SSI (deg/step) | Nominal | 10.07 ± 0.29 (n=8) | 1.06 ± 0.10 (n=8) | 0.60 ± 0.06 (n=8) | 1.10 ± 0.08 (n=8) |
| SSI (deg/step) | Noise | 5.08 ± 0.45 (n=10) | 1.82 ± 0.05 (n=10) | 1.04 ± 0.05 (n=10) | 1.92 ± 0.05 (n=10) |
| SSI (deg/step) | Shadow | 24.63 ± 0.25 (n=10) | 12.51 ± 0.55 (n=10) | 11.01 ± 0.18 (n=10) | 12.84 ± 0.39 (n=10) |
| SSI (deg/step) | Occlusion | 21.50 ± 0.43 (n=10) | 3.80 ± 0.13 (n=10) | 2.08 ± 0.05 (n=10) | 3.72 ± 0.17 (n=10) |
| SSI (deg/step) | Blur | 8.81 ± 0.16 (n=9) | 0.90 ± 0.07 (n=9) | 0.47 ± 0.04 (n=9) | 0.87 ± 0.21 (n=9) |
| Lane RMSE (px) | Nominal | 34.02 ± 2.39 (n=8) | 32.72 ± 6.72 (n=8) | 43.63 ± 7.32 (n=8) | 34.41 ± 2.62 (n=8) |
| Lane RMSE (px) | Noise | 16.12 ± 2.70 (n=10) | 16.42 ± 2.79 (n=10) | 14.83 ± 2.27 (n=10) | 16.16 ± 0.45 (n=10) |
| Lane RMSE (px) | Shadow | 163.76 ± 4.13 (n=10) | 147.31 ± 2.88 (n=10) | 142.42 ± 2.01 (n=10) | 147.03 ± 3.86 (n=10) |
| Lane RMSE (px) | Occlusion | 73.87 ± 8.80 (n=10) | 36.39 ± 1.80 (n=10) | 47.15 ± 3.77 (n=10) | 36.72 ± 7.00 (n=10) |
| Lane RMSE (px) | Blur | 32.22 ± 2.30 (n=9) | 34.65 ± 6.08 (n=9) | 39.50 ± 16.28 (n=9) | 41.94 ± 34.61 (n=9) |
| Steering variation per second (deg/s) | Nominal | 1468.52 ± 175.34 (n=8) | 146.30 ± 20.94 (n=8) | 87.37 ± 16.58 (n=8) | 158.14 ± 18.88 (n=8) |
| Steering variation per second (deg/s) | Noise | 96.04 ± 18.33 (n=10) | 34.70 ± 4.14 (n=10) | 19.62 ± 2.38 (n=10) | 37.26 ± 2.27 (n=10) |
| Steering variation per second (deg/s) | Shadow | 4642.79 ± 530.72 (n=10) | 2257.79 ± 323.69 (n=10) | 2012.24 ± 145.66 (n=10) | 2367.96 ± 203.88 (n=10) |
| Steering variation per second (deg/s) | Occlusion | 3426.08 ± 231.90 (n=10) | 555.23 ± 58.01 (n=10) | 317.11 ± 28.13 (n=10) | 573.29 ± 80.78 (n=10) |
| Steering variation per second (deg/s) | Blur | 927.23 ± 115.74 (n=9) | 103.12 ± 8.46 (n=9) | 50.43 ± 4.62 (n=9) | 97.20 ± 12.01 (n=9) |

**Mean control-loop rate per condition (Hz, non-stalled runs).** Per-step SSI is not comparable across conditions when the loop rate differs; the per-second variation above is.

| Condition | PID | Tuned PID | Manual SMC | Proposed BO-SMC |
|---|---|---|---|---|
| Nominal | 146 | 139 | 146 | 143 |
| Noise | 19 | 19 | 19 | 19 |
| Shadow | 188 | 180 | 183 | 184 |
| Occlusion | 159 | 147 | 152 | 154 |
| Blur | 105 | 115 | 108 | 120 |

**Paired tests, proposed vs PID and vs Tuned PID** (complete-case pairs, exact Wilcoxon, Holm over 20 tests). Smallest attainable exact p: n=8 → 0.0078, n=9 → 0.0039, n=10 → 0.0020.

| Comparator | Metric | Condition | n pairs | Means (proposed vs comparator) | Mean diff [95% CI] | p | Holm p | Sig. |
|---|---|---|---|---|---|---|---|---|
| PID | SSI | Nominal | 8 | 1.10 vs 10.07 | -8.97 [-9.16, -8.78] | 0.008 | 0.109 | no |
| PID | SSI | Noise | 10 | 1.92 vs 5.08 | -3.15 [-3.38, -2.80] | 0.002 | 0.039 | yes |
| PID | SSI | Shadow | 10 | 12.84 vs 24.63 | -11.79 [-12.01, -11.60] | 0.002 | 0.039 | yes |
| PID | SSI | Occlusion | 10 | 3.72 vs 21.50 | -17.78 [-18.06, -17.50] | 0.002 | 0.039 | yes |
| PID | SSI | Blur | 9 | 0.87 vs 8.81 | -7.94 [-8.09, -7.81] | 0.004 | 0.059 | no |
| PID | Lane RMSE | Nominal | 8 | 34.41 vs 34.02 | +0.39 [-1.71, +2.72] | 0.844 | 1.000 | no |
| PID | Lane RMSE | Noise | 10 | 16.16 vs 16.12 | +0.05 [-0.95, +3.20] | 0.131 | 1.000 | no |
| PID | Lane RMSE | Shadow | 10 | 147.03 vs 163.76 | -16.72 [-19.39, -14.94] | 0.002 | 0.039 | yes |
| PID | Lane RMSE | Occlusion | 10 | 36.72 vs 73.87 | -37.14 [-42.98, -32.39] | 0.002 | 0.039 | yes |
| PID | Lane RMSE | Blur | 9 | 41.94 vs 32.22 | +9.72 [-2.17, +53.92] | 0.652 | 1.000 | no |
| Tuned PID | SSI | Nominal | 9 | 1.10 vs 1.07 | +0.03 [-0.01, +0.06] | 0.129 | 1.000 | no |
| Tuned PID | SSI | Noise | 10 | 1.92 vs 1.82 | +0.10 [+0.04, +0.15] | 0.010 | 0.127 | no |
| Tuned PID | SSI | Shadow | 10 | 12.84 vs 12.51 | +0.32 [-0.19, +0.65] | 0.064 | 0.773 | no |
| Tuned PID | SSI | Occlusion | 10 | 3.72 vs 3.80 | -0.07 [-0.18, +0.06] | 0.275 | 1.000 | no |
| Tuned PID | SSI | Blur | 9 | 0.87 vs 0.90 | -0.03 [-0.27, +0.05] | 0.359 | 1.000 | no |
| Tuned PID | Lane RMSE | Nominal | 9 | 33.74 vs 32.84 | +0.90 [-3.30, +6.27] | 1.000 | 1.000 | no |
| Tuned PID | Lane RMSE | Noise | 10 | 16.16 vs 16.42 | -0.26 [-1.25, +3.21] | 0.084 | 0.924 | no |
| Tuned PID | Lane RMSE | Shadow | 10 | 147.03 vs 147.31 | -0.27 [-2.24, +2.86] | 0.695 | 1.000 | no |
| Tuned PID | Lane RMSE | Occlusion | 10 | 36.72 vs 36.39 | +0.33 [-3.34, +6.11] | 0.492 | 1.000 | no |
| Tuned PID | Lane RMSE | Blur | 9 | 41.94 vs 34.65 | +7.29 [-5.97, +55.43] | 0.203 | 1.000 | no |

Occlusion check: lane RMSE proposed 36.72 vs PID 73.87 (ratio 0.497); SSI 3.72 vs 21.50 (5.78-fold). Tuned PID under occlusion: lane 36.39, SSI 3.80.

---

## S5. Component analysis / ablation (Table 6), 10 runs per variant, no exclusions
Violation rate (the manuscript's 'sample frequency') = fraction of logged control steps with s_k·Δs_k > 0 on the steering channel. m6 = mean max(0, s_k·Δs_k), in px². Speed RMSE is driven by the number of simulator respawns per variant and is NOT comparable across rows.

| Variant | Min detection | Lane RMSE (px) | SSI (deg/step) | Jerk (deg/step²) | m6 (px²) | Violation rate (fraction of steps) | Speed RMSE median (respawn runs) | Steering gains found by BO |
|---|---|---|---|---|---|---|---|---|
| Sign switching, no stability term | 0.999 | 33.34 ± 3.12 | 2.08 ± 0.11 | 4.05 ± 0.22 | 282.25 ± 66.64 | 0.437 ± 0.027 | 1.09 (2) | λ=0.343, η=0.50, φ=inert |
| Boundary layer, no stability term | 1.000 | 39.48 ± 5.92 | 0.70 ± 0.06 | 1.36 ± 0.12 | 186.24 ± 67.02 | 0.458 ± 0.030 | 0.65 (3) | λ=0.107, η=0.50, φ=41.3 |
| Sign switching + stability term | 0.999 | 31.93 ± 4.42 | 1.35 ± 0.08 | 2.63 ± 0.16 | 230.56 ± 94.85 | 0.458 ± 0.032 | 0.62 (3) | λ=0.188, η=0.50, φ=inert |
| Boundary layer + stability term (full) | 1.000 | 31.26 ± 1.65 | 1.07 ± 0.06 | 2.08 ± 0.11 | 182.52 ± 17.60 | 0.443 ± 0.025 | 0.67 (1) | λ=0.167, η=0.50, φ=20.1 |

**Effect of adding a component** (unpaired exact Mann–Whitney, Holm over 16 tests; each comparison contrasts two separately optimised controllers, so it reflects the component AND the outcome of one optimisation run).

| Component added | Metric | Mean before → after | Diff [95% CI] | Cliff's δ | p | Holm p | Sig. |
|---|---|---|---|---|---|---|---|
| boundary layer (no stability term) | Lane RMSE (px) | 33.34 → 39.48 | +6.14 [+1.98, +9.82] | +0.62 | 0.019 | 0.074 | no |
| boundary layer (no stability term) | SSI (deg/step) | 2.08 → 0.70 | -1.37 [-1.45, -1.30] | -1.00 | <0.001 | <0.001 | yes |
| boundary layer (no stability term) | Jerk (deg/step²) | 4.05 → 1.36 | -2.69 [-2.83, -2.54] | -1.00 | <0.001 | <0.001 | yes |
| boundary layer (no stability term) | m6 (px²) | 282.25 → 186.24 | -96.02 [-153.87, -40.67] | -0.82 | 0.001 | 0.008 | yes |
| boundary layer (with stability term) | Lane RMSE (px) | 31.93 → 31.26 | -0.67 [-3.42, +2.09] | -0.18 | 0.529 | 0.786 | no |
| boundary layer (with stability term) | SSI (deg/step) | 1.35 → 1.07 | -0.28 [-0.34, -0.22] | -1.00 | <0.001 | <0.001 | yes |
| boundary layer (with stability term) | Jerk (deg/step²) | 2.63 → 2.08 | -0.54 [-0.65, -0.42] | -1.00 | <0.001 | <0.001 | yes |
| boundary layer (with stability term) | m6 (px²) | 230.56 → 182.52 | -48.04 [-111.88, -6.99] | -0.64 | 0.015 | 0.073 | no |
| stability term (sign switching) | Lane RMSE (px) | 33.34 → 31.93 | -1.41 [-4.55, +1.78] | -0.24 | 0.393 | 0.786 | no |
| stability term (sign switching) | SSI (deg/step) | 2.08 → 1.35 | -0.72 [-0.80, -0.65] | -1.00 | <0.001 | <0.001 | yes |
| stability term (sign switching) | Jerk (deg/step²) | 4.05 → 2.63 | -1.42 [-1.58, -1.27] | -1.00 | <0.001 | <0.001 | yes |
| stability term (sign switching) | m6 (px²) | 282.25 → 230.56 | -51.69 [-116.58, +23.42] | -0.80 | 0.002 | 0.011 | yes |
| stability term (boundary layer) | Lane RMSE (px) | 39.48 → 31.26 | -8.22 [-11.58, -4.46] | -0.74 | 0.004 | 0.023 | yes |
| stability term (boundary layer) | SSI (deg/step) | 0.70 → 1.07 | +0.37 [+0.32, +0.42] | +1.00 | <0.001 | <0.001 | yes |
| stability term (boundary layer) | Jerk (deg/step²) | 1.36 → 2.08 | +0.73 [+0.63, +0.82] | +1.00 | <0.001 | <0.001 | yes |
| stability term (boundary layer) | m6 (px²) | 186.24 → 182.52 | -3.71 [-49.78, +29.49] | +0.34 | 0.218 | 0.653 | no |

---

## S6. Revised Figure 3
- `fig3_steering_rev_grid.png`: PID SSI 10.31 deg/step (loop 176 Hz); Grid-search SMC SSI 2.74 deg/step (loop 172 Hz); Proposed BO-SMC SSI 1.04 deg/step (loop 155 Hz)
- `fig3_steering_rev_pso.png`: PID SSI 10.31 deg/step (loop 176 Hz); PSO SMC SSI 3.02 deg/step (loop 150 Hz); Proposed BO-SMC SSI 1.04 deg/step (loop 155 Hz)
The per-run SSI values of run 0 differ from the 15-run means in Table 4; the caption must say 'single representative run (run index 0)'.
