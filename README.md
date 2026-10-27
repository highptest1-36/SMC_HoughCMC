# Sample-Efficient Bayesian Optimization of Boundary-Layer SMC for Vision-Based Lane Keeping

Code, configuration, and data for the manuscript

> C.-P. Ha and P.-N. Le, "Sample-Efficient Bayesian Optimization of Boundary-Layer Sliding Mode
> Control for Real-Time Vision-Based Lane Keeping", *Computers, Materials & Continua* (under review,
> manuscript ID 92446).

## Contents

| Folder | Content |
|---|---|
| `journal_cmc/` | Experiment harness: Hough perception with perturbation injection (`perception.py`), boundary-layer SMC and PID controllers (`controller.py`), composite objective (`objective.py`), optimizers — Bayesian optimization (BoTorch), PSO (PySwarms), random and grid search (`optimizers.py`), simulator client (`simulator.py`), experiments (`experiments.py`, `fair_compare.py`), CLI (`run.py`), configuration (`config.py`) |
| `journal_cmc/results/optim/` | Per-run and per-evaluation results (JSON) of all experiments reported in the paper |
| `journal_cmc/results/logs/` | Per-frame logs (run index 0 of every controller/condition) |
| `revision_R1/` | Statistical analysis that produces every number, table and figure of the paper from the JSON files (`run_all.py`), and its output (`output/REPORT.md`, `output/tables`, `output/figures`) |

## Reproducing the tables and figures (no simulator needed)

```
pip install numpy scipy matplotlib
python revision_R1/run_all.py
```

The script reads only `journal_cmc/results/` and writes `revision_R1/output/`. It takes about 15 s.
Values are reported as mean (standard deviation), as in the paper, but with more decimals.

| Paper | Output files (`revision_R1/output/`) |
|---|---|
| Table 2, Figure 2 (matched-budget optimizer study) | `tables/table2_optimizers.csv`, `tables/table2_tests.csv`, `figures/fig2_convergence_rev.png` |
| Table 3 (deployed gains) | `tables/gains_deployed.csv`, `tables/grid_phi_v_effect.csv` (effect of the fixed φv of the grid-search SMC) |
| Section 3.3 (operating regime, objective sensitivity) | `tables/regime_all_bo_runs.csv`, `tables/bo_seeds_gains.csv`, `tables/sensitivity_offline.csv` |
| Table 4, Figure 3 (nominal comparison) | `tables/table4_nominal_paper.csv`, `tables/table4_tests_paper.csv` (all 20 paired tests with 95% CIs), `figures/fig3_steering_rev_grid.png` (panels (a) Classical PID, (b) Grid-search SMC, (c) Proposed BO-SMC) |
| Table 5 (robustness) | `tables/table5_variantA.csv` (as reported), `table5_variantB.csv` (no exclusion), `table5_variantC.csv` (balanced exclusion), `table5_tests.csv` |
| Table 6 (component analysis) | `tables/table6_ablation.csv`, `tables/table6_tests.csv` |
| All numbers quoted in the text | `REPORT.md` |

## Re-running the experiments

The experiments require the Unity driving simulator used in the paper (a Windows build provided by the
organizers of a student autonomous-driving competition). The authors are not permitted to redistribute
this build. The client communicates with it over TCP (`127.0.0.1:54321`): it sends `"<steering_deg> <speed_cmd>"`
and receives a JSON message `{"Speed", "Angle", "Img"}` with a base64-encoded 640x360 JPEG frame.
Any simulator implementing this protocol can be used. `journal_cmc/mock_server.py` implements the
protocol for testing without the simulator. See `journal_cmc/QUICKSTART.md` and `journal_cmc/README_RUN.md`.

Software used for the paper: Python 3.10.6, OpenCV 4.11.0, NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.7.1 (CUDA 12.6),
BoTorch 0.16.1, GPyTorch 1.15.2, PySwarms 1.3.0 (`journal_cmc/requirements_journal.txt`).

## Units and fixed settings of the controller gains

The gains are numerical coefficients acting on the raw signals of the implementation (no physical
normalization): lateral error `e`, its increment and the steering sliding variable in px; speed error,
its increment and the speed sliding variable in km/h; steering command in deg; speed command in the
integer units of the simulator (0 to 150). Hence `phi_a` is in px, `phi_v` in km/h, `eta_a` in deg,
`eta_v` in speed-command units, and `lambda_a`, `lambda_v` are scalar coefficients that act both as the
surface slope in `s` and as the gain of the nominal feedback in `u`. Only Bayesian optimization maps the
gains linearly from the search bounds (`journal_cmc/config.py`, `SEARCH_SPACE`) to the unit cube; PSO and
random search operate directly on the bounds.

Switching functions (`journal_cmc/controller.py`): the manual SMC uses `s/(|s|+1e-3)` in both channels
(no tunable width parameter); the grid-search and BO-tuned SMC use `sat(s/phi)` in both channels.

The grid-search SMC searched only the 3 steering gains. Its longitudinal gains `lambda_v = 70` and
`eta_v = 0.001` are inherited from the manual SMC, and `phi_v = 1` is a fixed value taken from the
configuration of the manual SMC (which does not use it); none of the three was searched. Because the
switching term is bounded by `eta_v = 0.001` for any `phi_v` and the speed command is truncated to an
integer, `phi_v` can affect the command only through this term. In nominal run 0 the term changed the
speed command in 1 of 7718 steps, by one unit (in that step the speed sliding variable lies outside the
boundary layer for every admissible `phi_v`), and the commands obtained with `phi_v = 1` and `phi_v = 50`
(and with 99 values of `phi_v` in [1, 50]) are identical in all steps; the effect of `phi_v` is therefore
negligible (`revision_R1/analysis/s2_gains_regime.py`, function `_phi_v_effect`; output
`tables/grid_phi_v_effect.csv`).

## Exclusion rules of the robustness experiment (Table 5)

An episode is regarded as a simulator stall, and excluded, when its lane-detection rate is below 0.1
(`STALL_DETECTION` in `revision_R1/analysis/common.py`). Six of the 200 robustness episodes meet this
criterion (listed in `REPORT.md`, section S4). Three variants are computed:

| Variant | Rule | Output |
|---|---|---|
| A (as reported in Table 5) | each stalled episode is excluded from the mean of its own controller; each paired test uses only the run indices valid for both controllers of the pair | `tables/table5_variantA.csv`, `tables/table5_tests.csv` |
| B (no exclusion) | all 10 runs are kept | `tables/table5_variantB.csv` |
| C (balanced exclusion) | for each condition, a run index that meets the criterion for any controller is excluded for all controllers (nominal: runs 6 and 7, n = 8; blur: run 5, n = 9) | `tables/table5_variantC.csv` |

Variants A and C are implemented in `revision_R1/analysis/s4_robustness.py` (function `cell`), and the
Holm-adjusted paired tests of variants B and C in `revision_R1/analysis/s7_text_stats.py`
(function `_robustness_variants`); their results are listed in `REPORT.md`.

## Notes on the data

- The deployed PSO tuning run (`pso_history.json`) was affected by a simulator stall after 13 of 80
  evaluations (detection rate 0 afterwards); its controller is not used in the revised paper.
- Runs whose speed RMSE exceeds 7 km/h contain a simulator respawn (vehicle returned to the start pose
  at zero speed); see Section 3.4 of the paper.

## License

MIT License (see `LICENSE`). The simulator build is not part of this repository.
