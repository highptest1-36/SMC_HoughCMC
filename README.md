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

## Re-running the experiments

The experiments require the Unity driving simulator used in the paper (a Windows build provided by the
organizers of a student autonomous-driving competition). The authors are not permitted to redistribute
this build. The client communicates with it over TCP (`127.0.0.1:54321`): it sends `"<steering_deg> <speed_cmd>"`
and receives a JSON message `{"Speed", "Angle", "Img"}` with a base64-encoded 640x360 JPEG frame.
Any simulator implementing this protocol can be used. `journal_cmc/mock_server.py` implements the
protocol for testing without the simulator. See `journal_cmc/QUICKSTART.md` and `journal_cmc/README_RUN.md`.

Software used for the paper: Python 3.10.6, OpenCV 4.11.0, NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.7.1 (CUDA 12.6),
BoTorch 0.16.1, GPyTorch 1.15.2, PySwarms 1.3.0 (`journal_cmc/requirements_journal.txt`).

## Notes on the data

- The deployed PSO tuning run (`pso_history.json`) was affected by a simulator stall after 13 of 80
  evaluations (detection rate 0 afterwards); its controller is not used in the revised paper.
- Runs whose speed RMSE exceeds 7 km/h contain a simulator respawn (vehicle returned to the start pose
  at zero speed); see Section 3.4 of the paper.

## License

MIT License (see `LICENSE`). The simulator build is not part of this repository.
