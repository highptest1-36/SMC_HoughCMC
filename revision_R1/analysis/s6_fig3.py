"""
s6_fig3.py — revised Figure 3 (terminology matched to the SSI definition).
Answers: Editor 6 (Figure 3 terminology).
Source: journal_cmc/results/logs/compare_<method>_seed0.json (run index 0 of Table 4).
Produces two variants: PID / Grid-search SMC / proposed (recommended if PSO-SMC is
withdrawn from Tables 3-4) and PID / PSO SMC / proposed (same panels as submitted).
"""
import numpy as np

import common as K


def _ssi(a):
    return float(np.mean(np.abs(np.diff(a))))


def _plot(methods, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"PID": "#1f77b4", "GridSMC": "#ff7f0e", "PSO_SMC": "#9467bd", "BO_SMC": "#2ca02c"}
    fig, axes = plt.subplots(len(methods), 1, figsize=(6.4, 6.2), sharex=True)
    info = []
    for ax, m in zip(axes, methods):
        lg = K.load_log(f"compare_{m}_seed0.json")
        t, a = np.array(lg["t"]), np.array(lg["angle"])
        s = _ssi(a)
        info.append((m, s, len(a) / t[-1]))
        ax.plot(t, a, color=col[m], lw=0.6)
        for y in (-20, 20):
            ax.axhline(y, color="0.6", ls="--", lw=0.8)
        ax.set_ylim(-24, 24)
        ax.set_ylabel(r"$\delta$ (deg)")
        ax.text(0.99, 0.93, f"{K.LABEL[m]}   SSI = {s:.2f} deg/step", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec=col[m]))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(r"Steering command $\delta_k$, run index 0; SSI = mean $|\delta_k-\delta_{k-1}|$ per control step "
                 "(lower = smoother)", fontsize=9)
    fig.tight_layout()
    fig.savefig(K.os.path.join(K.OUT_FIG, fname), dpi=400)
    plt.close(fig)
    return info


def run():
    md = ["## S6. Revised Figure 3"]
    for methods, fname in [(["PID", "GridSMC", "BO_SMC"], "fig3_steering_rev_grid.png"),
                           (["PID", "PSO_SMC", "BO_SMC"], "fig3_steering_rev_pso.png")]:
        info = _plot(methods, fname)
        md.append(f"- `{fname}`: " + "; ".join(f"{K.LABEL[m]} SSI {s:.2f} deg/step (loop {hz:.0f} Hz)"
                                               for m, s, hz in info))
    md.append("The per-run SSI values of run 0 differ from the 15-run means in Table 4; the caption must say "
              "'single representative run (run index 0)'.")
    return "\n".join(md)


if __name__ == "__main__":
    print(run())
