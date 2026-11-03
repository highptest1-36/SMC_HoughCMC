"""
make_readme_figures.py — figures shown in the repository README (light and dark versions).

Every value is read from journal_cmc/results/ (the same files that produce the paper's tables);
nothing is typed in by hand. Run from the repository root:

    python docs/make_readme_figures.py

Output: docs/figures/<name>_light.png and docs/figures/<name>_dark.png
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "revision_R1", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "journal_cmc"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import common as K                                                   # noqa: E402
import config as C                                                   # noqa: E402
import perception as P                                               # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
FRAMES = os.path.join(ROOT, "journal_cmc", "results", "perception_debug")
DPI = 200

# Colour roles (validated categorical palette; the proposed controller is always blue).
THEMES = {
    "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e7e6e2", axis="#b9b8b3",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray1="#a3a29d", gray2="#6f6e69"),
    "dark": dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#2f2f2d", axis="#5a5955",
                 blue="#3987e5", orange="#d95926", aqua="#199e70", gray1="#77766f", gray2="#a9a8a1"),
}
CTRL = [("PID", "Classical PID", "orange", "o"), ("TunedPID", "Tuned PID", "aqua", "s"),
        ("ManualSMC", "Manual SMC", "gray1", "^"), ("GridSMC", "Grid-search SMC", "gray2", "D"),
        ("BO_SMC", "Proposed BO-SMC", "blue", "o")]
OPT = [("random", "Random search", "aqua"), ("pso", "PSO", "orange"), ("bo", "Proposed BO", "blue")]


def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["fg2"], "ytick.color": t["fg2"], "axes.titlecolor": t["fg"],
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def save(fig, name, theme):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}_{theme}.png"), dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def caption(fig, text, t):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=9, color=t["fg2"], transform=fig.transFigure)


# --------------------------------------------------------------------------- #
# 1. What the vehicle sees: camera frame -> lane mask -> Hough segments -> error
# --------------------------------------------------------------------------- #
def _mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(C.HSV_LOWER_YELLOW, np.uint8), np.array(C.HSV_UPPER_YELLOW, np.uint8))
    return cv2.GaussianBlur(m, C.GAUSS_KERNEL, 0)


def fig_perception(theme):
    t = THEMES[theme]
    style(t)
    img = cv2.imread(os.path.join(FRAMES, "sample0_raw.png"))
    err, _, det, viz = P.detect_lane(img, draw=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.35))
    panels = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1  Onboard camera frame (640×360)", None),
              (_mask(img), "2  Yellow center-line mask (HSV band)", "gray"),
              (cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), f"3  Hough segments → lateral error e = {err:+.0f} px", None)]
    for a, (im, title, cmap) in zip(ax, panels):
        a.imshow(im, cmap=cmap)
        a.set_title(title, fontsize=11)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    caption(fig, "Panel 3: yellow = detected line segments, green = lane position (mean of segment midpoints), "
                 "red = image center; their horizontal distance is the error fed to the controller.", t)
    fig.tight_layout(w_pad=1.2)
    save(fig, "perception", theme)


def fig_perturbations(theme):
    t = THEMES[theme]
    style(t)
    img = cv2.imread(os.path.join(FRAMES, "sample2_raw.png"))
    kinds = [("clean", "Nominal"), ("noise", "Gaussian noise (σ = 25)"), ("shadow", "Shadow band (50 %)"),
             ("occlusion", "Occlusion (25 % × 30 %)"), ("blur", "Blur (σ = 3 px)")]
    fig, ax = plt.subplots(1, 5, figsize=(16, 2.55))
    for a, (k, name) in zip(ax, kinds):
        rng = np.random.default_rng(3)
        im = P.apply_perturbation(img, k, rng)
        err, _, det, viz = P.detect_lane(im, draw=True)
        a.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        a.set_title(name, fontsize=10.5)
        a.set_xlabel(f"e = {err:+.0f} px" if det else "lane lost", fontsize=9.5, color=t["fg2"])
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    caption(fig, "The four perturbations of the robustness experiment, applied to every camera frame before "
                 "lane detection (one example frame; positions are redrawn at every frame during a run).", t)
    fig.tight_layout(w_pad=0.6)
    save(fig, "perturbations", theme)


# --------------------------------------------------------------------------- #
# 2. Sample efficiency of the tuning methods (matched budget, 10 seeds each)
# --------------------------------------------------------------------------- #
def fig_optimizers(theme):
    t = THEMES[theme]
    style(t)
    fc = K.load("fair_compare_results.json")["data"]
    R = {o: np.array([r["running_best"] for r in fc[o]], float) for o, *_ in OPT}
    fin = {o: R[o][:, -1] for o in R}
    mean = {o: R[o].mean(0) for o in R}
    x = np.arange(1, R["bo"].shape[1] + 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.55, 1]})
    # end-of-line labels, spread apart vertically so that they never overlap
    ends = sorted(((mean[o][-1], o, name) for o, name, _ in OPT), reverse=True)
    ylab, prev = {}, None
    for v, o, _ in ends:
        ylab[o] = v if prev is None else min(v, prev - 0.32)
        prev = ylab[o]
    for o, name, c in OPT:
        sd = R[o].std(0, ddof=1)
        a1.fill_between(x, mean[o] - sd, mean[o] + sd, color=t[c], alpha=0.14, lw=0)
        a1.plot(x, mean[o], color=t[c], lw=2.2 if o == "bo" else 1.6)
        a1.text(x[-1] + 1.5, ylab[o], f"{name}  {mean[o][-1]:.2f}", va="center", fontsize=10,
                color=t["fg"], fontweight="bold" if o == "bo" else "normal")
    notes = {"pso": ("PSO", (19.0, 6.75)), "random": ("random-search", (24.5, 6.05))}
    for o in ("pso", "random"):
        k = int(np.argmax(mean["bo"] <= mean[o][-1])) + 1
        lab, ytxt = notes[o]
        a1.plot([k], [mean[o][-1]], "o", ms=7, color=t["blue"], mec=t["bg"], mew=1.5, zorder=5)
        a1.annotate(f"BO reaches the final {lab} level after {k} evaluations", (k, mean[o][-1]),
                    xytext=ytxt, fontsize=9.5, color=t["fg2"],
                    arrowprops=dict(arrowstyle="-", color=t["axis"], lw=0.9, shrinkB=4))
    a1.set_ylim(3.5, 10.5)
    a1.set_xlim(1, x[-1] + 18)
    a1.set_xticks([1, 12, 20, 30, 40, 50, 60])
    a1.set_xlabel("Closed-loop evaluations (one 45-s real-time driving episode each)")
    a1.set_ylabel("Best objective so far (lower is better)")
    a1.set_title("Convergence: mean over 10 seeds, band = ±1 SD")

    for i, (o, name, c) in enumerate(OPT):
        y = np.full(len(fin[o]), i) + np.linspace(-0.13, 0.13, len(fin[o]))
        a2.scatter(fin[o], y, s=34, color=t[c], alpha=0.75, edgecolor=t["bg"], lw=0.8, zorder=3)
        a2.plot([fin[o].mean()] * 2, [i - 0.3, i + 0.3], color=t["fg"], lw=2.2, zorder=4)
        a2.text(fin[o].mean(), i + 0.36, f"mean {fin[o].mean():.2f}", ha="center", va="bottom", fontsize=9.5,
                color=t["fg"], fontweight="bold" if o == "bo" else "normal")
    a2.set_yticks(range(len(OPT)))
    a2.set_yticklabels([n for _, n, _ in OPT])
    a2.set_ylim(-0.6, len(OPT) - 0.15)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("Best objective after 60 evaluations (lower is better)")
    a2.set_title("Final result of each seed")
    caption(fig, "Matched budget: 60 evaluations over the same 6 gains and bounds, seeds 0–9. BO vs PSO: "
                 "Holm-adjusted p ≤ 0.008 (Mann-Whitney and Wilcoxon); BO vs random search: p = 0.015 "
                 "(Mann-Whitney) and 0.084 (Wilcoxon).", t)
    fig.tight_layout(w_pad=3)
    save(fig, "optimizers", theme)


# --------------------------------------------------------------------------- #
# 3. Nominal driving: smoothness and accuracy (15 runs per controller)
# --------------------------------------------------------------------------- #
def _strip(ax, data, t, fmt, label_dx=0.0):
    for i, (m, name, c, mk) in enumerate(CTRL):
        v = data[m]
        y = np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v))
        ax.scatter(v, y, s=22, marker=mk, color=t[c], alpha=0.65, edgecolor="none", zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.32, i + 0.32], color=t["fg"], lw=2.2, zorder=4)
        ax.text(v.mean(), i + 0.38, fmt.format(v.mean()), ha="center", va="bottom", fontsize=9,
                color=t["fg"], fontweight="bold" if m == "BO_SMC" else "normal")
    ax.set_yticks(range(len(CTRL)))
    ax.set_ylim(-0.6, len(CTRL) - 0.1)
    ax.grid(axis="y", visible=False)


def fig_nominal(theme):
    t = THEMES[theme]
    style(t)
    cm = K.load("compare_results.json")["methods"]
    get = lambda k, s=1.0: {m: s * K.arr(cm[m], k) for m, *_ in CTRL}
    panels = [("ssi", 1.0, "Steering smoothness index (deg/step)", "{:.2f}"),
              ("sat_ratio", 100.0, "Steps at the ±20° steering limit (%)", "{:.1f}"),
              ("lane_rmse_px", 1.0, "Lateral error, RMSE (px)", "{:.1f}")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1), sharey=True)
    for a, (k, s, title, fmt) in zip(ax, panels):
        _strip(a, get(k, s), t, fmt)
        a.set_title(title, fontsize=10.5)
        a.set_xlim(left=0)
    ax[0].set_yticklabels([n for _, n, *_ in CTRL])
    for lab in ax[0].get_yticklabels():
        if lab.get_text().startswith("Proposed"):
            lab.set_fontweight("bold")
            lab.set_color(t["fg"])
    fig.suptitle("Nominal driving, 15 runs per controller (dots = runs, bar = mean; lower is better in every panel)",
                 x=0.0, ha="left", fontsize=12, fontweight="bold", color=t["fg"])
    caption(fig, "Proposed BO-SMC vs classical PID: SSI and saturation reduced (Holm-adjusted p = 0.001 each), "
                 "lateral error not significantly different. Vs grid-search SMC: smoother and more accurate. "
                 "No significant difference from a PID tuned by the same BO procedure.", t)
    fig.tight_layout(w_pad=1.5)
    save(fig, "nominal", theme)


def fig_steering(theme):
    t = THEMES[theme]
    style(t)
    win = (10.0, 20.0)
    fig, ax = plt.subplots(2, 1, figsize=(13, 4.6), sharex=True)
    for a, (m, name, c) in zip(ax, [("PID", "Classical PID", "orange"), ("BO_SMC", "Proposed BO-SMC", "blue")]):
        lg = K.load_log(f"compare_{m}_seed0.json")
        tt, d = np.array(lg["t"]), np.array(lg["angle"])
        ssi = np.mean(np.abs(np.diff(d)))
        sel = (tt >= win[0]) & (tt <= win[1])
        for y in (-20, 20):
            a.axhline(y, color=t["axis"], lw=0.9)
        a.plot(tt[sel], d[sel], color=t[c], lw=1.0)
        a.set_ylim(-25, 25)
        a.set_yticks([-20, 0, 20])
        a.set_ylabel("Steering (deg)")
        a.text(0.0, 1.04, f"{name}", transform=a.transAxes, fontsize=11.5, fontweight="bold", color=t["fg"])
        a.text(1.0, 1.04, f"SSI {ssi:.2f} deg/step over the whole run", transform=a.transAxes, ha="right",
               fontsize=10, color=t["fg2"])
    ax[1].set_xlabel("Time (s)")
    caption(fig, "Steering command of the same nominal run index (10-s window of a 45-s run; gray lines = steering "
                 "limit ±20°). The classical PID chatters between the limits; the tuned boundary-layer SMC steers "
                 "smoothly.", t)
    fig.tight_layout(h_pad=1.6)
    save(fig, "steering", theme)


def fig_tradeoff(theme):
    t = THEMES[theme]
    style(t)
    cm = K.load("compare_results.json")["methods"]
    fig, a = plt.subplots(figsize=(8.4, 5.2))
    for m, name, c, mk in CTRL:
        x, y = K.arr(cm[m], "lane_rmse_px"), K.arr(cm[m], "ssi")
        a.scatter(x, y, s=26, marker=mk, color=t[c], alpha=0.45, edgecolor="none")
        a.scatter([x.mean()], [y.mean()], s=150, marker=mk, color=t[c], edgecolor="none", zorder=5)
        lx, ly = {"PID": (41.0, 12.0), "TunedPID": (26.0, 1.75), "ManualSMC": (47.0, 0.40),
                  "GridSMC": (44.0, 3.6), "BO_SMC": (39.0, 1.35)}[m]
        a.annotate(name, (x.mean(), y.mean()), xytext=(lx, ly), fontsize=10, color=t["fg"], va="center",
                   fontweight="bold" if m == "BO_SMC" else "normal",
                   arrowprops=dict(arrowstyle="-", color=t["axis"], lw=0.9, shrinkA=2, shrinkB=7))
    a.set_yscale("log")
    a.set_yticks([0.5, 1, 2, 5, 10])
    a.set_yticklabels(["0.5", "1", "2", "5", "10"])
    a.minorticks_off()
    a.set_xlabel("Lateral error, RMSE (px)  →  less accurate")
    a.set_ylabel("Steering smoothness index (deg/step, log)  →  rougher")
    a.set_title("Accuracy–smoothness trade-off (small = runs, large = mean)")
    a.annotate("better (lower-left)", xy=(23.5, 4.2), xytext=(28.5, 7.0), fontsize=9.5, color=t["fg2"],
               ha="left", va="center", arrowprops=dict(arrowstyle="->", color=t["fg2"], lw=1.1))
    a.set_ylim(0.36, 15)
    fig.tight_layout()
    save(fig, "tradeoff", theme)


# --------------------------------------------------------------------------- #
# 4. Robustness under degraded vision (10 runs per cell, stalled episodes excluded)
# --------------------------------------------------------------------------- #
def fig_robustness(theme):
    t = THEMES[theme]
    style(t)
    rb = K.load("robustness_results.json")["data"]
    conds = [("clean", "Nominal"), ("noise", "Noise"), ("shadow", "Shadow"), ("occlusion", "Occlusion"),
             ("blur", "Blur")]
    ctrls = [c for c in CTRL if c[0] != "GridSMC"]
    ok = lambda m, c: [r for r in rb[m][c] if r["detection_rate"] >= K.STALL_DETECTION]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    off = np.linspace(-0.27, 0.27, len(ctrls))
    for j, (m, name, c, mk) in enumerate(ctrls):
        v = [np.mean([r["chatter_tv"] for r in ok(m, k)]) for k, _ in conds]
        a1.scatter(v, np.arange(len(conds)) + off[j], s=70, marker=mk, color=t[c], edgecolor=t["bg"], lw=1.2,
                   zorder=4, label=name)
    a1.set_xscale("log")
    a1.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000])
    a1.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k"])
    a1.minorticks_off()
    a1.set_yticks(range(len(conds)))
    a1.set_yticklabels([n for _, n in conds])
    a1.invert_yaxis()
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("Steering variation per second (deg/s, log scale; lower = smoother)")
    a1.set_title("Steering activity under visual perturbations (mean of 10 runs)", pad=30)
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9.5, handletextpad=0.3,
              columnspacing=1.4, borderaxespad=0.3)

    for i, (m, name, c, mk) in enumerate(ctrls):
        v = K.arr(ok(m, "occlusion"), "lane_rmse_px")
        y = np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v))
        a2.scatter(v, y, s=26, marker=mk, color=t[c], alpha=0.7, edgecolor="none", zorder=3)
        a2.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=t["fg"], lw=2.2, zorder=4)
        a2.text(v.mean(), i + 0.36, f"{v.mean():.1f} px", ha="center", va="bottom", fontsize=9.5, color=t["fg"],
                fontweight="bold" if m == "BO_SMC" else "normal")
    a2.set_yticks(range(len(ctrls)))
    a2.set_yticklabels([n for _, n, *_ in ctrls])
    a2.set_ylim(-0.6, len(ctrls) - 0.1)
    a2.set_xlim(left=0)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("Lateral error, RMSE (px; lower is better)")
    a2.set_title("Under occlusion")
    caption(fig, "Under occlusion the proposed controller halves the lateral error of the classical PID with about "
                 "6× less steering activity; it performs on par with the tuned PID (no significant difference).", t)
    fig.tight_layout(w_pad=3)
    save(fig, "robustness", theme)


if __name__ == "__main__":
    for th in THEMES:
        for f in (fig_perception, fig_perturbations, fig_optimizers, fig_nominal, fig_steering, fig_tradeoff,
                  fig_robustness):
            f(th)
            print(f"{f.__name__:20s} {th}")
