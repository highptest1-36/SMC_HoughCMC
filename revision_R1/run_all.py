"""
run_all.py — regenerate every number, table and figure needed for the Round-1 revision
of CMC manuscript ID 92446, using ONLY the data already in journal_cmc/results/.

    python revision_R1/run_all.py

No simulator is needed and no existing file is modified. Outputs:
    revision_R1/output/REPORT.md      all numbers, organised by section (read this first)
    revision_R1/output/tables/*.csv   tables for the manuscript / response letter
    revision_R1/output/tables/*.tex   LaTeX rows ready to paste
    revision_R1/output/figures/*.png  revised figures (400 dpi)
    revision_R1/output/run_log.txt    console log incl. any error
"""
import datetime
import os
import platform
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))

import common as K                                                          # noqa: E402
import s1_optimizer, s2_gains_regime, s3_nominal, s4_robustness, s5_ablation, s6_fig3, s7_text_stats  # noqa: E402

STEPS = [("S1 optimizer study (Table 2, Fig 2)", s1_optimizer.run),
         ("S2 tuning runs, gains, control law, regime, sensitivity", s2_gains_regime.run),
         ("S3 nominal comparison, all 6 controllers (record)", lambda: s3_nominal.run()),
         ("S3 nominal comparison, paper version without PSO-SMC", lambda: s3_nominal.run(exclude=("PSO_SMC",), tag="paper")),
         ("S4 robustness (Table 5)", s4_robustness.run),
         ("S5 ablation (Table 6)", s5_ablation.run),
         ("S6 revised Figure 3", s6_fig3.run),
         ("S7 statistics quoted in the text", s7_text_stats.run)]


def main():
    import numpy, scipy, matplotlib
    head = [f"# Revision R1 — analysis report (CMC ID 92446)",
            f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} · Python {platform.python_version()} · "
            f"numpy {numpy.__version__} · scipy {scipy.__version__} · matplotlib {matplotlib.__version__}",
            f"Data folder: {K.RESULTS}", ""]
    parts, log, ok = [], [], 0
    for name, fn in STEPS:
        print(f"[run_all] {name} ...", flush=True)
        try:
            parts.append(fn())
            log.append(f"OK    {name}")
            ok += 1
        except Exception:
            tb = traceback.format_exc()
            parts.append(f"## {name}\n\n**FAILED**\n\n```\n{tb}\n```")
            log.append(f"FAIL  {name}\n{tb}")
            print(tb)
    report = "\n".join(head) + "\n\n" + "\n\n---\n\n".join(parts) + "\n"
    K.write_text("REPORT.md", report, folder=K.OUT)
    K.write_text("run_log.txt", "\n".join(log) + "\n", folder=K.OUT)
    print(f"[run_all] {ok}/{len(STEPS)} steps OK")
    print(f"[run_all] report : {os.path.join(K.OUT, 'REPORT.md')}")
    print(f"[run_all] tables : {K.OUT_TAB}")
    print(f"[run_all] figures: {K.OUT_FIG}")
    return 0 if ok == len(STEPS) else 1


if __name__ == "__main__":
    sys.exit(main())
