"""
run.py — single CLI entry point for all journal_cmc experiments.

Usage (run from inside the journal_cmc/ folder):

    python run.py gpucheck
    python run.py demo        --method ManualSMC --show
    python run.py reference
    python run.py optimize    --which all         # or bo / pso / grid / random
    python run.py compare
    python run.py robustness
    python run.py ablation
    python run.py analyze                          # build tables + figures
    python run.py all                              # full pipeline end-to-end

Add --quick to any sim command for a fast smoke test (short episodes, few seeds).
Every sim command needs the Unity Map_demo_v3.exe (or mock_server.py) running.
See README_RUN.md for the full guide and expected outputs.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402


def _connect(args):
    from simulator import SimClient
    client = SimClient(host=args.host, port=args.port).connect()
    print(f"[run] connected to sim at {args.host}:{args.port}")
    return client


def _prof(args):
    """Resolve the run profile: explicit --profile wins, else --quick, else full."""
    if getattr(args, "profile", None):
        return args.profile
    return "quick" if args.quick else "full"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_gpucheck(args):
    try:
        import torch
        print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU  :", torch.cuda.get_device_name(0))
            print("mem  : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    except Exception as e:
        print("torch not available:", e)
    for mod in ("botorch", "gpytorch", "skopt", "pyswarms"):
        try:
            m = __import__(mod)
            print(f"{mod:9s}: OK {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{mod:9s}: MISSING ({type(e).__name__})")
    from optimizers import _resolve_backend
    print("BO backend that will be used:", _resolve_backend("auto"))


def cmd_demo(args):
    import controller as ctrl
    import metrics as M
    from simulator import run_episode
    t_settle, t_measure, _ = C.profile(_prof(args))
    client = _connect(args)
    try:
        if args.method in ("PID", "ManualSMC"):
            con = ctrl.build_controller(args.method)
        else:
            from experiments import controller_for
            con = controller_for(args.method)
        print(f"[demo] {args.method} | perturb={args.perturb} | "
              f"settle={t_settle}s measure={t_measure}s")
        log = run_episode(client, con, t_settle, t_measure,
                          perturb=args.perturb, seed=args.seed, show=args.show)
        met = M.compute_metrics(log)
        print("\n--- metrics ---")
        for k in ("lane_rmse_px", "lane_dev_pct", "ssi", "steer_jerk",
                  "chattering_index", "speed_rmse", "recovery_events",
                  "recovery_time", "sat_ratio", "lyap_violation_rate",
                  "lyap_penalty", "detection_rate", "fps", "n_frames"):
            print(f"  {k:20s} = {met.get(k)}")
    finally:
        client.close()


def cmd_reference(args):
    from experiments import set_reference
    client = _connect(args)
    try:
        set_reference(client, profile=_prof(args))
    finally:
        client.close()


def cmd_optimize(args):
    from experiments import run_optimize, run_all_opt
    client = _connect(args)
    try:
        if args.which == "all":
            run_all_opt(client, profile=_prof(args), seed=args.seed)
        else:
            run_optimize(client, args.which, profile=_prof(args), seed=args.seed,
                         save_logs=args.save_logs)
    finally:
        client.close()


def cmd_compare(args):
    from experiments import run_compare
    client = _connect(args)
    try:
        run_compare(client, profile=_prof(args), seed=args.seed, resume=not args.fresh)
    finally:
        client.close()


def cmd_robustness(args):
    from experiments import run_robustness
    client = _connect(args)
    try:
        run_robustness(client, profile=_prof(args), seed=args.seed, resume=not args.fresh)
    finally:
        client.close()


def cmd_ablation(args):
    from experiments import run_ablation
    client = _connect(args)
    try:
        run_ablation(client, profile=_prof(args), seed=args.seed, resume=not args.fresh)
    finally:
        client.close()


def cmd_faircompare(args):
    from fair_compare import run_fair_compare
    client = _connect(args)
    try:
        run_fair_compare(client, profile=_prof(args), n_evals=args.evals,
                         n_seeds=args.seeds, with_grid=args.with_grid,
                         resume=not args.fresh)
    finally:
        client.close()


def cmd_analyze(args):
    import analysis
    analysis.make_all()


def cmd_all(args):
    """Full pipeline in dependency order. LONG in --full profile (see README)."""
    from experiments import (set_reference, run_all_opt, run_compare,
                             run_robustness, run_ablation)
    import analysis
    prof = _prof(args)
    client = _connect(args)
    t0 = time.time()
    try:
        print(f"\n===== [1/6] reference (Manual-SMC scales) | profile={prof} =====")
        set_reference(client, profile=prof)
        print("\n===== [2/6] optimize (random/grid/pso/bo/pid) =====")
        run_all_opt(client, profile=prof, seed=args.seed)
        res = not args.fresh
        print("\n===== [3/6] compare (6 methods) =====")
        run_compare(client, profile=prof, seed=args.seed, resume=res)
        print("\n===== [4/6] robustness =====")
        run_robustness(client, profile=prof, seed=args.seed, resume=res)
        print("\n===== [5/6] ablation =====")
        run_ablation(client, profile=prof, seed=args.seed, resume=res)
    finally:
        client.close()
    print("\n===== [6/6] analyze (tables + figures) =====")
    analysis.make_all()
    print(f"\n[all] complete in {(time.time()-t0)/60:.1f} min. "
          f"See results/tables and results/figures.")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="journal_cmc experiment runner")
    p.add_argument("--host", default=C.HOST)
    p.add_argument("--port", type=int, default=C.PORT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", choices=["quick", "medium", "full"], default=None,
                   help="run size: quick (~25min smoke) / medium (~2-3h validation) / full (~11h paper)")
    p.add_argument("--quick", action="store_true", help="shortcut for --profile quick")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any checkpoint and recompute the phase from scratch "
                        "(default: resume, skipping episodes already saved)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gpucheck").set_defaults(func=cmd_gpucheck)

    d = sub.add_parser("demo")
    d.add_argument("--method", default="ManualSMC",
                   choices=["PID", "TunedPID", "ManualSMC", "GridSMC", "PSO_SMC", "BO_SMC"])
    d.add_argument("--perturb", default="clean", choices=C.PERTURBATIONS)
    d.add_argument("--show", action="store_true", help="live OpenCV visualization")
    d.set_defaults(func=cmd_demo)

    sub.add_parser("reference").set_defaults(func=cmd_reference)

    o = sub.add_parser("optimize")
    o.add_argument("--which", default="all",
                   choices=["all", "bo", "pso", "grid", "random", "pid"])
    o.add_argument("--save-logs", action="store_true")
    o.set_defaults(func=cmd_optimize)

    sub.add_parser("compare").set_defaults(func=cmd_compare)
    sub.add_parser("robustness").set_defaults(func=cmd_robustness)
    sub.add_parser("ablation").set_defaults(func=cmd_ablation)

    fc = sub.add_parser("faircompare",
                        help="reviewer fair optimizer comparison: equal 6-D space, "
                             "equal budget, >=10 seeds for random/pso/bo")
    fc.add_argument("--evals", type=int, default=60,
                    help="evaluations per optimizer run (same for all; default 60)")
    fc.add_argument("--seeds", type=int, default=10,
                    help="independent repeats per optimizer (default 10)")
    fc.add_argument("--with-grid", action="store_true",
                    help="also run the deterministic 3-D grid reference once")
    fc.set_defaults(func=cmd_faircompare)

    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("all").set_defaults(func=cmd_all)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
