"""
optimizers.py — the four tuning strategies, all minimising an Evaluator (theta -> J).

  bayes_opt   Bayesian Optimization.  backend='botorch' runs the GP surrogate on
              the GPU (RTX 3060); backend='skopt' is a CPU fallback; 'auto' picks
              botorch if CUDA+botorch are available, else skopt.
  pso         Particle Swarm Optimization (pyswarms).
  grid_search Coarse grid over the 3 steering params (speed params fixed to manual).
  random_search Uniform random sampling.

Each returns a result dict (best_theta/best_gains/best_J/history/wall_time/...).
The Evaluator records full per-eval metrics in `evaluator.history`; here we also
return a compact convergence history (running-best J) for Fig 2 / Table 6.
"""
import time

import numpy as np

import config as C
import controller as ctrl


def _finish(name, backend, evaluator, wall, device="cpu", extra=None):
    best = evaluator.best()
    hist = [{"call": h["call"], "J": h["J"]} for h in evaluator.history]
    running = []
    b = np.inf
    for h in hist:
        b = min(b, h["J"])
        running.append(b)
    res = {
        "name": name,
        "backend": backend,
        "device": device,
        "kind": getattr(evaluator, "kind", "smc"),
        "param_names": getattr(evaluator, "param_names", C.PARAM_NAMES),
        "n_dims": len(getattr(evaluator, "param_names", C.PARAM_NAMES)),
        "best_theta": best["theta"] if best else None,
        "best_gains": best["gains"] if best else None,
        "best_J": best["J"] if best else None,
        "best_metrics": best["metrics"] if best else None,
        "n_evals": evaluator.n_calls,
        "history": hist,
        "running_best": running,
        "wall_time": wall,
    }
    if extra:
        res.update(extra)
    return res


# --------------------------------------------------------------------------- #
# Random search
# --------------------------------------------------------------------------- #
def random_search(evaluator, n, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    low, high = np.array(evaluator.low), np.array(evaluator.high)
    for _ in range(n):
        theta = rng.uniform(low, high)
        evaluator(theta.tolist())
    return _finish("RandomSearch", "numpy", evaluator, time.time() - t0)


# --------------------------------------------------------------------------- #
# Grid search (coarse; steering params only, speed fixed to manual)
# --------------------------------------------------------------------------- #
def grid_search(evaluator, levels=3):
    # NOTE (disclosed in the paper): a full 6-D grid is combinatorially infeasible
    # (levels**6). Grid-SMC therefore tunes the 3 STEERING params and holds the speed
    # loop at the Manual-SMC values. This is recorded below and must be stated when
    # interpreting Table 3 (accuracy) and Table 6 (efficiency).
    searched = ["lambda_a", "eta_a", "phi_a"]
    t0 = time.time()
    idx = [C.PARAM_NAMES.index(n) for n in searched]
    grids = [np.linspace(C.BOUNDS_LOW[i], C.BOUNDS_HIGH[i], levels) for i in idx]
    fixed = ctrl.gains_to_vector(C.MANUAL_SMC_GAINS)  # speed params come from here
    import itertools
    for combo in itertools.product(*grids):
        theta = list(fixed)
        for j, i in enumerate(idx):
            theta[i] = float(combo[j])
        evaluator(theta)
    return _finish("GridSearch", "numpy", evaluator, time.time() - t0,
                   extra={"levels": levels, "dims_searched": searched,
                          "n_dims": len(searched),
                          "note": "steering loop only; speed params fixed to Manual-SMC"})


# --------------------------------------------------------------------------- #
# PSO (pyswarms)
# --------------------------------------------------------------------------- #
def pso(evaluator, n_particles=10, iters=8, seed=0):
    import pyswarms as ps
    t0 = time.time()
    low, high = np.array(evaluator.low), np.array(evaluator.high)
    dim = len(low)

    def cost(x):                      # x: (n_particles, dim)
        return np.array([evaluator(row.tolist()) for row in x])

    options = {"c1": 1.5, "c2": 1.5, "w": 0.7}
    np.random.seed(seed)              # pyswarms uses numpy global RNG for init
    opt = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim,
                                  options=options, bounds=(low, high))
    best_cost, best_pos = opt.optimize(cost, iters=iters, verbose=False)
    return _finish("PSO", "pyswarms", evaluator, time.time() - t0,
                   extra={"reported_best_cost": float(best_cost),
                          "reported_best_pos": [float(v) for v in best_pos]})


# --------------------------------------------------------------------------- #
# Bayesian Optimization
# --------------------------------------------------------------------------- #
def bayes_opt(evaluator, n_calls=60, n_init=12, seed=0, backend="auto"):
    backend = _resolve_backend(backend)
    if backend == "botorch":
        return _bo_botorch(evaluator, n_calls, n_init, seed)
    return _bo_skopt(evaluator, n_calls, n_init, seed)


def _resolve_backend(backend):
    if backend != "auto":
        return backend
    try:
        import torch
        import botorch  # noqa: F401
        if torch.cuda.is_available():
            return "botorch"
    except Exception:
        pass
    return "skopt"


def _bo_skopt(evaluator, n_calls, n_init, seed):
    from skopt import gp_minimize
    from skopt.space import Real
    t0 = time.time()
    dims = [Real(evaluator.low[i], evaluator.high[i], name=n)
            for i, n in enumerate(evaluator.param_names)]
    gp_minimize(func=lambda th: evaluator(list(th)), dimensions=dims,
                n_calls=n_calls, n_initial_points=n_init,
                acq_func="EI", random_state=seed)
    return _finish("BayesianOptimization", "skopt", evaluator, time.time() - t0,
                   device="cpu")


def _bo_botorch(evaluator, n_calls, n_init, seed):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import standardize
    from torch.quasirandom import SobolEngine

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    torch.manual_seed(seed)

    low = torch.tensor(evaluator.low, dtype=dtype, device=dev)
    high = torch.tensor(evaluator.high, dtype=dtype, device=dev)
    d = low.numel()
    unit_bounds = torch.stack([torch.zeros(d, dtype=dtype, device=dev),
                               torch.ones(d, dtype=dtype, device=dev)])

    def to_real(u):                      # unit cube -> real params
        return (low + (high - low) * u)

    def eval_unit(u):                    # returns +J (we minimise J => maximise -J)
        theta = to_real(u).detach().cpu().numpy().tolist()
        return evaluator(theta)

    # initial Sobol design
    sob = SobolEngine(dimension=d, scramble=True, seed=seed)
    X = sob.draw(n_init).to(dtype=dtype, device=dev)
    Y = torch.tensor([[-eval_unit(X[i])] for i in range(n_init)],
                     dtype=dtype, device=dev)   # maximise -J

    fail_y = -(C.FAILURE_OBJECTIVE * 0.9)      # Y (=-J) at/below this == a failed probe
    n_bo = max(0, n_calls - n_init)
    for _ in range(n_bo):
        # Fit the GP ONLY on successful evaluations: a failure sentinel would dominate
        # output standardisation and flatten the acquisition surface over the good region.
        keep = (Y.squeeze(-1) > fail_y)
        if int(keep.sum()) >= 2:
            trX, trY = X[keep], Y[keep]
        else:
            trX, trY = X, Y                    # too few successes yet; use all
        Ys = standardize(trY)
        gp = SingleTaskGP(trX, Ys)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acqf = LogExpectedImprovement(gp, best_f=Ys.max(), maximize=True)
        cand, _ = optimize_acqf(acqf, bounds=unit_bounds, q=1,
                                num_restarts=5, raw_samples=64)
        u_new = cand.detach().reshape(-1)
        y_new = -eval_unit(u_new)
        X = torch.cat([X, u_new.reshape(1, -1)], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_new]], dtype=dtype, device=dev)], dim=0)

    return _finish("BayesianOptimization", "botorch", evaluator, time.time() - t0,
                   device=str(dev))


OPTIMIZERS = {
    "random": random_search,
    "grid": grid_search,
    "pso": pso,
    "bo": bayes_opt,
}
