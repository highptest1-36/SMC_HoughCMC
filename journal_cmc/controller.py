"""
controller.py — controllers for lane keeping.

  * SlidingModeController : boundary-layer SMC (journal proposal) with a
    selectable switching function so the SAME class serves the Manual-SMC
    baseline AND the ablation variants.
  * PIDController         : the conference PID baseline (2106_PID_hough.py).

Every controller exposes the same interface:
    reset()
    angle(error_px)  -> (u_deg, s, ds)     # s, ds only meaningful for SMC
    speed(error_kmh) -> (u_cmd, s, ds)
so the harness and metrics treat them uniformly. `s` is the sliding-surface
value and `ds` its per-step change (used for the Lyapunov-inspired penalty).
"""
import numpy as np

import config as C


# --------------------------------------------------------------------------- #
# Sliding Mode Controller (boundary-layer)
# --------------------------------------------------------------------------- #
class SlidingModeController:
    """
    Discrete boundary-layer SMC, continuous with the conference control law:

        s = lambda * e + de           (de = e - e_prev)
        u = -(lambda * e + eta * switch(s))          [steering]
        u =  (lambda * e + eta * switch(s))          [speed]

    switch(s):
        'sat'        -> clip(s/phi, -1, 1)   (boundary layer — the PROPOSED form)
        'sign'       -> sign(s)              (classic SMC — chattering; ablation)
        'conference' -> s/(|s|+eps)          (the original conference soft-sign;
                                              used for the Manual-SMC baseline)

    gains : dict with keys lambda_a, eta_a, phi_a, lambda_v, eta_v, phi_v
    """

    def __init__(self, gains, switch="sat"):
        self.g = dict(gains)
        self.switch = switch
        self.reset()

    def reset(self):
        self._pre_e_a = 0.0
        self._pre_e_v = 0.0
        self._pre_s_a = 0.0
        self._pre_s_v = 0.0

    def _switch(self, s, phi):
        if self.switch == "sign":
            return np.sign(s)
        if self.switch == "conference":
            return s / (abs(s) + C.EPSILON)
        # default 'sat' — boundary layer
        phi = max(float(phi), 1e-6)
        return float(np.clip(s / phi, -1.0, 1.0))

    def angle(self, error, dt=None):          # dt ignored (SMC is dt-free here)
        de = error - self._pre_e_a
        self._pre_e_a = error
        s = self.g["lambda_a"] * error + de
        u = -(self.g["lambda_a"] * error + self.g["eta_a"] * self._switch(s, self.g["phi_a"]))
        u = float(np.clip(u, -C.STEER_CLIP, C.STEER_CLIP))
        ds = s - self._pre_s_a
        self._pre_s_a = s
        return u, s, ds

    def speed(self, error, dt=None):
        de = error - self._pre_e_v
        self._pre_e_v = error
        s = self.g["lambda_v"] * error + de
        u = self.g["lambda_v"] * error + self.g["eta_v"] * self._switch(s, self.g["phi_v"])
        u = float(np.clip(u, C.SPEED_CLIP[0], C.SPEED_CLIP[1]))
        ds = s - self._pre_s_v
        self._pre_s_v = s
        return u, s, ds


# --------------------------------------------------------------------------- #
# PID controller (conference baseline)
# --------------------------------------------------------------------------- #
class PIDController:
    """
    Conference PID (2106_PID_hough.py). Uses per-call dt from the harness clock
    rather than time.time() so it is deterministic w.r.t. the episode loop.
    Returns (u, 0.0, 0.0) for s/ds (not applicable to PID).
    """

    def __init__(self, gains=None, angle_hist=10, speed_hist=5):
        self.k = dict(gains or C.PID_GAINS)
        self._na = angle_hist
        self._ns = speed_hist
        self.reset()

    def reset(self):
        self._err_a = np.zeros(self._na)
        self._err_v = np.zeros(self._ns)

    def angle(self, error, dt):
        self._err_a[1:] = self._err_a[0:-1]
        self._err_a[0] = error
        P = self.k["Kp_angle"] * error
        I = self.k["Ki_angle"] * np.sum(self._err_a) * dt if dt > 0 else 0.0
        D = self.k["Kd_angle"] * (error - self._err_a[1]) / dt if dt > 0 else 0.0
        u = float(np.clip(-(P + I + D), -C.STEER_CLIP, C.STEER_CLIP))
        return u, 0.0, 0.0

    def speed(self, error, dt):
        self._err_v[1:] = self._err_v[0:-1]
        self._err_v[0] = error
        P = self.k["Kp_speed"] * error
        I = self.k["Ki_speed"] * np.sum(self._err_v) * dt if dt > 0 else 0.0
        D = self.k["Kd_speed"] * (error - self._err_v[1]) / dt if dt > 0 else 0.0
        u = float(np.clip(P + I + D, C.SPEED_CLIP[0], C.SPEED_CLIP[1]))
        return u, 0.0, 0.0


# --------------------------------------------------------------------------- #
# Factory: build a controller for a named method / variant
# --------------------------------------------------------------------------- #
def build_controller(method, gains=None, switch=None):
    """
    method:
        'PID'                 -> PIDController (conference gains)
        'ManualSMC'           -> SMC, conference switch + conference gains
        'SMC'/optimized names -> SMC with `gains` and `switch` (default 'sat')
    For ablation, pass switch in {'sat','sign'} and the optimized gains.
    """
    if method == "PID":
        return PIDController()
    if method == "ManualSMC":
        return SlidingModeController(C.MANUAL_SMC_GAINS, switch=C.MANUAL_SMC_SWITCH)
    # any optimized SMC method (GridSMC / PSO_SMC / BO_SMC / ablation variants)
    if gains is None:
        raise ValueError(f"method '{method}' needs gains")
    return SlidingModeController(gains, switch=(switch or "sat"))


def vector_to_gains(theta):
    """Map a 6-vector [lambda_a, eta_a, phi_a, lambda_v, eta_v, phi_v] -> dict."""
    return {name: float(v) for name, v in zip(C.PARAM_NAMES, theta)}


def gains_to_vector(gains):
    return [float(gains[name]) for name in C.PARAM_NAMES]


def pid_vector_to_gains(theta):
    """Map a 6-vector [Kp_a, Ki_a, Kd_a, Kp_v, Ki_v, Kd_v] -> PID gains dict."""
    return {name: float(v) for name, v in zip(C.PID_PARAM_NAMES, theta)}
