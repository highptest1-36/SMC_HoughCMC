"""
simulator.py — the Option-A harness.

SimClient keeps ONE persistent TCP connection to the Unity Map_demo_v3.exe
server (request/response protocol identical to the conference client.py).
run_episode() drives the car for a settle window (not logged) then a measurement
window (logged), applying an optional visual perturbation each frame.

Because the connection persists, many trials run back-to-back changing only the
controller/gains between them — no manual restart of the exe. The car keeps
driving on the looping track; the settle window absorbs the handover transient.
"""
import base64
import json
import socket
import time
from collections import defaultdict

import cv2
import numpy as np

import config as C
import perception as P


class SimServerError(RuntimeError):
    pass


class SimClient:
    def __init__(self, host=C.HOST, port=C.PORT, timeout=10.0,
                 reconnect_tries=40, reconnect_delay=0.5):
        self.host, self.port, self.timeout = host, port, timeout
        self.reconnect_tries = reconnect_tries
        self.reconnect_delay = reconnect_delay
        self.sock = None
        self.reconnects = 0          # counts auto-recoveries (server restarts)
        self.dead_streak = 0         # consecutive near-blank (detection~0) episodes

    def _open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))

    def connect(self):
        try:
            self._open()
        except OSError as e:
            raise SimServerError(
                f"Cannot connect to sim server at {self.host}:{self.port}. "
                f"Is Map_demo_v3.exe running? ({e})")
        return self

    def reconnect(self):
        """Re-open the socket after the sim dropped it. The Unity server closes the
        connection when it resets/restarts (e.g. the car went off-track and the
        'Score X/3' ran out -> Restart Server). Retries because the server needs a
        moment to start accepting again. Raises only if the sim is truly gone."""
        self.close()
        last = None
        for _ in range(self.reconnect_tries):
            try:
                self._open()
                self.reconnects += 1
                print(f"[sim] connection dropped -> reconnected (recovery #{self.reconnects})")
                return True
            except OSError as e:
                last = e
                time.sleep(self.reconnect_delay)
        raise SimServerError(
            f"Sim did not come back after {self.reconnect_tries} reconnect attempts "
            f"({last}). Is Map_demo_v3.exe still open?")

    def _raw_step(self, angle, speed):
        """One send + receive. Raises on any socket problem (caller handles retry)."""
        self.sock.sendall(bytes(f"{angle} {speed}", "utf-8"))
        buf = b""
        while True:
            chunk = self.sock.recv(C.RECV_BUF)   # a big base64 image can span segments
            if not chunk:
                raise ConnectionError("empty recv (server closed the connection)")
            buf += chunk
            try:
                return json.loads(buf)
            except json.JSONDecodeError:
                continue                          # partial message; keep reading

    def step(self, angle, speed):
        """Send (angle, speed), receive one state frame -> (speed, angle_state, bgr).
        AUTO-RECONNECTS once if the sim dropped the connection, so a car reset /
        server restart does NOT crash the run — the pipeline self-heals and keeps
        going unattended."""
        data = None
        for attempt in (0, 1):
            try:
                data = self._raw_step(angle, speed)
                break
            except (OSError, ConnectionError) as e:
                if attempt == 0:
                    self.reconnect()              # recover, then retry the step once
                    continue
                raise SimServerError(f"socket error after reconnect: {e}")
        speed_kmh = float(data["Speed"])
        angle_state = float(data.get("Angle", 0.0))
        jpg = base64.b64decode(data["Img"])
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SimServerError("failed to decode image from sim")
        return speed_kmh, angle_state, img

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()


def _drive_once(client, controller, last_angle, last_speed, perturb, rng,
                dt, want_log, draw=False):
    """One control step. Returns (row_or_None, angle_cmd, speed_cmd, viz_or_None)."""
    speed_kmh, _, image = client.step(last_angle, last_speed)
    if perturb and perturb != "clean":
        image = P.apply_perturbation(image, perturb, rng)
    lane_err, lane_w, detected, viz = P.detect_lane(image, draw=draw)

    angle, s_a, ds_a = controller.angle(lane_err, dt)
    speed_err = C.SPEED_TARGET - speed_kmh
    speed_cmd_f, s_v, ds_v = controller.speed(speed_err, dt)
    speed_cmd = int(speed_cmd_f)

    row = None
    if want_log:
        row = dict(lane_error=lane_err, lane_width=lane_w, detected=float(detected),
                   angle=angle, speed_cmd=speed_cmd, speed=speed_kmh,
                   speed_err=speed_err, s_angle=s_a, ds_angle=ds_a,
                   s_speed=s_v, ds_speed=ds_v)
    return row, angle, speed_cmd, viz


def run_episode(client, controller, t_settle, t_measure, perturb="clean",
                seed=0, show=False, on_frame=None):
    """
    Drive one episode and return a `log` dict of equal-length lists.

    client     : connected SimClient
    controller : a controller with .reset()/.angle(e,dt)/.speed(e,dt)
    t_settle   : warm-up seconds (driven, NOT logged)
    t_measure  : logged measurement seconds
    perturb    : one of config.PERTURBATIONS
    seed       : seeds the perturbation RNG (and is recorded)
    show       : if True, cv2.imshow a live visualization (manual runs)
    on_frame   : optional callback(row) each logged frame
    """
    controller.reset()
    rng = np.random.default_rng(seed)
    last_angle, last_speed = 0, 0
    prev_t = time.time()

    # ---- settle (not logged) ---- #
    # Add a seed-dependent random extra warm-up so different seeds start the logged
    # window at DIFFERENT track positions -> genuine replication (not just timing
    # replicates). Deterministic given the seed.
    extra = float(rng.uniform(0, C.SEED_JITTER_SETTLE))
    settle_total = t_settle + extra
    t0 = time.time()
    while time.time() - t0 < settle_total:
        now = time.time()
        dt = now - prev_t
        prev_t = now
        _, last_angle, last_speed, _ = _drive_once(
            client, controller, last_angle, last_speed, perturb, rng, dt, want_log=False)

    # ---- measurement (logged) ---- #
    log = defaultdict(list)
    t0 = time.time()
    while time.time() - t0 < t_measure:
        now = time.time()
        dt = now - prev_t
        prev_t = now
        row, last_angle, last_speed, viz = _drive_once(
            client, controller, last_angle, last_speed, perturb, rng, dt,
            want_log=True, draw=show)
        row["t"] = now - t0
        row["dt"] = dt
        for k, v in row.items():
            log[k].append(v)
        if on_frame:
            on_frame(row)
        if show and viz is not None:
            cv2.putText(viz, f"e={row['lane_error']:+.0f}px  steer={row['angle']:+.1f}"
                             f"  v={row['speed']:.0f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("journal_cmc live", viz)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if show:
        cv2.destroyAllWindows()

    # ---- sim-health guard ---- #
    # A working map detects the lane ~100%. If episodes come back near-blank, the car is
    # stuck off-track / the sim froze. First try to AUTO-RECOVER by forcing a reconnect
    # (which makes the Unity server restart and reset the car to start); if that keeps
    # failing, abort with a clear message so we don't waste hours on garbage data.
    det_series = log.get("detected", [])
    if det_series:
        det_rate = sum(det_series) / len(det_series)
        if det_rate < C.SIM_HEALTH_MIN_DETECTION:
            client.dead_streak += 1
            print(f"[sim] WARNING: episode detection_rate={det_rate:.2f} (near-blank) "
                  f"— dead streak {client.dead_streak}/{C.SIM_HEALTH_MAX_DEAD_EPISODES}")
            if client.dead_streak in C.SIM_HEALTH_RECOVER_AT:
                print("[sim] car appears stuck — forcing a reconnect to reset the sim ...")
                try:
                    client.reconnect()          # may trigger the server to restart & reset
                except Exception as e:
                    print(f"[sim] recovery reconnect failed: {e}")
            if client.dead_streak >= C.SIM_HEALTH_MAX_DEAD_EPISODES:
                raise SimServerError(
                    f"Sim returned near-blank frames for {client.dead_streak} consecutive "
                    f"episodes (lane detection ~0) and auto-recovery failed. The car is "
                    f"most likely stuck/flipped and the map did not reset. STOP, click "
                    f"'Restart Server' (or reopen Map_demo_v3.exe), and re-run this phase.")
        else:
            client.dead_streak = 0

    return dict(log)
