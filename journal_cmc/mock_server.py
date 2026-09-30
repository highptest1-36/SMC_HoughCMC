"""
mock_server.py — a stand-in for Map_demo_v3.exe for OFFLINE testing.

Speaks the same TCP/JSON protocol as the Unity build (request/response on
127.0.0.1:54321): it receives "angle speed", advances a tiny lane-following
vehicle model, renders a synthetic two-line yellow lane, and replies with
{"Speed","Angle","Img"(base64 jpg)}.

Purpose: smoke-test the whole journal_cmc pipeline (perception -> controller ->
metrics -> objective -> optimizers -> tables/figures) WITHOUT the Unity sim.
It is NOT a physics model and must NOT be used for paper numbers — those come
from the real Map_demo_v3.exe.

Run:  python mock_server.py     (then run run.py commands in another terminal)
"""
import base64
import json
import math
import socket

import cv2
import numpy as np

import config as C

W, H = 320, 240
CENTER = W / 2.0


def render(pos, half_width=30):
    """Dark road with two yellow lane markers centred at CENTER+pos."""
    img = np.full((H, W, 3), 40, np.uint8)                 # grey road
    xL = int(np.clip(CENTER + pos - half_width, 2, W - 2))
    xR = int(np.clip(CENTER + pos + half_width, 2, W - 2))
    y0, y1 = H // 2, H - 2
    cv2.line(img, (xL, y0), (xL + 4, y1), (0, 255, 255), 4)   # slight slant for Hough
    cv2.line(img, (xR, y0), (xR - 4, y1), (0, 255, 255), 4)
    return img


def serve_once(conn):
    pos = 0.0            # lane offset (px): mid_x - CENTER
    speed = 0.0          # km/h
    step = 0
    while True:
        raw = conn.recv(C.RECV_BUF)
        if not raw:
            return
        try:
            angle_cmd, speed_cmd = map(float, raw.decode(errors="ignore").split()[:2])
        except Exception:
            angle_cmd, speed_cmd = 0.0, 0.0

        # --- tiny closed-loop-able dynamics ---
        t = step * 0.03
        road = 3.0 * math.sin(2 * math.pi * 0.05 * t)        # gentle curve disturbance
        pos = float(np.clip(pos + road - 0.15 * angle_cmd, -W / 2, W / 2))
        speed += 0.2 * (speed_cmd - speed)                   # first-order speed lag
        speed = float(np.clip(speed, 0, 200))

        img = render(pos)
        ok, buf = cv2.imencode(".jpg", img)
        payload = {
            "Speed": round(speed, 2),
            "Angle": round(angle_cmd, 2),
            "Img": base64.b64encode(buf).decode("utf-8"),
        }
        conn.sendall(json.dumps(payload).encode("utf-8"))
        step += 1


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((C.HOST, C.PORT))
    srv.listen(1)
    print(f"[mock_server] listening on {C.HOST}:{C.PORT} (Ctrl+C to stop)")
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[mock_server] client connected: {addr}")
            try:
                serve_once(conn)
            except (ConnectionResetError, OSError) as e:
                print(f"[mock_server] client dropped: {e}")
            finally:
                conn.close()
                print("[mock_server] waiting for next client ...")
    except KeyboardInterrupt:
        print("\n[mock_server] stopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
