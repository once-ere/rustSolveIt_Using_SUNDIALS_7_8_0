#!/usr/bin/env python3
"""A live GUI for the gyroscope on a two-ring gimbal.

Standard library only. The server owns one `posim --machine` child — the
same pure-Rust engine, same model as videos/scenes/gyroscope_gimbal.posim
— and steps it with IDA. Nothing here integrates anything: the page only
draws the states the solver produces.

  GET  /            the page
  GET  /api/state   latest state as JSON
  POST /api/start   run
  POST /api/stop    pause
  POST /api/reset   fresh solver, fresh model, t = 0
  POST /api/dt/<x>  seconds of simulated time per frame
"""
import json, math, os, re, shutil, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # version-7.8.0/
PORT = 8897

MODEL = [
    "set system.g_constant = 0",
    "set system.uniform_gravity = [0, -3.0, 0]",
    "collide off",
    "new point as base { mass = 1, position = [0, 0, 0], inverse_mass = 0 }",
    "new torus as outer { mass = 0.5, ring_radius = 0.9, tube_radius = 0.04, position = [0, 0, 0] }",
    "new torus as inner { mass = 0.4, ring_radius = 0.7, tube_radius = 0.04, position = [0, 0, 0] }",
    "new cylinder as rotor { mass = 2, radius = 0.5, half_height = 0.06, position = [0, 0, 0] }",
    "set inner.orientation = [0.7071067811865476, 0.7071067811865476, 0, 0]",
    "hinge base outer [0, 1, 0]",
    "hinge outer inner [1, 0, 0]",
    "hinge inner rotor [0, 0, 1]",
    "method ida",
    "set outer.angular_velocity = [0, 1.0, 0]",
    "set inner.angular_velocity = [0, 1.0, 0]",
    "set rotor.angular_velocity = [0, 1.0, 15]",
]

I3 = 0.25          # rotor axial moment: cylinder, m r^2 / 2 = 2*0.25/2

def find_posim():
    env = os.environ.get("POSIM_BIN")
    if env and Path(env).is_file():
        return env
    cand = ROOT / "target" / "release" / "posim"
    if cand.is_file():
        return str(cand)
    onpath = shutil.which("posim")
    if onpath:
        return onpath
    raise SystemExit("build the engine first: cargo build --release -p posim")

def axial_spin(q, w):
    """I3 * (omega . rotor axis), the rotor axis being its local z in world."""
    qw, qx, qy, qz = q
    ax = 2 * (qx * qz + qy * qw)
    ay = 2 * (qy * qz - qx * qw)
    az = 1 - 2 * (qx * qx + qy * qy)
    return I3 * (w[0] * ax + w[1] * ay + w[2] * az)

class Sim:
    """The posim child and the latest state, shared across requests."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.dt = 0.01
        self.state = {}
        self._spawn()
        self._start_loop()

    def _start_loop(self):
        self.loop = threading.Thread(target=self._loop, daemon=True)
        self.loop.start()

    # ---- the child ----------------------------------------------------
    def _spawn(self):
        self.proc = subprocess.Popen(
            [find_posim(), "--machine"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
            env=dict(os.environ, POSIM_NO_BROWSER="1"),
        )
        for cmd in MODEL:
            self._exec(cmd)
        self.t = 0.0
        # the two exact invariants, captured at the true start
        L = self._angmom()
        self.L0y = L[1]
        self.spin0 = axial_spin(self._get("rotor.orientation"),
                                self._get("rotor.angular_velocity"))
        self._refresh()

    def _rpc(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("the engine closed the connection")
            reply = json.loads(line)
            if "event" in reply:
                continue
            return reply

    def _exec(self, code):
        r = self._rpc({"op": "exec", "code": code})
        if not r.get("ok"):
            raise RuntimeError(f"the engine refused {code!r}: {r.get('error')}")
        return r

    def _get(self, path):
        r = self._rpc({"op": "get", "path": path})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return r["result"]

    def _angmom(self):
        """Total angular momentum, from the engine's own ANGMOM command."""
        return json.loads(self._exec("angmom").get("display") or "[0,0,0]")

    # ---- state --------------------------------------------------------
    def _refresh(self):
        g = gd = None
        r = self._exec("constraints")
        m = re.search(r"worst \|g\| = ([^,]+), worst \|g_dot\| = (\S+)",
                      r.get("display") or "")
        if m:
            g, gd = float(m.group(1)), float(m.group(2))
        rq = self._get("rotor.orientation")
        rw = self._get("rotor.angular_velocity")
        # swapped atomically: /api/state reads it without the lock
        self.state = {
            "t": self.t,
            "outer_q": self._get("outer.orientation"),
            "inner_q": self._get("inner.orientation"),
            "rotor_q": rq,
            "L": self._angmom(),
            "L0y": self.L0y,
            "spin": axial_spin(rq, rw),
            "spin0": self.spin0,
            "running": self.running,
            "dt": self.dt,
            "g": g, "gdot": gd,
        }

    # ---- the stepping thread ------------------------------------------
    def _loop(self):
        try:
            self._loop_body()
        except Exception as exc:
            # the child died or hung and was killed: stop cleanly; reset()
            # will spawn a fresh child and a fresh thread.
            self.running = False
            print(f"stepping thread stopped: {exc}", flush=True)

    def _loop_body(self):
        while True:
            if self.running:
                t0 = time.monotonic()
                with self.lock:
                    self._exec(f"step {self.dt}")
                    self.t += self.dt
                    self._refresh()
                # aim for real time: one dt of simulation per dt of wall clock
                time.sleep(max(0.0, self.dt - (time.monotonic() - t0)))
            else:
                time.sleep(0.05)

    # ---- controls ------------------------------------------------------
    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        with self.lock:
            self._refresh()

    def reset(self):
        self.running = False
        # Kill first, before touching the lock: if the stepping thread is
        # deep in a solve, this ends it and unblocks its readline.
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        with self.lock:
            for p in (self.proc.stdin, self.proc.stdout):
                try: p.close()
                except Exception: pass
            self._spawn()
            if not self.loop.is_alive():
                self._start_loop()

    def set_dt(self, dt):
        self.dt = min(0.05, max(0.001, dt))

SIM = Sim()

class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            # self.state is swapped atomically by _refresh, so no lock:
            # a long IDA solve must never make the page unresponsive.
            self._send(dict(SIM.state, running=SIM.running, dt=SIM.dt))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/start":
            SIM.start(); self._send({"ok": True})
        elif self.path == "/api/stop":
            SIM.stop(); self._send({"ok": True})
        elif self.path == "/api/reset":
            SIM.reset(); self._send({"ok": True})
        elif self.path.startswith("/api/dt/"):
            try:
                SIM.set_dt(float(self.path.rsplit("/", 1)[1]))
                self._send({"ok": True, "dt": SIM.dt})
            except ValueError:
                self.send_error(400)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass                                   # keep the terminal quiet

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"gyroscope-gimbal GUI: http://127.0.0.1:{PORT}/", flush=True)
    srv.serve_forever()
