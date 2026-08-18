#!/usr/bin/env python3
"""A live GUI for the mixed-shape box rattle.

Standard library only. The server owns one `posim --machine` child — the
same pure-Rust engine, same model as videos/scenes/box_of_shapes.posim —
and steps it with CVODE Adams. Every bounce is a real rootfinding event
inside the solver; the page only draws the states the solver produces.

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
PORT = 8907

MODEL = [
    "set system.g_constant = 0",
    "box 4",
    "collide on",
    "new cylinder { mass = 2, radius = 0.25, height = 1.5, position = [-1, 0.3, 0], velocity = [2, 0.5, -1] }",
    "new disk { mass = 0.66666666666666663, radius = 1, position = [1, -0.8, 0.5], velocity = [-1, 1, 0.6] }",
    "new cuboid { mass = 1.6666666666666667, half_extents = [0.5, 0.5, 0.5], position = [0.2, 1.1, -0.9], velocity = [0.5, -2, 1] }",
    "method adams",
]
BODIES = ["obj6", "obj7", "obj8"]           # cylinder, disk, cuboid
WALLS  = ["obj0", "obj1", "obj2", "obj3", "obj4", "obj5"]
WALL_HOME = [[3,0,0], [-3,0,0], [0,3,0], [0,-3,0], [0,0,3], [0,0,-3]]

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
        self.E0 = self._energy()
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

    def _energy(self):
        return float(self._exec("energy").get("display") or "nan")

    def _impulses(self):
        """Collision count, from the engine's own COLLIDE ON report
        (idempotent when already armed)."""
        d = self._exec("collide on").get("display") or ""
        m = re.search(r"(\d+) impulse", d)
        return int(m.group(1)) if m else 0

    # ---- state --------------------------------------------------------
    def _refresh(self):
        # the walls must never move: their worst drift from home, exactly
        wall_drift = 0.0
        for name, home in zip(WALLS, WALL_HOME):
            p = self._get(f"{name}.position")
            wall_drift = max(wall_drift, math.dist(p, home))
        # swapped atomically: /api/state reads it without the lock
        self.state = {
            "t": self.t,
            "q": [self._get(f"{n}.orientation") for n in BODIES],
            "p": [self._get(f"{n}.position") for n in BODIES],
            "E": self._energy(),
            "E0": self.E0,
            "hits": self._impulses(),
            "wall_drift": wall_drift,
            "running": self.running,
            "dt": self.dt,
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
            # a long solve must never make the page unresponsive.
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
    print(f"box-of-shapes GUI: http://127.0.0.1:{PORT}/", flush=True)
    srv.serve_forever()
