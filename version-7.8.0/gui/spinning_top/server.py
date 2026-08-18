#!/usr/bin/env python3
"""A live GUI for the spinning top, precessing on its ball-jointed tip.

Standard library only. The server owns one `posim --machine` child — the
same pure-Rust engine, same model as videos/scenes/spinning_top.posim —
and steps it with IDA. Nothing here integrates anything: the page only
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
PORT = 8900

MODEL = [
    # The one scene that must run LOOSER than the orientation-joint floor:
    # a gyroscope at 20 rad/s stepped as per-frame cold restarts fails at
    # the floor tolerance, exactly as the scene header records.
    "set system.rtol = 1e-5",
    "set system.atol = 1e-7",
    "set system.g_constant = 0",
    "set system.uniform_gravity = [0, -3.0, 0]",
    "collide off",
    "new point as pivot { mass = 1, position = [0, 0, -0.6], inverse_mass = 0 }",
    "new cylinder as top { mass = 1, radius = 0.42, half_height = 0.06, position = [0, 0, 0.6], velocity = [0.6122448979591837, 0, 0], angular_velocity = [0, 1.0204081632653061, 20] }",
    "ball pivot top",
    "method ida",
]

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
        self.L0y = self._angmom()[1]
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

    def _angmom(self):
        return json.loads(self._exec("angmom").get("display") or "[0,0,0]")

    # ---- state --------------------------------------------------------
    def _refresh(self):
        g = gd = None
        r = self._exec("constraints")
        m = re.search(r"worst \|g\| = ([^,]+), worst \|g_dot\| = (\S+)",
                      r.get("display") or "")
        if m:
            g, gd = float(m.group(1)), float(m.group(2))
        # swapped atomically: /api/state reads it without the lock
        self.state = {
            "t": self.t,
            "top_q": self._get("top.orientation"),
            "top_p": self._get("top.position"),
            "top_w": self._get("top.angular_velocity"),
            "L": self._angmom(),
            "L0y": self.L0y,
            "E": self._energy(),
            "E0": self.E0,
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
    print(f"spinning-top GUI: http://127.0.0.1:{PORT}/", flush=True)
    srv.serve_forever()
