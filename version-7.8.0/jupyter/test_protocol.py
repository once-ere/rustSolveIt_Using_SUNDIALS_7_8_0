#!/usr/bin/env python3
"""Protocol test for `posim --machine` — needs only the Python stdlib.

Verifies the exact request/reply flow the JupyterLab wrapper kernel
uses: exec / get / set / state round-trips against a live subprocess,
including a sundials integration step with an analytic check.
"""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)


def find_posim():
    env = os.environ.get("POSIM_BIN")
    if env:
        return env
    for profile in ("release", "debug"):
        cand = os.path.join(WORKSPACE, "target", profile, "posim")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    sys.exit("posim binary not found: cargo build first (or set $POSIM_BIN)")


class Posim:
    def __init__(self, binary):
        env = dict(os.environ, POSIM_NO_BROWSER="1")
        self.proc = subprocess.Popen(
            [binary, "--machine"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.events = []

    def request(self, **obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            assert line, "backend exited unexpectedly"
            msg = json.loads(line)
            # unsolicited scene events may interleave with replies
            if isinstance(msg, dict) and "event" in msg:
                self.events.append(msg)
                continue
            return msg

    def close(self):
        self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=10)


def approx(a, b, tol=1e-8):
    return abs(a - b) <= tol


def main():
    sim = Posim(find_posim())
    failures = 0

    def check(name, cond, extra=""):
        nonlocal failures
        if cond:
            print(f"  ok   {name}")
        else:
            failures += 1
            print(f"  FAIL {name} {extra}")

    print("posim --machine protocol test")

    r = sim.request(op="exec", code="new sphere { mass = 2, radius = 0.5, charge = -1.5 }")
    check("exec new", r.get("ok") and r.get("result") == "obj0", str(r))

    r = sim.request(op="set", path="obj0.velocity", value=[1, 0, -0.5])
    check("set velocity", r.get("ok"), str(r))

    r = sim.request(op="get", path="obj0.momentum")
    check("get momentum = m*v", r.get("ok") and r.get("result") == [2.0, 0.0, -1.0], str(r))

    r = sim.request(op="set", path="system.uniform_gravity", value=[0, -9.81, 0])
    check("set gravity", r.get("ok"), str(r))

    r = sim.request(op="exec", code="set obj0.position = [0, 10, 0]")
    check("exec set position", r.get("ok"), str(r))

    r = sim.request(op="exec", code="step 1")
    check("sundials step", r.get("ok"), str(r))

    r = sim.request(op="get", path="obj0.position.y")
    # y(1) = 10 + 0*1 - 9.81/2 = 5.095 (analytic parabola)
    check(
        "analytic y(1) = 5.095",
        r.get("ok") and approx(r.get("result"), 10.0 - 9.81 / 2.0),
        str(r),
    )

    r = sim.request(op="state")
    ok = (
        r.get("ok")
        and approx(r["result"]["time"], 1.0)
        and len(r["result"]["objects"]) == 1
        and approx(r["result"]["objects"][0]["mass"], 2.0)
    )
    check("state dump", ok, str(r)[:200])

    r = sim.request(op="get", path="obj9.mass")
    check("error path reported", not r.get("ok") and "no object" in r.get("error", ""), str(r))

    r = sim.request(op="exec", code="energy")
    check("energy observable", r.get("ok") and isinstance(r.get("result"), float), str(r))

    # quaternion after a spin: run a torque and confirm |q| stays 1
    sim.request(op="exec", code="set obj0.torque = [0.1, 0, 0]")
    sim.request(op="exec", code="run 2 steps 4")
    r = sim.request(op="get", path="obj0.orientation")
    q = r.get("result")
    check(
        "orientation stays unit",
        r.get("ok") and approx(math.sqrt(sum(x * x for x in q)), 1.0, 1e-9),
        str(r),
    )

    # ---- graphical scene commands over the same protocol ----
    r = sim.request(op="exec", code="scene status")
    check("scene needs create first", not r.get("ok") and "SCENE CREATE" in r.get("error", ""), str(r))

    r = sim.request(op="exec", code="scene create")
    check("scene create", r.get("ok") and "http://127.0.0.1:" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene set_time_step 0.005")
    check("scene set_time_step", r.get("ok") and "0.005" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene translate 1 2 3")
    check("scene translate", r.get("ok") and "[1, 2, 3]" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene rotate 30 -10")
    check("scene rotate", r.get("ok") and "yaw" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene zoom in")
    check("scene zoom", r.get("ok") and "distance" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene hide 0")
    check("scene hide", r.get("ok") and "1 object(s) hidden" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene show all")
    check("scene show all", r.get("ok") and "0 object(s) hidden" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene start")
    check("scene start", r.get("ok") and "running" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene pause")
    check("scene pause", r.get("ok") and "paused" in r.get("display", ""), str(r))

    r = sim.request(op="exec", code="scene status")
    check("scene status", r.get("ok") and "mode = paused" in r.get("display", ""), str(r))

    r = sim.request(op="events")
    check("events op", r.get("ok") and isinstance(r.get("result"), list), str(r))

    r = sim.request(op="exec", code="scene close")
    check("scene close", r.get("ok") and "scene closed" in r.get("display", ""), str(r))

    sim.close()
    if failures:
        sys.exit(f"{failures} protocol check(s) failed")
    print("all protocol checks passed")


if __name__ == "__main__":
    main()
