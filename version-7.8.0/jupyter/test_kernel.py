#!/usr/bin/env python3
"""End-to-end Jupyter kernel test: starts the posim wrapper kernel with
jupyter_client's KernelManager (the same machinery JupyterLab uses),
executes notebook cells over the ZMQ messaging protocol, and checks
the streamed results — verifying that JupyterLab can get and set the
simulator's data through this kernel.

Run with the venv python: `.venv/bin/python test_kernel.py`
"""

import os
import queue
import sys

from jupyter_client.manager import KernelManager

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cell(kc, code, timeout=60):
    """Executes one cell; returns (status, [result/stream texts])."""
    msg_id = kc.execute(code)
    outputs = []
    status = None
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            sys.exit(f"timeout waiting for kernel output of {code!r}")
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        t = msg["msg_type"]
        if t == "execute_result":
            outputs.append(msg["content"]["data"]["text/plain"])
        elif t == "stream":
            outputs.append(msg["content"]["text"].rstrip("\n"))
        elif t == "error":
            outputs.append("\n".join(msg["content"].get("traceback", [])))
        elif t == "status" and msg["content"]["execution_state"] == "idle":
            break
    reply = kc.get_shell_msg(timeout=timeout)
    status = reply["content"]["status"]
    return status, outputs


def install_test_kernelspec():
    """Writes a posim kernelspec into a local dir and points JUPYTER_PATH
    at it (no changes outside the repo's jupyter/ directory)."""
    import json

    spec_dir = os.path.join(HERE, ".kernels", "kernels", "posim")
    os.makedirs(spec_dir, exist_ok=True)
    spec = {
        "argv": [sys.executable, "-m", "posim_kernel", "-f", "{connection_file}"],
        "display_name": "posim (physical_object)",
        "language": "posim",
        "env": {"PYTHONPATH": HERE},
    }
    with open(os.path.join(spec_dir, "kernel.json"), "w") as f:
        json.dump(spec, f, indent=2)
    os.environ["JUPYTER_PATH"] = os.path.join(HERE, ".kernels")


def main():
    install_test_kernelspec()
    km = KernelManager(kernel_name="posim")
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)

    failures = 0

    def check(name, cond, extra=""):
        nonlocal failures
        print(("  ok   " if cond else "  FAIL ") + name + (" " + extra if extra and not cond else ""))
        if not cond:
            failures += 1

    print("posim Jupyter kernel end-to-end test (jupyter_client over ZMQ)")

    status, out = run_cell(kc, "new sphere { mass = 2, radius = 0.5 }")
    check("cell 1: NEW", status == "ok" and out == ["obj0"], f"{status} {out}")

    status, out = run_cell(
        kc,
        "set system.gravity = [0, -9.81, 0]\nset obj0.position = [0, 10, 0]\nstep 1",
    )
    check("cell 2: multi-line SET + STEP", status == "ok" and any("t = 1" in o for o in out), f"{status} {out}")

    status, out = run_cell(kc, "get obj0.position.y")
    check("cell 3: GET analytic y", status == "ok" and out and abs(float(out[0]) - 5.095) < 1e-8, f"{status} {out}")

    status, out = run_cell(kc, "get obj0.bogus_field")
    check("cell 4: error surfaces", status == "error" and any("unknown object field" in o for o in out), f"{status} {out}")

    status, out = run_cell(kc, "energy")
    check("cell 5: ENERGY observable", status == "ok" and len(out) == 1, f"{status} {out}")

    # A MULTI-LINE DEF in one Jupyter cell must travel as ONE exec
    # (the kernel joins brace-continuation lines like the notebook).
    status, out = run_cell(
        kc,
        'def probe(m = 4) {\n  new sphere as pk { mass = m }\n}\nprobe()',
    )
    check(
        "cell 6: multi-line DEF defines and calls",
        status == "ok" and any("obj" in o and "pk" in o for o in out),
        f"{status} {out}",
    )
    status, out = run_cell(kc, "get pk.mass")
    check("cell 7: named path after DEF call", status == "ok" and out == ["4"], f"{status} {out}")

    kc.stop_channels()
    km.shutdown_kernel(now=False)

    if failures:
        sys.exit(f"{failures} kernel check(s) failed")
    print("all kernel checks passed — JupyterLab can drive this kernel")


if __name__ == "__main__":
    main()
