"""Jupyter wrapper kernel for posim.

Each notebook cell is sent to the Rust simulator as one or more
``{"op": "exec", "code": ...}`` requests (one per non-empty line, so a
cell can hold a whole command script); replies stream back as
``execute_result`` / ``error`` messages. Because it is a real Jupyter
kernel, JupyterLab's own notebook UI supplies the shift-enter
execution, cell history, and cell editing the simulator spec asks for.
"""

import json
import os
import queue
import subprocess
import threading

from ipykernel.kernelbase import Kernel

from . import __version__


def find_posim():
    """Locate the posim binary: $POSIM_BIN, PATH, or the workspace target dir."""
    env = os.environ.get("POSIM_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    from shutil import which

    on_path = which("posim")
    if on_path:
        return on_path
    here = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(os.path.dirname(here))
    for profile in ("release", "debug"):
        cand = os.path.join(workspace, "target", profile, "posim")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "posim binary not found: set $POSIM_BIN or run "
        "`cargo build --release` at the workspace root"
    )


class PosimKernel(Kernel):
    implementation = "posim"
    implementation_version = __version__
    language = "posim"
    language_version = "0.1"
    language_info = {
        "name": "posim",
        "mimetype": "text/plain",
        "file_extension": ".posim",
    }
    banner = "posim — physical_object simulator (pure-Rust sundials_rs backend)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._proc = subprocess.Popen(
            [find_posim(), "--machine"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Replies to requests land on this queue; asynchronous
        # {"event": ...} lines (pushed by the graphical scene window at
        # any time) are streamed straight to the notebook front end.
        self._replies = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        """Routes backend stdout: replies vs. unsolicited scene events."""
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict) and "event" in msg:
                self._push_event(msg)
            else:
                self._replies.put(msg)
        self._replies.put(None)  # backend exited

    def _push_event(self, msg):
        """Async scene-window -> notebook message (error, request, ...)."""
        text = msg.get("message", "")
        stream = "stderr" if text.startswith("error") else "stdout"
        try:
            self.send_response(
                self.iopub_socket,
                "stream",
                {"name": stream, "text": "[scene] " + text + "\n"},
            )
        except Exception:
            pass  # front end not ready; the event is still in posim's log

    def _request(self, obj, timeout=120.0):
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()
        reply = self._replies.get(timeout=timeout)
        if reply is None:
            raise RuntimeError("posim backend exited")
        return reply

    @staticmethod
    def _brace_delta(line):
        """Net {/} depth of a line, ignoring braces inside string
        literals and after # comments — mirrors the posim notebook's
        continuation logic so a multi-line DEF travels as ONE exec."""
        depth = 0
        in_str = False
        escape = False
        for c in line:
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == "#":
                break
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
        return depth

    @classmethod
    def _join_cells(cls, code):
        """Splits a Jupyter cell into posim commands, joining
        brace-continuation lines (DEF bodies) into single commands."""
        cells = []
        pending = ""
        depth = 0
        for raw in code.splitlines():
            line = raw.rstrip()
            if depth == 0:
                t = line.strip()
                if not t or t.startswith("#"):
                    continue
                pending = t
                depth = cls._brace_delta(t)
            else:
                depth += cls._brace_delta(line)
                pending += "\n" + line
            if depth <= 0:
                cells.append(pending)
                pending = ""
                depth = 0
        if pending:
            cells.append(pending)  # unbalanced input: let posim report it
        return cells

    def do_execute(
        self, code, silent, store_history=True, user_expressions=None, allow_stdin=False
    ):
        status = "ok"
        for line in self._join_cells(code):
            try:
                reply = self._request({"op": "exec", "code": line})
            except Exception as exc:  # backend died
                reply = {"ok": False, "error": str(exc)}
            if reply.get("ok"):
                display = reply.get("display") or ""
                if display and not silent:
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": self.execution_count,
                            "data": {"text/plain": display},
                            "metadata": {},
                        },
                    )
            else:
                status = "error"
                err = reply.get("error", "unknown error")
                if not silent:
                    self.send_response(
                        self.iopub_socket,
                        "stream",
                        {"name": "stderr", "text": err + "\n"},
                    )
                break
        return {
            "status": status,
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def do_shutdown(self, restart):
        try:
            self._request({"op": "quit"})
        except Exception:
            pass
        self._proc.terminate()
        return {"status": "ok", "restart": restart}


if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance(kernel_class=PosimKernel)
