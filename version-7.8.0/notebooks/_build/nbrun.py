#!/usr/bin/env python3
"""Execute a generated notebook's code cells and write the outputs back.

This is a miniature of what a Jupyter kernel does: run each code cell in
one shared namespace, in order, capturing whatever it prints. Cells
tagged "interactive" are skipped, because they wait for a human to type
a name and pick a folder in a dialog.

Exit status is 0 only if every non-interactive cell ran without raising.
"""
import contextlib, io, json, sys, traceback
from pathlib import Path

def stream(text):
    return {"output_type": "stream", "name": "stdout",
            "text": text.splitlines(keepends=True)}

def run(path, write=True):
    nb = json.loads(Path(path).read_text())
    ns = {"__name__": "__main__"}
    count = 0
    ok = True
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "interactive" in cell.get("metadata", {}).get("tags", []):
            cell["outputs"] = []
            cell["execution_count"] = None
            continue
        src = "".join(cell["source"])
        count += 1
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(src, f"{path}:cell{count}", "exec"), ns)
        except BaseException:
            ok = False
            print(f"FAIL {path} cell {count}\n{'-'*60}\n{src}\n{'-'*60}", file=sys.stderr)
            print(buf.getvalue(), file=sys.stderr)
            traceback.print_exc()
            break
        cell["execution_count"] = count
        text = buf.getvalue()
        cell["outputs"] = [stream(text)] if text else []
    # never leave a simulator process behind
    sim = ns.get("sim")
    if sim is not None:
        try: sim.close()
        except Exception: pass
    if write and ok:
        Path(path).write_text(json.dumps(nb, indent=1) + "\n")
    return ok, count

if __name__ == "__main__":
    good, bad = [], []
    for p in sys.argv[1:]:
        ok, n = run(p)
        (good if ok else bad).append(p)
        print(f"{'ok  ' if ok else 'FAIL'} {p}  ({n} cells)")
    print(f"\n{len(good)} ok, {len(bad)} failed")
    sys.exit(1 if bad else 0)
