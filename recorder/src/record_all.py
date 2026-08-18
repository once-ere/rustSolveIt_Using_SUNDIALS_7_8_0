#!/usr/bin/env python3
"""Record every entry in `recordings.json`, or check that they still match.

The parameters a recording was made with — frame count, `dt`, opening
camera, title, caption — are not recoverable from the recording itself
without picking the HTML apart, and guessing them wrong silently
produces a *different* video under the same name.  So they live in
`recordings.json`, and this script is the only thing that reads them.

    record_all.py               re-record all of them
    record_all.py --check       record to a temporary directory and
                                compare, byte for byte, against what is
                                committed; write nothing
    record_all.py --only NAME   just the one

`--check` is the interesting mode.  The recorder never integrates
anything itself, so a recording is a pure function of (scene, frames,
dt, view, title, caption) and the posim binary.  If a check fails,
either the physics moved or the player template changed — and the
difference says which.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
MANIFEST = PACKAGE / "recordings.json"
RECORDER = HERE / "record_video.py"


def load(manifest: pathlib.Path):
    """Manifest paths are relative to the manifest itself, so moving the
    package moves its notion of where the videos live with it."""
    doc = json.loads(manifest.read_text())
    base = (manifest.parent / doc["base"]).resolve()
    ws = doc.get("workspace")
    # Pinned, not searched: a checkout can hold more than one posim
    # workspace, and recording against the wrong one is silent.
    ws = (manifest.parent / ws).resolve() if ws else None
    return base, ws, doc["recordings"]


def record_one(entry: dict, base: pathlib.Path, out: pathlib.Path,
               workspace=None) -> None:
    cmd = [
        sys.executable, str(RECORDER), str(base / entry["scene"]),
        "-o", str(out),
        "--frames", str(entry["frames"]), "--dt", str(entry["dt"]),
        "--view", entry["view"], "--title", entry["title"],
        "--caption", entry["caption"],
    ]
    if workspace:
        cmd += ["--workspace", str(workspace)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{entry['name']}: recorder failed\n{r.stdout}{r.stderr}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed files, write nothing")
    ap.add_argument("--only", default=None, help="one recording, by name")
    ap.add_argument("--workspace", type=pathlib.Path, default=None)
    args = ap.parse_args()

    base, manifest_ws, entries = load(args.manifest)
    workspace = args.workspace or manifest_ws
    if args.only:
        entries = [e for e in entries if e["name"] == args.only]
        if not entries:
            sys.exit(f"no recording named {args.only!r} in {args.manifest}")

    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for e in entries:
            committed = base / e["out"]
            target = pathlib.Path(tmp) / f"{e['name']}.html" if args.check \
                else committed
            record_one(e, base, target, workspace)
            if args.check:
                if not committed.is_file():
                    print(f"MISSING  {e['name']}: {committed} does not exist")
                    failed.append(e["name"])
                elif committed.read_bytes() != target.read_bytes():
                    print(f"DIFFERS  {e['name']}: {committed}")
                    failed.append(e["name"])
                    # keep the mismatch where a human can look at it
                    shutil.copy(target, committed.with_suffix(".html.new"))
                else:
                    print(f"ok       {e['name']}: byte-identical "
                          f"({committed.stat().st_size} bytes)")
            else:
                print(f"wrote    {e['name']}: {committed}")

    if failed:
        print(f"\n{len(failed)} of {len(entries)} differ: {', '.join(failed)}")
        print("Each mismatch was written alongside the original as *.html.new")
        return 1
    if args.check:
        print(f"\nall {len(entries)} recordings reproduce byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
