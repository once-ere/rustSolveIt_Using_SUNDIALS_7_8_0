#!/usr/bin/env python3
"""Phase-4 verifier: run every catalog example and record what it actually did.

Reads index_data/catalog.json, executes each example, and writes the captured
output back into `expected` with the run date in `verified`. A fragment that
fails leaves BOTH fields null, so an unverified example is visibly unverified
rather than silently presented as working.

The failure rules are not this script's opinion — they are the
`_meta.failure_rules` block in catalog.json, put there in Phase 3 after a
sample run reported 91/91 while 14 of those passes were worthless:

  * exit status alone is not enough. A failing magic — notably `%load` on a
    path that does not exist — exits 0, prints `%load ... failed: ...`, and
    carries on. Any line matching ^%\\w+ .* failed: is a failure.
  * cwd must be the repository root: %load and %save resolve relative paths.
  * POSIM_NO_BROWSER=1, so SCENE CREATE opens no browser.

Usage:
    python3 tools/verify_index_examples.py [--only KIND] [--jobs N] [--dry-run]

Stdlib only.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(ROOT, "index_data", "catalog.json")
REPORT = os.path.join(ROOT, "index_data", "verification_report.json")
BIN = os.path.join(ROOT, "target", "release", "posim")
# The day this actually ran. Hardcoding it is how the per-example
# dates drifted away from the catalog's own verified_date: a stamp
# that does not move is a stamp that stops being true.
TODAY = datetime.date.today().isoformat()

# The media this project EXECUTES, as opposed to quotes or points at.
# One definition, used for both the pass and the total.
EXECUTED_MEDIA = ("posim", "rust", "machine")

MAGIC_FAILED = re.compile(r"^%\w+ .* failed:")
ERR_LINE = re.compile(r"^Err\[\d+\]:")
TIMEOUT = 300


def failures_in(stdout):
    """Every reason this run should be considered a failure."""
    bad = []
    for line in stdout.splitlines():
        if ERR_LINE.match(line) or MAGIC_FAILED.match(line):
            bad.append(line.strip())
    return bad


def run_machine(code):
    """Feed JSONL requests to `posim --machine` and capture the replies.

    Machine mode is every bit as runnable as the notebook — it is what the
    JupyterLab kernel drives — so its examples are executed rather than left
    as stubs. A reply line carrying "ok":false is a failure, the machine-mode
    equivalent of an Err[] line.
    """
    try:
        r = subprocess.run(
            [BIN, "--machine"], input=code.rstrip("\n") + "\n",
            capture_output=True, text=True, timeout=TIMEOUT,
            env=dict(os.environ, POSIM_NO_BROWSER="1"), cwd=ROOT)
    except subprocess.TimeoutExpired:
        return False, "", ["timed out after %ds" % TIMEOUT]
    reasons = [l.strip()[:120] for l in r.stdout.splitlines() if '"ok":false' in l]
    if r.returncode != 0:
        reasons.append("exit status %d" % r.returncode)
    return (not reasons), r.stdout, reasons


def run_posim(code):
    """Run one fragment. Returns (ok, stdout, [reasons])."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fragment.posim")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code.rstrip("\n") + "\n")
        try:
            r = subprocess.run(
                [BIN, "--script", path],
                capture_output=True, text=True, timeout=TIMEOUT,
                env=dict(os.environ, POSIM_NO_BROWSER="1"),
                cwd=ROOT,                      # %load / %save are cwd-relative
            )
        except subprocess.TimeoutExpired:
            return False, "", ["timed out after %ds" % TIMEOUT]
        reasons = failures_in(r.stdout)
        if r.returncode != 0:
            reasons.append("exit status %d" % r.returncode)
        if r.stderr.strip():
            reasons.append("stderr: " + r.stderr.strip().splitlines()[0])
        return (not reasons), r.stdout, reasons


# Values that genuinely differ between runs. Showing the captured literal
# would be a small lie: the reader re-runs the fragment and gets a different
# number, and then has no way to tell which of the other numbers to trust.
# Solver step counts are NOT in this list — those are deterministic, and they
# are exactly the anchors the documentation pins.
VARIES = [
    (re.compile(r"(127\.0\.0\.1:)\d+"), r"\1<port>"),   # OS-assigned each CREATE
    (re.compile(r"steps = [1-9]\d*"), "steps = <varies>"),  # playback is wall-clock
    (re.compile(r"history = [1-9]\d* frame"), "history = <varies> frame"),
]


def transcript(stdout):
    """What a reader sees: the Out[]/Err[] lines and any bare replies."""
    keep = [l for l in stdout.splitlines()
            if not l.startswith("In[") or "Out[" in l or "Err[" in l]
    text = "\n".join(keep).strip()
    for pat, sub in VARIES:
        text = pat.sub(sub, text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="verify one kind only (command, builtin, ...)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything but do not write catalog.json back")
    args = ap.parse_args()

    if not os.path.exists(BIN):
        sys.exit("%s missing — run: cargo build --release -p posim" % BIN)

    catalog = json.load(open(CATALOG, encoding="utf-8"))
    meta = catalog[0] if catalog and catalog[0].get("_meta") else None
    entries = [e for e in catalog if not e.get("_meta")]
    if meta:
        print("failure rules in force:")
        for r in meta["failure_rules"]:
            print("  -", r)
        print()

    jobs = []
    for e in entries:
        if args.only and e["kind"] != args.only:
            continue
        for i, x in enumerate(e["examples"]):
            if x["medium"] in ("posim", "machine"):
                jobs.append((e["id"], e["kind"], i, x["code"], x["medium"]))

    print("verifying %d fragments (%d posim, %d machine) with %d workers\n"
          % (len(jobs), sum(1 for j in jobs if j[4] == "posim"),
             sum(1 for j in jobs if j[4] == "machine"), args.jobs))
    results = {}
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_machine if med == "machine" else run_posim,
                              code): (eid, kind, i, med)
                   for eid, kind, i, code, med in jobs}
        for fut in concurrent.futures.as_completed(futures):
            eid, kind, i, _med = futures[fut]
            ok, out, reasons = fut.result()
            results[(eid, i)] = (ok, out, reasons)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                bad = sum(1 for v in results.values() if not v[0])
                print("  %d/%d  failures so far: %d  (%.0fs)"
                      % (done, len(jobs), bad, time.time() - t0))

    report = []
    passed = 0
    for e in entries:
        for i, x in enumerate(e["examples"]):
            if (e["id"], i) not in results:
                continue
            ok, out, reasons = results[(e["id"], i)]
            if ok:
                x["expected"] = (out.strip() if x["medium"] == "machine"
                                 else transcript(out))
                x["verified"] = TODAY
                passed += 1
            else:
                x["expected"] = None
                x["verified"] = None
            report.append({"id": e["id"], "kind": e["kind"], "level": x["level"],
                           "ok": ok, "reasons": reasons[:3]})

    n = len(report)
    print("\nPASS %d/%d  (%.1f%%)   %.0fs wall clock"
          % (passed, n, 100.0 * passed / n, time.time() - t0))

    fails = [r for r in report if not r["ok"]]
    if fails:
        print("\n%d failing fragment(s):" % len(fails))
        for r in fails[:40]:
            print("  %s [%s] %s" % (r["id"], r["level"], r["reasons"]))
        if len(fails) > 40:
            print("  ... and %d more (see verification_report.json)" % (len(fails) - 40))

    if args.dry_run:
        print("\n--dry-run: catalog.json not written")
        return 1 if fails else 0

    if meta:
        # count BOTH media: this pass only executes posim fragments, but the
        # Rust snippets were compiled by tools/verify_tierb_examples.py and
        # overwriting the total with the posim count alone would erase them.
        meta["examples_verified"] = True
        meta["verified_date"] = TODAY
        meta["verified_pass"] = sum(1 for e in entries for x in e["examples"]
                                    if x["verified"])
        # EVERY executed medium. This runs after build_catalog and overwrites
        # what it wrote, so a fix applied only there does not hold — which is
        # exactly how "1435 of 1411" reached the home screen twice.
        meta["verified_total"] = sum(1 for e in entries for x in e["examples"]
                                     if x["medium"] in EXECUTED_MEDIA)
    # A pass count cannot exceed its total. Assert it rather than trusting the
    # two sums to agree, because they have now disagreed twice.
    if meta and meta["verified_pass"] > meta["verified_total"]:
        sys.exit("BUG: verified_pass %d > verified_total %d — the media sets "
                 "have drifted apart again"
                 % (meta["verified_pass"], meta["verified_total"]))
    json.dump(catalog, open(CATALOG, "w", encoding="utf-8"), indent=1)
    json.dump(report, open(REPORT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s\nwrote %s" % (CATALOG, REPORT))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
