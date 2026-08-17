#!/usr/bin/env python3
"""compare_results.py — pair every C example run with its Rust counterpart
and classify the difference.

    python3 tools/compare_results.py

Reads   c-results/index.tsv      and c-results/raw/**
        rust-results/index.tsv   and rust-results/raw/**
Writes  differences/index.tsv
        differences/diffs/<dir>/<variant>.diff     (unified diff, when they differ)
        differences/diffs/<dir>/<variant>.numbers  (worst numeric deltas)

Classification, in the order it is tried:

  NOT_PORTED       no Rust binary exists for this example
  IDENTICAL        the two stdout streams are byte-for-byte equal
  WHITESPACE       equal once runs of blanks are collapsed -- every printed
                   character matches, only column padding differs
  NUMERIC          same text skeleton, same count of numeric tokens, and
                   every difference is in the value of a number. The worst
                   relative difference and the worst distance in ulp are
                   both reported.
  STRUCTURAL       anything else: different line counts, different words,
                   different number of numeric fields

Only stdlib is used. Nothing is rounded, smoothed or filtered before the
comparison: IDENTICAL means the bytes matched.
"""

import os
import re
import struct
import sys
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "c-results"
R_DIR = ROOT / "rust-results"
D_DIR = ROOT / "differences"

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def read_index(p):
    rows = []
    with open(p) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(head, parts)))
    return rows


def ulp_distance(a, b):
    """Distance between two doubles counted in representable steps."""
    if a == b:
        return 0
    if a != a or b != b:
        return None
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    if ia < 0:
        ia = -0x8000000000000000 - ia
    if ib < 0:
        ib = -0x8000000000000000 - ib
    return abs(ia - ib)


def skeleton(line):
    """The line with every numeric token replaced by a placeholder.

    Runs of blanks are collapsed as well. A printf field such as `%5ld`
    pads to a fixed width, so a value that gains a digit (976 -> 1000)
    eats one space. Comparing the raw spacing would call that a
    *structural* difference when it is purely a numeric one, and would
    hide the actual change behind a misleading classification.
    """
    return re.sub(r"[ \t]+", " ", NUM.sub("\0", line))


def numbers(line):
    out = []
    for tok in NUM.findall(line):
        try:
            out.append(float(tok.replace("D", "E").replace("d", "e")))
        except ValueError:
            out.append(None)
    return out


def classify(ctext, rtext):
    if ctext == rtext:
        return "IDENTICAL", {}

    # A capture ends with a newline, so splitting on "\n" yields a phantom
    # empty final element and every total_lines was one too high (48 reported
    # for a 47-line file). Strip exactly one trailing newline, not all of
    # them -- a genuine difference in trailing blank lines must still show up.
    ctext = ctext[:-1] if ctext.endswith("\n") else ctext
    rtext = rtext[:-1] if rtext.endswith("\n") else rtext
    if ctext == rtext:
        return "IDENTICAL", {}
    # whitespace-insensitive: collapse runs of blanks, drop trailing blanks
    def squash(t):
        return "\n".join(re.sub(r"[ \t]+", " ", ln).rstrip() for ln in t.split("\n"))

    if squash(ctext) == squash(rtext):
        return "WHITESPACE", {}

    clines = ctext.split("\n")
    rlines = rtext.split("\n")
    if len(clines) != len(rlines):
        return "STRUCTURAL", {"reason": f"line count {len(clines)} vs {len(rlines)}"}

    worst_rel = 0.0
    worst_ulp = 0
    worst_at = None
    ndiff_lines = 0
    for i, (cl, rl) in enumerate(zip(clines, rlines), start=1):
        if cl == rl:
            continue
        ndiff_lines += 1
        if skeleton(cl) != skeleton(rl):
            return "STRUCTURAL", {"reason": f"line {i}: text differs, not just numbers"}
        cn, rn = numbers(cl), numbers(rl)
        if len(cn) != len(rn):
            return "STRUCTURAL", {"reason": f"line {i}: {len(cn)} vs {len(rn)} numbers"}
        for j, (a, b) in enumerate(zip(cn, rn)):
            if a is None or b is None or a == b:
                continue
            denom = max(abs(a), abs(b))
            rel = abs(a - b) / denom if denom else 0.0
            u = ulp_distance(a, b)
            if rel > worst_rel:
                worst_rel = rel
                worst_at = (i, j, a, b)
            if u is not None and u > worst_ulp:
                worst_ulp = u
    if worst_at is None:
        # Same skeleton, same numeric fields, same values, yet the bytes
        # differ: the difference is in formatting the numbers, not in the
        # numbers. Calling that NUMERIC would overstate it.
        return "FORMATTING", {
            "reason": "numeric values all equal; the printed form differs",
            "diff_lines": ndiff_lines,
            "total_lines": len(clines),
        }
    return "NUMERIC", {
        "diff_lines": ndiff_lines,
        "total_lines": len(clines),
        "worst_rel": worst_rel,
        "worst_ulp": worst_ulp,
        "worst_at": worst_at,
    }


def main():
    c_rows = {(r["dir"], r["variant"]): r for r in read_index(C_DIR / "index.tsv")}
    r_rows = {(r["dir"], r["variant"]): r for r in read_index(R_DIR / "index.tsv")}

    # Clear the previous run's diffs before writing this one's. Without this
    # the directory only ever grows: a variant that used to differ and now
    # matches keeps its stale .diff and .numbers, so the tree ends up
    # contradicting the index beside it. That happened -- idaRoberts_klu and
    # idasRoberts_klu kept diffs from before the pivoting fix while
    # index.tsv called them IDENTICAL, which is exactly the kind of internal
    # contradiction that makes an evidence directory worthless.
    diffs = D_DIR / "diffs"
    if diffs.exists():
        for stale in diffs.rglob("*"):
            if stale.is_file() and stale.suffix in (".diff", ".numbers"):
                stale.unlink()
    diffs.mkdir(parents=True, exist_ok=True)
    out = open(D_DIR / "index.tsv", "w")
    out.write(
        "dir\texample\targv\tvariant\tclass\tc_status\trust_status\t"
        "diff_lines\ttotal_lines\tworst_rel\tworst_ulp\tdetail\n"
    )

    # Only variants the Rust port claims to cover are comparable; the Rust
    # index is therefore the driving list.
    counts = {}
    for key in sorted(r_rows):
        rr = r_rows[key]
        cr = c_rows.get(key)
        d, vid = key
        if rr["status"] == "NOT_PORTED":
            cls, info = "NOT_PORTED", {"reason": "no pure-Rust counterpart (KLU/SuperLU backend)"}
            ctext = rtext = ""
        elif cr is None:
            cls, info = "NO_C_RUN", {"reason": "the C example was not built on this machine"}
            ctext = rtext = ""
        else:
            ctext = (C_DIR / "raw" / d / f"{vid}.stdout").read_text(errors="replace")
            rtext = (R_DIR / "raw" / d / f"{vid}.stdout").read_text(errors="replace")
            cls, info = classify(ctext, rtext)

        counts[cls] = counts.get(cls, 0) + 1

        if cls not in ("IDENTICAL", "NOT_PORTED", "NO_C_RUN"):
            dd = D_DIR / "diffs" / d
            dd.mkdir(parents=True, exist_ok=True)
            diff = difflib.unified_diff(
                ctext.splitlines(keepends=True),
                rtext.splitlines(keepends=True),
                fromfile=f"c-results/raw/{d}/{vid}.stdout",
                tofile=f"rust-results/raw/{d}/{vid}.stdout",
                n=1,
            )
            (dd / f"{vid}.diff").write_text("".join(diff))
            if cls == "NUMERIC" and info.get("worst_at"):
                ln, col, a, b = info["worst_at"]
                (dd / f"{vid}.numbers").write_text(
                    f"worst relative difference {info['worst_rel']:.3e} "
                    f"({info['worst_ulp']} ulp)\n"
                    f"  line {ln}, numeric field {col}\n"
                    f"  C    = {a!r}\n"
                    f"  Rust = {b!r}\n"
                    f"lines differing: {info['diff_lines']} of {info['total_lines']}\n"
                )

        detail = info.get("reason", "")
        if cls == "NUMERIC":
            wa = info.get("worst_at")
            detail = f"line {wa[0]} field {wa[1]}" if wa else ""
        out.write(
            "\t".join(
                [
                    d,
                    rr["example"],
                    rr["argv"],
                    vid,
                    cls,
                    cr["status"] if cr else "-",
                    rr["status"],
                    str(info.get("diff_lines", "")),
                    str(info.get("total_lines", "")),
                    f"{info['worst_rel']:.3e}" if "worst_rel" in info else "",
                    str(info.get("worst_ulp", "")),
                    detail,
                ]
            )
            + "\n"
        )
    out.close()

    for d in sorted(diffs.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    print("classification of the C-vs-Rust comparison:")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {k:12s} {counts[k]}")
    print(f"\nindex: {D_DIR / 'index.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
