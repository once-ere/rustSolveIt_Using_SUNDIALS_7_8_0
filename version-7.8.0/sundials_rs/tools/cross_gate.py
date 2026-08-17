#!/usr/bin/env python3
"""cross_gate.py — put this repository's two independent gates side by side.

    python3 tools/cross_gate.py

The port is measured twice, against two different things, on two different
machines. Both measurements cover the *same* 199 `(example, argv)` variants,
which is what makes them comparable at all — the script asserts that rather
than assuming it.

  Gate A  Rust  vs  the `.out` reference files shipped inside SUNDIALS 7.8.0
          Ubuntu 24.04, glibc 2.39, rustc 1.93.1, host libm.
          `tools/verify_examples.sh all`
          -> evidence/linux-x86_64-glibc239/summary.txt

  Gate B  Rust  vs  the upstream C sources compiled from scratch on the same
          machine, at the same time, by the same compiler.
          Ubuntu 26.04, glibc 2.43, rustc 1.96.1, pure-Rust libm.
          `tools/c_build.sh && tools/c_examples_run.sh`,
          `tools/rust_examples_run.sh`, `tools/compare_results.py`
          -> differences/index.tsv

They answer different questions. Gate A asks whether the port reproduces the
*published* reference output. Gate B asks whether the port agrees with the C
it was translated from, holding the machine fixed. A shipped `.out` was
generated years ago on somebody else's libm, so Gate A necessarily charges the
port for that drift; Gate B cannot, because both binaries are built here.

Neither supersedes the other, and the cross-tabulation is more informative
than either alone: it shows where a variant lands in one gate given where it
landed in the other.

Only stdlib is used, and every number printed is computed from the two files
named above.
"""

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE_A = ROOT / "evidence" / "linux-x86_64-glibc239" / "summary.txt"
GATE_B = ROOT / "differences" / "index.tsv"
AB = ROOT / "differences" / "ab-host-libm.tsv"

# `<example>  [<argv>]  <class>` — argv may be empty, class may carry a detail
# such as `DIFF(70 lines)` or `EXCLUDED excluded(klu)`.
ROW = re.compile(r"^(\S+)\s+\[(.*)\]\s+(.*)$")


def load_gate_a(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("###") or not line.strip():
            continue
        m = ROW.match(line)
        if not m:
            raise SystemExit(f"{path}: cannot parse {line!r}")
        example, argv, cls = m.group(1), m.group(2), m.group(3).strip()
        if cls.startswith("DIFF"):
            cls = "DIFF"
        elif cls.startswith("EXCLUDED"):
            # keep the reason: excluded(klu) and excluded(superlu) are
            # different scope decisions with different fates in gate B
            cls = cls.replace("EXCLUDED ", "EXCLUDED ")
        out[(example, argv)] = cls
    return out


def load_tsv(path, column):
    out = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(head, line.rstrip("\n").split("\t")))
            out[(row["example"], row["argv"])] = row[column]
    return out


def table(rows, cols, cell, corner="", rlabel=str):
    w = max([len(rlabel(r)) for r in rows] + [len(corner)]) + 2
    cw = max(max((len(c) for c in cols), default=0), 8) + 3
    lines = [corner.ljust(w) + "".join(c.rjust(cw) for c in cols) + "total".rjust(cw)]
    lines.append(" " * w + "".join("-" * (cw - 2) + "  " for _ in cols) + "-" * (cw - 2))
    for r in rows:
        vals = [cell(r, c) for c in cols]
        lines.append(
            rlabel(r).ljust(w)
            + "".join((str(v) if v else ".").rjust(cw) for v in vals)
            + str(sum(vals)).rjust(cw)
        )
    tot = [sum(cell(r, c) for r in rows) for c in cols]
    lines.append("total".ljust(w) + "".join(str(v).rjust(cw) for v in tot) + str(sum(tot)).rjust(cw))
    return "\n".join(lines)


def main():
    for p in (GATE_A, GATE_B, AB):
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(ROOT)}")

    a = load_gate_a(GATE_A)
    b = load_tsv(GATE_B, "class")

    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        raise SystemExit(
            "the two gates do not cover the same variants, so they are not\n"
            f"comparable: {len(only_a)} only in A, {len(only_b)} only in B\n"
            f"  A only: {only_a[:5]}\n  B only: {only_b[:5]}"
        )

    print(f"Both gates cover the same {len(a)} (example, argv) variants.\n")

    counts = collections.Counter((a[k], b[k]) for k in a)
    arows = sorted({x for x, _ in counts})
    bcols = sorted({y for _, y in counts})
    print("            gate B: Rust vs pristine C rebuilt here (Ubuntu 26.04 / glibc 2.43)")
    print(table(arows, bcols, lambda r, c: counts[(r, c)], corner="gate A vs .out"))

    # --- the two results worth stating in prose ----------------------------
    diff_a = {k for k in a if a[k] == "DIFF"}
    same_in_b = {k for k in diff_a if b[k] == "IDENTICAL"}
    print(
        f"\n1. Of the {len(diff_a)} variants that differ from the shipped .out files, "
        f"{len(same_in_b)} are\n   byte-identical to pristine C compiled here. "
        + (
            "That is all of them, and it is\n   an independent confirmation of "
            '"0 port defects" — a second host, a second\n   glibc, and a reference '
            "built rather than downloaded."
            if same_in_b == diff_a
            else f"{len(diff_a - same_in_b)} are not, and each\n   of those is a port defect to be fixed: "
            + ", ".join(sorted(e for e, _ in diff_a - same_in_b))
        )
    )

    # --- decompose gate B's divergences ------------------------------------
    default_cls = load_tsv(AB, "default_class")
    host_cls = load_tsv(AB, "host_libm_class")
    numeric = {k for k in b if b[k] == "NUMERIC"}
    libm = {k for k in default_cls if default_cls[k] != "IDENTICAL" and host_cls[k] == "IDENTICAL"}
    klu = {k for k in host_cls if host_cls[k] != "IDENTICAL"}

    print(f"\n2. Gate B's {len(numeric)} divergences decompose into two substitutions:")
    print(f"     {len(libm):2d}  the pure-Rust libm    — proven by --features host-libm restoring them")
    print(f"     {len(klu):2d}  the pure-Rust sparse LU — differ under both builds, all *_klu")
    residue = numeric - libm - klu
    print(
        f"     {len(residue):2d}  unaccounted for"
        + ("" if residue else "   <- nothing is left over, which is the point")
    )
    if residue:
        print("        " + ", ".join(f"{e} [{v}]" for e, v in sorted(residue)))
    if not all("_klu" in e for e, _ in klu):
        print("     WARNING: a non-klu variant survives the host-libm switch — investigate")

    # --- the cross-check that makes the libm attribution independent -------
    flipped = {k for k in a if a[k] == "IDENTICAL" and b[k] == "NUMERIC"}
    print(
        f"\n3. {len(flipped)} variants match the shipped .out in gate A but not pristine C in\n"
        f"   gate B. The host-libm control independently names {len(libm)}. "
        + (
            "They are the same\n   set — two experiments, different machines, same answer."
            if flipped == libm
            else f"They differ:\n   {sorted(flipped ^ libm)}"
        )
    )
    for e, v in sorted(libm):
        print(f"     {e}" + (f"  [{v}]" if v else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
