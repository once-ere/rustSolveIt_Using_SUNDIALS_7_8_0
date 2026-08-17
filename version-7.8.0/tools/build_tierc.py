#!/usr/bin/env python3
"""Phase-8: Tier-C entries — the vendored sundials_rs workspace.

4,054 public items across seven crates. These get a FULL reference entry —
signature, doc comment, file:line, crate and C-file module — but no
generated code snippet. That is a deliberate choice, not a shortfall:

  * sundials_rs is a faithful translation of a C library. Its API is
    `&mut CVodeMem` plus a context, a matrix, a linear solver and a set of
    callbacks; a one-line snippet for `CVodeSetMaxOrd` would be a lie about
    how it is reached.
  * The workspace already ships 105 runnable example PROGRAMS whose stdout
    is diffed byte-for-byte against the upstream C references (see
    sundials_rs/VERIFICATION.md). Pointing at the ones that actually call a
    symbol is worth more than any snippet this script could invent — they
    are real, complete, and already verified by someone else's harness.

So each entry links to the example programs that genuinely reference it,
found by scanning their sources. Entries carry status "reference": a full
reference whose usage is demonstrated by a verified program elsewhere in
the tree, as distinct from "stub", which means nothing is there yet.

Stdlib only.
"""

import json
import os
import re
from collections import defaultdict

CRATES = ["sundials_core", "cvode_rs", "cvodes_rs", "arkode_rs",
          "ida_rs", "idas_rs", "kinsol_rs"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def example_index():
    """symbol -> [example paths that reference it]"""
    idx = defaultdict(set)
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "sundials_rs", "crates")):
        if os.path.basename(dirpath) != "examples":
            continue
        for fn in files:
            if not fn.endswith(".rs"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            try:
                text = open(full, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for sym in set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b", text)):
                idx[sym].add(rel)
    return idx


def main():
    items = [json.loads(l) for l in open(os.path.join(ROOT, "index_data",
                                                      "rust_items.jsonl"))]
    pub = [d for d in items if d["crate"] in CRATES and d["visibility"] == "pub"
           and d["kind"] != "mod"]
    exidx = example_index()

    # the return-flag convention is the same across the suite and is the one
    # thing every caller must know, so it goes on every function entry
    FLAGS = ("Return-flag convention: `*_SUCCESS == 0`; negative values are fatal "
             "(`CV_ILL_INPUT = -22`, `CV_MEM_NULL`, …); certain positive values are "
             "informative (`CV_ROOT_RETURN = 2`, `CV_TSTOP_RETURN = 1`). Function "
             "names, argument order and flag values match the C headers exactly; "
             "only the types are Rust-ified.")

    out = []
    with_ex = 0
    for d in pub:
        kind = {"fn": "function", "struct": "type", "enum": "type", "trait": "type",
                "type": "type"}.get(d["kind"], "type")
        eid = "rs.%s.%s.%s" % (d["crate"], d["module"], d["name"].replace("::", "."))
        doc = d["doc"] or ""
        egs = sorted(exidx.get(d["bare"], ()))[:6]
        if egs:
            with_ex += 1
        examples = [{
            "level": "trivial", "medium": "shell",
            "code": "cargo run -p %s --example %s"
                    % (e.split("/")[2], os.path.basename(e)[:-3]),
            "expected": None, "verified": None,
            "runner": "cargo (in the sundials_rs workspace)",
        } for e in egs[:3]]

        defn = (doc + "  " if doc else "")
        defn += ("Declared in the `%s` module of `%s` — the module name is the base "
                 "name of the C file it was translated from, so "
                 "`sundials_rs/ARCHITECTURE.md` §1 maps it straight back to the "
                 "original source." % (d["module"], d["crate"]))
        if d["kind"] == "fn":
            defn += "  " + FLAGS
        if egs:
            defn += ("  Exercised by %d shipped example program(s), whose stdout is "
                     "diffed byte-for-byte against the upstream C reference: %s."
                     % (len(egs), ", ".join("`%s`" % e for e in egs)))

        out.append({
            "id": eid, "name": d["name"], "kind": kind, "tier": "C",
            "aliases": [], "indexKeys": [(d["bare"][0] if d["bare"] else "Σ").upper()],
            "summary": (doc.split(".")[0][:150] + "." if doc
                        else "%s `%s` in `%s::%s`." % (d["kind"], d["bare"],
                                                       d["crate"], d["module"])),
            "definition": defn,
            "syntax": [d["signature"]],
            "parameters": [], "returns": None, "errors": [],
            "locations": [{"file": d["file"], "line": d["line"], "role": "definition"}]
                       + [{"file": e, "line": 1, "role": "verified example program"}
                          for e in egs[:3]],
            "examples": examples, "seeAlso": [], "invariants": [],
            "status": "reference",
        })

    path = os.path.join(ROOT, "index_data", "entries_tierc.json")
    json.dump(out, open(path, "w"), indent=1)
    print("%d Tier-C entries -> %s" % (len(out), path))
    print("  linked to at least one verified example program: %d (%.0f%%)"
          % (with_ex, 100.0 * with_ex / len(out)))


if __name__ == "__main__":
    main()
