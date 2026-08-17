#!/usr/bin/env python3
"""Phase-2 extractor: every public Rust item in the project, with file:line.

Scans the crate roots given on the command line (or the project defaults),
emits one JSON object per item to stdout as JSONL:

    {"name", "bare", "kind", "visibility", "crate", "module",
     "file", "line", "signature", "doc"}

Stdlib only, no external dependencies — the same spirit as the existing
scripts/gen_airy_uniform_series.py.

Every line of every scanned file is examined; items are matched on the
`pub` item forms Rust actually uses. Impl blocks are tracked so that
methods are attributed to their type (`NVector::linear_sum_with`).
`#[cfg(test)]` modules are excluded: their items are not API.
"""

import json
import os
import re
import sys

ROOTS = [
    "sundials_rs/crates",
    "physical_object/src",
    "posim/src",
    "special_functions/src",
    "quantum/src",
]

# `pub`, `pub(crate)`, `pub(super)` … — pub(crate) is recorded with a
# visibility marker rather than dropped, so nothing is silently lost.
VIS = r"pub(?:\s*\(\s*(?P<scope>crate|super|self|in\s+[\w:]+)\s*\))?"

ITEM = re.compile(
    r"^(?P<indent>\s*)" + VIS + r"\s+"
    r"(?:(?P<mods>(?:const\s+|async\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*))"
    r"(?P<kind>fn|struct|enum|trait|type|const|static|mod|union)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)

IMPL = re.compile(
    r"^\s*impl(?:\s*<[^>]*>)?\s+"
    r"(?:(?P<trait>[\w:]+(?:\s*<[^>]*>)?)\s+for\s+)?"
    r"(?P<type>[\w:]+)"
)

DOC = re.compile(r"^\s*//[/!]\s?(?P<text>.*)$")


def crate_of(path):
    parts = path.split(os.sep)
    if "crates" in parts:
        return parts[parts.index("crates") + 1]
    return parts[0]


def module_of(path):
    base = os.path.basename(path)
    if base == "mod.rs":
        return os.path.basename(os.path.dirname(path))
    return base[:-3]


def signature(lines, i):
    """The item's declaration, joined until the body or `;` — capped at 6 lines."""
    out = []
    depth = 0
    for line in lines[i:i + 6]:
        s = line.rstrip()
        out.append(s.strip())
        depth += s.count("(") - s.count(")")
        t = s.rstrip()
        if depth <= 0 and (t.endswith("{") or t.endswith(";") or t.endswith(")")):
            break
    sig = " ".join(out).split("{")[0].strip().rstrip(";").strip()
    return re.sub(r"\s+", " ", sig)


def doc_above(lines, i):
    """Contiguous `///` / `//!` block immediately above line i (attributes skipped)."""
    out = []
    j = i - 1
    while j >= 0:
        m = DOC.match(lines[j])
        if m:
            out.append(m.group("text"))
            j -= 1
            continue
        if lines[j].lstrip().startswith("#["):
            j -= 1
            continue
        break
    return " ".join(reversed(out)).strip()


def scan(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    impl_stack = []          # [(type_name, brace_depth_at_entry)]
    depth = 0
    pending_test = False     # saw #[cfg(test)]; decide on its target line
    test_depth = None        # brace depth at which the #[cfg(test)] mod opened
    out = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Exclude #[cfg(test)] MODULES only (the stated policy): the
        # flag arms on the attribute and fires when its target turns
        # out to be a `mod`. An attribute-gated single item (a helper
        # fn compiled for tests) stays indexed. The old version set and
        # cleared the depth marker on the attribute line itself, so the
        # exclusion could never fire.
        if test_depth is None and stripped.startswith("#[cfg(test)]"):
            pending_test = True
        elif pending_test and stripped and not stripped.startswith(("#[", "//")):
            if re.match(r"(pub(\([a-z]+\))?\s+)?mod\b", stripped):
                test_depth = depth
            pending_test = False
        in_test = test_depth is not None

        if not in_test:
            m_impl = IMPL.match(line)
            if m_impl:
                impl_stack.append((m_impl.group("type"), depth))

            m = ITEM.match(line)
            if m:
                kind = m.group("kind")
                name = m.group("name")
                owner = impl_stack[-1][0] if (impl_stack and kind == "fn") else None
                scope = m.group("scope")
                out.append({
                    "name": "%s::%s" % (owner, name) if owner else name,
                    "bare": name,
                    "kind": kind,
                    "visibility": "pub" if not scope else "pub(%s)" % scope,
                    "crate": crate_of(path),
                    "module": module_of(path),
                    "file": path,
                    "line": i + 1,
                    "signature": signature(lines, i),
                    "doc": doc_above(lines, i)[:400],
                })

        depth += line.count("{") - line.count("}")

        if in_test and depth <= test_depth:
            test_depth = None
        while impl_stack and depth <= impl_stack[-1][1]:
            impl_stack.pop()

    return out


def main():
    roots = sys.argv[1:] or ROOTS
    total = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("target", ".git")]
            for fn in sorted(filenames):
                if not fn.endswith(".rs"):
                    continue
                for item in scan(os.path.join(dirpath, fn)):
                    print(json.dumps(item, sort_keys=True))
                    total += 1
    print("# %d items" % total, file=sys.stderr)


if __name__ == "__main__":
    main()
