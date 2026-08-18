#!/usr/bin/env python3
"""Check every generated notebook against the seven stated requirements.

  R1  it explains how to launch a notebook from a terminal CLI
  R2  it never sends the reader to another notebook
  R3  every code cell is preceded by explanatory markdown
  R4  it asks the user to name the notebook
  R5  it opens a graphical save dialog, with a fallback
  R6  it describes objects, equations of motion, constraint equations,
      and the second-order -> first-order reduction
  R7  it is valid nbformat-4 JSON and pairs with a real example file
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # the repository root

# Phrases that would send a reader somewhere else. R2.
CROSSREF = re.compile(
    r"(see|refer to|as (?:explained|described|shown|covered) in|documented in)\s+"
    r"(the\s+)?(other|another|previous|next|companion|first|earlier|following)?\s*"
    r"notebook", re.I)

def check(path):
    bad = []
    try:
        nb = json.loads(Path(path).read_text())
    except Exception as exc:
        return [f"R7 not valid JSON: {exc}"]
    if nb.get("nbformat") != 4:
        bad.append("R7 nbformat is not 4")
    cells = nb["cells"]
    text  = "\n".join("".join(c["source"]) for c in cells)
    mdtext = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")

    # R1
    for needle in ["jupyter lab", "New → Notebook", "Python 3 (ipykernel)",
                   "cargo build --release -p posim", "Shift+Enter"]:
        if needle not in text:
            bad.append(f"R1 launch instructions missing {needle!r}")
    # R2
    m = CROSSREF.search(mdtext)
    if m:
        bad.append(f"R2 sends the reader elsewhere: {m.group(0)!r}")
    # Naming the paired .posim example is required, not a cross-reference.
    # Naming ANOTHER .ipynb would be one.
    for m in re.finditer(r"[\w./-]+\.ipynb", mdtext):
        bad.append(f"R2 markdown names another notebook file: {m.group(0)!r}")
    # R3
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            if i == 0 or cells[i-1]["cell_type"] != "markdown":
                bad.append(f"R3 code cell {i} has no explanation before it")
            elif len("".join(cells[i-1]["source"]).strip()) < 80:
                bad.append(f"R3 explanation before code cell {i} is too thin")
    # R4 / R5
    if "Name for this notebook" not in text:
        bad.append("R4 does not ask the user to name the notebook")
    for needle in ["tkinter", "asksaveasfilename", "Falling back to a typed folder path"]:
        if needle not in text:
            bad.append(f"R5 save dialog missing {needle!r}")
    # R6
    for heading in ["## 4. The physical situation", "### 4.1 Equations of motion",
                    "### 4.2 Constraint equations",
                    "### 4.3 The first-order system actually handed to SUNDIALS",
                    "## 3. How a second-order mechanics problem becomes a first-order system"]:
        if heading not in mdtext:
            bad.append(f"R6 missing section {heading!r}")
    # R7
    src = nb.get("metadata", {}).get("posim", {}).get("pairs_with")
    if not src:
        bad.append("R7 no paired example recorded in metadata")
    elif not (ROOT / src).exists():
        bad.append(f"R7 paired example {src} does not exist")
    # every non-interactive code cell must have run
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "interactive" not in c.get("metadata", {}).get("tags", []):
            if c.get("execution_count") is None:
                bad.append(f"R7 code cell {i} was never executed")
    return bad

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(str(p) for p in Path("notebooks").glob("*.ipynb"))
    failed = 0
    for p in paths:
        problems = check(p)
        if problems:
            failed += 1
            print(f"FAIL {p}")
            for q in problems:
                print(f"       {q}")
    print(f"\n{len(paths) - failed}/{len(paths)} notebooks pass all seven requirements")
    sys.exit(1 if failed else 0)
