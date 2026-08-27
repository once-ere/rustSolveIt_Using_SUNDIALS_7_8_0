#!/usr/bin/env python3
"""Render one .ipynb from one spec, using the invariant prose in nbtext.

A spec is a JSON document with these keys:

  key         unique notebook stem, e.g. "video_piston_crankshaft"
  title       human title, e.g. "A piston driven by a crankshaft"
  source      repo-relative path of the example this notebook pairs with
  category    video | rust | collision | solveit | dynamic
  howtorun    the shell command that runs the paired example directly
  abstract    one or two paragraphs: what this notebook shows
  situation   markdown: every object, its type and its properties
  eom         markdown: the equations of motion for THIS system
  constraints markdown: the constraint equations for THIS system
  reduction   markdown: the second-order -> first-order reduction, sized
  steps       [ {explain: markdown, code: python} ... ]
  discussion  markdown: what the numbers above mean

Every notebook gets the full invariant prose, verbatim. Nothing is ever
elided with a pointer to another notebook.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nbtext

def md(text, **meta):
    return {"cell_type": "markdown", "metadata": meta, "source": text.splitlines(keepends=True)}

def code(text, outputs=None, **meta):
    return {"cell_type": "code", "execution_count": None, "metadata": meta,
            "outputs": outputs or [], "source": text.splitlines(keepends=True)}

def build(spec):
    key    = spec["key"]
    title  = spec["title"]
    source = spec["source"]
    nbfile = f"{key}.ipynb"

    cells = []

    # ---- title and abstract -------------------------------------------
    cells.append(md(
f"""# {title}

**This notebook is one half of a pair.** The other half is the example
file `{source}` in this repository, which contains the same simulation
written directly in the simulator's own command language. This notebook
runs that same simulation from Python, and explains every line of it.

**This notebook is complete in itself.** Everything needed to understand
and run it is written below. You will never be asked to open another
file to find an explanation.

{spec["abstract"]}

To run the paired example directly, without Python and without Jupyter:

```bash
{spec["howtorun"]}
```

---
"""))

    # ---- invariant sections 1, 2, 3 -----------------------------------
    cells.append(md(nbtext.LAUNCH + "\n---\n"))
    cells.append(md(nbtext.GLOSSARY + "\n---\n"))
    cells.append(md(nbtext.FIRST_ORDER + "\n---\n"))

    # ---- the physics of THIS example ----------------------------------
    cells.append(md(
f"""## 4. The physical situation

{spec["situation"]}

### 4.1 Equations of motion

{spec["eom"]}

### 4.2 Constraint equations

{spec["constraints"]}

### 4.3 The first-order system actually handed to SUNDIALS

{spec["reduction"]}

---
"""))

    # ---- the driver ----------------------------------------------------
    cells.append(md(nbtext.DRIVER_MD.replace("## 4.", "## 5.")))
    cells.append(code(nbtext.DRIVER_CODE))

    # ---- the run, one explained step at a time -------------------------
    cells.append(md(
"""## 6. Building and running the simulation

From here on, the notebook alternates: a section of text explaining
exactly what the next cell asks the simulator to do and what each value
in it means, then the cell itself. Run them in order from here down.
"""))
    for n, step in enumerate(spec["steps"], start=1):
        cells.append(md(f"### 6.{n} {step['title']}\n\n{step['explain']}"))
        cells.append(code(step["code"]))

    # ---- discussion ----------------------------------------------------
    cells.append(md(f"""## 7. What those numbers mean

{spec["discussion"]}
"""))

    # ---- shut down ------------------------------------------------------
    cells.append(md(
"""## 8. Close the simulator

The simulator is a separate operating-system process that this notebook
started. Closing it releases that process and its two pipes. If you skip
this cell the process is closed anyway when the Jupyter kernel shuts
down, but doing it explicitly means you can re-run the notebook from the
top without leaving a stray process behind each time.
"""))
    cells.append(code("sim.close()"))

    # ---- name and save (requirements 4 and 5) ---------------------------
    cells.append(md(nbtext.SAVE_MD))
    cells.append(code(
        nbtext.SAVE_CODE
            .replace("__DEFAULT_NAME__", nbfile)
            .replace("__SOURCE_PATH__", f"notebooks/{nbfile}"),
        **{"tags": ["interactive"]}))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3", "file_extension": ".py",
                              "mimetype": "text/x-python", "nbconvert_exporter": "python",
                              "pygments_lexer": "ipython3"},
            "posim": {"pairs_with": source, "category": spec["category"]},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

REQUIRED = ["key","title","source","category","howtorun","abstract","situation",
            "eom","constraints","reduction","steps","discussion"]

def main():
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("notebooks")
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if not spec.get(k)]
    if missing:
        sys.exit(f"{sys.argv[1]}: spec is missing {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{spec['key']}.ipynb"
    dest.write_text(json.dumps(build(spec), indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    print(dest)

if __name__ == "__main__":
    main()
