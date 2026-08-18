# notebooks/ — one Jupyter notebook per example, 109 of them

Every example in this repository has exactly one Jupyter notebook, and
every notebook here pairs with exactly one example:

| prefix | count | paired with | example form |
|---|---|---|---|
| `video_*` | 13 | `videos/scenes/*.posim` | the scripts behind the recorded browser videos |
| `rust_*` | 6 | `physical_object/examples/*.rs` | the self-checking compiled Rust examples |
| `collision_*` | 12 | `scripts/collisions/*.posim` | the collision-detection walkthroughs |
| `solveit_*` | 19 | `scripts/solveit/*.posim` | the SolveIt worked examples |
| `dynamic_*` | 59 | `dynamic_notebooks/*.posim` | the dynamic notebooks, incl. the 34 Routh problems |

Each notebook is a **Python 3** notebook that starts the simulator as a
child process in machine mode (`posim --machine`) and drives it over
JSON Lines — the same wire protocol the `jupyter/` kernel uses. Each is
completely stand-alone: the launch instructions, the command-language
glossary, the second-order-to-first-order reduction, the physical
situation (objects, properties, interactions, equations of motion,
constraint equations, state-vector sizing), an explanation before every
code cell, and the name-and-save cells are all written out in full in
every notebook, on purpose. No notebook ever refers to another.

The committed notebooks carry **real outputs**: every code cell was
executed against the release build, and the `rust_*` six also run their
compiled example and assert its `SUCCESS` verdict.

## Running one

```bash
cargo build --release -p posim        # once
python3 -m pip install --user jupyterlab
jupyter lab notebooks/
```

Open any notebook and run the cells top to bottom. The final two cells
ask you to name the notebook and pick a folder in a save dialog; they
are interactive and are the only cells the batch runner skips.

## Regenerating and re-verifying all 109

```bash
cargo build --release -p posim
python3 notebooks/_build/regen.py          # specs -> .ipynb, all 109
ls notebooks/*.ipynb | POSIM_NO_BROWSER=1 xargs -P 6 -I{} \
    python3 notebooks/_build/nbrun.py {}   # execute, embed real outputs
python3 notebooks/_build/nbcheck.py        # audit the 7 requirements
```

`_build/` holds the machinery: `nbtext.py` (the invariant prose),
`lang.py` (what every command and field means), `nbgen.py` (derives a
spec from a `.posim` example), `nbbuild.py` (spec → notebook),
`nbrun.py` (executes code cells and embeds outputs), `nbcheck.py`
(audits every requirement), `specs/` (the 109 derived specs), and
`rust_equivalents/` (posim reproductions of the six compiled examples,
each verified against the same analytic anchors).
