# Export Provenance & Verification Report

*What this repository is, exactly where every file came from, what was
deliberately left behind, what was deliberately changed, and the
evidence that the result works. Written so a reviewer can re-check every
claim rather than trust it.*

**Export date:** 2026-07-26
**Toolchain:** `rustc 1.96.1 (31fca3adb 2026-06-26)`

---

## 1. Lineage

| item | value |
|---|---|
| source repository | `/home/youruser/Developer/code/rust/realtime_orbit` |
| source commit | `bc6a2a38b6ff9e36307cdf8c1d5defeab33000ff` (`bc6a2a3`) |
| that commit as published | `0406926b2b2a5ced1e81fb78acee111d4fd501a4` on `once-ere/SolveIt_rust` `main` |
| source subtree exported | `physical_object_simulator/` (91 files) + `sundials_rs/` (394 files) |
| `sundials_rs` subtree hash | `f616de8242c967fe6db67f8b0b52a55461587b0f` |
| files exported | **485** |
| repository size | 13 MB (excluding `target/`) |

The `sundials_rs` engine is itself a vendored copy of
`once-ere/sundials_rs@faabb7f` less five artifact files (two macOS
`.DS_Store`, three generated `*_stats.csv`) that its own `.gitignore`
classifies as build output. Its 394 remaining files are unchanged here —
see §4.

The export was taken with `git archive` from the source commit, so
**only tracked files moved**: untracked scratch, build output and
`.gitignore`d material could not leak in by construction.

---

## 2. What changed: the flatten

In the source, the simulator lived in a `physical_object_simulator/`
subdirectory beside `sundials_rs/`. Here the cargo workspace **is** the
repository root, so `git clone && cargo run` works with no subdirectory
hop:

```
rustSimulate/
├── Cargo.toml           workspace root (members: physical_object, posim)
├── physical_object/     the library      posim/          the front end
├── dynamic_notebooks/   23 notebooks     scripts/collisions/  12 scripts
├── jupyter/             wrapper kernel   sundials_rs/    the numerical engine
└── docs: README, CLAUDE, ARCHITECTURE, PLAN, NOTEBOOKS, grammar,
          physical_object_simulator, scene_info, collision_detection,
          box_of_shapes_m32  (.md, and .tex/.pdf where they exist)
```

---

## 3. What was excluded, and why

| excluded | reason |
|---|---|
| `sundials-7.7.0/` (61 MB of C) | third-party reference, not needed to build, test or run — `sundials_rs` is the completed, verified pure-Rust translation. Decided with the user. |
| `src/main.rs`, `src/solver.rs`, `RigidBody.rs`, `RigidBody3D.rs`, `obsolete_or_old/` | the deprecated donor types the union struct replaced. Their mapping into `physical_object` is recorded in `PLAN.md`. |
| root `Cargo.toml` / `Cargo.lock` (nalgebra + eframe) | the legacy `realtime_orbit` GUI app — it depends on crates.io, which this project forbids. |
| `FIX_REPORT.md`, `New_SolveIt_rust_Repo.md`, the old root `README.md` | documents about the *previous repository's* history, not about the simulator. |
| `prompt*.txt`, `output-*.txt`, the Gmail PDF | user scratch and personal reference material. |
| `target/`, `.backups/`, `jupyter/.venv`, `jupyter/.kernels` | build output and local scratch. |

**Deliberately kept:** all seven `sundials_rs` crates. Only three
(`sundials_core`, `cvode_rs`, `arkode_rs`) are used today; the other
four (`cvodes_rs`, `ida_rs`, `idas_rs`, `kinsol_rs`) are members of a
verified vendored workspace. Dropping them would mean editing
`sundials_rs/Cargo.toml` and forfeiting the byte-identity guarantee in
§4 — a bad trade for ~5 MB.

---

## 4. Byte-identity proof

Every exported file was SHA-256'd and compared against its source
counterpart (`sundials_rs/X` → source `sundials_rs/X`; everything else
→ source `physical_object_simulator/X`):

```
TOTAL=485   IDENTICAL=474   DIFFERENT=11   NO_SOURCE=0
sundials_rs: 394 files, 0 differ
```

**Zero files** appeared without a source counterpart, and the numerical
engine is untouched. The 11 differing files are exactly the ones the
re-layout required — no file changed by accident:

| file | ± lines | why it had to change |
|---|---|---|
| `physical_object/Cargo.toml` | +3 −3 | path deps `../../sundials_rs/…` → `../sundials_rs/…` |
| `Cargo.toml` | +5 −0 | added `exclude = ["sundials_rs"]` — see §5 defect 1 |
| `.gitignore` | +22 −5 | merged, LaTeX rules anchored — see §5 defect 2 |
| `README.md` | +22 −6 | retitled for this repo; clone/build quick-start; `sundials_rs/` listed as a component |
| `CLAUDE.md` | +16 −9 | layout paths; rule 5 repointed at the upstream SUNDIALS release (C tree not vendored); donor paths removed |
| `physical_object_simulator.md` | +6 −5 | layout tree root → `rustSimulate/` + `sundials_rs/` entry; `cd` instruction; stale "52 tests" → 104 |
| `physical_object_simulator.tex` | +4 −3 | same edits, kept in lockstep with the `.md` |
| `physical_object_simulator.pdf` | binary | recompiled from the edited `.tex` (`pdflatex` ×2, 19 pages) |
| `box_of_shapes_m32.md` | +4 −4 | clone URL → `rustSimulate`; `cd` instruction |
| `NOTEBOOKS.md` | +2 −1 | added "(the repository this project was exported from)" — see below |
| `jupyter/posim_kernel/kernel.py` | +1 −1 | error-message text named the old directory |

`NOTEBOOKS.md` still cites `once-ere/SolveIt_rust` in one place. That is
a **historical verification record** — those notebooks genuinely were
re-executed in a clone of that repository. Rewriting the URL would
falsify provenance, so the statement stands and a clarifying clause was
added instead.

---

## 5. Two defects found *in the export itself*, and fixed

Both were caught by verification, not by inspection. They are recorded
because an export that hides its own near-misses is not a provenance
report.

**Defect 1 — the flatten silently downgraded the engine's version.**
Moving `sundials_rs/` from a *sibling* to a *child* of the workspace
root made cargo absorb its crates into the rustSimulate workspace. Their
`version.workspace = true` then inherited `0.1.0` from this repo instead
of `7.7.0` from their own workspace — the crates would have misreported
which SUNDIALS they port, and `sundials_rs/Cargo.toml`'s `[workspace]`
was being ignored entirely. Caught by reading the build output
(`Compiling sundials_core v0.1.0`). Fixed with `exclude =
["sundials_rs"]` in the root `Cargo.toml`; the build now reports
`v7.7.0` and `Cargo.lock` records it.

**Defect 2 — an ignore rule silently dropped 77 reference files.**
A newly written `.gitignore` used unanchored `*.out` for LaTeX
intermediates, which also matched `sundials_rs/localref/*.out` — the
byte-exact C reference outputs the example-verification harness compares
against. Only 408 of 485 files staged. Caught by comparing the staged
count against the expected file count. Fixed by anchoring the LaTeX
rules to the repository root (`/*.out`); all 485 files are now tracked
and 77 `localref` files are present.

---

## 6. Verification

Every gate below was executed in this repository. Commands and their
real output; nothing is asserted that was not run.

### Build — zero errors, zero warnings

```
Compiling sundials_core v7.7.0 (…/rustSimulate/sundials_rs/crates/sundials_core)
Compiling arkode_rs v7.7.0     (…/rustSimulate/sundials_rs/crates/arkode_rs)
Compiling cvode_rs v7.7.0      (…/rustSimulate/sundials_rs/crates/cvode_rs)
Compiling physical_object v0.1.0 (…/rustSimulate/physical_object)
Compiling posim v0.1.0           (…/rustSimulate/posim)
Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.77s
warnings+errors = 0
```

### Tests — 104 passed, 0 failed

```
running 40 tests   ok. 40 passed   (library)
running 16 tests   ok. 16 passed   (collision)
running  9 tests   ok.  9 passed   (conservation)
running 39 tests   ok. 39 passed   (posim)
TOTAL: 104 passed, 0 failures
```

### The six self-checking physics examples — 6/6 SUCCESS, exit 0

```
kepler_orbit               SUCCESS: all conservation checks passed
outer_solar_system         SUCCESS: energy drift within 1e-6
tumbling_body              SUCCESS: rigid-body invariants conserved
charged_in_b_field         SUCCESS: gyration matches analytic solution
newtons_cradle             SUCCESS: impulse propagated through the chain, momentum conserved
bouncing_ball_restitution  SUCCESS: rebound apex = e^2 x drop height, impact at the exact TOI
```

### Collision scripts — 12/12

All of `scripts/collisions/01_head_on_exchange…12_two_dumbbells` exit 0
with zero `Err[` lines and no failing-cell message.

### Scenario baselines — exact

| notebook | energy | momentum |
|---|---|---|
| `box_of_shapes.posim` | `30000` | `[100, 200, 100]` |
| `box_of_shapes_m32.posim` | `960000` | `[3200, 6400, 3200]` |

23 dynamic notebooks present.

### REPL smoke test

```
In[3]:= step 1        →  t = 1 (advanced by 1, 12 solver steps)
In[4]:= get obj0.position →  [1, 5.095000000000006, 0]
```

The analytic value is `y = 10 − 9.81/2 = 5.095`.

### Scene window — live browser attached

```
scene: http://127.0.0.1:7900/  (1 window(s) connected)
mode = paused, t = 0.6050000000000004, dt = 0.005, steps = 121, history = 121 frame(s)
entities = 2 (hidden: none)
camera: yaw = -60°, pitch = 55°, dist = 12, target = [0, 0, 0]
```

The window's conserved-quantity readout showed `E = -0.50000003`,
`P = [0, 2, 0]`, `L = [0, 0, 0.8]` for the two-body Kepler setup, with
the full toolbar (Start, Pause, Stop, Reverse, Reset, step ±, dt/Set,
zoom, View, Grid, Trails, Labels, Contacts, ?) rendering correctly.

### Jupyter bridge

`python3 jupyter/test_protocol.py` → `all protocol checks passed`, exit 0
(stdlib only; covers the SCENE command family and the `events` op).

The full ZMQ kernel test — the same machinery JupyterLab itself uses —
passes all seven cells:

```
posim Jupyter kernel end-to-end test (jupyter_client over ZMQ)
  ok   cell 1: NEW                       ok   cell 5: ENERGY observable
  ok   cell 2: multi-line SET + STEP     ok   cell 6: multi-line DEF defines and calls
  ok   cell 3: GET analytic y            ok   cell 7: named path after DEF call
  ok   cell 4: error surfaces
all kernel checks passed — JupyterLab can drive this kernel
```

### Invariants

| check | result |
|---|---|
| `Cargo.lock` contents | exactly 5 local crates: `sundials_core` `cvode_rs` `arkode_rs` (7.7.0), `physical_object` `posim` (0.1.0) — proof of zero external dependencies |
| `unsafe` outside `#![forbid(unsafe_code)]` | none (only doc comments that mention the word) |
| `git ls-files -ci --exclude-standard` | empty — nothing ignored was committed |
| tracked files | 485, including all 77 `sundials_rs/localref` reference outputs |

---

## 7. Known limitations carried over

These are properties of the simulator, not of the export, and are
restated here so the move does not quietly bury them.

1. **Output granularity changes the trajectory.** With several
   collisions inside one solver output interval, energy conservation
   collapses: on `box_of_shapes_m32`, `run 1 steps 1000` (interval
   0.001) gives `|dE/E| = 6.9e-8`, while `run 1 steps 8` (interval
   0.125) gives `2.3e-1` — 23 % of the energy gone. The same failure
   hits the scene playback at large `dt` (at `dt = 0.1`, E falls
   30000 → 78.66 while the window reports `mode = running`). The rule of
   thumb was to keep the output interval below the mean free time
   between collisions; full measurements in `box_of_shapes_m32.md`.
   **Repaired in Stage 2C.** The cause was the Zeno guard counting
   events per *output interval*, so ordinary elastic collisions were
   forced plastic when fewer snapshots were asked for. It now counts a
   time-local burst, and the coarse run conserves energy to 1.0e-7
   while resolving 9160 collisions instead of 898.
2. **Parallel face-on disk–disk crossings are invisible to
   rootfinding** (`|dz|` touches zero without a sign change). Pinned
   deliberately, with a test asserting the documented behaviour; tilt a
   disk or model a thin cylinder to work around it.

---

## 8. Re-checking this report

```bash
git clone https://github.com/once-ere/rustSimulate.git
cd rustSimulate
cargo build --workspace          # expect zero warnings
cargo test --workspace           # expect 568 passed
cargo run -p physical_object --release --example kepler_orbit   # expect SUCCESS
python3 jupyter/test_protocol.py
git ls-files -ci --exclude-standard    # expect no output
grep -E '^name = ' Cargo.lock          # expect exactly the 5 local crates
```
