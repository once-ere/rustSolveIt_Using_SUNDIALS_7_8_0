# rustSolveIt — on pure-Rust SUNDIALS 7.8.0

Pure-Rust physics simulator. This repository is
[`once-ere/rustSolveIt`](https://github.com/once-ere/rustSolveIt) with
its numerical engine upgraded from a pure-Rust translation of SUNDIALS
**7.7.0** to one of **7.8.0** — and with the evidence that the upgrade
changed no physics: the six self-checking examples, the twelve
collision scripts and all 59 dynamic notebooks produce **byte-identical
output** under both engines. See
[PORT_7.8.0_PROVENANCE.md](PORT_7.8.0_PROVENANCE.md) and
[evidence/port-7.8.0/](evidence/port-7.8.0).

rustSolveIt itself is the refined and refactored export of
[`once-ere/rustSimulate`](https://github.com/once-ere/rustSimulate)
(full history preserved; the refinement pass and its evidence are
recorded in [REFINE_PROVENANCE.md](REFINE_PROVENANCE.md)).



Based on a Claude-Fable-5/Opus-5 port of my
2002 'SolveIt' code (c++ v.14, c, fortran, x86 assembly codes)
anything that could not be ported to pure rust has been omitted.
Numerical integration runs through
[`sundials_rs/`](sundials_rs) — a pure-Rust translation of SUNDIALS
7.8.0 vendored in this repository (CVODE Adams/BDF, ARKODE symplectic
SPRK). Zero `unsafe`, zero external crate dependencies, zero warnings.

The repository is **self-contained**: an ordinary clone is all you need
— no submodules, no network access during the build, nothing from
crates.io.

```bash
git clone https://github.com/once-ere/rustSolveIt_Using_SUNDIALS_7_8_0.git
cd rustSolveIt_Using_SUNDIALS_7_8_0/version-7.8.0
cargo run                 # the notebook REPL (type HELP)
```

This project is a standalone export of the simulator; its lineage,
byte-identity manifest and full verification transcript are recorded in
[EXPORT_PROVENANCE.md](EXPORT_PROVENANCE.md).

Latest release: a programmable notebook and a compound rigid body —
user-defined functions (`DEF name(param = default, ...) { body }`,
every body line syntax-checked at definition, `FUNCS`/`SHOW` to list
and edit), named objects (`NEW ... AS name`, plus `LET` variables and
string literals), and the rigid `DUMBBELL` (two solid spheres plus a
rod as ONE rigid body, exact part-wise collisions conserving E, P and
L through real solver events); the scene window gains a permanent
Reset button (with `SCENE RESET` — bit-identical re-initialization,
Start re-runs) and a live labeled conserved-quantities readout (E, P
and L); 605 passed workspace-wide (40 physical_object lib +
19 collision + 9 conservation + 109 posim + 92 quantum +
233 special_functions + 11 vendored identities + 55 doctests).

- `physical_object/` — library: `pub struct physical_object`, the
  unique union of the legacy `PointParticle`, `RigidBody` and
  `RigidBody3D`, with get/set for every field; `PhysicalObjectSystem`;
  the sundials integration drivers; validated examples.
- `posim/` — the simulator front end: lexer → grammar compiler → stack
  machine, a notebook REPL (`In[n]`/`Out[n]` cells), script batch mode,
  and a JSON machine mode.
- `gui/` — thirteen live GUI web pages, one per recorded video scene:
  each a standard-library Python server driving the engine over the
  machine protocol plus a canvas page with Start/Pause/Reset and live
  readouts verifying that scene's closed forms (`gui/README.md`).
- `notebooks/` — one executed Jupyter notebook per example — 109 in
  all, one per video scene, Rust example, collision script, SolveIt
  script and dynamic notebook. Each is a stand-alone Python notebook
  that starts the simulator in machine mode, explains every command it
  sends, derives the equations of motion and constraint equations, and
  carries real outputs (`notebooks/README.md`).
- `jupyter/` — JupyterLab wrapper kernel so notebooks can get/set the
  simulator's data (see `jupyter/README.md`).
- `sundials_rs/` — the numerical engine: a pure-Rust, zero-`unsafe`,
  dependency-free translation of SUNDIALS 7.8.0, vendored here as a
  self-contained workspace (read-only; upstream any changes).
- `PLAN.md` — the integration plan / design record (union mapping,
  grammar, solver mapping, verification results).

## Six solver families, four questions

| you type | question | solver |
|---|---|---|
| `STEP` / `RUN` | what happens next? | CVODE / ARKODE |
| `CONSTRAIN` / `GEAR` / `RACK` / `BALL` / `HINGE` / `UNIVERSAL` / `PRISMATIC` + `METHOD IDA` + `RUN` | …with this geometry held **exactly** | IDA |
| `EQUILIBRIUM` | where does it come to **rest**? | KINSOL |
| `SENSITIVITY 3 "gravity.y"` | how much does the answer **depend** on an input? | CVODES / IDAS |

A rod is a geometric fact, not a stiff spring, so `CONSTRAIN` turns the
equations of motion into a differential-algebraic equation. Over a full
pendulum period IDA holds the rod to `1.0000000000000082` — one bit —
and the bob closes to 3.7e-10. `EQUILIBRIUM` hangs the same pendulum
straight down to 13 digits. `SENSITIVITY` returns `∂y/∂g = T²/2` on free
fall to 1.3 parts in 10⁸, and **exactly zero** for `∂y/∂mass`, because
in uniform gravity there is no dependence to find.

See grammar.md §5.13 for the commands and SolveIt.md examples 17–18 for
the worked cases.

## Documentation

- **[SolveIt.md](SolveIt.md) / [SolveIt.pdf](SolveIt.pdf)** — the
  complete solution guide, written for a reader who has never used the
  program, with **sixteen** fully documented worked examples (scripts in
  [`scripts/solveit/`](scripts/solveit)).
- **[grammar.md](grammar.md) / [grammar.pdf](grammar.pdf)** — the
  complete command-language and notebook specification: the lexer, the
  full EBNF, the type system, every command, the stack machine, the
  SUNDIALS 7.8.0 engine, the browser videos, and **eighteen** worked
  examples.
- [PORT_7.8.0_PROVENANCE.md](PORT_7.8.0_PROVENANCE.md) — the 7.8.0
  upgrade: the API delta call by call, and the evidence that nothing
  else moved.
- [physical_object_simulator.md](physical_object_simulator.md) /
  [physical_object_simulator.pdf](physical_object_simulator.pdf) — the
  predecessor user guide, with fourteen more worked examples.
- [scene_info.md](scene_info.md) / [scene_info.pdf](scene_info.pdf) —
  the graphical scene window: the simulator research survey, the
  protocol, and the UI.
- [collision_detection.md](collision_detection.md) /
  [collision_detection.pdf](collision_detection.pdf) — the collision
  science reference, with documented example scripts in
  `scripts/collisions/` (01–12).
- [ARCHITECTURE.md](ARCHITECTURE.md) — module responsibilities and
  pinned cross-module contracts (solver driving §3.3, the video
  recorder §7).
- [CLAUDE.md](CLAUDE.md) — working rules for contributors and agents.

## Browser videos

Thirteen recorded runs you can open offline — no server, no CDN, nothing
fetched. Scrub, orbit, and read the conserved quantities off whichever
frame you stopped on.

| video | what to watch | measured over the recording |
|---|---|---|
| [videos/kepler_ellipse.html](videos/kepler_ellipse.html) | the speed swinging between perihelion and aphelion on an `e = 0.6` ellipse | \|dE\|/E = 9.8e-8, \|dL\|/\|L\| = 1.3e-7 |
| [videos/tumbling_racket.html](videos/tumbling_racket.html) | the Dzhanibekov flip: a torque-free cuboid spun about its **intermediate** axis turns over, and over | \|d\|L\|\|/\|L\| = **0 exactly**; \|dE\|/E = 6.4e-9 |
| [videos/box_of_shapes.html](videos/box_of_shapes.html) | a cylinder, a disk and a cuboid rattling in a rigid `BOX 4`; gold arrows are the analytic contact normals, sized by impulse | 36 collision events, \|dE\|/E = 3.4e-16 |
| [videos/double_pendulum_hinges.html](videos/double_pendulum_hinges.html) | **two `HINGE` joints** made into the chaotic double pendulum; gold rings are the joints | the joints hold to \|g\| = 5.6e-8 |
| [videos/universal_joint.html](videos/universal_joint.html) | a **`UNIVERSAL` joint** carrying a driven shaft's rotation to a second shaft; the bend flattens straight and folds back, and the speed across the joint swings with it | the bend stops at cos b = 0.6000004 against a geometric bound of exactly 0.6; joints hold to \|g\| = 4.0e-7 |
| [videos/ball_joint_chain.html](videos/ball_joint_chain.html) | four links on **`BALL` joints**, whirling as they collapse out of the plane they started on | joints hold to \|g\| = 3.3e-9; \|z\| runs from exactly 0 to 1.7147 |
| [videos/rod_pendulum_chain.html](videos/rod_pendulum_chain.html) | four bobs on four **`CONSTRAIN` rods** — one row each, the cheapest linkage there is — going chaotic | run continuously at the default tolerance, \|g\| = 5.4e-15 |
| [videos/spinning_top.html](videos/spinning_top.html) | a top held at its tip by a **`BALL` joint**, precessing under gravity | 1.020440 rad/s vs the exact 1.020408 — 3 parts in 10⁵ |
| [videos/gyroscope_gimbal.html](videos/gyroscope_gimbal.html) | a rotor in **two gimbal rings** on three perpendicular `HINGE` axes; push it one way, it moves another | `L·ŷ` conserved to 1.4e-14 |
| [videos/cardan_compass.html](videos/cardan_compass.html) | the same two rings with a **pendulous** bowl: a ship's compass held level by its own weight | two physical-pendulum periods matched, 1.878587 and 2.307339 s |
| [videos/cardan_gear.html](videos/cardan_gear.html) | **Cardan gears**: a wheel inside a ring of twice its radius rolling on a `GEAR` row, rim point on a straight line | line held to 1.1e-8, against 1.8e-3 with the ratio merely imposed |
| [videos/rack_and_pinion.html](videos/rack_and_pinion.html) | a weight on a **`RACK`** winding up a flywheel, guided by a **`PRISMATIC`** | falls at exactly `g/2`, independent of the pitch radius |
| [videos/piston_crankshaft.html](videos/piston_crankshaft.html) | the **slider-crank**: piston, connecting rod and crankshaft, free-running | follows the exact kinematics to 8.4e-8 |

Record your own from any posim script:

```bash
cargo build --release -p posim
recorder/src/record_video.py videos/scenes/kepler_ellipse.posim \
     -o /tmp/mine.html --frames 360 --dt 0.02 --title "..."
```

Every advance in a recording is a real SUNDIALS step — the recorder
drives `posim --machine` and asks it to `step`. It is a camera, not a
physics engine.

## Some Beautiful Examples from
Routh Part1:
A TREATISE ON
DYNAMICS OF A PARTICLE
WITH NUMEROUS EXAMPLES
BY
EDWARD JOHN ROUTH, Sc.D., LL.D., M.A., F.R.S., &c.,
HON. FELLOW OF PETERHOUSE, CAMBRIDGE;
FELLOW OF THE UNIVERSITY OF LONDON.
CAMBRIDGE
AT THE UNIVERSITY PRESS
1898

Routh Part2:
THE ADVANCED PART
OF A TREATISE ON THE
DYNAMICS OF A SYSTEM OF
RIGID BODIES.
BEING PART II. OF A TREATISE ON THE WHOLE
SUBJECT.
WITH NUMEROUS EXAMPLES.
BY
EDWARD JOHN ROUTH, Sc.D., LL.D., F.R.S., &c.
HON. FELLOW OF PETERHOUSE, CAMBRIDGE;
FELLOW OF THE SENATE OF THE UNIVERSITY OF LONDON.
SIXTH EDITION, REVISED AND ENLARGED.
London:
MACMILLAN AND CO., LIMITED
NEW YORK: THE MACMILLAN COMPANY
1905

Thirty-four of Routh's problems are solved here as runnable notebooks —
seventeen from each Part (a first pair, then sixteen more from each book).
Every one derives the closed-form answer in its own header and
then checks it against the integrator, so the numbers below are measured, not
quoted. Launch any of them by name:

```bash
tools/posim_notebook routh_p1_hodograph_circle
```

### Part I — *Dynamics of a Particle* (1898)

| notebook | Arts. | what it shows | measured |
|---|---|---|---|
| `routh_p1_sphere_exchange` | 85, 87 | equal elastic spheres exchange velocities; `m = e·m'` stops the striker dead | `v = 0` exactly, momentum 1 |
| `routh_p1_geometric_progression` | 88 | masses in geometric progression give velocities in geometric progression | 1, 2/3, 4/9, 8/27; `\|dE/E\| = 1.1e-16` |
| `routh_p1_oblique_impact` | 89 | equal spheres leave an oblique impact at a right angle | `v₀·v₁ = -8.3e-17` |
| `routh_p1_centre_of_gravity` | 92 | impacts cannot move the centre of gravity | 5.0e-15 from `P/M·t` through 8 impacts |
| `routh_p1_three_projectiles` | 158 | the plane through three projectiles stays parallel to itself | normal fixed to 1e-15 while the triangle grows 4× |
| `routh_p1_parabola_of_safety` | 159–160 | two trajectories reach any point inside the envelope | both arcs within 3.6e-15; every clearance positive |
| `routh_p1_expanding_sphere` | 167 | equal speeds in all directions give an expanding sphere | radii 5, 5, 5 then 10, 10, 10 |
| `routh_p1_escape_velocity` | 312, 335 | the velocity from infinity, and the three conics | `E` = −0.19 / **exactly 0** / +0.21 |
| `routh_p1_equal_periods` | 335 | speed alone fixes the orbit's size, hence its period | four eccentricities 0.87→0.21, one period, all home within 1e-9 |
| `routh_p1_two_trajectories` | 339 | two orbits of one speed reach the same target | 1.3e-6 and 9.0e-7 |
| `routh_p1_kepler_equation` | 342–346 | Kepler's equation and the equation of the centre | radius predicted to 1e-11 |
| `routh_p1_lambert_theorem` | 350–355 | the time depends only on chord, `r₁+r₂` and `a` | two dissimilar ellipses, both 1.3453158122479596 |
| `routh_p1_hodograph_circle` | 394–398 | the hodograph of a Kepler orbit is a circle | radius 1.25 while the speed swings 2 → 0.5 |
| `routh_double_star_period` | 400 | a double star's period depends only on the **sum** of the masses | ratios 1:1, 3:1, 19:1 all close to ~1e-11 |
| `routh_p1_equilateral_three_body` | 407–408, 412 | Lagrange's equilateral solution, and why it cannot hold | home to 3e-10; nudged 0.001 → sides 0.405 / 1.802 / 1.409 |
| `routh_p1_collinear_three_body` | 409–412 | Euler's collinear solution, and its instability | home to 6e-11; nudged → spacings 3.44 / 0.34 |
| `routh_p1_apsidal_symmetry` | 419–420 | an apsidal radius divides the orbit symmetrically | `r(1) = r(2π−1)` to 3e-9; apses at exactly 0.4 and 1.6 |

### Part II — *Advanced Rigid Dynamics* (1905)

| notebook | Arts. | what it shows | measured |
|---|---|---|---|
| `routh_p2_two_quadrics` | 140–142 | the two quadrics whose intersection is the polhode | `G` bit-identical, `E` to 12 digits, `\|ω\|` wanders 3.017 → 3.407 |
| `routh_p2_invariable_line` | 141 | the invariable line is fixed in space; the instantaneous axis is not | `normalize(L)` bit-identical; `normalize(ω)` swings |
| `routh_p2_poinsot_rolling` | 143 | the momental ellipsoid rolls on a fixed plane | `ω·L̂ = 2E/G` constant to 12 digits |
| `routh_p2_thin_rod` | 144 | a thin rod's momental ellipsoid is a circular cylinder | a 50× axial spin carries 0.0075 of `L` |
| `routh_p2_impulsive_couple` | 146 | an impulsive couple tilts the invariable line | `\|L'\| = √(G²+25)`; tilt 21.2785° |
| `routh_p2_fixed_couple` | 148 | a couple whose axis is fixed in space | `L(t) = (0,0,2t)` **exactly** |
| `routh_p2_polhode_quarter_period` | 150a | the quarter period as an elliptic integral | `k² = 0.25` exactly; `ω₃` vanishes at `K/λ` to 6e-12 |
| `routh_rectangle_diagonal` | 150b Ex. 1 | a rectangle spun about one diagonal ends on the other | at `T = 4.285129705594264`, error 1.2e-12 |
| `routh_p2_period_ratio` | 150b Ex. 2 | the ratio of the two periods is independent of the disturbance | sixfold different wobbles, identical period |
| `routh_p2_mean_axis_instability` | 155 | the three principal axes have unequal stability | mean axis reverses; the others stay within 0.005 and 0.018 |
| `routh_p2_spin_stabilisation` | 156 | rapid rotation steadies a body | one kick at n = 1, 3, 9 → 20.171°, 8.326°, 2.854° |
| `routh_p2_rolling_cones` | 157–159 | the body and space cones roll on one another | both half-angles constant to 13 digits |
| `routh_p2_principal_axes_in_space` | 176–179 | `cos α = Aω₁/G` — the invariable line seen from the body | fixed in space, sweeping `(1.5,12.75,0.125)` → `(11.52,−4.90,2.86)` in the body |
| `routh_p2_uniaxal_precession` | 180–183 | steady precession when `A = B` — two periods at once | space 4.683, body 3.491; `\|ω\|` constant |
| `routh_p2_separatrix` | 184–185 | `G² = BT` decides which axis the polhode encircles | +0.3 / 0 / −0.337 pins `ω_x`, nothing, `ω_z` |
| `routh_p2_correlated_bodies` | 192–195 | confocal ellipsoids of gyration | `ω − ω' = kL` exact at correspondence, with the drift measured |
| `routh_p2_sylvester_time` | 196–198 | the clock Poinsot's rolling cannot supply | rate `kG = 1.283854061020956` |

Provenance and the full run record for every notebook — including the other 27
in the directory — are in
[dynamic_notebooks/MANIFEST.md](dynamic_notebooks/MANIFEST.md); the reader's
catalogue is [dynamic_notebooks/README.md](dynamic_notebooks/README.md).

## The Index of Functions

`index_of_entities.html` is a browsable catalog of **every named entity in this
repository** — 6,309 of them — with a definition, the `file:line` where it is
defined, its complete syntax, and examples you can paste into a notebook and
run. Open it directly; it needs no server and fetches nothing.

```bash
open index_of_entities.html          # macOS  (xdg-open on Linux)
```

Keep `catalog-c.js` beside it: the page loads that second payload on demand
when you open a bucket or search, which is what keeps the main file at 1.7 MB
instead of 6.

| tier | what | entries | examples |
|---|---|---|---|
| A | the notebook surface — commands, keywords, field paths, builtins, types, notebooks, worked examples | 452 | 1,141 posim + 24 machine-mode fragments, **all executed** |
| B | the first-party Rust API — `physical_object`, `special_functions`, `quantum`, `posim` | 493 | 258 snippets, **all compiled** |
| C | the vendored `sundials_rs` workspace (SUNDIALS 7.8.0: CVODE, CVODES, IDA, IDAS, KINSOL, ARKODE, core) | 5,331 | linked to the shipped example programs that call them |

**Every example is checked, and the page says which kind of check it got:**
a posim fragment is *executed* (`posim --script`, output captured), a
machine-mode fragment is *executed* through the JSONL protocol, a Rust snippet
is *compiled*, and a shell command reads "run this yourself" because it points
at a program verified elsewhere rather than one this page ran. 1,435 of 1,435
runnable examples pass.

The 68 documented worked examples now carry their real transcripts. Where the
typed half replays, it is **executed** and shown beside the document's own
published output so you can compare; where it cannot — a deliberate refusal
that *is* the lesson, an elided session, or a `SCENE REVERSE` that depends on
wall-clock playback — the transcript is quoted and the reason is stated.

Navigation is A–Z / 0–9 / special-character buckets, with search and kind
filters. The whole thing is operable from the keyboard — press <kbd>?</kbd> for
the map — and has BACK/FORWARD history that agrees with the browser's own
buttons. You can edit, add and hide entries; changes live in a local overlay
you can export and re-import, and the generated catalog is never mutated.

### What it does not claim

The status page inside the app is the authority, and it is blunt:

- **There are no stubs left.** Every Tier-A and Tier-B entry is either
  `complete` (it carries examples) or `reference` — 235 of the latter, all in
  `posim`, which is a binary crate with no lib target, so nothing can
  `use posim::…` and no Rust snippet is possible. Those link instead to the
  notebook command each one implements.
- **5,331 Tier-C entries carry status `reference`**, not `complete`.
  `sundials_rs` is a faithful translation of a C library whose API is
  `&mut CVodeMem` plus a context, a matrix, a linear solver and callbacks; a
  one-line snippet would misrepresent how any of it is reached. Instead each
  entry links to the shipped example *programs* that actually call it — those
  are diffed byte-for-byte against the upstream C references
  ([sundials_rs/VERIFICATION.md](sundials_rs/VERIFICATION.md)).
- **All 254 Tier-A commands, builtins, field paths and types carry the full
  four-rung ladder.** Closing the last 24 meant giving the machine-mode JSON
  ops real executed examples (`posim --machine` is as runnable as the
  notebook), and giving each shape a rung that *checks* its documented inertia
  formula against the printed tensor rather than restating it.
- Captured output has genuine run-to-run variation normalised — the
  OS-assigned scene port becomes `<port>`, playback counters become
  `<varies>`. Solver step counts are **not** normalised: those are
  deterministic, and they are the anchors the documentation pins.

[index_data/DIVERGENCES.md](index_data/DIVERGENCES.md) records six places where
the prose and the code disagreed, each settled by running a probe rather than
by reading. One was a defect in code-adjacent material and has been fixed with
a test that now gates it in both directions; the rest are documentation drift,
and the index carries the code's behaviour.

### Rebuilding it

```bash
python3 tools/extract_rust_items.py > index_data/rust_items.jsonl
python3 tools/build_commands.py && python3 tools/build_tierb.py && python3 tools/build_tierc.py
python3 tools/build_catalog.py && python3 tools/build_app.py
python3 tools/verify_index_examples.py      # runs every posim fragment
python3 tools/verify_tierb_examples.py      # compiles every Rust snippet
```

## Quick start

```bash
cargo run                 # notebook REPL (type HELP)
cargo test --workspace    # all tests
cargo run -p physical_object --release --example kepler_orbit
cargo run -p physical_object --release --example outer_solar_system
cargo run -p physical_object --release --example tumbling_body
cargo run -p physical_object --release --example charged_in_b_field
cargo run -p posim -- --script scripts/collisions/12_two_dumbbells.posim
cargo run -p posim -- --script my_session.posim
cargo run -p posim --release -- --notebook dynamic_notebooks/kepler_orbit.posim
cargo run -p posim -- --machine   # JSON protocol for front ends
```

Example session:

```
In[1]:= new sphere { mass = 2, radius = 0.5, position = [0, 10, 0], velocity = [1, 0, 0] }
Out[1]= obj0
In[2]:= set system.gravity = [0, -9.81, 0]
In[3]:= step 1
Out[3]= t = 1 (advanced by 1, 12 solver steps)
In[4]:= get obj0.position
Out[4]= [1, 5.095000000000006, 0]
In[5]:= method sprk leapfrog_2_2 0.001
In[6]:= help
```
