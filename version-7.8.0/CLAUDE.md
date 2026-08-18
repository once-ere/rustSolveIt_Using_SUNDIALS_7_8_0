# CLAUDE.md — rustSolveIt on pure-Rust SUNDIALS 7.8.0

Pure-Rust physics simulator. The **only** numerical-integration backend
is the vendored `sundials_rs/` workspace (pure-Rust **SUNDIALS 7.8.0**)
— no exceptions. Treat `sundials_rs/` as a **read-only vendored
library**: it is byte-identical to
`once-ere/SUNDIALS_7_8_Rust_port_for_Linux@780b916` (2,929 files) and
that identity is a verifiable property — never edit it in place;
upstream it instead.

```bash
# the byte-identity claim, reproducible
git clone https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux.git
diff -rq --exclude=.git --exclude=target \
     SUNDIALS_7_8_Rust_port_for_Linux sundials_rs      # silent
```

This repository is `once-ere/rustSolveIt` with its 7.7.0 engine replaced
by the 7.8.0 one. What changed, and the evidence that nothing else did,
is in `PORT_7.8.0_PROVENANCE.md`. The original export's provenance —
including which donor sources were deliberately not carried over — is in
`EXPORT_PROVENANCE.md` and `PLAN.md`.

## Commands

- Build: `cargo build --workspace --all-targets 2>&1 | tee /tmp/build.log`
- Tests: `cargo test --workspace 2>&1 | tee /tmp/test.log` (605 expected)
- Notebook: `cargo run` (type `HELP`); batch: `cargo run -p posim -- --script <f>`
- Dynamic notebook (loads a file, opens its scene window, stays
  interactive): `cargo run -p posim --release -- --notebook
  dynamic_notebooks/<name>.posim` — see `dynamic_notebooks/README.md`
- Scene window: type `SCENE CREATE` in the notebook (opens a browser
  page; `SCENE START/PAUSE/REVERSE/RESET`, arrows/drag/wheel in the
  window). Headless runs: `POSIM_NO_BROWSER=1` suppresses `xdg-open`.
- Self-checking physics examples:
  `cargo run -p physical_object --release --example
  {kepler_orbit|outer_solar_system|tumbling_body|charged_in_b_field|newtons_cradle|bouncing_ball_restitution}`
  (each prints SUCCESS/FAILURE and exits nonzero on failure)
- Collision example scripts: `cargo run -p posim -- --script
  scripts/collisions/NN_name.posim` (01–12; documented with captured
  output in `collision_detection.md` §9)
- The sixteen SolveIt examples: `cargo run -p posim --release --
  --script scripts/solveit/NN_name.posim` (01–16; documented with
  captured output in `SolveIt.md` §7)
- Browser video: `cargo build --release -p posim` then
  `recorder/src/record_video.py videos/scenes/<x>.posim -o videos/<x>.html
  --frames N --dt DT --title "..." --caption "..."`
- Wire protocol test: `python3 jupyter/test_protocol.py` (needs
  `cargo build --release` first — it prefers `target/release/posim`,
  so a stale release binary silently shadows your debug build)
- Kernel test: `POSIM_NO_BROWSER=1 jupyter/.venv/bin/python jupyter/test_kernel.py`
  (needs `uv venv jupyter/.venv && uv pip install -p jupyter/.venv/bin/python ipykernel jupyter_client`)
- Docs: `pdflatex -interaction=nonstopmode <name>.tex` (run **twice** —
  the table of contents needs the second pass) for `SolveIt`,
  `grammar`, `physical_object_simulator`, `scene_info`,
  `collision_detection`. Keep `.md` and `.tex` in sync — **the `.md` is
  the source of truth.**

## Layout

- `physical_object/` — library: `constrain` (rigid rods + the constraint
  Jacobian), `equilibrium` (KINSOL rest states), `sensitivity` (CVODES /
  IDAS derivatives), `linalg` (Vec3/Mat3/Quat + skew/outer),
  `boundary` (enum + `Sdf` trait; **every round shape is symmetric
  about its local z axis**), `physical_object` (the union struct,
  get/set), `system` (collection + 13N pack/unpack), `integrate`
  (**all** time integration: CVODE Adams/BDF + ARKODE SPRK, with
  collision event rootfinding armed when collidable pairs exist),
  `collide` (contact geometry + impulse response; conventions pinned in
  ARCHITECTURE §3.8 — normal points i→j, separation > 0 apart and IS
  the root function).
- `posim/` — binary: `lexer` → `parser` (EBNF in its header) → `vm`
  (stack machine) → `notebook` (REPL) / `machine` (JSONL) / `scene`
  (graphical window: `mod.rs` server+playback, `ws.rs` hand-rolled
  SHA-1/base64/RFC 6455, `scene.html` embedded page); `qm`/`qm2`/`qm3`
  and `special` are the quantum and special-function command families.
- `quantum/`, `special_functions/`, `vendor/spec_math/` — the physics
  and mathematics libraries the language reaches.
- `jupyter/` — Python wrapper kernel (outside the Rust constraints);
  a reader thread streams async `{"event":...}` lines to the notebook;
  `.venv/` and `.kernels/` are gitignored scratch.
- `tools/` — the index pipeline, the example verifiers, and
  the video recorder in `recorder/` (also outside the Rust constraints).
- `videos/`, `videos/scenes/` — recorded browser videos and the posim
  scripts that produced them.
- `gui/` — thirteen live GUI web pages, one per recorded video scene:
  a stdlib-Python server owning a `posim --machine` child plus a
  vanilla-JS canvas page with live physics readouts (`gui/README.md`).
- `notebooks/` — one executed Jupyter notebook per example, 109 in all
  (13 video scenes, 6 Rust examples, 12 collision scripts, 19 SolveIt
  scripts, 59 dynamic notebooks). Python notebooks driving
  `posim --machine`; stand-alone by design — every explanation is
  repeated in full in every notebook. Machinery and re-verification
  loop in `notebooks/_build/` (see `notebooks/README.md`).
- `evidence/port-7.8.0/` — the logs behind every claim in
  `PORT_7.8.0_PROVENANCE.md`.
- `ARCHITECTURE.md` — pinned cross-module contracts (state layout,
  solver driving §3.7, setter invariants, wire protocol, scene
  subsystem, collision subsystem, the video recorder §7). **Read it
  before touching anything cross-module; update it when a contract
  moves.**
- `SolveIt.md`/`.pdf` — the complete solution guide (16 worked
  examples); `grammar.md`/`.pdf` — the command-language spec (18 more);
  `scene_info.md`/`.pdf`; `collision_detection.md`/`.pdf`;
  `physical_object_simulator.md`/`.pdf` (the predecessor guide, 14
  more).
- `.backups/` — pre-modification file backups (gitignored). Back up
  before modifying; this repo's git history is the real undo.

## Hard rules

1. **Sundials-only integration.** All stepping goes through
   `physical_object/src/integrate.rs` calling `cvode_rs`/`arkode_rs`.
   Never add a hand-rolled Euler/Verlet/RK stepper anywhere — including
   in examples, docs, tooling (`recorder/` drives `step`;
   it does not integrate), and the scene playback thread (it calls
   `integrate::step`; reverse is snapshot replay from the history ring,
   never negative-dt integration).
2. **Zero `unsafe`, zero external dependencies, zero warnings.** Every
   crate root carries `#![forbid(unsafe_code)]`, `#![deny(warnings)]`,
   and allows `non_snake_case`, `non_camel_case_types`,
   `non_upper_case_globals`. `Cargo.lock` must list **only local
   crates**; today: `arkode_rs 7.8.0`, `cvode_rs 7.8.0`,
   `sundials_core 7.8.0`, `physical_object 0.1.0`, `posim 0.1.0`,
   `quantum 0.1.0`, `spec_math 0.1.6`, `special_functions 0.1.0`. Any
   entry with a registry source means you broke the rule. This applies
   to networking too: the scene server's HTTP/WebSocket/SHA-1/base64 are
   hand-rolled on `std::net` in `posim/src/scene/ws.rs`, `scene.html`
   uses vanilla JS + canvas, and the recorded video pages fetch nothing.
3. **Fidelity to donor physics.** Formulas ported from the legacy types
   (softened gravity, Laplace vector incl. its `r = 0` guard, inertia
   formulas, kinetic energy, SDF central-difference normal) keep their
   arithmetic order — floating point is not associative.
4. **All state access through get/set.** The VM, machine mode, the
   scene subsystem, and any new front end reach `physical_object`
   fields only via the setters (they enforce the coupled invariants:
   mass↔inverse, inertia↔inverse, unit quaternions,
   momentum-canonical velocity).
5. **Missing sundials symbols are reported, not invented.** If an API
   you need is absent from `sundials_rs/`, stop and say exactly which
   symbol is missing, naming the C original's file (`src/` or
   `include/` of the upstream **SUNDIALS 7.8.0** release, at
   `/home/nsh/Developer/sundials-7.8.0/` — that reference tree is not
   vendored here). Do not reimplement solver numerics locally. The same
   applies to a `None` from a 7.8.0 constructor or from
   `N_VGetArrayPointer`: turn it into a named `Err`, never an `unwrap`.
6. **Quaternions are w-first everywhere** (packing, VM literals, JSON,
   scene frame messages, video frames, docs). The 13-per-object state
   layout in `system.rs` is a pinned contract (ARCHITECTURE §3.2).
7. Public error style: `Result<_, String>` with actionable messages
   naming the field/column/feature. No panics in library code paths;
   scene threads turn bad window input into error events, never crashes.
8. **The struct name is `physical_object`** (lower-case, by
   specification). Import as
   `use physical_object::physical_object::physical_object;`; in files
   that import it, prefix other crate paths with `::`. Do not rename it
   and do not try to re-export it at the crate root (namespace clash).
9. **The scene evolves a copy.** The playback thread owns a synced
   clone of the system (`SCENE CREATE`/`SCENE REFRESH`/`RESET` sync
   it); notebook `STEP`/`RUN` never move the window and window playback
   never moves the notebook. Do not "fix" this by sharing the system —
   it is the isolation that keeps the VM lock-free (ARCHITECTURE §3.7).

## Working with the 7.8.0 API

Read ARCHITECTURE §3.3 before touching `integrate.rs`. The four things
that will bite you:

- **Handles are shared, not mutable.** `CVodeMem` and `ARKodeMem` are
  `Rc<RefCell<…>>`; every entry point takes `&`, never `&mut`. If you
  reach for `&mut`, you are reading 7.7.0-era code.
- **Vector data comes with a borrow guard.**
  `N_VGetArrayPointer` returns a `RefMut`. Holding it across a solver
  call on the same vector panics. Use `with_data` / `with_data_mut`,
  which take the guard, do the work, and drop it.
- **Constructors return `Option`, destructors take ownership.**
  `SUNContext_Create(SUN_COMM_NULL, &mut opt) -> SUNErrCode`;
  `CVodeFree(&mut Some(mem))`; `SPRKStepCreate(f1, f2, …)` takes the
  two right-hand sides **by value**, not `Option`.
- **`fmt_e(x, prec)` is `%.*e`; `fmt_ew(x, width, prec)` is `%*.*e`.**
  7.7.0's three-argument `fmt_e` split in two. Never use Rust's `{:e}` —
  its exponent form differs from C's and breaks byte-identity with the
  reference outputs.

## Workflow

- **Make backups before modifying files** (copy into
  `.backups/<date>/`), then edit.
- After EVERY build/test/run command: `2>&1 | tee <log>`, then read the
  log before editing. Never re-run a command that returned no visible
  output. ≤2 attempts per failing command, then switch strategy.
- Commit after every coherent file group; keep
  `cargo build --workspace --all-targets` warning-free and
  `cargo test --workspace` green at every commit (**605 tests**:
  46 physical_object lib + 19 collision + 9 conservation +
  16 constrained/DAE + 111 posim + 92 quantum + 233 special_functions +
  11 vendored identities + 55 doctests).
- New solver features need: a unit or conservation test with an
  **analytic** expectation (not a golden-output snapshot), a grammar
  hook if user-facing (lexer keyword → parser production → VM
  instruction → `HELP_TEXT` → `grammar.md`/`.tex`), and an
  ARCHITECTURE.md update if a contract changed.
- When touching the language, update **all four** in lockstep: the EBNF
  comment in `parser.rs`, `HELP_TEXT` in `vm.rs`, `grammar.md`, and
  `grammar.tex` (then recompile the PDF). Scene-visible behavior also
  updates `scene_info.md`/`.tex`; user-visible behavior also updates
  `SolveIt.md`/`.tex`.
- Scene argument parsing is **term-level** (so `scene rotate 15 -5` is
  two arguments); if you change it, `parser::scene_command` and the
  grammar docs move together.
- **Documentation numbers are measured, not remembered.** Every figure
  quoted in `SolveIt.md`, `grammar.md` and
  `PORT_7.8.0_PROVENANCE.md` came out of a run captured in
  `evidence/port-7.8.0/`. If you change something that moves a number,
  re-run and re-paste; do not adjust the prose to fit.

## The constrained / equilibrium / sensitivity paths

Read ARCHITECTURE §3.9 before touching `constrain.rs`, `equilibrium.rs`
or `sensitivity.rs`. Four things that already cost a day:

- **The GGL projection is mass-weighted: `q̇ = v - M⁻¹Jᵀμ`.** Dropping
  `M⁻¹` is dimensionally wrong the moment a joint grips orientation
  (`J_ω` carries the attachment arm), and *invisible* for rods, where it
  is only a rescaling of μ. This cost a day; do not undo it.
- **`IDACalcIC` cannot find the multipliers and is not called.** The
  index-2 residual is satisfied by free fall with λ = 0, so there is
  nothing for it to solve. `seed_multipliers` solves `g̈ = 0` directly.
- **`g = |d| - L`, not `|d|² - L²`.** The squared form is better
  behaved algebraically and *worse* numerically: its gradient scales
  with `L`, and the index-2 corrector stops converging for `L ≠ 1`. Do
  not "simplify" it back.
- **GGL index-2, not index-1.** Both `g` and `ġ` are carried as
  algebraic equations. Dropping `μ` gives an index-1 system whose `g`
  drifts quadratically — and nothing fails loudly when it does.
- **Do not call `IDACalcIC` when the ICs are already consistent.** A
  bare `CONSTRAIN` guarantees they are, and `seed_multipliers` supplies
  the exact tension; asking IDA to re-derive it at trajectory tolerance
  fails with `IDA_CONV_FAIL`.
- **The sensitivity parameter vector is shared (`Rc<RefCell<Vec<f64>>>`),
  and the RHS must read every value out of it.** Capture a copy and the
  difference-quotient sensitivities come back as zeros, silently.

## Traps that have already cost time

- **`RUN` takes a duration, not an absolute time.** `run 1.7` then
  `run 2.89` lands you at t = 4.59.
- **`system.g_constant` defaults to 1.** Bodies rattling in a box also
  attract each other, and the softened force at `softening = 1e-6` is
  nearly singular when surfaces touch. The same box scenario conserves
  energy to 3e-16 with `G = 0` and drifts 3.2 % with `G = 1`. **A
  conservation claim carries its system settings with it** — the
  `free_system` helper in `physical_object/tests/collision.rs` sets
  `G = 0` for exactly this reason.
- **A magnetic moment tensor's third column is what a B along z can
  grip.** `[[0, 0.5, 0], [-0.5, 0, 0], [0, 0, 0]]` looks like a
  reasonable antisymmetric tensor and produces exactly zero torque.
- **`BALL`, `HINGE`, `UNIVERSAL`, `GEAR`, `RACK` and `PRISMATIC` are CONTEXTUAL keywords** — commands
  only at the start of a line. `ball` is exactly what a physics user
  calls a sphere, and `new sphere as ball` / `get ball.mass` were already
  in the tests and docs; reserving it outright broke them.
- **A joint constrains velocity, so starting velocities are PROJECTED**
  (`project_initial_velocities`). A body turning about an offset pivot
  must have its centre moving; a caller who sets ω and leaves v at zero
  is off the manifold, and IDA then fails on the first step at every
  tolerance. Reported via `RunReport::initial_velocity_projected`.
  A rod has `J_ω = 0` and never needed it — which is why this hid.
- **Orientation joints carry a tolerance floor** of `rtol = 1e-6`
  (`ROT_JOINT_RTOL_FLOOR`): the index-2 accuracy ceiling is real, and
  measured sharp across twelve pendulums.
- **A constrained system refuses every method but `METHOD IDA`**, and a
  fully-free system has no isolated equilibrium at all (translate it and
  nothing changes) — `EQUILIBRIUM` says so and tells you to pin a body.
- **Quantum discretisation error converges; a bug does not.** Before
  calling a QM disagreement a defect, refine the grid and check the
  ratio: the square-barrier transmission at E = 2 goes 0.069368 →
  0.070328 → 0.070570 toward the analytic 0.070651 as the grid goes
  4k → 16k → 64k.

## Verified physical facts to protect (regression anchors)

Under 7.8.0 every one of these is **byte-identical** to its 7.7.0 value;
that is the substance of `PORT_7.8.0_PROVENANCE.md` §3.

- Outer solar system matches the donor `solar_system` example: Pluto
  `x = 31.78592516  y = 38.63618957  z = 3.19279415` at t = 500,000
  days, energy drift `7.835809e-07`, `12581` internal steps.
- Kepler e = 0.6 conserves E/L/Laplace < 1e-6
  (`|dA|/|A| = 1.131858e-07`); gyroradius = mv/(qB) to 1e-4; tumbling
  body conserves L **exactly** (`|dL|/|L| = 0.000000e+00`).
- Kepler's third law: `T²/a³ = 0.039478417604357434` for a = 1, 2, 3, 4,
  identical to the last bit.
- Ball dropped on a plate: TOI `0.8944271909999157` against analytic
  `sqrt(0.8) = 0.8944271909999159` — 1.24e-16 relative.
- Mixed shapes in a rigid BOX 4 with G = 0: `|dE|/E = 3.4e-16` through
  36 collision events.
- Scene reverse replays history to **exactly** t = 0 (bit-identical
  snapshots, `playback_forward_then_reverse_restores_state`).
- Zero-collidable-pair systems are **bit-identical** with COLLIDE ON vs
  OFF (structural invariance — protect it).
- The tilted-torus fit: an axis-aligned torus of outer radius 2 exactly
  inscribes BOX 4; tilted to axis (1,1,1)/√3 its per-axis extent is
  1.5·√(2/3) + 0.5 ≈ 1.7247, clearing every wall
  (`tilted_torus_fits_a_4x4_box_where_flat_does_not`).
- A ball rattling in a rigid BOX conserves E with the walls
  **bit-identically at rest**
  (`ball_in_a_rigid_box_conserves_energy_and_walls_never_move`).
- A point threads the torus hole while the fat ball bounces with
  TOI = (3−√1.45)/4
  (`point_threads_the_torus_hole_but_a_fat_ball_bounces`).
- The BOX grammar family (BOX `<size>`/OFF/status, `system.box`,
  `[wall: static, inverse_mass=0]` LIST tags, machine
  `box`/`wall`/`inverse_mass`, scene init `box` + wall flags)
  round-trips end-to-end (`box_family_and_infinite_mass_walls`,
  `state_reports_box_walls_and_inverse_mass`,
  `box_shapes_and_wall_flags_reach_the_init_message`); and `BOX <size>`
  after a wall deletion removes the surviving tracked slabs before
  building the new box — no orphan leak
  (`box_recreate_after_wall_deletion_leaks_nothing`).
- Parallel round shapes touch side-on at the exact lateral gap — the
  radial rejection axes make cylinder-cylinder side contacts exact
  (`parallel_cylinders_side_contact_is_exact`).
- A ball released FROM REST above a thin plate is caught at the
  analytic TOI to 1e-6 — the anti-tunneling cap is acceleration-aware
  (`ball_released_from_rest_does_not_tunnel_the_thin_plate`).
- **A known limitation, pinned deliberately:** two disks with PARALLEL
  planes have separation |dz|, which touches zero without a sign
  change, so face-on disk-disk crossings are invisible to
  downward-crossing rootfinding
  (`parallel_disk_disk_separation_is_the_documented_limitation`; tilt
  one disk or model a thin cylinder).
- NEW is transactional (a failing initializer or final validation
  leaves no ghost object) and the torus inner/outer pair is genuinely
  order-independent, horn torus `inner_radius = 0` valid
  (`torus_pair_is_order_independent_and_new_is_transactional`).
- Two tumbling dumbbells colliding off-center conserve E, P **and L
  (about the origin)** to 1e-8 through real CVODE events — the
  part-wise exact narrow phase puts the impulse pair at one shared
  contact point
  (`colliding_dumbbells_conserve_energy_momentum_and_angular_momentum`).
- Scene Reset (toolbar button / window `reset` / `SCENE RESET` — one
  primitive) restores the playback's initial state **bit-identically**,
  clears history and the step counter, returns the mode to Stopped, and
  Start re-runs from the beginning
  (`reset_restores_the_initial_state_and_start_reruns`).
- The `create_dumbell` flow — define/call/members/shorthands/renumber/
  errors/redefine/LET-defaults/ghost-free failing calls — round-trips
  end-to-end (`def_call_named_objects_and_dumbbell_members`); the
  directed supports are exact for BOTH dumbbell ends (asymmetric wall
  gaps exact, the light end pokes farther when m2 > m1, ball-vs-pole
  and ball-vs-rod contacts exact —
  `dumbbell_wall_gaps_and_ball_contacts_are_exact`,
  `dumbbell_constructor_com_sdf_and_supports`); part masses whose SUM
  overflows f64 are refused, and a fat-rod dumbbell reports honest
  side-on supports — rank 1 with footprint = the rod half-length
  (`fat_rod_dumbbell_reports_honest_support_ranks`).
- The dumbbell composite inertia is exact: with a massless rod and two
  2 kg spheres of radius 0.3 at z = ±1, the tensor is
  `[[4.144,0,0],[0,4.144,0],[0,0,0.144]]`.
- DEF is hardened: parameters cannot be reserved words, builtins,
  pi/tau or duplicates; defaults are expressions only (a command like
  `reset` is refused at definition time); a `}` inside a `#` comment
  does not close the body; `NEW ... AS n` errors when `n` is bound to a
  number; and a function that runs NEW inside another NEW's initializer
  stashes/restores the outer NEW context
  (`def_hardening_reserved_words_duplicates_defaults_and_nesting`).
- Contact records arriving with a notebook system never reach the scene
  playback copy — cleared at create, refresh-sync and reset
  (`stale_notebook_contacts_never_reach_the_playback_copy`).

## Verified UI facts to protect

Re-check with a headless-Chrome CDP session after touching
`scene.html`: arrow keys translate the view right/left (and up/down),
left-drag orbits yaw+pitch, mouse wheel and `+`/`-` zoom, Space toggles
start/pause, toolbar Start/Pause/Stop/Reverse work, the statusbar shows
mode/t/dt/E/bodies/contacts/camera/fps, the Contacts button and `C`
toggle the golden contact-normal arrows and the frame protocol carries
the exact analytic normal/point/impulse for a head-on impact, the
dashed `#5d84a8` interior box wireframe draws when a BOX exists (**the
six wall slabs are never drawn as bodies**), and torus (outer/inner
equators + tube rings + 4 cross-sections), disk (rim + 2 diameters) and
cylinder (2 rims + 4 side lines) wireframes render quaternion-rotated so
spin is visible; the permanent toolbar Reset button (`bt-reset`)
re-initializes the playback and Start re-runs from the beginning; the
labeled conserved-quantities readout (`hud`: E, P and L with components
and magnitudes) updates live and reads identically before and after a
dumbbell impact; entity labels show the registered user names
(`dumbell0`) instead of `objN`; and dumbbells render as one rigid body —
two shaded spheres at their rotated COM offsets joined by the rod's four
silhouette lines.

The recorded video player (`recorder/src/record_video.py`) follows the same
rules: wall slabs excluded from both the draw list and the camera
auto-fit, round shapes rotated about their local z axis, contact arrows
along the exact analytic normal scaled by impulse.
