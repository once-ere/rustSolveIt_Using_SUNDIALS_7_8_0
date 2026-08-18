# ARCHITECTURE — physical_object_simulator

This document pins the structural decisions and cross-module contracts.
Read it before changing anything that crosses a module boundary.
The numerical engine is the vendored pure-Rust translation of
**SUNDIALS 7.8.0** (§3.3, §5).

Companion documents: [SolveIt.md](SolveIt.md) (the complete solution
guide, for users), [grammar.md](grammar.md) (language spec),
[PLAN.md](PLAN.md) (design record / union mapping),
[PORT_7.8.0_PROVENANCE.md](PORT_7.8.0_PROVENANCE.md) (what the 7.8.0
upgrade changed, and the evidence it changed nothing else),
[physical_object_simulator.md](physical_object_simulator.md) (the
predecessor user guide), [scene_info.md](scene_info.md) (graphical scene
window: the research survey, the protocol, and the UI),
[CLAUDE.md](CLAUDE.md) (working rules).

## 1. System overview

```
                 ┌───────────────────────────── posim (binary) ─────────────────────────────┐
 user input ───► │ lexer.rs ──tokens──► parser.rs ──Vec<Instr>──► vm.rs (stack machine)      │
                 │     ▲                                            │ get/set API only       │
                 │ notebook.rs (REPL cells/magics)                  ▼                        │
                 │ machine.rs (JSONL server)  ◄────────────► SimState{ PhysicalObjectSystem, │
                 │        ▲  unsolicited {"event":...} lines        scene: Option<Scene> }   │
                 │        │                                          │ SCENE commands        │
                 │        │            ┌─ scene/ (std-only HTTP + WebSocket server) ─┐       │
                 │        └── events ──┤ mod.rs   SceneHandle, Shared, playback loop │       │
                 │                     │ ws.rs    SHA-1, base64, RFC 6455 frames     │       │
                 │                     │ scene.html  embedded window page            │       │
                 │                     └───────────────▲─────────────────────────────┘       │
                 └─────────────────────────────────────┼──────────────────────┬──────────────┘
                                            WebSocket (JSON frames)           │
                                                       ▼                      │
                                     web browser: scene window                │
                                     (toolbar, statusbar, canvas,            │
                                      arrows/drag/wheel gestures)            │
                                                                             │
                 ┌───────────────────────── physical_object (lib) ───────────▼──────────────┐
                 │ physical_object.rs   the union struct + get/set + observables            │
                 │ system.rs            PhysicalObjectSystem + pack/unpack (13N)            │
                 │ integrate.rs         ALL time integration                                │
                 │ boundary.rs          Boundary enum + Sdf trait + analytic inertia        │
                 │ linalg.rs            Vec3 / Mat3 / Quat (pure std)                       │
                 └──────────────┬───────────────────────────┬───────────────────────────────┘
                                │                           │
                        cvode_rs (CVODE)             arkode_rs (SPRKStep)
                                └───────── sundials_core ───┘        (path deps → ../sundials_rs)
                    pure-Rust SUNDIALS 7.8.0, vendored byte-identical; the suite also
                    ships cvodes_rs, ida_rs, idas_rs, kinsol_rs (built, not yet used)

 JupyterLab ◄─ZMQ─► jupyter/posim_kernel (Python) ◄─stdin/stdout JSONL─► posim --machine
                    (reader thread routes replies vs. async scene events → iopub streams)
```

## 2. Crate & module responsibilities

### 2.1 `physical_object` (library)

| module | owns | must NOT do |
|---|---|---|
| `linalg` | `Vec3`, `Mat3`, `Quat`: all vector/matrix/quaternion math. `Copy + Clone + Debug + PartialEq`. Zero-guards: `Vec3::normalize`(0)→0, `Quat::normalize`(degenerate)→identity, `Mat3::try_inverse`→`Option` | depend on any other module |
| `boundary` | `Boundary` enum (`Point`, `Sphere`, `Cuboid`, `Torus`, `Disk`, `Cylinder`, `Dumbbell` — §3.8), `Sdf` trait (`signed_distance` + default central-difference `surface_normal`, ε = 1e-6, ported verbatim; the torus/disk/cylinder SDFs are exact closed forms — the ideal zero-thickness disk's is the *unsigned* distance, zero exactly on the disk), `analytic_inertia_tensor` (Point → zero matrix; torus Iz = m(c²+¾a²), Ixy = m(½c²+⅝a²); disk Ixy = ¼ma², Iz = ½ma²; cylinder Iz = ½mr², Ixy = m(3r²+4h²)/12), the support family `support_extent`/`support_point`/`support_rank`/`bounding_radius` (conventions in §3.8) | hold `dyn` trait objects (keeps the object `Clone`) |
| `physical_object` | the union struct; **all invariant enforcement lives in its setters** (see §3.1); observables; three constructors | integrate time (delegates to `integrate`) |
| `system` | `PhysicalObjectSystem`; observables over the collection; `add_object`/`remove_object` (keeps `external_forces`/`external_torques` index-aligned); `pack_state`/`unpack_state` | touch solvers |
| `integrate` | **the only module that advances time** — CVODE and ARKODE drivers, RHS functions, `Method`, `RunReport`/`Snapshot`, `propagate_single` | be bypassed: no other module may implement stepping |

### 2.2 `posim` (binary)

| module | owns |
|---|---|
| `lexer` | `Token`/`TokKind`/`Keyword` (incl. the 20 scene keywords); `tokenize(line)`; column-carrying errors; `#` comments |
| `parser` | the EBNF (documented in its file header, incl. `scenecmd`); `compile_line(line) -> Vec<Instr>`; keyword-as-field-name acceptance after `.` and in `NEW {}` |
| `vm` | `Value`, `Path`/`PathRoot`, `Instr`, `MethodSpec`, `ShapeKind`, `SceneCmd`, `BoxMode`, `SimState` (incl. `box_size`/`wall_indices` backing the `BOX` command and the deferred-torus slot `pending_torus` — §3.8); `execute`/`execute_line`; the path→get/set dispatch tables; `exec_scene`, `exec_box`; `HELP_TEXT` |
| `notebook` | `Notebook`/`Cell`; the REPL loop; magics; `run_script` |
| `machine` | minimal JSON (`Json` enum, `parse`, `to_string`), the request ops (`exec/get/set/state/events/help/quit`), `serve()`, async `{"event":...}` line emission |
| `scene` | the graphical window subsystem: `SceneHandle` (VM-facing API), `Shared` state under `Arc<Mutex<_>>`, the listener/playback/per-client threads, the JSON message builders; `scene::ws` (SHA-1, base64, RFC 6455 server frames); `scene.html` (the embedded page: toolbar, statusbar, canvas renderer, gestures) |
| `main` | mode dispatch only |

### 2.3 `jupyter/` (Python, outside the Rust constraints)

`posim_kernel/kernel.py` — ipykernel wrapper kernel subprocessing
`posim --machine`; a background **reader thread** routes backend stdout:
reply lines go to a queue consumed by `_request`, unsolicited
`{"event": ...}` lines are pushed immediately to the notebook as
`[scene] ...` iopub stream messages. `kernelspec/kernel.json`;
`test_protocol.py` (stdlib-only wire test, incl. the scene command
flow); `test_kernel.py` (jupyter_client ZMQ test, generates a throwaway
kernelspec under `.kernels/` via `JUPYTER_PATH`).

## 3. Pinned contracts (change these only deliberately, updating every listed dependent)

### 3.1 Setter invariants (enforced in `physical_object.rs`, relied on everywhere)

- `set_mass(m)` ⇒ `inverse_mass = if m > 0 { 1/m } else { 0 }`;
  `set_inverse_mass` back-computes symmetrically. `inverse_mass == 0`
  is the *static body* convention.
- `set_inertia_tensor(I)` ⇒ inverse via `Mat3::try_inverse`, singular →
  **zero matrix** (= cannot rotate); `set_inverse_inertia_tensor`
  symmetric.
- `set_orientation(q)` renormalizes; `unpack_state` renormalizes.
- `set_velocity(v)` writes `momentum = mass * v` (momentum is
  canonical); `set_angular_velocity(w)` writes `L = (R I Rᵀ) w`.
- The VM (`vm.rs::store_path`) and the machine mode reach state **only**
  through these setters. Never add a raw-field write path.

### 3.2 State packing (system.rs ↔ integrate.rs ↔ tests)

`VARS_PER_OBJECT = 13`; layout per object, in order:
`[pos.x .y .z | momentum.x .y .z | quat.w .x .y .z | L.x .y .z]`.
Quaternion is **w-first** — everywhere (packing, the VM's 4-element
literals, the JSON protocol, the scene frame messages, the docs).

### 3.3 Solver driving (integrate.rs ↔ sundials_rs 7.8.0)

The vendored engine is the pure-Rust translation of **SUNDIALS 7.8.0**
(`once-ere/SUNDIALS_7_8_Rust_port_for_Linux@780b916`, byte-identical,
2,929 files). Its API models C's opaque pointers directly, which fixes
the shape of everything in `integrate.rs`.

**Handle model.**

- `N_Vector = Rc<_generic_N_Vector>` with `RefCell` content and an ops
  table — C's handle-with-vtable, not a struct with a public `data`
  field.
- `CVodeMem = Rc<RefCell<CVodeMemRec>>`, `ARKodeMem` likewise. Every
  entry point therefore takes `&CVodeMem` / `&ARKodeMem` (shared, C's
  "pass the pointer"), **never `&mut`**.
- Constructors return `Option<…>` where C returns a possibly-`NULL`
  pointer: `SUNContext_Create(SUN_COMM_NULL, &mut Option<SUNContext>)
  -> SUNErrCode`, `N_VNew_Serial -> Option<N_Vector>`,
  `SUNDenseMatrix`, `SUNLinSol_Dense`, `CVodeCreate`, `SPRKStepCreate`.
  Each `None` becomes a **named `Err`**, never an `unwrap` — hard rule 5
  applies to missing allocations as well as missing symbols.
- Destructors take ownership (`N_VDestroy(v)`, `SUNMatDestroy(A)`,
  `SUNLinSolFree(Some(LS))`) or blank an `Option`
  (`CVodeFree(&mut Option<CVodeMem>)`, `ARKodeFree`,
  `SUNContext_Free`) — C's `free(p); p = NULL;`.

**The borrow-guard rule (the one that bites).** Vector payloads are
reached through `N_VGetArrayPointer(&v) -> Option<RefMut<Vec<f64>>>`.
That guard is a **live borrow**: holding it across a solver call that
touches the same vector panics at runtime. `integrate.rs` therefore
routes *every* access through two helpers that take the guard, do the
work, and drop it before returning:

```rust
fn with_data<R>(v: &N_Vector, f: impl FnOnce(&[f64]) -> R) -> Option<R>;
fn with_data_mut<R>(v: &N_Vector, f: impl FnOnce(&mut [f64]) -> R) -> Option<R>;
```

Inside a callback the two guards on `y` and `ydot` coexist safely
because CVODE/ARKODE never pass the same handle for both.

**Callbacks.** Plain `fn` pointers, matching the 7.8.0 types verbatim:

```rust
type CVRhsFn  = fn(sunrealtype, &N_Vector, &N_Vector, &mut Option<Box<dyn Any>>) -> i32;
type CVRootFn = fn(sunrealtype, &N_Vector, &mut [sunrealtype], &mut Option<Box<dyn Any>>) -> i32;
```

Note `ydot` is a **shared** reference. Parameters travel as a `Clone`
snapshot in `user_data: Option<Box<dyn Any>>`, downcast with
`downcast_mut`; downcast failure returns **−1** (unrecoverable), never
panics. `CVodeSetUserData` consumes the box → re-clone per run.

**CVODE path.** `CVodeCreate(lmm)` → `CVodeInit` → `CVodeSStolerances`
→ `SUNDenseMatrix` + `SUNLinSol_Dense` +
`CVodeSetLinearSolver(&mem, &LS, Some(&A))` →
`CVodeSetMaxNumSteps(500_000)` → `CVodeSetUserData` → loop
`CVode(&mem, tout, &y, &mut t, CV_NORMAL)`. Teardown at the single
success return, in the C example order: `CVodeFree`, `SUNLinSolFree`,
`SUNMatDestroy`, `N_VDestroy`, `SUNContext_Free`.

- Quaternion renormalization happens **only at output points**, only
  when drift > `QUAT_RENORM_TOL = 1e-10`, and must be followed by
  `CVodeReInit` (mutating `y` invalidates the Nordsieck history).
  `CVodeReInit` **zeroes the stats counters** → accumulate
  `nst/nfe/nni/netf` before every ReInit and once at the end.
- `CVodeSetNoInactiveRootWarn` now returns `i32`; the return **is
  checked**, like every other flag.

**SPRK path.** Layout `[q(3N) | p(3N)]`;
`SPRKStepCreate(force, velocity, t0, &y, &sunctx)` — 7.8.0 takes the two
right-hand sides **by value** (`ARKRhsFn`, not `Option<ARKRhsFn>`),
because C requires both. `f1 = force` writes the p-half, `f2 = velocity`
writes the q-half (per `ark_kepler.rs`); fixed step via
`ARKodeSetFixedStep`; **separability gate** rejects (with a message
naming the feature): nonzero `b_field`, any nonzero
`magnetic_moment_tensor`, any nonzero external torque, any object with
both invertible inertia and nonzero spin.

**Formatting.** 7.8.0 splits the padded and unpadded `printf`
conversions: `fmt_e(x, prec)` is C's `%.*e`, `fmt_ew(x, width, prec)` is
`%*.*e` (same for `f` and `g`). 7.7.0's single `fmt_e(x, width, prec)`
maps to `fmt_ew` when `width > 0` and to `fmt_e` otherwise — the emitted
bytes are identical either way. Never use Rust's `{:e}`.

**Import hygiene.** Never glob-import both `cvode_rs::*` and
`arkode_rs::*` in one module (both re-export `sundials_core` →
ambiguity); each crate ships a `prelude` for the flat form.
`usize → i64` casts for sundials lengths stay inside `integrate.rs`.

**Unused families.** 7.8.0 also vendors `cvodes_rs`, `ida_rs`,
`idas_rs` and `kinsol_rs`. They build and carry verified examples;
`physical_object` does not depend on them. A constrained mechanism
belongs in `ida_rs`, a parameter derivative in `cvodes_rs` — do not
reimplement either locally.

**If a sundials symbol you need is missing, report it — do not
reimplement numerics locally** (see CLAUDE.md rule 5).

### 3.4 Language pipeline (lexer ↔ parser ↔ vm)

- Every token carries a 1-based column; every parse error cites it.
- Keywords are reserved only in command position; after `.` and inside
  `NEW { }`, keywords are accepted as field names
  (`parser::expect_field`). This includes the scene keywords (`start`,
  `show`, `in`, ... remain legal field names).
- Bracket literals are shape-directed **in the VM's `PackList`**:
  3 numbers → `Vec3`, 4 numbers → `Quat` (w-first), 3 `Vec3` → `Mat3`,
  else `List` (fields reject lists with typed errors).
- `NEW` initializer semantics: `velocity`/`angular_velocity` are
  *deferred* to `FinishNew` (order-independence w.r.t. mass/inertia);
  `FinishNew { recompute_inertia }` recomputes shape inertia **unless**
  the initializer list mentioned `inertia_tensor`/
  `inverse_inertia_tensor` (the parser sets the flag).
- `STEP`/`RUN` advance **by a duration** (relative); `integrate::run`
  takes an **absolute** t_end. The VM does the addition.
- Object handles are **positional indices** (`objN` = `objects[N]`);
  `DEL` renumbers. Documented in grammar.md; do not silently change.
- **Scene arguments are term-level**, not expr-level
  (`scene rotate 15 -5` is two arguments; a sum must be parenthesized).
  This is deliberate: space-separated argument lists and infix `-` are
  otherwise ambiguous. Keep `scene_command()` on `term()`.
- **`DEF` is a line form handled ahead of the grammar**
  (`DEF name(param [= default], ...) { body }`; multi-line bodies —
  the notebook shows `...:=` continuation prompts, scripts join lines
  by brace depth). The body is newline/`;`-separated ordinary commands
  using the parameters as variables; **every body line is
  syntax-checked at definition time**; defaults are ordinary
  expressions evaluated **once at definition** (`LET` variables
  visible). The source is preserved verbatim: redefinition replaces,
  and `SHOW name` prints the verbatim source — that is the edit loop;
  `FUNCS` lists signatures.
- **Calls run on an env stack** (`SimState.env_stack`): ordinary call
  syntax with trailing defaults; each call pushes a frame
  (`MAX_CALL_DEPTH = 32`), the body returns the last line's value, and
  a failing line **rolls the whole call back**, naming the function
  and the offending line in the error.
- **The name registry** (`SimState.names`): `NEW <shape> AS <name>`
  registers a user name — a string literal, or a bare identifier that
  resolves first against a parameter/`LET` string binding (so a
  function names the objects it creates from its arguments). Named
  paths work everywhere paths do (`ball.mass`, `dumbell0.position.x`);
  `DEL`/`BOX` renumber the registry; duplicate and reserved names
  (`objN`/`contactK`/`system`) are refused. Every object gains the
  component shorthands `.x .y .z .vx .vy .vz`. String literals
  (`"..."`) join the value system; `LET name = expr` stores session
  variables. Machine-mode objects carry their registered `"name"`.
- **Bare identifiers compile** (`Instr::LoadIdent`) and resolve at
  execution; a mistyped root's runtime error still teaches
  `objN`/`contactK`/`system`.
- **Deferred + validated-once constructors**: torus geometry
  (`pending_torus`, §3.8) and the dumbbell (`NEW DUMBBELL AS d
  { m1, m2, m_rod, r1, r2, rod_radius, length, ... }`,
  `pending_dumbbell`) collect their parameters order-independently
  and are resolved + validated **once** at `FinishNew` (the dumbbell
  via `boundary::dumbbell`).
  `d.m1/.m2/.m_rod/.r1/.r2/.rod_radius/.length` read AND write — the
  mass fractions stored in the boundary keep every part recoverable,
  and a member write rebuilds total mass, COM offsets and the inertia
  tensor in one step. New keywords: `as`, `let`, `funcs`|`functions`,
  `dumbbell`|`dumbell` (aliases).

### 3.5 Wire protocol (machine.rs ↔ jupyter/)

One JSON document per line, flush after every reply. Requests:
`{"op":"exec"|"get"|"set"|"state"|"events"|"help"|"quit", ...}`.
Replies: `{"ok":true,"result":...,"display":"..."}` or
`{"ok":false,"error":"..."}`. Numbers serialize via Rust `{:?}`
(shortest round-trip). `get`/`set` are implemented by *compiling a
command line* (`get <path>` / `set <path> = <literal>`) so the wire
protocol can never bypass VM validation. The Python kernel splits a
cell into lines, sends one `exec` per line, stops at first error.

**State-dump additions** (shapes + BOX release): `{"op":"state"}`
reports, per object, `inverse_mass` (0 = static body) and `wall`
(`true` for `BOX` wall slabs), plus a top-level `box` — the box's
inner side length as a number, or an **explicit `null`** when no box
exists (front ends can bind it without a presence check). The
`boundary` string names all six shapes (`torus ring=… tube=…`,
`disk r=…`, `cylinder r=… h=…` — h is the *full* height).

**Asynchronous lines**: while a scene window is open in `--machine`
mode, the scene threads may print unsolicited
`{"event":"scene","message":"..."}` lines at any time. Each event is
one complete `println!` (line-atomic). Front ends MUST tolerate event
lines interleaved between a request and its reply — the Python kernel
does this with a dedicated reader thread; `test_protocol.py` shows the
minimal pattern. `{"op":"events"}` drains the same queue by polling
(for front ends that cannot read asynchronously).

### 3.6 Naming constraint

`pub struct physical_object` (spec-mandated lower-case) shares its name
with its module and shadows the crate name in importing files. Living
with it: crate roots allow `non_camel_case_types`; consumers write
`use physical_object::physical_object::physical_object;` and prefix
sibling imports with `::`. The struct cannot be re-exported at the
crate root (module/type namespace collision) — do not try.

### 3.7 Scene subsystem (posim/src/scene/ ↔ vm ↔ machine ↔ browser)

- **Zero-dependency networking**: the HTTP server, SHA-1, base64 and
  RFC 6455 framing are hand-rolled on `std::net` (server binds
  `127.0.0.1` only). Do not add crates.io networking or crypto; the
  SHA-1 is a protocol checksum, not a security boundary.
- **The playback thread owns a COPY** of the notebook's
  `PhysicalObjectSystem` (synced by `SCENE CREATE`, `SCENE REFRESH`,
  and `RESET`). Notebook `STEP`/`RUN` do **not** move the window;
  window playback does **not** move the notebook. This isolation is
  what lets both sides stay responsive without locking the VM.
- **`Shared.initial` / `reset_playback`**: `Shared` keeps an
  `initial` snapshot of the playback system — the state last synced
  at `SCENE CREATE` or `SCENE REFRESH`. Reset is reachable **three
  ways sharing one primitive** (`Shared::reset_playback`): the
  permanent toolbar **Reset** button (`bt-reset`), the window `reset`
  command, and the `SCENE RESET` notebook command. The primitive
  restores the snapshot **bit-identically** — every mutable value and
  the time return to their initial values — clears history and the
  step counter, and returns the mode to `Stopped`; the Start button
  then re-starts the simulation from the beginning
  (`reset_restores_the_initial_state_and_start_reruns`).
- **Forward evolution goes through `integrate::step` only** (rule 1
  applies inside the scene too). **Reverse is snapshot replay**: each
  forward tick pushes a `Clone` of the system into a ring buffer
  (`HISTORY_CAP = 20_000`); `SCENE REVERSE` pops it. Never integrate
  with a negative dt.
- **RunMode state machine** (`Stopped | Running | Paused | Reversing`):
  `Stopped` clears history; `Reversing` with an empty buffer refuses
  (VM) or auto-pauses with an event (playback). Tick period
  `TICK_MS = 33` (~30 fps broadcast).
- **Lock discipline**: `Shared` sits under `Arc<Mutex<_>>`; the lock is
  never held across blocking network I/O — broadcasts go through
  per-client `mpsc` channels; each client has a writer thread, and the
  socket write-half is mutex-wrapped so ping/pong replies cannot
  interleave mid-frame.
- **Message protocol** (JSON text frames):
  server→window `init` (entity geometry — each entity carries its
  shape plus that shape's parameters (`radius`, `half_extents`,
  `ring_radius`/`tube_radius`, `half_height`, and for dumbbells
  `r1`/`r2`/`rod_radius`/`z1`/`z2`), its registered user `"name"`
  when one exists, and `"wall":true` on `BOX` slabs; a top-level
  `"box":<size>` appears when a box exists — plus camera + playback
  state, sent on connect / REFRESH / REDRAW / structural change),
  `frame` (t, dt, mode, steps, history, energy, `p` = total momentum
  and `l` = total angular momentum about the origin — computed on the
  playback copy every tick — hidden, per-body
  `[x,y,z,qw,qx,qy,qz]` — w-first, §3.2), `camera`;
  window→server `cmd` (`start/pause/stop/reverse/step/step_back/
  set_dt/refresh`), `camera` (gesture sync, throttled to 10 Hz),
  `request_state`, `event` (level + message). Unknown types become
  error events, never panics.
- **Conserved-quantity readout + name labels** (`scene.html`): the
  window shows a permanent labeled readout (`hud`) — E, P and L lines
  with components and magnitudes — updated live from each frame's
  `energy`/`p`/`l`. Entity labels prefer the registered user name
  (`dumbell0`) over `objN`; dumbbells draw as one rigid body — two
  shaded spheres at their rotated COM offsets joined by the rod's
  four silhouette lines, so spin is visible.
  `SceneHandle::set_box(box_size, walls, names)` also ships the
  user-name registry (the `SCENE CREATE`/`SCENE REFRESH`/`RESET`
  call sites keep it current).
- **Camera**: z-up orbit — `yaw`/`pitch` in degrees (pitch clamped to
  ±89°), `dist > 0`, `target` in world units. The browser owns gesture
  handling (arrows translate along the view basis, left-drag orbits,
  wheel and `+`/`-` zoom, shift/right-drag pans) and syncs back;
  `SCENE TRANSLATE/ROTATE/ZOOM` mutate the server copy and broadcast.
- **Event queue**: bounded at 1000 (oldest dropped). Drained by
  `SCENE EVENTS` / `{"op":"events"}`; under `--machine` each event is
  also pushed immediately as an async line (§3.5).
- **Lifecycle**: `SimState.scene: Option<SceneHandle>`; dropping the
  handle (SCENE CLOSE, quit) sets `shutdown`, clears outboxes, and all
  threads exit on their next poll tick (reader timeout 500 ms, accept
  poll 50 ms). `RESET` keeps the window open and re-syncs it.
- **Browser opening** is best-effort `xdg-open`, suppressed by
  `$POSIM_NO_BROWSER` (set in every headless test).

### 3.8 Collision subsystem (physical_object/src/collide.rs ↔ integrate.rs ↔ vm ↔ scene)

- **Pinned conventions**: a `Contact { i, j, t, point, normal, depth,
  rel_vel_n, impulse_n }` between objects `i < j` carries a **unit
  normal pointing from body i toward body j** — the action–reaction
  line (`+J·n̂` on j, `−J·n̂` on i). `pair_separation` is **positive
  when separated** and is *exactly* the sundials root-function value;
  `depth = max(0, −separation)`. These signs are load-bearing across
  collide.rs, integrate.rs, the VM paths, machine JSON and the scene
  protocol — change them nowhere or everywhere.
- **Narrow phase is closed-form** (no GJK/EPA/MPR — the shape set
  {Point, Sphere, Cuboid, Torus, Disk, Cylinder, Dumbbell} is closed;
  the dumbbell decomposes over its parts, below), in
  **three exactness tiers**:
  (1) *exact ball-vs-anything*: sphere-sphere by center distance,
  sphere-cuboid by box SDF (interior branch → nearest face), and
  ball (sphere, or point as the zero-radius case) vs
  torus/disk/cylinder via the shape's exact SDF closest point — so a
  small ball genuinely threads a torus hole, and a point passes
  through the ideal zero-thickness disk;
  (2) *exact cuboid-cuboid*: the 15-axis SAT, unchanged (deepest axis
  = normal; support vertex or edge-edge closest points = contact
  point);
  (3) *support-axis tests* for every remaining extended-vs-extended
  pair: candidate axes are each cuboid's three face axes, each round
  shape's symmetry axis, their pairwise cross products, the **radial
  rejection axes** (the component of the center offset perpendicular
  to each primary/cross axis — for round shapes with (near-)parallel
  axes and an axial offset this is the true lateral separating
  direction, where the parallel-axis cross product vanishes and the
  raw center line is tilted), and the center line — **exact for
  face-on contacts** (in particular every wall-slab contact) **and
  for side-side contacts of parallel round shapes** such as
  cylinder-cylinder (`parallel_cylinders_side_contact_is_exact`),
  conservative only on genuinely skew corner-on configurations (may
  report contact slightly early, along a candidate axis), and this
  tier sees the torus as its convex hull (only tier 1 can use the
  hole). Known ideal-body limitation, pinned by
  `parallel_disk_disk_separation_is_the_documented_limitation`: two
  disks with PARALLEL planes both have zero extent along the shared
  normal, so their separation is |dz| — it touches zero at plane
  coincidence **without a sign change**, and face-on disk-disk
  crossings are invisible to downward-crossing rootfinding (exactly
  like the point-through-disk case; tilt one disk or model a thin
  cylinder for a detectable version). The root function
  `g_contacts` evaluates the **same**
  `separation_at` geometry on the packed state, so detection and
  response cannot disagree.
- **Support conventions** (`boundary::support_extent`/`support_point`/
  `support_rank`): `support_point` returns the **centroid of the
  supporting set** whenever that set is a face, edge, circle or cap
  rather than a single vertex — a flat-on contact places its point at
  the center of the touching patch and carries no spurious lever arm;
  the invariant `p·u = support_extent(u)` holds in every case.
  `support_rank` is the supporting set's dimension (0 = vertex/rim
  point, 1 = edge/line/circle, 2 = face/disk/cap); a tier-3 contact
  point is the **lower-rank (incident) body's support point** — the
  true deepest point, since the higher-rank partner only contributes
  a face whose centroid may sit far from the contact. At **equal
  rank** (face-on-face, edge-on-edge) the point prefers the body with
  the clearly smaller flat footprint (`support_footprint_radius` —
  the supporting set's lateral circumradius about its centroid;
  smaller than half the partner's wins, midpoint only when they are
  comparable): a small cap landing on a big wall face contacts at the
  CAP center, carrying no spurious lever arm
  (`small_cap_on_large_face_contacts_at_the_cap_center`).
- **Dumbbell** (`Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, f1,
  f2 }`): two solid spheres plus a solid rod as **ONE rigid body** —
  sphere 1 at `(0,0,z1)`, sphere 2 at `(0,0,z2)`, the rod between
  them, with the part mass fractions `f1`/`f2` stored so every part
  stays recoverable. The constructor
  `boundary::dumbbell(m1, m2, m_rod, r1, r2, rod_radius, length)`
  places `z1 = −(m2 + m_rod/2)·L/M`, `z2 = (m1 + m_rod/2)·L/M`, so
  the body-frame origin **IS** the center of mass (identity
  `m1·z1 + m2·z2 + m_rod·(z1 + z2)/2 = 0`, pinned by
  `dumbbell_constructor_com_sdf_and_supports`). The union SDF is
  exact (min of the parts); the composite inertia is exact
  (2/5·m·r² spheres + parallel-axis + rod cylinder terms). It is the
  first **non-centrally-symmetric** shape: `support_extent` is now
  the true **directed** `h(u) = max x·u` (bit-identical formulas for
  all symmetric shapes), the support-axis tier evaluates the general
  directed SAT gap `d·l − h_a(l) − h_b(−l)` for **both** axis
  orientations — **it reduces exactly to `|d·l| − e_a − e_b` for
  symmetric shapes** — and `world_aabb` takes the directed ± extents.
- **Dumbbell narrow phase DECOMPOSES over the parts**:
  dumbbell-vs-anything splits into its two spheres and its rod — each
  sphere part uses the other shape's **exact SDF** at its center
  (exact against every shape, including another dumbbell's union
  SDF), and the rod recurses as a free-standing cylinder — so only
  rod-vs-rod ever reaches the approximate support-axis tier; contact
  selection takes the deepest part
  (`dumbbell_wall_gaps_and_ball_contacts_are_exact`: asymmetric wall
  gaps exact for both ends — the light end pokes farther,
  `|z1| > |z2|` when `m2 > m1` — ball-vs-pole and ball-vs-rod exact).
  This exactness is what makes the anchor pass: two tumbling
  dumbbells colliding off-center conserve E, P **and L (about the
  origin)** to 1e-8 through the real CVODE event path
  (`colliding_dumbbells_conserve_energy_momentum_and_angular_momentum`)
  — the impulse pair acts at one shared contact point, so the net
  torque is zero.
- **Arming condition**: `collide_enabled && !collidable_pairs.is_empty()`
  — pairs exclude Point-Point and static-static. **Step-selection
  invariance**: CVODE's root check runs after each completed internal
  step on the interpolant only (cvode.rs:912-1031), so armed-but-quiet
  runs take the identical step sequence; zero-pair systems are
  **bit-identical** to the pre-collision build (test-enforced). While
  armed, `CVodeSetMaxStep` caps the step at (smallest crossable
  feature)/(2·max surface speed) — the anti-tunneling bound. The
  surface speed is the **reachable** speed over the horizon Δt to the
  next refresh (one output interval), not the instantaneous one:
  relative center speed plus `(a_i + a_j)·Δt` — acceleration from
  pairwise gravity at the current configuration, uniform gravity, qE
  and external forces (the magnetic force is ⊥ v and never grows
  speed) — **plus each body's spin bound times its bounding radius**
  (a rotating edge can sweep into contact with the centers at rest),
  where `|ω| ≤ √3·‖I⁻¹‖∞·(|L| + Δt·(|τ_ext| + ‖M‖∞·|B|))` — this
  covers torque-free tumbling exactly (L is conserved, but the
  polhode can spike |ω| up to |L|/I_min mid-interval). A ball
  released FROM REST above a thin plate is therefore still caught
  (`ball_released_from_rest_does_not_tunnel_the_thin_plate`, TOI to
  1e-6); no-cap (0.0) is returned only when literally nothing can
  move. The thinnest crossable features of the new
  shapes are the torus tube radius, the disk radius (the ideal disk
  has zero thickness — the pairwise min picks the approaching ball's
  radius), and the cylinder's min(radius, half_height). The cap is
  refreshed after each event **and at every output-interval start**,
  and its clamp span is the **remaining run `t_end − t` — never
  `tout − t`**: an event landing exactly on an output boundary would
  collapse `tout − t` to 0, pin hmax at the 1e-12 clamp floor, and
  starve every later interval into CV_TOO_MUCH_WORK (a latent bug
  fixed in the shapes + box release). The speed-**growth** horizon
  Δt, by contrast, IS the interval `tout − t` — collapsing it merely
  drops the growth term, never the cap itself. Approach-only root direction
  (−1) + CVODE's zero-at-restart deactivation (`cvRcheck1`) prevent
  post-impulse re-triggering.
- **Event loop** (both CVODE and ARKODE/SPRK paths): `CV_ROOT_RETURN` →
  `GetRootInfo` → unpack → `resolve_impulses` (Gauss–Seidel over all
  pairs, ≤10 passes — simultaneity and cradle propagation) → repack →
  accumulate stats → `ReInit`/`Reset` → continue toward the same tout;
  a tout-boundary guard breaks when the event lands on tout itself.
  **Zeno tiers**: >64 events in one **burst** → restitution forced
  0; >128 → disarm roots, `resolve_penetrations(plastic = true)`
  (a *burst* is a run of events separated by less than `1e-9 max(|t|,1)`
  — chattering; any real flight between two events resets the count, so
  the tiers do **not** depend on the output interval, which is the
  defect they used to cause)
  (impulse + positional projection split by inverse mass beyond
  `contact_slop`), re-arm next interval. End-of-interval sweep
  (read-only pre-check first) catches deep initial overlaps, which
  produce no sign crossing — the sweep runs **elastic**
  (`plastic = false`) so slowly-grinding approximate contacts cannot
  bleed energy; only the Zeno tier-2 projection stays plastic.
- **Response**: `J = −(1+e)(v_rel·n̂)/(n̂ᵀKn̂)`,
  `K = (1/m_i+1/m_j)·1 − [r_i]×I⁻¹ᵢ,w[r_i]× − [r_j]×I⁻¹ⱼ,w[r_j]×`
  (world inertia via `world_inverse_inertia()` = `R I⁻¹ Rᵀ`, the same
  expression `get_angular_velocity` uses). All writes via
  `set_momentum`/`set_angular_momentum`; static sides receive **no**
  writes. `e = min(e_i, e_j)`, default 1.0; forced 0 under
  `restitution_threshold`. Degenerate normals (concentric) are skipped,
  never divided by.
- **Exposure**: `system.contacts` (cleared per run, cap 1024),
  `collision_count`, `RunReport.{nge, ncollisions}`; VM `contactK.*`
  read-only paths + `COLLIDE`/`CONTACTS`; machine `state` gains
  `contacts`/`collide_enabled`/`collision_count` (and, with BOX,
  `box`/`wall`/`inverse_mass` — §3.5); scene `frame` gains a
  `contacts` array and the page draws fading golden normal arrows
  (toggle: Contacts button / `C`). The word `collisions` is
  deliberately NOT a lexer keyword so `system.collisions` resolves as
  a field.
- **SPRK path**: same event pattern on the 6N layout with orientations
  snapshotted (the separability gate forbids spin); sampling bound =
  the fixed dt (documented). Friction: designed (Coulomb cone clamp),
  deliberately not shipped.
- **BOX is a posim-level construct** (`vm.rs::exec_box`):
  `BOX <size>` creates six *static* `Boundary::Cuboid` wall slabs
  (slab cross-sections overlap past the corners) with
  `inverse_mass = 0` and zero inverse inertia; `BOX OFF` removes
  them; bare `BOX` reports. The VM tracks `SimState.box_size` +
  `wall_indices` (`DEL` keeps the indices renumbered; deleting any
  wall **dissolves** the box — `box_size` cleared, `system.box`
  reads 0 — but the surviving slabs **stay tracked**, keeping their
  `LIST` tag, until `BOX <size>` replaces them or `BOX OFF` removes
  them; a recreate removes the tracked survivors first, so nothing
  leaks — `box_recreate_after_wall_deletion_leaks_nothing` — and
  bare `BOX` on a dissolved box says so and names both exits),
  exposes `system.box` (0 = none) and tags walls in `LIST` as
  `[wall: static, inverse_mass=0]`. `physical_object` has **no box
  concept at all** — deliberately: the equations of motion only ever
  see the *inverse* mass (`v = p·m⁻¹`; impulse denominator
  `n·Kn = m_i⁻¹ + m_j⁻¹ + angular terms`), so the existing
  static-body convention `inverse_mass == 0` (§3.1) already *is*
  infinite mass — walls contribute 0 to every denominator and receive
  no state writes (they stay bit-identically at rest), with zero
  special-casing in the library. The scene learns of the box via
  `SceneHandle::set_box(box_size, walls, names)` (the third argument
  is the user-name registry — §3.7) and draws a dashed interior
  wireframe instead of the six slabs (§3.7 `init`: per-entity
  `"wall":true`, top-level `"box":<size>`); `RESET` with an open
  window pushes `set_box` too, clearing the window's box wireframe
  and wall flags along with the synced copy.
- **NEW is transactional** (`vm.rs::execute`): a failing initializer
  or a failing final validation removes the half-built object — no
  ghosts. A nested call in the initializer may have appended objects
  after it, so the rollback renumbers the name registry and the wall
  index list exactly as `DEL` does
  (`failing_new_with_nested_creation_renumbers_names`). The converse
  holds too: a `DEL`/`BOX` executed *during* the initializer (through
  a user-function call) renumbers every live NEW rollback target —
  the active `last_new` and the frames `Instr::Call` stashes in
  `SimState.stashed_last_new` (`shift_new_targets`); deleting the
  half-built object itself disarms its rollback
  (`del_inside_a_new_initializer_keeps_the_rollback_on_target`,
  `box_off_inside_a_new_initializer_keeps_the_rollback_on_target`).
  Torus geometry in `NEW` is **deferred**
  (`SimState.pending_torus`; the dumbbell's seven parameters likewise
  in `SimState.pending_dumbbell` — §3.4) and resolved + validated
  once at `FinishNew`: ring/tube apply first, inner/outer override
  the derived pair — so `inner_radius`/`outer_radius` are genuinely
  order-independent (including pairs like
  `{ outer_radius = 0.5, inner_radius = 0.2 }` that sequential
  validation used to reject in one order). `inner_radius = 0` (horn
  torus) is valid on NEW and SET; `SET objN.radius` on a torus errors
  (it has two radii — set ring/tube or inner/outer) instead of
  silently making it a sphere
  (`torus_pair_is_order_independent_and_new_is_transactional`).

### 3.9 Constrained dynamics, equilibrium and sensitivity (constrain.rs ↔ integrate.rs ↔ equilibrium.rs ↔ sensitivity.rs)

The four solver families the 7.8.0 engine added, and the contracts that
bind them.

**Four joints, one abstraction.** `Joint::{Distance, Ball, Hinge,
Universal}` — 1, 3, 5 and 4 scalar rows, i.e. degrees of freedom removed
from the pair's six. Everything is expressed as a **velocity Jacobian**
`J` with `ġ = J·u`, `u = [v₀ ω₀ v₁ ω₁ …]`, walked by
`ConstraintSet::for_each_block`. The constraint wrench is `Jᵀλ`: `J_vᵀλ`
a force, `J_ωᵀλ` a torque. Blocks: distance `J_v = ∓d̂`; ball
`J_v = ±I`, `J_ω = ∓[r]ₓ` (because `e_k·(ω×r) = ω·(r×e_k)`); axis
alignment `J_ω = ±(a×b)`. A hinge is a ball plus two alignment rows, a
universal joint a ball plus one. `constrain.rs`'s
`the_velocity_jacobian_matches_a_finite_difference_of_g` is what keeps
every block honest.

**The GGL projection is MASS-WEIGHTED: `q̇ = v - M⁻¹Jᵀμ`.** Not
`v - Jᵀμ`. `J_v` is dimensionless but `J_ω` carries the attachment arm,
so `J_ωᵀμ` has units of length × μ and subtracting it from an angular
velocity is dimensionally wrong by length². For translation alone `M⁻¹`
is one scalar and omitting it merely rescales μ — which is exactly why
rods never revealed the error. Add a hinge and it breaks the
integration outright.

**Consistent initial velocities are PROJECTED, not assumed.**
`project_initial_velocities` is the answer to what looked for a long time
like an index-2 wall. A joint constrains velocity as well as position: a
ball joint's `ġ = 0` reads `v_i + ω_i×r_i = v_j + ω_j×r_j`, so a body
turning about an offset pivot must have its centre moving. A caller who
sets `ω` and leaves `v` at zero hands in a state **off the manifold**,
and IDA fails on the first step at every tolerance — which is exactly
what was observed and misdiagnosed. The fix is the standard impulsive
projection,

```text
  minimise ½ δuᵀ M δu  subject to  J(u + δu) = 0
  ⟹  (J M⁻¹ Jᵀ) ν = J·u,   δu = -M⁻¹Jᵀν,   δ(p, L) = -Jᵀν
```

reusing the same `S = J M⁻¹ Jᵀ` the multiplier seed builds. It is
reported via `RunReport::initial_velocity_projected` rather than done
silently, because it changes the state the caller handed in. A rod has
`J_ω = 0`, so spin never enters its `ġ` — which is why rods carried
spinning bodies from the start and hid this for so long.

**`ROT_JOINT_RTOL_FLOOR = 1e-6`** floors the tolerance, because the
index-2 system has a real accuracy ceiling: across twelve compound
pendulums `1e-6` converges in every one and `1e-8` in none.
`RunReport::tolerance_floored` reports when it bit. (Before the velocity
projection this boundary was *erratic* in the release angle, which is
what first suggested a conditioning problem. It was not.)

**EQUILIBRIUM and SENSITIVITY stay translational** and refuse
orientation joints by name: solving for a mechanism's rest *pose* means
solving for orientations too, which they do not do.

**The constraint function is NOT squared.** `g = |d| - L` with
`∂g/∂q_j = -∂g/∂q_i = d̂`. The squared form `|d|² - L²` is a polynomial
and needs no division, and it was tried first: its gradient is `2d`, of
magnitude `2L`, and its value is in units of length *squared*, so the
constraint rows of the DAE's iteration matrix are scaled by `L` relative
to the differential rows. At `L = 1` that is invisible; at `L = 1.3` the
index-2 corrector stops converging and the step collapses to `1e-17`.
The unsquared form has a unit gradient everywhere. **Do not "simplify"
it back.**

**GGL index-2, not acceleration-level index-1.** State is the ordinary
13-per-object packing plus multipliers,
`y = [pos, momentum, quat, angmom]ⁿ ⧺ λ(m) ⧺ μ(m)`, `neq = 13N + 2m` —
the same packing `system.rs` defines, which is what lets joints grip
orientation and lets a rod carry a spinning body:

```text
  0 = q̇ - v + Gᵀμ         (position, projected by the GGL multiplier)
  0 = M v̇ - F(q,v) + Gᵀλ   (momentum balance with the constraint force)
  0 = g(q)                 (the constraint)
  0 = G v                  (and its derivative)
```

Carrying **both** `g` and `ġ` as algebraic equations is what pins them at
roundoff; index-1 lets `g` drift quadratically. Same shape as the
reference mechanism example `sundials_rs/crates/ida_rs/examples/idaSlCrank_dns.rs`.
`id` marks `q, v` differential (1) and `λ, μ` algebraic (0);
`IDASetSuppressAlg(true)` keeps the multipliers out of the error test.

**Anchors.** `inverse_mass == 0` ⇒ the body receives no multiplier force
(`ConstraintSet::add_jacobian_transpose` skips it) and gets `0 = v̇`
instead of the momentum balance. A rod to an anchor is a pin joint to
the world. `equilibrium::solve` additionally restores anchor positions
**exactly** after the solve rather than accepting the pin row's ~1e-27
residual — "a wall never moves" is relied on bit-for-bit.

**Initial conditions: `IDACalcIC` cannot help, and is not called.**
Every joint is built from the pose the bodies are already in, so `g = 0`
and `J·u = 0` hold at t₀. The multipliers are then solved for directly
by `seed_multipliers`, from `g̈ = 0`:

```text
  (J M⁻¹ Jᵀ) λ = J u̇_applied + (dJ/dt) u
```

with `M⁻¹` block-diagonal (`1/m`, and `A = R I⁻¹ Rᵀ`), the angular
acceleration `ω̇ = A(L̇ - ω × L)` (differentiating `ω = A L` gives
`Ȧ = [ω]ₓA - A[ω]ₓ`; the `ω × L` term is the gyroscopic one), and
`(dJ/dt)u` as a central difference of `J·u` along the motion. The dense
`m × m` solve is `solve_dense`.

**This is required, not an optimisation.** The GGL system carries `g`
and `ġ` but *not* `g̈`, so at an instant where everything is at rest
`ġ = 0` holds whatever the accelerations are — free fall satisfies the
residual exactly with `λ = 0`. `IDACalcIC` therefore has nothing to
solve and leaves the derivative in free fall, after which BDF spends its
first step discovering that a hinge is attached and the step collapses
to `1e-15`.

**Which solver, and the gates.**

| entry point | solver | refuses |
|---|---|---|
| `integrate::run` with `Method::Ida` | `ida_rs` | spinning bodies, external torques (`ConstraintSet::gate`) |
| `integrate::run` with any other method and `!constraints.is_empty()` | — | names `METHOD IDA` |
| `equilibrium::solve` | `kinsol_rs`, `KIN_LINESEARCH` | all-anchor systems; a fully-free system gets the translation-invariance hint |
| `sensitivity::run`, unconstrained | `cvodes_rs` | spinning bodies (`sensitivity::gate`) |
| `sensitivity::run`, constrained | `idas_rs` | as above, plus the constraint gate |

**The sensitivity parameter vector is shared, not captured.**
`SensParams = Rc<RefCell<Vec<f64>>>` laid out
`[g | gravity(3) | e_field(3) | b_field(3) | mass(N) | charge(N)]`. The
right-hand side reads **every** value out of that vector on every call;
`plist` selects which entries are differentiated. That indirection is
what makes `fS: None` (internal difference quotients) correct — the
solver perturbs an entry and re-calls the same RHS. A right-hand side
that captured a copy would silently return zero sensitivities.

**All three paths are translational.** Positions and velocities, no
orientation, no 13N packing — deliberately narrower than
`integrate::rhs_full`, and gated rather than silently different. The
force block is byte-for-byte the same arithmetic order as `rhs_full`'s
(hard rule 3).

**Grammar.** `CONSTRAIN`/`CONSTRAINTS`/`EQUILIBRIUM`/`SENSITIVITY` and
`METHOD IDA` follow the §3.4 lockstep. `SENSITIVITY` takes its parameter
names as **STRING** tokens because `mass 0` is two tokens and would be
ambiguous against a duration followed by a number.

## 4. Error-handling policy

Library and VM return `Result<_, String>` with human-readable,
actionable messages (name the field, the column, the offending
feature). No panics in library code paths; `assert!`/`expect` only in
tests and examples. Solver failures propagate the SUNDIALS return code
in the message (negative = fatal, per SUNDIALS conventions). Scene
threads never panic on client input: malformed window messages become
error events; a solver failure during playback pauses the scene and
queues an event.

## 5. Dependency policy

- Allowed: `std`, and path deps `sundials_core`, `cvode_rs`,
  `arkode_rs` (and the four further `../sundials_rs` crates —
  `cvodes_rs`, `ida_rs`, `idas_rs`, `kinsol_rs` — if a future feature
  needs them), plus `special_functions` and its vendored `spec_math`.
- Forbidden: anything from crates.io — including for networking and
  the WebSocket layer (hand-rolled in `scene/ws.rs` on purpose). The
  gate: `Cargo.lock` must list **only local crates**. Today that is
  `arkode_rs 7.8.0`, `cvode_rs 7.8.0`, `sundials_core 7.8.0`,
  `physical_object 0.1.0`, `posim 0.1.0`, `quantum 0.1.0`,
  `spec_math 0.1.6`, `special_functions 0.1.0`. Any entry with a
  registry source means the rule was broken.
  `#![forbid(unsafe_code)]` + `#![deny(warnings)]` + the three `non_*`
  allows in every crate root.
- The browser page (`scene.html`) is dependency-free too: vanilla
  JS + canvas 2D, embedded via `include_str!`, no CDN fetches.
- The Python side (`jupyter/`) may use `ipykernel`/`jupyterlab` — it is
  outside the Rust constraint boundary by design.

## 6. Testing map

| layer | where | what it proves |
|---|---|---|
| linalg/boundary/struct/system units | `physical_object/src/*` `#[cfg(test)]` | math identities, setter invariants, donor-formula equivalence, pack/unpack round-trip; torus/disk/cylinder SDF values and inertia tensors, support extent/point exactness (`p·u = h(u)`, `torus_disk_cylinder_sdfs`, `support_extents_and_points_are_exact`); the dumbbell constructor, COM identity, union SDF, directed supports and composite inertia (`dumbbell_constructor_com_sdf_and_supports`) |
| solver integration | `physical_object/tests/conservation.rs` | analytic solutions (parabola, constant torque, gyration period), conservation, SPRK gate, empty-system edges |
| self-checking examples | `physical_object/examples/*` | long-horizon physics: donor solar-system cross-check, Kepler LRL, Dzhanibekov, gyroradius |
| language units | `posim/src/*` `#[cfg(test)]` | token streams (incl. scene keywords), postfix programs (incl. `scenecmd`), VM sessions, notebook magics, JSON round-trip; the shape/BOX surfaces end-to-end: `new_shapes_and_parameter_paths`, `box_family_and_infinite_mass_walls`, `torus_pair_is_order_independent_and_new_is_transactional` (transactional NEW + deferred torus), `box_recreate_after_wall_deletion_leaks_nothing` (vm), `state_reports_box_walls_and_inverse_mass` (machine); the function/name surface: `def_call_named_objects_and_dumbbell_members` (define/call/members/shorthands/renumber/errors/redefine/LET-defaults/ghost-free failing calls), and the parser pins that bare identifiers compile to `LoadIdent` while a mistyped root's runtime error still teaches `objN`/`contactK`/`system` |
| websocket primitives | `posim/src/scene/ws.rs` `#[cfg(test)]` | SHA-1 FIPS 180-4 vectors, base64 RFC 4648 vectors, the RFC 6455 §1.3 accept-key example |
| scene server integration | `posim/src/scene/mod.rs` `#[cfg(test)]` | HTTP page serving (toolbar/statusbar/gesture wiring present, incl. `bt-reset`), full WS session (handshake, init, camera sync, cmd start/pause, event path, frame broadcasts), forward-then-reverse playback landing exactly on t = 0, `box_shapes_and_wall_flags_reach_the_init_message`; `reset_restores_the_initial_state_and_start_reruns` (bit-identical restore of the initial state + restart), and the protocol tests assert `p`/`l` in every frame and the `name` in init |
| collision geometry/impulse units | `physical_object/src/collide.rs` `#[cfg(test)]` | known separations/depths/normals, SAT face vs. edge, separation continuity, n·Kn closed form, static-wall reflection (wall bit-unchanged), separating pairs untouched, projection; `ball_vs_torus_exact_contact_and_hole_passage`, `ball_vs_disk_and_cylinder_contacts`, `tilted_torus_fits_a_4x4_box_where_flat_does_not`, `static_slab_reflects_a_cylinder_elastically`, `parallel_cylinders_side_contact_is_exact` (radial rejection axes), `parallel_disk_disk_separation_is_the_documented_limitation` (the pinned face-on disk-disk limitation), `small_cap_on_large_face_contacts_at_the_cap_center` (equal-rank footprint rule), `dumbbell_wall_gaps_and_ball_contacts_are_exact` (asymmetric wall gaps exact for both ends, ball-vs-pole and ball-vs-rod contacts exact) |
| collision event integration | `physical_object/tests/collision.rs` | real CVODE/ARKODE events vs. closed forms: exchange + TOI < 1e-9, restitution ratios/e², cradle, ΔL = r×J, apex e²h, thin-wall no-tunnel, bit-identical zero-pair invariance, armed-but-quiet ≤ 1e-12, SPRK path + gate; `ball_in_a_rigid_box_conserves_energy_and_walls_never_move`, `point_threads_the_torus_hole_but_a_fat_ball_bounces` (TOI = (3−√1.45)/4), `mixed_shapes_rattle_in_the_box_conserving_energy` (|ΔE|/E < 1e-6), `ball_released_from_rest_does_not_tunnel_the_thin_plate` (acceleration-aware cap, from-rest TOI to 1e-6), `colliding_dumbbells_conserve_energy_momentum_and_angular_momentum` (two tumbling dumbbells off-center: E, P and L about the origin to 1e-8 through real CVODE events) |
| wire/kernel | `jupyter/test_protocol.py`, `jupyter/test_kernel.py` | machine protocol incl. the scene command family and the `events` op; full Jupyter ZMQ path |
| real-browser gestures | scratchpad `verify_gestures.py` (headless Chrome CDP; not committed) | genuine key/mouse/wheel input: arrows translate right/left/up/down, left-drag rotates, wheel and +/- zoom, toolbar Start/Pause/Reverse, statusbar reporting |

Regression invariant: `cargo test --workspace` green (**605 tests**:
52 physical_object lib + 19 collision + 9 conservation +
25 constrained/DAE + 113 posim + 92 quantum + 233 special_functions +
11 vendored identities + 55 doctests) and `cargo build --workspace --all-targets` warning-free at
every commit.

## 7. The video recorder (`recorder/`)

Outside the Rust constraint boundary, like `jupyter/`, but a **package
rather than a script**: it has its own CMake build, tests and
documentation ([`recorder/README.md`](recorder/README.md),
[`recorder/docs/FRAME_FORMAT.md`](recorder/docs/FRAME_FORMAT.md)), and
nothing in the cargo workspace refers to it. It drives `posim --machine`
over the §3.5 protocol: run the setup script line by line, then alternate
`{"op":"state"}` and `{"op":"exec","code":"step <dt>"}`, and emit one
self-contained HTML page with the frames embedded and a vanilla-JS canvas
player around them.

Two properties it keeps regardless of where the package sits, both
pinned:

- **The workspace is never inferred from what sits beside what.** The
  recorder does not assume the workspace is its own parent, because a
  checkout can hold more than one posim workspace — this port beside the
  upstream it came from — and the two agree bit for bit on everything the
  older grammar can express, so recording against the wrong one is
  silent. Resolution order is `--workspace` / `$POSIM_WORKSPACE`, then
  the manifest's `workspace` key, then the **scene script's** own
  ancestors, then the current directory's. Ancestors only, never
  siblings: the scene lives inside the workspace it belongs to. This is
  what lets the package be moved without editing any code.
- **The parameters of every shipped recording are recorded, not
  remembered.** `recorder/recordings.json` holds each one's scene,
  frame count, `dt`, opening camera, title and caption, because none of
  that is recoverable from a finished recording without picking the HTML
  apart. `record_all.py --check` re-records every entry into a temporary
  directory and compares byte for byte; that check is CTest test
  `recorder-e2e`.

Pinned properties:

- **It never integrates.** Every advance is a `step` through
  `integrate::run` — hard rule 1 holds for tooling too.
- **The page fetches nothing.** No CDN, no font server; it must work
  from `file://` on a machine with no network.
- **Wall slabs are never drawn as bodies**, and the camera auto-fit
  excludes them — the same rule the live window follows. A `BOX` is the
  dashed interior wireframe only.
- **Round shapes are symmetric about their local z axis** (boundary.rs),
  so the player rotates every ring by the w-first quaternion; this is
  what makes spin visible.
- Contact arrows carry the exact analytic normal from the frame's
  contact records, scaled by the applied impulse.
- **Joints are drawn from the machine protocol, not re-derived.** The
  `{"op":"state"}` dump carries a `joints` array with each joint's kind,
  bodies, row count and *world-frame* pivot and axis, plus
  `joint_drift`/`joint_rate_drift`. The body-frame arms a `Joint` stores
  are no use to a viewer; resolving them once, on the Rust side, keeps
  every front end honest about what the joint actually is.
- **A joint's two ends are both carried, and the player compares them.**
  Each frame's `j` entry is `[point, axis, point_j]`. BALL, HINGE and
  UNIVERSAL hold *one shared point*, so the two coincide and only the
  ring and axis are drawn; a rod holds *two points apart*, so they
  differ and the strut between them is drawn as well. Nothing in the
  player is keyed on the joint's name — the geometry decides, which is
  why a new joint kind needs no player change.
- **`PRISMATIC` is the mirror of `HINGE`**: same argument, same five
  rows, one freedom left — a rotation for the hinge, a translation for
  the slider. Two rows kill the offset across the axis and three lock
  the relative orientation. Its two translational rows have `R_i c`
  and the normal both riding on body `i`, and the contributions
  collapse to `δ_i·(n̂ × Δ)` with body `j`'s orientation dropping out
  entirely — checked by central differences rather than assumed.
- **A `RACK` solves the same wrapping problem the other way**, and the
  contrast is the point. A gear hides the wrapping in a sine and pays
  with a rational ratio; a rack has an unbounded coordinate — the
  travel — so it unwraps the angle *from the travel*,
  `k = round((Δs/r − θ_w)/2π)`, and needs no restriction on the
  radius. `k` is locally constant, so the Jacobian is the plain one.
  It misreads only once the joint is violated by `πr`. `RACK` stacks
  on bearings like `GEAR`, and is written against the pinion's turn
  **relative to the rack**, since the joint set has no prismatic
  constraint to hold an unguided rack square.
- **A `GEAR` is the one joint that is not geometric**, and two rules
  follow from it. It may **stack on a bearing** — `check_pair` refuses a
  second geometric joint on a pair but allows `HINGE 5 + GEAR 1`,
  because a wheel needs a bearing to hold it and a gear to drive it, and
  the rows are independent. And its ratio must be **rational**: the
  position-level row is `sin(q θ_i + p θ_j)` on wrapped angles, which is
  faithful only for integer `p, q`, since wrapping then moves the
  argument by a multiple of `2π`. A trivial `g ≡ 0` row was not an
  option — the DAE carries `g` and `J·u` as separate blocks (§3.3), so a
  row that is identically zero leaves its multiplier undetermined and
  the Jacobian singular.
- The camera auto-fit centres on the **content bounding box**, not the
  origin — a pendulum hangs below its pivot — and `--view front` opens
  looking down z for a planar linkage.

Recordings live in `videos/`, the scripts that produced them in
`videos/scenes/`.
