# rustSolveIt_Using_SUNDIALS_7_8_0

[`once-ere/rustSolveIt`](https://github.com/once-ere/rustSolveIt) — a
pure-Rust physics simulator — with its numerical engine upgraded from a
pure-Rust translation of **SUNDIALS 7.7.0** to one of **SUNDIALS
7.8.0**.

**➜ The release lives in [`version-7.8.0/`](version-7.8.0).**

```bash
git clone https://github.com/once-ere/rustSolveIt_Using_SUNDIALS_7_8_0.git
cd rustSolveIt_Using_SUNDIALS_7_8_0/version-7.8.0
cargo run                 # the notebook REPL (type HELP)
cargo test --workspace    # 605 tests
```

Zero `unsafe`, zero crates.io dependencies, zero warnings. The clone is
all you need — nothing is fetched at build time.

Descendants of this repository:
[`rustSolveIt_Win11_SUNDIALS_7_8_0`](https://github.com/once-ere/rustSolveIt_Win11_SUNDIALS_7_8_0)
is its Windows 11 port (byte-identical physics via a glibc-translated
math library), and out of that port's physics acceptance test came
[`rebound_rust`](https://github.com/once-ere/rebound_rust) — a
pure-Rust translation of the
[REBOUND](https://github.com/hannorein/rebound) 5.1.1 N-body code,
built with the same discipline (zero `unsafe`, zero dependencies, zero
warnings, C names preserved) and verified bit-for-bit against its
MSVC-compiled C reference.

## What changed, and what did not

| | before | after |
|---|---|---|
| engine | `once-ere/sundials_rs@faabb7f` (SUNDIALS 7.7.0, 394 files) | `once-ere/SUNDIALS_7_8_Rust_port_for_Linux@780b916` (SUNDIALS **7.8.0**, 2,929 files, vendored byte-identically) |
| solver families | CVODE, ARKODE | CVODE, ARKODE, **CVODES, IDA, IDAS, KINSOL** — all six now reachable from the language |
| first-party code touched | — | `physical_object/src/integrate.rs` and four examples (the port); then `constrain.rs`, `equilibrium.rs`, `sensitivity.rs` and the grammar |
| physics | — | **unchanged, and proven so** |

The 7.8.0 translation models C's opaque pointers directly — handles are
`Rc<RefCell<…>>` passed by shared reference, vector payloads come with a
borrow guard, constructors return `Option` where C returns `NULL`. That
is why the integration layer changed; not one constant, tolerance or
heuristic moved.

**The evidence that the physics did not move.** The six self-checking
physics examples, the twelve collision scripts and all **59 dynamic
notebooks** were run against both engines and diffed. Every one is
**byte-identical**. At the moment of the port 568 tests passed,
unchanged in count and composition — that equality is the point. The
suite is now **592** after the four further solver families were wired
into the simulator.
See
[`version-7.8.0/PORT_7.8.0_PROVENANCE.md`](version-7.8.0/PORT_7.8.0_PROVENANCE.md)
and [`version-7.8.0/evidence/port-7.8.0/`](version-7.8.0/evidence/port-7.8.0).

## What the six solver families do for you

| you type | question | solver |
|---|---|---|
| `STEP` / `RUN` | what happens next? | CVODE / ARKODE |
| `CONSTRAIN` / `GEAR` / `RACK` / `BALL` / `HINGE` / `UNIVERSAL` / `PRISMATIC` + `METHOD IDA` | …with this geometry held **exactly** | IDA |
| `EQUILIBRIUM` | where does it come to **rest**? | KINSOL |
| `SENSITIVITY 3 "gravity.y"` | how much does the answer **depend** on an input? | CVODES, or IDAS when constrained |

A rod is a geometric fact, not a stiff spring — so a `CONSTRAIN` turns
the equations of motion into a differential-algebraic equation, and IDA
holds the rod to **one bit** over a full pendulum period. `EQUILIBRIUM`
hangs that pendulum straight down to 13 digits. And `SENSITIVITY` gets
`∂y/∂g = T²/2` on free fall to 1.3 parts in 10⁸ — while returning
**exactly zero** for `∂y/∂mass`, because in uniform gravity there is no
dependence to find.

## Start here

| document | for |
|---|---|
| [`version-7.8.0/SolveIt.md`](version-7.8.0/SolveIt.md) · [`.pdf`](version-7.8.0/SolveIt.pdf) | the complete solution guide, written for a first-time reader, with **16** fully documented worked examples |
| [`version-7.8.0/grammar.md`](version-7.8.0/grammar.md) · [`.pdf`](version-7.8.0/grammar.pdf) | the complete command language: lexer, full EBNF, type system, stack machine, the 7.8.0 engine, browser videos, and **18** more worked examples |
| [`version-7.8.0/ARCHITECTURE.md`](version-7.8.0/ARCHITECTURE.md) | the pinned cross-module contracts, for anyone changing the code |
| [`version-7.8.0/CLAUDE.md`](version-7.8.0/CLAUDE.md) | the working rules for contributors and agents |
| [`version-7.8.0/README.md`](version-7.8.0/README.md) | the full project README, including the 34 Routh notebooks and the index of 6,309 entities |

## Browser videos

Thirteen recorded runs, openable offline — no server, no CDN, nothing
fetched. Scrub, orbit, and read the conserved quantities off whichever
frame you stopped on.

| video | measured over the recording |
|---|---|
| [`kepler_ellipse.html`](version-7.8.0/videos/kepler_ellipse.html) — an `e = 0.6` orbit | \|dE\|/E = 9.8e-8 |
| [`tumbling_racket.html`](version-7.8.0/videos/tumbling_racket.html) — the Dzhanibekov flip | \|d\|L\|\|/\|L\| = **0 exactly** |
| [`box_of_shapes.html`](version-7.8.0/videos/box_of_shapes.html) — three shapes in a rigid box | 36 collisions, \|dE\|/E = 3.4e-16 |
| [`double_pendulum_hinges.html`](version-7.8.0/videos/double_pendulum_hinges.html) — **two hinges**, the chaotic linkage | joints hold to \|g\| = 5.6e-8 |
| [`universal_joint.html`](version-7.8.0/videos/universal_joint.html) — a **universal joint** driving a shaft | bend bounded at cos b = 0.6000004 vs exactly 0.6 |
| [`ball_joint_chain.html`](version-7.8.0/videos/ball_joint_chain.html) — a **chain on ball joints** | leaves its plane: \|z\| from exactly 0 to 1.7147 |
| [`rod_pendulum_chain.html`](version-7.8.0/videos/rod_pendulum_chain.html) — **four rods**, the cheapest joint | \|g\| = 5.4e-15 run continuously |
| [`spinning_top.html`](version-7.8.0/videos/spinning_top.html) — a **precessing top** | 1.020440 rad/s vs the exact 1.020408 |
| [`gyroscope_gimbal.html`](version-7.8.0/videos/gyroscope_gimbal.html) — a **gimballed gyroscope** | L·ŷ conserved to 1.4e-14 |
| [`cardan_compass.html`](version-7.8.0/videos/cardan_compass.html) — a **ship's compass** in gimbals | two pendulum periods matched |
| [`cardan_gear.html`](version-7.8.0/videos/cardan_gear.html) — **Cardan gears** on a `GEAR` row | line held to 1.1e-8 |
| [`rack_and_pinion.html`](version-7.8.0/videos/rack_and_pinion.html) — a **rack and pinion drive** | falls at exactly g/2 |
| [`piston_crankshaft.html`](version-7.8.0/videos/piston_crankshaft.html) — the **slider-crank** | exact kinematics to 8.4e-8 |

Every advance in a recording is a real SUNDIALS step; the recorder
(`version-7.8.0/recorder/`) is a camera, not a physics
engine.

## Licence

BSD-3-Clause. The vendored `sundials_rs/` carries its own upstream
`LICENSE` and `NOTICE` (Lawrence Livermore National Security et al.).
See [`version-7.8.0/THIRD_PARTY.md`](version-7.8.0/THIRD_PARTY.md).
