# SolveIt — the complete solution guide

*Written for a reader who has never used this program, has never run a
physics simulator, and does not write Rust. Every term is defined the
first time it is used. Every number printed below was produced by the
program on the machine that wrote this file; nothing is quoted from
memory.*

---

## Table of contents

1. [What this is, in one page](#1-what-this-is-in-one-page)
2. [Installing and running it](#2-installing-and-running-it)
3. [The five things you need to know](#3-the-five-things-you-need-to-know)
4. [What the program is made of](#4-what-the-program-is-made-of)
5. [The engine: SUNDIALS 7.8.0 in pure Rust](#5-the-engine-sundials-780-in-pure-rust)
6. [How collisions actually work](#6-how-collisions-actually-work)
7. [Sixteen worked examples](#7-sixteen-worked-examples)
8. [Watching it: the scene window and browser videos](#8-watching-it-the-scene-window-and-browser-videos)
9. [How you know the numbers are right](#9-how-you-know-the-numbers-are-right)
10. [When something goes wrong](#10-when-something-goes-wrong)
11. [Where everything lives](#11-where-everything-lives)

---

## 1. What this is, in one page

**SolveIt** simulates classical mechanics and one-, two- and
three-dimensional quantum mechanics. You describe a situation — some
bodies, some fields, maybe a rigid box to put them in — and it tells you
what happens next, to as many digits as the arithmetic allows.

You talk to it by typing short commands, one per line:

```
In[1]:= new sphere { mass = 2, radius = 0.5, position = [0, 10, 0], velocity = [1, 0, 0] }
Out[1]= obj0
In[2]:= set system.gravity = [0, -9.81, 0]
In[3]:= step 1
Out[3]= t = 1 (advanced by 1, 12 solver steps)
In[4]:= get obj0.position
Out[4]= [1, 5.095000000000006, 0]
```

That is the whole interface. No configuration files, no build system to
learn, no graphics API.

**The one thing that makes this different from a game engine.** Game
physics is allowed to be approximately right — it only has to look
plausible for the next sixteen milliseconds. This program's job is to be
*measurably* right. Every trajectory it prints comes out of a
production-grade differential-equation solver with genuine error
control, and every claim in the documentation is a number you can
reproduce by running the command underneath it.

Concretely, and these are the actual measured values:

| claim | measured |
|---|---|
| the outer solar system, integrated for 1,369 years | Pluto's position matches the reference to 8 decimal places; total energy drifts by 7.8 parts in 10⁷ |
| a torque-free tumbling body | angular momentum changes by **exactly zero** |
| a ball dropped on a plate | the moment of impact is right to 1 part in 10¹⁶ |
| three shapes rattling in a rigid box, 36 collisions | energy conserved to 3 parts in 10¹⁶ |
| Kepler's third law across four orbits | *T²/a³* identical to the last bit for all four |

### The rules the code lives under

Four of them, and they are worth stating because they explain a lot
about how this program behaves:

1. **Everything is integrated by SUNDIALS.** There is no hand-written
   stepper anywhere in the program — not in the examples, not in the
   graphics playback, not as a "fast path".
2. **No `unsafe` code.** Rust's memory-safety checks are on everywhere,
   with no escape hatches.
3. **No dependencies at all.** Not one package from the internet. The
   solver library, the web server behind the graphics window, the
   WebSocket protocol, the SHA-1 implementation, the special-function
   library — all of it is in this repository.
4. **No compiler warnings.** The build is clean, and it is set up to
   *fail* if it stops being clean.

Rule 3 has a consequence worth spelling out: `git clone`, `cargo build`,
done. No network access during the build, nothing to go stale, nothing
that can be pulled out from under you.

---

## 2. Installing and running it

You need Rust. If you do not have it:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then:

```bash
git clone https://github.com/once-ere/rustSolveIt_Using_SUNDIALS_7_8_0.git
cd rustSolveIt_Using_SUNDIALS_7_8_0/version-7.8.0
cargo run
```

The last command builds everything and drops you at the `In[1]:=`
prompt. Type `help` for the reference card, `quit` to leave.

### The four ways to run it

| how | command | when |
|---|---|---|
| **interactive notebook** | `cargo run` | you are exploring |
| **batch script** | `cargo run -p posim -- --script my_run.posim` | you want it reproducible |
| **dynamic notebook** | `cargo run -p posim --release -- --notebook dynamic_notebooks/kepler_orbit.posim` | run a saved session, open its graphics window, then keep typing |
| **machine mode** | `cargo run -p posim -- --machine` | another program is driving; this is what JupyterLab uses |

A "script" is just a text file of the same commands you would type.
Comments start with `#`. Every example in section 7 is a script in
[`scripts/solveit/`](scripts/solveit); run any of them directly.

### Checking your build

```bash
cargo test --workspace
```

605 tests should pass. If they do, your build is the same build this
documentation was written against.

---

## 3. The five things you need to know

Everything else is detail. These five will save you an afternoon each.

### 3.1 `RUN` takes a duration, not a destination

```
In[1]:= run 1.7 steps 170        # advances BY 1.7, so t is now 1.7
In[2]:= run 1.19 steps 120       # advances BY 1.19, so t is now 2.89
```

If you want to stop at an absolute time, subtract. This trips up
everyone once. `STEP dt` is the same thing with one output point.

### 3.2 Gravity is on by default, and it is between *bodies*

`system.g_constant` defaults to **1**, which means every pair of bodies
attracts every other pair. That is what you want for a solar system and
emphatically not what you want for three blocks in a box: two touching
surfaces are at nearly zero separation, and the attraction between them
is nearly singular.

If you are studying collisions, turn it off:

```
set system.g_constant = 0
```

The difference is not subtle. The same three-shapes-in-a-box scenario
conserves energy to **3 parts in 10¹⁶** with `G = 0`, and drifts by
**3.2 %** with `G = 1`. Neither number is a bug — they are different
physical systems. `system.uniform_gravity` is the separate thing you
usually mean by "gravity": a constant downward field, like a laboratory.

### 3.3 Objects are numbered, and deleting renumbers them

`obj0`, `obj1`, … in creation order. `DEL 1` removes `obj1` **and
renumbers everything after it**. If that sounds fragile, it is — so give
things names instead:

```
In[1]:= new sphere as earth { mass = 1, radius = 0.1 }
Out[1]= obj0 as earth
In[2]:= get earth.position
```

Names survive deletions of other objects.

### 3.4 Collisions must be switched on

```
collide on
```

Without it, bodies pass through each other. This is a deliberate
default: collision detection changes how the solver picks its steps,
and a great many problems (orbits, fields, quantum) never need it.

### 3.5 `|dE/E|` is printed for free, and you should read it

Every `RUN` reports the relative change in total energy:

```
Out[7]= t = 2 (53 solver steps, 200 snapshots, |dE/E| = 0.000e0, 1 collision(s))
```

For a closed system this should be tiny. If it is not, either the
physics genuinely does not conserve energy (a magnetic torque, an
inelastic bounce, a driven potential — see examples 4 and 8) or
something is wrong with your setup. It is the cheapest sanity check you
will ever get.

---

## 4. What the program is made of

Six pieces. You only ever type at the first one.

| piece | what it does |
|---|---|
| **`posim`** | the command language: reads your line, checks it, compiles it to a tiny stack program, runs it. Also the graphics window and the machine-mode protocol. |
| **`physical_object`** | the physics: one `physical_object` type that is simultaneously a point mass, a rigid body and a shaped, collidable solid. Plus all the time integration. |
| **`quantum`** | 1-D, 2-D and 3-D quantum mechanics: bound states, wavepacket propagation, tunnelling, absorbing boundaries. |
| **`special_functions`** | Bessel, Legendre, Hankel, Airy, gamma, Wigner symbols, quadrature, eigenproblems — the mathematical machinery the other two need. |
| **`sundials_rs`** | the differential-equation solvers. Section 5. |
| **`jupyter`** | a JupyterLab kernel, so you can run all of this in a notebook in your browser. |

### The one type that does everything

The original 2002 SolveIt had three separate types: a point particle, a
rigid body, and a 3-D rigid body. Each had its own integrator, its own
bugs, and its own idea of what "position" meant. This version has
**one** type, `physical_object`, which is the union of all three. A
point mass is just a `physical_object` whose shape is `POINT` and whose
inertia tensor is zero.

That sounds like a detail. It is why a point mass, a spinning cuboid and
a dumbbell can all be in the same collision and the same conservation
sum, without a single special case.

### The state vector

Every object carries **13 numbers**:

```
[ position (3) | momentum (3) | orientation quaternion (4) | angular momentum (3) ]
```

Momentum rather than velocity, and angular *momentum* rather than
angular velocity, both on purpose: those are the quantities that are
conserved, and the ones that behave correctly when you change a mass
mid-run. Quaternions are written **w first**: `[w, x, y, z]`. That
convention holds everywhere — in the language, in the JSON protocol, in
the graphics, and in this document.

---

## 5. The engine: SUNDIALS 7.8.0 in pure Rust

### 5.1 What SUNDIALS is

**SUNDIALS** — SUite of Nonlinear and DIfferential/ALgebraic equation
Solvers — is Lawrence Livermore National Laboratory's library of
differential-equation solvers. It has been in production use in physics
and engineering codes since the 1990s. It is written in C.

`sundials_rs/` in this repository is a **line-for-line translation of
SUNDIALS 7.8.0 into pure Rust**. Not a wrapper, not bindings: there is
no C compiler involved, no `.so` to link, no FFI boundary. It carries
the same control flow, the same constants, the same tolerances, and — a
subtler point that matters a great deal — the same *order of arithmetic
operations*, because floating-point addition is not associative and
changing the order changes the answer in the last digits.

### 5.2 Which solver runs when

| you type | what runs |
|---|---|
| `method adams` (the default), then `step`/`run` | **CVODE Adams–Moulton**: variable order, variable step size, with Newton iteration and a dense Jacobian estimated by finite differences. The general-purpose choice. |
| `method bdf`, then `step`/`run` | **CVODE BDF**: the backward differentiation formulas, for *stiff* problems — ones with two very different timescales, like a fast magnetic gyration inside a slow drift. |
| `method sprk <table> [dt]`, then `step`/`run` | **ARKODE SPRKStep**: a *symplectic* partitioned Runge–Kutta method at a fixed step. Only valid for separable systems, and worth it when you have one — see example 6. |

"Adaptive" means the solver chooses its own internal step size to keep
the estimated local error under the tolerance you asked for. When it
tells you `1289 solver steps` for a run you asked to sample at 400
points, those 1289 are its own steps; the 400 are just where it stopped
to tell you the answer. **The output cadence does not affect the
physics.** That property is tested, not assumed.

### 5.3 What the 7.8.0 upgrade brought

This repository is `once-ere/rustSolveIt` with its 7.7.0 engine replaced
by the 7.8.0 one. Two things came with it.

**Four more solver families.** The 7.7.0 vendoring shipped CVODE and
ARKODE. 7.8.0 brings the whole suite:

| crate | solves | used here? |
|---|---|---|
| `cvode_rs` | ordinary differential equations (Adams / BDF) | yes |
| `arkode_rs` | Runge–Kutta: explicit, implicit, IMEX, multirate, symplectic | yes |
| `ida_rs` | **differential-algebraic** systems, `F(t, y, ẏ) = 0` | yes — `CONSTRAIN` + `METHOD IDA` |
| `kinsol_rs` | **nonlinear algebraic** systems, with Anderson acceleration | yes — `EQUILIBRIUM` |
| `cvodes_rs` | CVODE plus forward and adjoint **sensitivity** analysis | yes — `SENSITIVITY` |
| `idas_rs` | IDA plus sensitivity analysis | yes — `SENSITIVITY` while constrained |

All six are compared byte-for-byte against the output of the original C
programs. The last four arrived with the 7.8.0 engine, and §5.4 is what
they do for you.

### 5.4 The three questions that are not "what happens next"

`STEP` and `RUN` answer one question. The four families above answer
three more.

**Four joints.** `CONSTRAIN a b` is a rod (1 row, 5 freedoms left).
`GEAR a b axis ratio` locks two turns in proportion (1 row), and `RACK pinion bar axis dir radius` ties a turn to a slide (1 row) — between them the two joints that cross from rotation to rotation and from rotation to translation, which is what a gear train, a chain drive or a steering rack is made of. `PRISMATIC a b axis` is the slider (5 rows): the mirror of a hinge, leaving one translation instead of one rotation, and what a rack or a piston runs in. `BALL a b` shares a point (3 rows). `UNIVERSAL a b u w` shares a point
and keeps two shafts square — a Cardan joint (4 rows). `HINGE a b axis`
shares a point *and* an axis, leaving exactly one freedom (5 rows): a
door, a knee, a pendulum. The last three grip orientation, which is why
`METHOD IDA` carries the full 13-numbers-per-object state.

**A rigid rod, held exactly** — `CONSTRAIN a b` then `METHOD IDA`. A rod
is a geometric fact, not a very stiff spring: the bob is *exactly* `L`
from the pivot, always. Saying so turns the equations of motion from an
ODE into a differential-algebraic equation, which is what IDA solves. A
body with `inverse_mass = 0` becomes an **anchor** — it never moves and
absorbs the reaction — so anchor + bob + rod is a pendulum. `CONSTRAINTS`
reports how far the rod has strayed from its length; over a 20-second
swing it stays at roundoff.

**Where it comes to rest** — `EQUILIBRIUM`. Not an integration: KINSOL
solves for a configuration where every free body has zero net force,
every anchor is where it started and every rod is the right length, then
puts the bodies there and stops them. It says nothing about *stability* —
a pencil on its point is an equilibrium too — so perturb the answer and
`RUN` if you need to know.

**How much the answer depends on an input** — `SENSITIVITY 3 "gravity.y"`.
Running twice with slightly different inputs and subtracting throws away
most of your digits. This integrates the derivative alongside the state
instead. CVODES does it for an ordinary system; IDAS does it when a
`CONSTRAIN` is active, because only then are the equations a DAE.

Examples 17 and 18 in §7 work all three, with the closed forms to check
them against.

**A more faithful API.** The 7.8.0 translation models C's opaque
pointers directly rather than smoothing them into idiomatic Rust. Vector
data is reached through a borrow guard, constructors return `Option`
where C returns a possibly-null pointer, and destructors blank the
handle the way `free(p); p = NULL;` does. None of that is visible from
the command language — but it is why `physical_object/src/integrate.rs`
changed, and the full call-by-call table is in
[`PORT_7.8.0_PROVENANCE.md`](PORT_7.8.0_PROVENANCE.md).

**And the physics did not move.** All six self-checking physics
examples, all twelve collision scripts and all 59 dynamic notebooks
produce output that is byte-for-byte identical under 7.7.0 and 7.8.0.
The evidence is in [`evidence/port-7.8.0/`](evidence/port-7.8.0).

---

## 6. How collisions actually work

This is the part of the program most worth understanding, because the
approach is genuinely unusual.

### 6.1 The problem with checking every frame

The obvious way to detect a collision is to advance time a little, then
ask "are these two objects overlapping?" If yes, back up and resolve it.

This fails in a specific, embarrassing way. Consider a bullet at 100 m/s
and a plate 5 mm thick. In one hundredth of a second the bullet moves a
full metre — two hundred times the plate's thickness. Check before the
step: not overlapping. Check after: not overlapping. The bullet is now
on the other side, and the program never noticed. This is called
**tunnelling**, and it is why fast objects in games sometimes fall
through floors.

### 6.2 What this program does instead

It hands the solver a **continuous function** — the signed gap between
two surfaces, positive when apart, negative when overlapping — and asks
it to find the moment that function crosses zero. This is called
**rootfinding**, and it is a first-class feature of CVODE, not something
bolted on top.

Three things follow.

**The impact time is exact, not quantised to a frame.** The solver
interpolates its own solution back to the crossing. Example 10 below
catches the bullet at `t = 0.01995500000000067`; the analytic answer,
1.9955 m at 100 m/s, is `0.019955`.

**The solver caps its own step size** while rootfinding is armed, so
that no single step can carry one body clear through another. The cap is
computed from the smallest feature among the paired bodies divided by
the fastest surface approach speed — and it accounts for *acceleration*,
so a body released from rest above a thin plate still gets a finite cap
rather than an infinite one.

**Systems with nothing to collide are unaffected.** If there are no
collidable pairs, the code path is bit-identical to running with
collisions off. That is a tested property, not a hope.

### 6.3 What happens at the moment of contact

The solver stops at the crossing and hands control back. The program
then:

1. reads the state at exactly that instant;
2. computes the contact point and the contact normal from the two
   shapes' actual geometry (a *support function*, not a bounding
   sphere — see example 13 for why that matters);
3. applies an impulse along the normal, sized by the two bodies' masses,
   inertias and coefficients of restitution;
4. re-initialises the solver at the new state and carries on toward the
   same output time.

The impulse is applied at *one shared point* for both bodies, which is
what makes angular momentum come out right for off-centre hits.

### 6.4 The Zeno problem, and what is done about it

A bouncing ball with restitution below 1 makes infinitely many bounces
in finite time. Its bounces get closer and closer together and a naive
event loop will grind to a halt trying to resolve them all.

The program counts events per *burst* — a run of events with no real
flight between them. Past a threshold it forces the remaining contacts
plastic; past twice the threshold it projects out any overlap and
disarms rootfinding for the rest of the output interval. The counting is
deliberately per-burst rather than per-output-interval, because
per-interval counting made the physics depend on how often you asked for
output, which is exactly the property section 5.2 promises you have.

---

## 7. Nineteen worked examples

Every transcript below is genuine program output. The scripts are in
[`scripts/solveit/`](scripts/solveit) and you can run any of them:

```bash
cargo run -p posim --release -- --script scripts/solveit/01_elastic_head_on.posim
```

The examples are ordered roughly by difficulty and each one checks the
simulator against something known independently — a closed-form
solution, a textbook identity, or a hand calculation.

---

### Example 1 — a head-on elastic collision between unequal masses

**The physics.** Two spheres, masses 3 and 1, approach head-on at
+2 and −4. Elementary mechanics says that for a perfectly elastic
collision,

*v₁′ = ((m₁ − m₂)u₁ + 2m₂u₂)/(m₁ + m₂)*  and the mirror image for *v₂′*.

**Why it is a real test.** The answer is exact, and both energy and
momentum must survive the impulse untouched. Any error in the impulse
solver shows up immediately.

```
In[1]:= set system.g_constant = 0
In[2]:= collide on
In[3]:= new sphere as heavy { mass = 3, radius = 0.5, position = [-3, 0, 0], velocity = [2, 0, 0] }
Out[3]= obj0 as heavy
In[4]:= new sphere as light { mass = 1, radius = 0.5, position = [ 3, 0, 0], velocity = [-4, 0, 0] }
Out[4]= obj1 as light
In[5]:= energy
Out[5]= 14
In[6]:= momentum
Out[6]= [2, 0, 0]
In[7]:= run 2 steps 200
Out[7]= t = 2 (53 solver steps, 200 snapshots, |dE/E| = 0.000e0, 1 collision(s) — CONTACTS lists them)
In[8]:= get heavy.velocity
Out[8]= [-1, 0, 0]
In[9]:= get light.velocity
Out[9]= [5, 0, 0]
```

Now the same numbers from the formula, computed **inside the language** —
`LET` makes a session variable, and a line that is just an expression is
evaluated and printed:

```
In[10]:= let m1 = 3
In[11]:= let m2 = 1
In[12]:= let u1 = 2
In[13]:= let u2 = -4
In[14]:= ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
Out[14]= -1
In[15]:= ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
Out[15]= 5
In[16]:= energy
Out[16]= 14
In[17]:= momentum
Out[17]= [2, 0, 0]
```

**What to notice.** `-1` and `5` exactly, not approximately. Energy 14
before and 14 after; momentum `[2, 0, 0]` before and after. And
`|dE/E| = 0.000e0` — the run lost nothing at all.

---

### Example 2 — Kepler's third law, four orbits at once

**The physics.** For a circular orbit of radius *a* about a mass *M*,
the period is *T = 2πa^{3/2}/√(GM)*. Kepler's third law is the statement
that *T²/a³* is the same number for every orbit around the same central
mass — with *G* = 1 and *M* = 1000 that number is 4π²/1000.

**Why it is a real test.** Four different orbits, four different
periods, one shared constant. And each planet must come *home* after
exactly one period — a closed-loop check that catches any systematic
drift.

```
In[1]:= set system.g_constant = 1
In[2]:= new sphere as sun { mass = 1000, radius = 0.2 }
In[3]:= new sphere as p1  { mass = 0.000001, radius = 0.02, position = [1, 0, 0], velocity = [0, 0, 31.622776601683793] }
In[4]:= run 0.19869176531592203 steps 200
Out[4]= t = 0.19869176531592203 (661 solver steps, 200 snapshots, |dE/E| = 1.731e-8)
In[5]:= get p1.position
Out[5]= [1.0000000044544244, 0, -0.00000014431704032681625]
```

…and the same for radius 2, 3 and 4, each run for its own period:

```
Out[11]= [2.000000000309673, 0, 0.000000026045939587396316]
Out[17]= [3.0000000004230944, 0, 0.00000004071881288571222]
Out[23]= [4.000000000077065, 0, 0.00000007248745821777447]
```

Every planet is back where it started, to between 4 and 300 parts in a
billion. Now *T²/a³* for the four:

```
In[24]:= 0.19869176531592203 * 0.19869176531592203 / 1
Out[24]= 0.039478417604357434
In[25]:= 0.5619851784832581 * 0.5619851784832581 / (2 * 2 * 2)
Out[25]= 0.039478417604357434
In[26]:= 1.0324326977181857 * 1.0324326977181857 / (3 * 3 * 3)
Out[26]= 0.039478417604357434
In[27]:= 1.5895341225273762 * 1.5895341225273762 / (4 * 4 * 4)
Out[27]= 0.039478417604357434
```

**What to notice.** All four are the same 17 digits. Not "close" —
identical. That is 4π²/1000 = 0.039478417604357434.

---

### Example 3 — bound, parabolic and hyperbolic, decided by a sign

**The physics.** The shape of an orbit is decided by one number: the
specific orbital energy *ε = v²/2 − GM/r*. Negative is an ellipse
(bound), exactly zero is a parabola (escaping, but only just), positive
is a hyperbola (escaping with speed to spare). The escape speed at
radius *r* is *√(2GM/r)*.

**Why it is a real test.** The parabolic case must come out *exactly*
zero. Any error in the potential energy formula shows up as a nonzero
third number.

```
In[1]:= set system.g_constant = 1
In[2]:= new sphere as sun { mass = 1000, radius = 0.2 }
In[3]:= new sphere as slow { mass = 0.000001, radius = 0.02, position = [4, 0, 0],  velocity = [0, 0, 20] }
In[4]:= new sphere as esc  { mass = 0.000001, radius = 0.02, position = [4, 0, 10], velocity = [0, 0, 22.360679774997898] }
In[5]:= new sphere as fast { mass = 0.000001, radius = 0.02, position = [4, 0, 20], velocity = [0, 0, 24] }
In[6]:= let gm = 1000
In[7]:= 20 * 20 / 2 - gm / 4
Out[7]= -50
In[8]:= 22.360679774997898 * 22.360679774997898 / 2 - gm / 4
Out[8]= 0.00000000000002842170943040401
In[9]:= 24 * 24 / 2 - gm / 4
Out[9]= 38
```

**What to notice.** −50, then 2.8 × 10⁻¹⁴ (the closest a 64-bit float
gets to zero after squaring an irrational number), then +38. Three
conics from one formula. The escape speed at *r* = 4 with *GM* = 1000 is
√500 = 22.360679774997898, and squaring it recovers 500 to the last bit
the format allows.

---

### Example 4 — the restitution ladder, and honest energy loss

**The physics.** A ball with coefficient of restitution *e* leaves each
bounce at *e* times its arrival speed, so it reaches *e²* of its
previous height. After *n* bounces it reaches *e^{2n}h*.

**Why it is a real test.** It checks the impulse magnitude, the impact
timing and the free-flight integration all at once, three times over,
against a formula a schoolchild can verify.

The ball's *centre* starts at 5.5 with radius 0.5, so it falls 5.0 to
the floor's top face. With *g* = 10 and *e* = 0.7 the schedule is:
bounce at *t* = 1.0, apex at 1.7; bounce at 2.4, apex at 2.89; bounce at
3.38, apex at 3.723. **Remember `RUN` takes a duration** — the three
runs are 1.7, then 1.19, then 0.833.

```
In[3]:= collide on
In[4]:= new cuboid as floor { mass = 1, half_extents = [50, 0.05, 50], position = [0, -0.05, 0], inverse_mass = 0 }
In[5]:= new sphere as ball  { mass = 1, radius = 0.5, position = [0, 5.5, 0], restitution = 0.7 }
In[6]:= run 1.7 steps 170
Out[6]= t = 1.7 (321 solver steps, 170 snapshots, |dE/E| = 4.636e-1, 1 collision(s) — CONTACTS lists them)
In[7]:= get ball.position.y
Out[7]= 2.949999999869974
In[8]:= 5 * 0.7 * 0.7 + 0.5
Out[8]= 2.9499999999999997
In[9]:= run 1.19 steps 120
Out[9]= t = 2.8899999999999997 (164 solver steps, 120 snapshots, |dE/E| = 4.236e-1, 1 collision(s) — CONTACTS lists them)
In[10]:= get ball.position.y
Out[10]= 1.7004999998062575
In[11]:= 5 * 0.7 * 0.7 * 0.7 * 0.7 + 0.5
Out[11]= 1.7004999999999997
In[12]:= run 0.833 steps 84
Out[12]= t = 3.723 (87 solver steps, 84 snapshots, |dE/E| = 3.600e-1, 1 collision(s) — CONTACTS lists them)
In[13]:= get ball.position.y
Out[13]= 1.0882449997748824
In[14]:= 5 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 + 0.5
Out[14]= 1.0882449999999997
In[15]:= get system.collisions
Out[15]= 3
```

**What to notice.** Three apexes, each matching the formula to about
2 × 10⁻¹⁰ — and each *arrived at at the right moment*, which is the
harder half.

Also notice `|dE/E| = 4.636e-1`. Energy is **not** conserved here, and
should not be: *e* = 0.7 means each bounce deliberately throws away
51 % of the kinetic energy. A conservation check is only meaningful when
the physics is conservative. This is the first place most people
misread the diagnostic.

---

### Example 5 — cyclotron motion, and when to reach for BDF

**The physics.** A charge *q* moving at speed *v* perpendicular to a
magnetic field *B* travels in a circle of radius *r = m|v|/(|q|B)* with
period *T = 2πm/(|q|B)*. With *m* = 2, *q* = −3, *B* = 6 and
*|v|* = 5 that is *r* = 5/9 = 0.5555… and *T* = 0.6981317007977318.

**Why it is a real test.** The Lorentz force *qv×B* depends on velocity,
which rules out symplectic methods entirely (see example 6), and it is
fast compared with anything else in the problem — the textbook
definition of a *stiff* system. This is what `METHOD BDF` is for.

Half a period puts the ion diametrically opposite its start, so the
distance from the origin must be exactly *2r*.

```
In[2]:= set system.b_field = [0, 0, 6]
In[3]:= new sphere as ion { mass = 2, charge = -3, radius = 0.05, position = [0, 0, 0], velocity = [5, 0, 0] }
In[4]:= method bdf
Out[4]= method = CVODE BDF
In[5]:= run 0.3490658503988659 steps 200
Out[5]= t = 0.34906585039886584 (178 solver steps, 200 snapshots, |dE/E| = 8.696e-9)
In[6]:= get ion.position
Out[6]= [-0.00000000020913167204927863, 1.1111111086955652, 0]
In[7]:= norm(ion.position)
Out[7]= 1.1111111086955652
In[8]:= 2 * (2 * 5 / (3 * 6))
Out[8]= 1.1111111111111112
In[9]:= run 0.3490658503988659 steps 200
In[10]:= get ion.position
Out[10]= [0.000000000408039691444928, 0.000000005381884658509583, 0]
In[11]:= get ion.velocity
Out[11]= [4.999999951563042, 0.0000000036723595761567474, 0]
```

**What to notice.** The half-period diameter is 1.1111111086955652
against an analytic 1.1111111111111112 — right to 8 significant figures.
After the full period the ion is back at the origin to 5 parts in a
billion, with its original velocity restored. The magnetic force does no
work, and the speed confirms it.

---

### Example 6 — symplectic versus adaptive, over a long run

**The physics.** Two ways to integrate an orbit for a long time. CVODE
Adams controls the *local* error at every step, which is the right thing
to ask for in general. A **symplectic** method controls something else
entirely: it exactly preserves the geometric structure of Hamiltonian
mechanics, which means the energy error does not accumulate — it
oscillates around a constant.

**Why it is a real test.** Same system, same span, two methods,
and the comparison is not close.

```
In[4]:= method adams
In[5]:= run 60 steps 60
Out[5]= t = 60 (507075 solver steps, 60 snapshots, |dE/E| = 1.459e-8)
In[6]:= energy
Out[6]= -0.20000000291885495
In[7]:= reset
In[11]:= method sprk mclachlan_4_4 0.001
Out[11]= method = ARKODE SPRK ARKODE_SPRK_MCLACHLAN_4_4, fixed dt = 0.001
In[12]:= run 60 steps 60
Out[12]= t = 60 (60000 solver steps, 60 snapshots, |dE/E| = 8.957e-11)
In[13]:= energy
Out[13]= -0.20000000001790574
```

**What to notice.** The symplectic method took **60,000** steps to the
adaptive method's **507,075**, and its energy error is **163 times
smaller**. Eight times less work, two orders of magnitude better answer.

**The catch, and it is a real one.** SPRK needs a *separable*
Hamiltonian — the energy has to split cleanly into a kinetic part
depending only on momentum and a potential part depending only on
position. A magnetic field breaks that, and so does rotation. The
program does not let you get this wrong quietly:

```
Err: SPRK method requires a separable Hamiltonian: magnetic field B must be
     zero (the Lorentz force q v x B is velocity-dependent); use METHOD ADAMS or BDF
```

The error names the feature that blocks it. That refusal is why example
5 uses BDF.

---

### Example 7 — the tennis-racket theorem, measured

**The physics.** A rigid body has three principal axes with three
different moments of inertia. Spin it about the largest or the smallest
and the motion is stable. Spin it about the **middle** one and it is
not: the body flips end over end, over and over. Cosmonaut Vladimir
Dzhanibekov noticed this on Salyut 7 in 1985 with a wing nut, and the
video is famous.

**Why it is a real test.** The flip is a genuine instability, so it is
exquisitely sensitive to integration error — and yet the angular
momentum must be *exactly* conserved throughout, because nothing is
applying a torque. Those two demands pull in opposite directions.

```
In[1]:= new cuboid as racket { mass = 1, half_extents = [1.6, 0.9, 0.25], angular_momentum = [0.02, 3, 0.02] }
In[2]:= angmom
Out[2]= [0.02, 3, 0.02]
In[3]:= energy
Out[3]= 5.148625491836798
In[4]:= run 0.5 steps 5
In[5]:= get racket.orientation.x
Out[5]= 0.013359726984815766
In[7]:= get racket.orientation.x
Out[7]= 0.009214316848229325
In[9]:= get racket.orientation.x
Out[9]= -0.06928171314906612
In[11]:= get racket.orientation.x
Out[11]= -0.32837233114496406
In[13]:= get racket.orientation.x
Out[13]= -0.4991829732950659
In[15]:= get racket.orientation.x
Out[15]= 0.1093367245470612
In[17]:= get racket.orientation.x
Out[17]= 0.8230403847701171
In[19]:= get racket.orientation.x
Out[19]= 0.9661018228588313
In[20]:= angmom
Out[20]= [0.02, 3, 0.02]
In[21]:= energy
Out[21]= 5.148625488700016
```

**What to notice.** The *x* component of the orientation quaternion —
which is, near enough, "how far over has it rolled" — sweeps from 0.013
through negative territory to 0.966. That is the flip.

And through all of it, `angmom` reads `[0.02, 3, 0.02]` at the end
exactly as it did at the start. Not to eight decimals: the identical
printed value. The energy moves in the 9th digit, which is the local
error control doing its job.

You can watch this one: [`videos/tumbling_racket.html`](videos/tumbling_racket.html).

---

### Example 8 — when energy *should* grow

**The physics.** A magnetised body in a magnetic field feels a torque
*τ = (R M Rᵀ)B*, where *M* is the body's magnetic moment tensor and *R*
rotates it into the world frame. The field is an outside agent: it does
work on the body, so kinetic energy grows and angular momentum about the
origin is *not* conserved.

**Why it is a real test.** Half of understanding a conservation
diagnostic is knowing when it should *fail*. And there is a subtlety
worth having: only the tensor's **third column** is what a field along
*z* can grip.

```
In[2]:= set system.b_field = [0, 0, 2]
In[3]:= new cuboid as rotor { mass = 1, half_extents = [1, 0.4, 0.4], magnetic_moment_tensor = [[0, 0, 0.5], [0, 0, 0], [-0.5, 0, 0]] }
In[4]:= energy
Out[4]= 0
In[5]:= angmom
Out[5]= [0, 0, 0]
In[6]:= run 3 steps 30
Out[6]= t = 3 (550 solver steps, 30 snapshots, |dE/E| = 9.928e-1)
In[7]:= energy
Out[7]= 0.9928369509454675
In[8]:= angmom
Out[8]= [0.46022300703213415, 0, 0]
```

**What to notice.** Starts at rest, ends spinning. `|dE/E| = 9.928e-1`
is not a warning here; it is the answer. The initial torque is
*M·B* = (0.5·2, 0, 0) = (1, 0, 0), and the angular momentum duly grows
along *x*.

**The trap.** With `magnetic_moment_tensor = [[0, 0.5, 0], [-0.5, 0, 0],
[0, 0, 0]]` — a perfectly reasonable-looking antisymmetric tensor — the
third column is zero, *M·B* is the zero vector, and absolutely nothing
happens. No error, no warning, just a body that sits there. Multiply the
tensor by your field by hand before you blame the solver.

---

### Example 9 — Newton's cradle

**The physics.** Five identical spheres, four of them touching. Roll one
into the line and *only the far one* leaves, at exactly the incoming
speed. Momentum alone does not force this — two balls leaving at half
speed would conserve momentum fine. It is momentum **and** energy
together that pick out the answer.

**Why it is a real test.** The impulse has to propagate through a chain
of simultaneous contacts in the right order, and both conserved
quantities have to survive all four.

```
In[3]:= new sphere as b0 { mass = 1, radius = 0.5, position = [-3, 0, 0], velocity = [1, 0, 0] }
In[4]:= new sphere as b1 { mass = 1, radius = 0.5, position = [ 0, 0, 0] }
In[5]:= new sphere as b2 { mass = 1, radius = 0.5, position = [ 1, 0, 0] }
In[6]:= new sphere as b3 { mass = 1, radius = 0.5, position = [ 2, 0, 0] }
In[7]:= new sphere as b4 { mass = 1, radius = 0.5, position = [ 3, 0, 0] }
In[8]:= energy
Out[8]= 0.5
In[9]:= momentum
Out[9]= [1, 0, 0]
In[10]:= run 6 steps 600
Out[10]= t = 6 (30 solver steps, 600 snapshots, |dE/E| = 0.000e0, 4 collision(s) — CONTACTS lists them)
In[11]:= get b0.velocity
Out[11]= [0, 0, 0]
In[12]:= get b1.velocity
Out[12]= [0, 0, 0]
In[13]:= get b2.velocity
Out[13]= [0, 0, 0]
In[14]:= get b3.velocity
Out[14]= [0, 0, 0]
In[15]:= get b4.velocity
Out[15]= [1, 0, 0]
In[16]:= energy
Out[16]= 0.5
In[17]:= momentum
Out[17]= [1, 0, 0]
```

**What to notice.** Four balls at rest, one moving at exactly 1. Energy
0.5 before and after; momentum `[1, 0, 0]` before and after;
`|dE/E| = 0.000e0`. Four impulses, and not one bit lost.

---

### Example 10 — the bullet that does not tunnel

**The physics.** A 4 mm bullet at 100 m/s meets a 5 mm plate. Between
output frames the bullet travels a metre — two hundred plate
thicknesses.

**Why it is a real test.** This is precisely the case that defeats
frame-sampled collision detection. Section 6 explains why rootfinding
does not have the problem; here is the number.

```
In[3]:= new cuboid as plate  { mass = 1, half_extents = [5, 0.0025, 5], position = [0, 0, 0], inverse_mass = 0 }
In[4]:= new sphere as bullet { mass = 0.01, radius = 0.002, position = [0, 2, 0], velocity = [0, -100, 0] }
In[5]:= run 0.1 steps 10
Out[5]= t = 0.1 (10003 solver steps, 10 snapshots, |dE/E| = 0.000e0, 1 collision(s) — CONTACTS lists them)
In[6]:= get bullet.position.y
Out[6]= 8.009000000001455
In[7]:= get bullet.velocity.y
Out[7]= 100
In[9]:= contacts
Out[9]= contact0: obj0 <-> obj1 at t = 0.01995500000000067
  point  = [0, 0.0025, 0]
  normal = [-0, 1, -0]  (from obj0 toward obj1)
  depth = 0.00000000000002214157676649897, approach speed = 100, impulse = 2
```

**What to notice.** The bullet bounced (velocity `+100`) and ended up at
*y* = 8.009 — above where it started, because it is elastic and there is
no gravity. The impact is recorded at `t = 0.01995500000000067`; the
bullet had 1.9955 m of gap to close at 100 m/s, so the exact answer is
`0.019955`.

Look also at the step count: **10,003 solver steps** for ten output
points. That is the anti-tunnelling cap at work — the solver was told it
may not take a step large enough to skip the plate, and it obeyed.

---

### Example 11 — Lagrange's equilateral three-body solution

**The physics.** Three equal masses at the corners of an equilateral
triangle, each moving at the right speed, rotate rigidly forever. This
is one of only a handful of exactly solvable configurations of the
three-body problem, found by Lagrange in 1772. The same geometry is why
Jupiter has Trojan asteroids. With *G* = *m* = 1 and side *a*, the
angular rate is *ω = √(3Gm/a³)*.

**Why it is a real test.** A rigidly rotating three-body solution is
*unstable* for large mass ratios and marginal here, so it is a
demanding integration; and there are three independent things to check —
each body comes home, the centre of mass never moves, and the triangle
stays equilateral.

At *a* = 1 the circumradius is 1/√3 = 0.5773502691896258, *ω* = √3, the
orbital speed is *ωR* = 1 and the period is 2π/√3 = 3.6275987284684357.

```
In[2]:= new sphere as a { mass = 1, radius = 0.01, position = [0.5773502691896258, 0, 0], velocity = [0, 0, 1] }
In[3]:= new sphere as b { mass = 1, radius = 0.01, position = [-0.2886751345948129, 0, 0.5], velocity = [-0.8660254037844387, 0, -0.5] }
In[4]:= new sphere as c { mass = 1, radius = 0.01, position = [-0.2886751345948129, 0, -0.5], velocity = [0.8660254037844387, 0, -0.5] }
In[5]:= com
Out[5]= [0, 0, 0]
In[6]:= energy
Out[6]= -1.4999999999984994
In[7]:= run 3.6275987284684357 steps 400
Out[7]= t = 3.6275987284684357 (1289 solver steps, 400 snapshots, |dE/E| = 4.689e-10)
In[8]:= get a.position
Out[8]= [0.5773502692574497, 0, -0.0000000025627822854319693]
In[9]:= get b.position
Out[9]= [-0.2886751324104037, 0, 0.5000000013400956]
In[10]:= get c.position
Out[10]= [-0.28867513684704826, 0, -0.49999999877731977]
In[11]:= com
Out[11]= [-0.0000000000000007586524001605236, 0, -0.000000000000002146431180941969]
In[12]:= norm(a.position - b.position)
Out[12]= 1.0000000001184224
```

**What to notice.** After a full revolution all three are back within
about 3 × 10⁻⁹. The centre of mass has moved by 2 × 10⁻¹⁵ — five
million times less, which is what you would expect, since it is
protected by momentum conservation rather than by accuracy. And the side
length is still 1 to 1 part in 10¹⁰: the triangle is still equilateral.

Note also `norm(a.position - b.position)` — you can do vector algebra on
live simulator fields directly in the language.

---

### Example 12 — a user-defined constructor, and an inertia tensor

**The physics.** A `DUMBBELL` is two solid spheres joined by a rod,
treated as **one rigid body**. Its inertia tensor about the centre of
mass is the exact composite: ⅖*mr²* for each sphere, plus *mz²* from
the parallel-axis theorem for moving it off centre, plus the rod's own
terms.

**Why it is a real test.** The inertia tensor is where compound-body
code usually goes wrong, and the answer is a hand calculation. It also
shows off `DEF`, which lets you name a construction and reuse it.

```
In[1]:= def barbell(m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.3, r2 = 0.3, rod_r = 0.05, len = 2) {
  new dumbbell as bar { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, rod_radius = rod_r, length = len }
}
Out[1]= function barbell(7 parameter(s)) defined — 1 body line(s)
In[2]:= funcs
Out[2]= barbell(m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.3, r2 = 0.3, rod_r = 0.05, len = 2) — 1 body line(s); SHOW barbell prints it
In[3]:= barbell(2, 2, 0, 0.3, 0.3, 0.05, 2)
Out[3]= obj0 as bar
In[4]:= list
Out[4]= obj0: dumbbell r1=0.3 r2=0.3 rod_r=0.05 len=2, mass=4, charge=0, pos=[0, 0, 0]
In[9]:= get bar.inertia_tensor
Out[9]= [[4.144, 0, 0], [0, 4.144, 0], [0, 0, 0.144]]
```

Now the hand calculation, again inside the language. With a massless rod
and two equal 2 kg spheres of radius 0.3 at *z* = ±1:

```
In[10]:= 2 * (0.4 * 2 * 0.3 * 0.3 + 2 * 1 * 1)
Out[10]= 4.144
In[11]:= 2 * (0.4 * 2 * 0.3 * 0.3)
Out[11]= 0.144
```

**What to notice.** `4.144` and `0.144`, exactly. The off-diagonal
entries are exactly zero, as they must be for a body symmetric about its
*z* axis. Every round shape in this program — torus, disk, cylinder,
dumbbell — is symmetric about its **local *z* axis**; that is the
convention.

Note the `m_rod = 0` in the call: `DEF` parameters fill in left to
right, and passing 0 for the rod's mass is what makes the composite
reduce to the two-sphere formula you can check by hand.

---

### Example 13 — geometry a bounding sphere gets wrong

**The physics.** An axis-aligned torus with outer radius 2 exactly
inscribes a cube of inner side 4 — it touches all four side walls.
Tilt that same torus so its axis points along (1,1,1)/√3 and its
extent along each coordinate axis *drops* to
1.5·√(2/3) + 0.5 ≈ 1.7247, well inside the walls.

**Why it is a real test.** A bounding sphere would say the opposite: the
torus's bounding sphere has radius 2 no matter how you turn it, so a
sphere test would report the tilted torus as *still touching*. Getting
this right requires the actual **support function** of the shape — the
farthest point of the body in a given direction — evaluated in the
rotated frame.

```
In[2]:= box 4
Out[2]= box: inner size 4 x 4 x 4 — six static walls obj0..obj5 with inverse_mass = 0
In[3]:= collide on
In[4]:= new torus as flat { mass = 1, ring_radius = 1.5, tube_radius = 0.5, position = [0, 0, 0] }
Out[4]= obj6 as flat
In[5]:= get flat.boundary
Out[5]= Torus { ring_radius: 1.5, tube_radius: 0.5 }
In[6]:= del 6
In[7]:= new torus as tilt { mass = 1, ring_radius = 1.5, tube_radius = 0.5, position = [0, 0, 0], orientation = [0.8880738339771154, -0.32505758367186816, 0.32505758367186816, 0] }
In[8]:= run 1 steps 10
Out[8]= t = 1 (4 solver steps, 10 snapshots, |dE/E| = 0.000e0)
In[10]:= get system.collisions
Out[10]= 0
In[11]:= 1.5 * sqrt(2 / 3) + 0.5
Out[11]= 1.724744871391589
```

**What to notice.** Zero collisions. The tilted torus sits inside a
`BOX 4` without touching anything, and the clearance
(2 − 1.7247 = 0.2753 per side) is exactly what the geometry predicts.

Note also the `BOX` machinery itself: six wall slabs, created
automatically, with `inverse_mass = 0` — *infinite* mass, so they can
absorb any impulse without ever moving. In the graphics they are drawn
as a dashed wireframe, never as bodies.

---

### Example 14 — the infinite square well, solved in the language

**The physics.** Quantum mechanics' first exercise. A particle confined
to a box of width *L* has energies *Eₙ = n²π²ħ²/(2mL²)*. With
*ħ* = *m* = 1 on the interval [0, 1] those are 4.9348…, 19.7392…,
44.4132…, 78.9568….

**Why it is a real test.** The eigenvalues are exact, so any error is
visible; and `QM GRID` pins the wavefunction to zero just outside the
domain, which *is* an infinite square well — the program is solving the
real problem, not a special case of it.

```
In[1]:= qm grid 0 1 800
Out[1]= grid [0, 1] with 800 interior points, h = 0.001248 (potential and psi cleared)
In[2]:= qm potential zero
In[5]:= qm states 4
Out[5]= 4 lowest bound state(s):
  E[0] = 4.934795874467
  E[1] = 19.739107587630
  E[2] = 44.412707408131
  E[3] = 78.955215788088
In[6]:= pi * pi / 2
Out[6]= 4.934802200544679
In[7]:= 4 * pi * pi / 2
Out[7]= 19.739208802178716
In[8]:= 9 * pi * pi / 2
Out[8]= 44.41321980490211
In[9]:= 16 * pi * pi / 2
Out[9]= 78.95683520871486
```

The agreement is to about 1.3 parts in a million, and it gets *worse*
for higher levels — 4.93480 vs 4.93480, then 78.9552 vs 78.9568. That
is not sloppiness: it is the finite-difference approximation to the
second derivative, whose error grows with the curvature of the state,
and it shrinks as *h²* when you refine the grid.

A physical cross-check on the loaded state:

```
In[10]:= qm state 1
Out[10]= psi = bound state 1, E = 19.739107587630, t reset to 0
In[11]:= qm norm
Out[11]= 1.000000000000345
In[12]:= qm position
Out[12]= 0.500000000001550
In[14]:= qm prob 0 0.5
Out[14]= 0.499999999998032
```

**What to notice.** Total probability 1, mean position dead centre, and
exactly half the probability in each half of the box — all to 12 digits.
The *eigenvalue* carries discretisation error; the *state* is properly
normalised and properly symmetric.

---

### Example 15 — tunnelling, and watching a grid converge

**The physics.** A particle with energy *E* below a barrier of height
*V₀* and width *a* still gets through, with probability

*T = 1 / (1 + V₀² sinh²(κa) / (4E(V₀ − E)))*,  *κ = √(2m(V₀ − E))/ħ*.

With *V₀* = 4, *a* = 1, *E* = 2, *ħ* = *m* = 1 that is 0.07065082485….

**Why it is a real test.** Transmission depends on the barrier width
*exponentially*, so a grid that gets the width slightly wrong gets *T*
badly wrong. That makes it an unusually sharp probe of the
discretisation — and a good lesson in how to tell discretisation error
from a bug.

```
In[7]:= 1 / (1 + v0 * v0 * s * s / (4 * e * (v0 - e)))
Out[7]= 0.07065082485316446

In[8]:= qm grid -30 30 4000
In[10]:= qm transmission 2
Out[10]= E = 2: T = 0.069368404960, R = 0.930631595040, T + R = 1.000000000000
In[11]:= qm grid -30 30 16000
In[13]:= qm transmission 2
Out[13]= E = 2: T = 0.070328037206, R = 0.929671962794, T + R = 1.000000000000
In[14]:= qm grid -30 30 64000
In[16]:= qm transmission 2
Out[16]= E = 2: T = 0.070569990790, R = 0.929430009210, T + R = 1.000000000000
```

**What to notice.** Three grids, each four times finer, and the answer
walks steadily toward the analytic value:

| points | *T* | error |
|---|---|---|
| 4,000 | 0.069368 | 1.3 × 10⁻³ |
| 16,000 | 0.070328 | 3.2 × 10⁻⁴ |
| 64,000 | 0.070570 | 8.1 × 10⁻⁵ |

The error falls by a factor of ~4 each time the grid is refined by a
factor of 4 — first-order convergence, exactly what a staircase-sampled
barrier gives you. **This is what "discretisation error" looks like from
the outside**, and it is how you tell it from a bug: a bug does not
converge.

Notice too that `T + R = 1.000000000000` on every grid. Probability is
conserved exactly regardless of how coarse the grid is, because that is
a structural property of the transfer matrix, not an accuracy property.
The two kinds of correctness are independent, and it is worth learning
to see which is which.

Above the barrier there are resonant energies where the barrier becomes
perfectly transparent:

```
In[19]:= qm scan 4.1 20 200
Out[19]= 200 energies in [4.1, 20] (0 refused), T from 3.535e-1 to 1.000e0
  1 resonance(s), strongest first:
    E = 8.893969849   T = 0.999992036
```

---

### Example 16 — the special-function library, checked by identities

**The physics — or rather, the mathematics.** Everything in
`special_functions` is reachable from ordinary call syntax. The way to
test such a library is not to compare against a table (tables have
typos) but against **identities** — relations that a wrong
implementation cannot satisfy by accident.

```
In[1]:= bessel_j(0, 2.404825557695773)
Out[1]= 0
```

*J₀* at its first zero, and the answer is a clean 0.

```
In[2]:= bessel_j_nu(1, 3) * bessel_y_nu(0, 3) - bessel_j_nu(0, 3) * bessel_y_nu(1, 3)
Out[2]= 0.21220659078919374 + 0i
In[3]:= 2 / (pi * 3)
Out[3]= 0.2122065907891938
```

The Wronskian identity *J*ᵥ₊₁*Y*ᵥ − *J*ᵥ*Y*ᵥ₊₁ = 2/(π*x*), matched to
16 digits. This one is a genuinely strong test: it couples *four*
different function evaluations, so an error in any one of them breaks
it.

```
In[4]:= legendre_p(5, 1)
Out[4]= 1
In[5]:= legendre_p(5, 0.906179845938664)
Out[5]= -0.00000000000000017763568394002506
In[6]:= gamma_z(0.5) * gamma_z(0.5)
Out[6]= 3.1415926535898397 + 0i
In[7]:= pi
Out[7]= 3.141592653589793
In[8]:= sph_harm_real(1, 0, 0, 0)
Out[8]= 0.48860251190291987
In[9]:= sqrt(3 / (4 * pi))
Out[9]= 0.4886025119029199
In[10]:= clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0)
Out[10]= 0.7071067811865475
In[11]:= sqrt(0.5)
Out[11]= 0.7071067811865476
In[12]:= gauss_legendre(4)
Out[12]= [[-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
          [0.34785484513745374, 0.6521451548625461, 0.6521451548625461, 0.34785484513745374]]
```

**What to notice**, line by line:

- *P₅*(1) = 1 exactly, and 0.906179845938664 really is one of its roots
  (the residual is 1.8 × 10⁻¹⁶, one bit).
- Γ(½)² = π to 14 digits, via the *complex* gamma function evaluated on
  the real axis.
- The spherical harmonic *Y₁⁰* at *θ* = 0 is √(3/4π), agreeing to 15
  digits.
- Two spin-½ particles coupling to the *m* = 0 triplet state have
  Clebsch–Gordan coefficient 1/√2, agreeing to the last bit.
- The 4-point Gauss–Legendre rule reproduces the classical nodes and
  weights, and the weights sum to 2 — the length of [−1, 1].

**One design decision worth knowing.** Where an argument is an integer
order, a fractional value is refused rather than truncated:

```
In[1]:= hermite_h(2.5, 1)
Err[1]: hermite_h(): argument 1 must be a whole number (an integer order), got 2.5
```

Truncating would have returned a confident, wrong number with no way for
you to notice. Angular momenta, on the other hand, *may* be
half-integers, so the Wigner and Clebsch–Gordan routines take plain
numbers.

---

---

### Example 17 — a pendulum as a constraint, not a force

**The physics.** A pendulum bob hangs on a rigid rod. The small-amplitude
period is `T = 2π√(L/g)`. Released 0.02 rad off vertical with `L = 1` and
`g = 9.81`, that is `T = 2.0060666807106475` s, and after exactly one
period the bob must be back where it started.

**Why it is a real test.** The rod is not a force, it is a geometric
condition — and the *only* way to hold it exactly is to solve the
differential-algebraic equation. The test has two independent halves: the
period must come out right (physics) and the rod must not stretch
(numerics).

```
In[4]:= new sphere as pivot { mass = 1, radius = 0.02, position = [0, 0, 0], inverse_mass = 0 }
Out[4]= obj0 as pivot
In[5]:= new sphere as bob { mass = 1, radius = 0.05, position = [0.019998666693333084, -0.9998000066665778, 0] }
Out[5]= obj1 as bob
In[6]:= constrain pivot bob
Out[6]= constraint0: obj0 <-> obj1 held at 1 (METHOD IDA is required to integrate it)
In[7]:= constraints
Out[7]= constraint0: obj0 <-> obj1 length 1 (currently 1)
In[8]:= method ida
Out[8]= method = IDA (constrained DAE, GGL index-2)
In[9]:= run 2.0060666807106475 steps 200
Out[9]= t = 2.0060666807106475 (339 solver steps, 200 snapshots, |dE/E| = 2.530e-12)
In[10]:= get bob.position
Out[10]= [0.019998666320588818, -0.9998000066740419, 0]
In[11]:= constraints
Out[11]= constraint0: obj0 <-> obj1 length 1 (currently 1.0000000000000082)
```

**What to notice.** The bob started at `(0.019998666693333084,
−0.9998000066665778)` and came back to `(0.019998666320588818,
−0.9998000066740419)` — a closure of **3.7 × 10⁻¹⁰** after a full
period. Energy held to 2.5 parts in 10¹².

And the rod: `1.0000000000000082`. Eight parts in 10¹⁵ — one bit — after
339 solver steps. That is not luck. The formulation carries *both* the
rod's length and its rate of change as equations; the cheaper scheme that
constrains only the acceleration lets the length drift quadratically, and
you find out much later.

**The `inverse_mass = 0` on the pivot is what makes it a pendulum.** It
marks the pivot as an anchor: immovable, and absorbing whatever the rod
pulls with. Without it you would have two bodies tumbling around their
shared centre of mass, joined by a stick — also a perfectly good
simulation, just not a pendulum.

---

### Example 18 — where it rests, and what the answer depends on

**The physics.** Two questions that integration does not answer. A
pendulum released anywhere comes to rest hanging straight down, one
rod-length below the pivot. And free fall, `y(T) = y₀ + v₀T + ½gT²`, has
the exact derivative `∂y/∂g = T²/2`, which at `T = 3` is 4.5.

**Why it is a real test.** Both answers are exact, and the second one has
a second exact answer hiding in it.

```
In[6]:= constrain pivot bob
Out[6]= constraint0: obj0 <-> obj1 held at 1 (METHOD IDA is required to integrate it)
In[7]:= equilibrium
Out[7]= equilibrium found in 17 Newton iteration(s), 67 residual evaluation(s);
        largest net force on any free body = 7.459152323898993e-13,
        worst |g| = 1.9317880628477724e-14
In[8]:= get bob.position
Out[8]= [0.0000000000000735262479725006, -0.9999999999999807, 0]
```

Released 57° off vertical, the bob lands at `x = 7.4 × 10⁻¹⁴`,
`y = −0.9999999999999807`. Straight down, one rod-length, to 13 digits —
and the largest net force left anywhere is 7 × 10⁻¹³.

Now the derivative, on a free body:

```
In[15]:= sensitivity 3 "gravity.y" "mass 0"
Out[15]= t = 3 (CVODES, 129 solver steps)
d/d(gravity.y):
  obj0 position [0, 4.500000056696235, 0]
d/d(mass 0):
  obj0 position [0, 0, 0]
In[16]:= get stone.position
Out[16]= [3.000000000000001, -44.14500000000045, 0]
```

**What to notice.** `4.500000056696235` against an analytic `4.5` — 1.3
parts in 10⁸, carried alongside a trajectory that itself landed on
`−44.14500000000045` where `−½ · 9.81 · 9 = −44.145`.

And the second derivative is **exactly zero**. In uniform gravity every
mass accelerates equally, so the trajectory does not depend on the mass
at all. A finite-difference estimate would have returned some small
number and left you guessing whether it was noise or physics. The
sensitivity equations return the zero.

---

---

### Example 19 — a door on a hinge

**The physics.** A hinge fixes a point *and* an axis, leaving one
freedom. A hinged rigid body is a **compound** pendulum: its
small-amplitude period is

*T = 2π√(I_pivot/(mgd))*,  *I_pivot = I_com + md²*

— the moment of inertia about the pivot, not just the distance to the
centre of mass. Using the point-mass formula instead is wrong by 15 %
for the slab below.

**Why it is a real test.** Three independent things must come out right:
the period (physics), the rod-and-axis holding (numerics), and the fact
that the body turns about the hinge axis and *nothing else*.

```
In[4]:= new sphere as jamb { mass = 1, radius = 0.02, position = [0, 0, 0], inverse_mass = 0 }
In[5]:= new cuboid as door { mass = 1, half_extents = [0.2, 0.4, 0.2], position = [0.0199986666933331, -0.9998000066665778, 0] }
In[6]:= hinge jamb door [0, 0, 1]
In[8]:= method ida
Out[8]= method = IDA (constrained DAE, GGL index-2)
In[9]:= run 1 steps 10
Out[9]= t = 1 (70 solver steps, 10 snapshots, |dE/E| = 1.613e-9)
In[10]:= constraints
Out[10]= constraint0: hinge obj0 <-> obj1, 5 row(s)
worst |g| = 2.73750133672479e-10, worst |g_dot| = 2.3769240437118873e-9
In[11]:= get door.angular_momentum
Out[11]= [0, 0, 0.003742219622967182]
```

**What to notice.** The angular momentum is `[0, 0, 0.0037]` — *exactly*
zero on x and y. The two rows a hinge has over a ball joint are the ones
that forbid those axes, and they are doing their job to the last bit.

The joint itself is held to `2.7 × 10⁻¹⁰` after 70 solver steps, and
energy to 1.6 parts in 10⁹. Run the same thing for one compound-pendulum
period and the slab returns to where it started to about `3 × 10⁻⁸`.

**The `inverse_mass = 0` on the jamb is what makes it a door.** It marks
the jamb as an anchor: immovable, and absorbing whatever the hinge pulls
with. Without it you would have two slabs tumbling about their shared
centre of mass, hinged together — a perfectly good simulation, just not
a door.

**A joint constrains velocity, not just position.** A ball joint says
the two bodies share a point, so at the velocity level it says
`v_i + ω_i×r_i = v_j + ω_j×r_j`: a body turning about a pivot offset from
its centre **must have its centre moving**. Give it a spin and leave its
velocity at zero and the state is not on the constraint manifold. The run
projects the starting velocities onto it — the smallest mass-weighted
change that satisfies the joint, which is precisely the impulse a real
coupling delivers when clutched onto a spinning shaft — and reports how
big that change was. An already-consistent state is left exactly alone.

For a cube spun at 3 rad/s on a half-metre arm, the projection leaves the
turn nearly untouched and sets the *centre* moving instead: the pivot was
running at 1.5 m/s and giving a 1 kg body some velocity is cheaper than
fighting the spin.

**One real limit.** Orientation joints carry a tolerance floor of
`rtol = 1e-6`, because the differential-algebraic system a hinge produces
is *index 2* and has an accuracy ceiling no tolerance can push past.
Measured across twelve compound pendulums the boundary is sharp: `1e-6`
converges in every one, `1e-8` in none. `RUN` says when the floor bit.

---

## 8. Watching it: the scene window and browser videos

### 8.1 The live window

Type `SCENE CREATE` and a browser tab opens. That tab **is** the
simulator's window: it draws every body on a 3-D canvas, with a toolbar
(Start, Pause, Stop, Reverse, Reset, single-step, a *dt* box, zoom,
grid/trails/labels toggles) and a status bar showing the playback mode,
the time, *dt*, the total energy, the body count, the camera and the
frame rate.

Behind it is a web server written on the Rust standard library —
including the HTTP parsing, the WebSocket handshake, the SHA-1 and the
base64. No dependency, and the page itself is vanilla JavaScript and
canvas with no CDN fetches.

Two things about it are worth knowing:

- **The window evolves its own copy of the system.** Typing `STEP` in
  the notebook does not move the window, and pressing Start in the
  window does not move the notebook. `SCENE CREATE`, `SCENE REFRESH` and
  `SCENE RESET` are what synchronise them. That isolation is what lets
  the window run without ever locking the language.
- **Reverse is replay, not negative-time integration.** The playback
  keeps a ring of snapshots; running backward walks it. It returns to
  *t* = 0 bit-identically.

For a headless machine, `POSIM_NO_BROWSER=1` suppresses the attempt to
open a browser.

### 8.2 Recorded videos

A live window needs a live program. For something you can send to
someone else, record it:

```bash
cargo build --release -p posim
recorder/src/record_video.py videos/scenes/kepler_ellipse.posim \
     -o videos/kepler_ellipse.html --frames 360 --dt 0.02 \
     --title "Kepler orbit, e = 0.6"
```

The result is a single HTML file: the frames embedded as data plus a
canvas player. It fetches nothing, so it works from `file://` on a
machine with no network, forever.

The recorder drives `posim --machine` and asks it to `step` — **every
advance is a real SUNDIALS step**. The tool is a camera, not a physics
engine.

Thirteen recordings ship with the repository:

| file | what to watch | measured over the recording |
|---|---|---|
| [`videos/kepler_ellipse.html`](videos/kepler_ellipse.html) | the speed swinging between perihelion and aphelion | \|d*E*\|/*E* = 9.8 × 10⁻⁸, \|d*L*\|/\|*L*\| = 1.3 × 10⁻⁷ |
| [`videos/tumbling_racket.html`](videos/tumbling_racket.html) | example 7's flip, seen from outside | \|d\|*L*\|\|/\|*L*\| = **0 exactly**; \|d*E*\|/*E* = 6.4 × 10⁻⁹ |
| [`videos/box_of_shapes.html`](videos/box_of_shapes.html) | a cylinder, a disk and a cuboid in a rigid `BOX 4`; gold arrows are the analytic contact normals, sized by impulse | 36 collisions, \|d*E*\|/*E* = 3.4 × 10⁻¹⁶ |
| [`videos/double_pendulum_hinges.html`](videos/double_pendulum_hinges.html) | **two `HINGE` joints** assembled into the classic chaotic linkage — gold rings mark the joints, gold lines their axes | the joints hold to \|*g*\| = 5.6 × 10⁻⁸ through 400 IDA steps |
| [`videos/universal_joint.html`](videos/universal_joint.html) | a **`UNIVERSAL` joint** carrying a driven shaft's rotation across to a second shaft, braced by a rod to a post | the bend stops at cos *β* = 0.6000004 against a geometric bound of exactly 0.6; three joints hold to \|*g*\| = 4.0 × 10⁻⁷ |
| [`videos/ball_joint_chain.html`](videos/ball_joint_chain.html) | four links on **`BALL` joints** — the chain whirls out of the plane it started on, which a hinged chain cannot do | the four joints hold to \|*g*\| = 3.3 × 10⁻⁹; \|*z*\| runs from exactly 0 to 1.7147 |
| [`videos/rod_pendulum_chain.html`](videos/rod_pendulum_chain.html) | four bobs on four **`CONSTRAIN` rods** — one row each, the cheapest linkage in the language, going chaotic | run continuously at the default tolerance the rods hold to \|*g*\| = 5.4 × 10⁻¹⁵, i.e. roundoff |
| [`videos/spinning_top.html`](videos/spinning_top.html) | a top on a **`BALL` joint**, precessing under gravity | 1.020440 rad/s against the exact Ω = *Mgr*/(*I₃ω₃*) = 1.020408, 3 parts in 10⁵ |
| [`videos/gyroscope_gimbal.html`](videos/gyroscope_gimbal.html) | a rotor in **two gimbal rings**, three perpendicular `HINGE` axes | total *L·ŷ* conserved to 1.4 × 10⁻¹⁴; every centre stays on the pivot to 1.2 × 10⁻³⁴ |
| [`videos/cardan_compass.html`](videos/cardan_compass.html) | the same rings with a **pendulous** bowl — a ship's compass, seeking level | periods 1.878587 and 2.307339 s against measured 1.883426 and 2.313653 |
| [`videos/cardan_gear.html`](videos/cardan_gear.html) | **Cardan gears** — a wheel in a ring of twice its radius, rolling on a `GEAR` row, its rim point tracing a straight line | line held to 1.1 × 10⁻⁸; stroke exactly ±2*r* |
| [`videos/rack_and_pinion.html`](videos/rack_and_pinion.html) | a weight on a **`RACK`** winding up a flywheel, on a **`PRISMATIC`** guide | falls at exactly *g*/2, and at the same rate for two different pitch radii |
| [`videos/piston_crankshaft.html`](videos/piston_crankshaft.html) | the **slider-crank** — a piston, a connecting rod and a crankshaft | follows the exact kinematics to 8.4 × 10⁻⁸; stroke exactly *L*−*a* to *L*+*a* |

That last one is the one to open if you only open one, because it shows
a joint doing something a rod cannot. A `UNIVERSAL` holds a shared point
and one right angle between its trunnions; it deliberately does **not**
hold the two shafts straight. Left dangling, the output shaft therefore
does not transmit rotation at all — it swings down past the joint like a
pendulum and folds back at 176°. Bracing it with a second bearing looks
right and fails at once, because `HINGE 5 + UNIVERSAL 4 + HINGE 5` is 14
rows on 12 freedoms and three of them are redundant, which leaves the
constrained solve singular. One `CONSTRAIN` — a single row — braces it
instead, and pins the bend to a closed form: the shaft sweeps a cone,
and cos *β* cannot go below 0.6. The recording touches 0.6000004.
Grammar §12.10 works the arithmetic through.

The player has Play/Pause (or Space), frame stepping (or ← →), a scrub
bar, speed from 0.25× to 4×, drag to orbit, wheel to zoom, and toggles
for trails, labels, contact arrows and joints. A mechanism gets its
joints drawn — a ring at each shared point and a line along each hinge
axis — because that is the thing the video is about. The corner readout shows the
frame number, *t*, *E*, \|*P*\|, \|*L*\| and the collision count **for the
frame you are on** — so you can stop on the moment something looks
wrong and read the conserved quantities off it.

---

## 9. How you know the numbers are right

This project's central claim is not that the code is elegant. It is that
the numbers are checkable. Here is how, at four levels.

### 9.1 The solver library against its C original

`sundials_rs/` ships the upstream SUNDIALS example programs, translated.
Each one is run and its output **diffed byte-for-byte** against the
output of the original C program. The results, the divergences, and the
reasons for each divergence, are in `sundials_rs/VERIFICATION.md`,
`sundials_rs/differences/` and `sundials_rs/c-results/`.

Two consequences of taking this seriously: printed floating-point values
go through helpers that reproduce C's `printf` conversions exactly
(Rust's own `{:e}` formats exponents differently), and `pow` was made
host-independent so that at least that function cannot vary between
machines.

### 9.2 The simulator against closed-form physics

Six example programs, each of which prints `SUCCESS` or `FAILURE` and
exits nonzero if it fails:

```bash
cargo run -p physical_object --release --example kepler_orbit
cargo run -p physical_object --release --example outer_solar_system
cargo run -p physical_object --release --example tumbling_body
cargo run -p physical_object --release --example charged_in_b_field
cargo run -p physical_object --release --example newtons_cradle
cargo run -p physical_object --release --example bouncing_ball_restitution
```

### 9.3 The test suite

605 tests: 40 library, 19 collision, 9 conservation, 109 language,
92 quantum, 233 special-function, 11 vendored identities and 55
documentation examples that are compiled and run as written.

The collision tests in particular are analytic rather than
regression-style: they assert the time of impact against a closed form,
not against last week's output.

### 9.4 The notebooks

`dynamic_notebooks/` holds 59 runnable sessions, including 34 problems
from Routh's *Dynamics of a Particle* (1898) and *Dynamics of a System
of Rigid Bodies* (1905). Each one derives its closed-form answer in its
own header and then checks the integrator against it — so the numbers in
the catalogue are measured, not quoted.

### 9.5 And for the 7.8.0 upgrade specifically

Everything in 9.2, 9.3 and 9.4 was run against **both** the old 7.7.0
engine and the new 7.8.0 one, and the outputs diffed. They are identical
byte for byte. See [`PORT_7.8.0_PROVENANCE.md`](PORT_7.8.0_PROVENANCE.md)
and [`evidence/port-7.8.0/`](evidence/port-7.8.0).

---

## 10. When something goes wrong

Errors name the column, the field, or the feature at fault, and never
abort your session. The common ones:

| symptom | what is actually happening |
|---|---|
| `|dE/E|` is huge and you expected conservation | Check `system.g_constant` (§3.2), the restitution of your bodies (§7, example 4), and whether a field is doing work (example 8). |
| `RUN` overshoots where you meant to stop | `RUN` takes a **duration** (§3.1). |
| bodies pass through each other | You did not `COLLIDE ON` (§3.4). |
| `SPRK method requires a separable Hamiltonian: …` | A magnetic field, a magnetic moment, an external torque or a spinning body is present. The message names which. Use `METHOD ADAMS` or `BDF`. |
| `no object obj7` after a `DEL` | Deleting renumbers (§3.3). Use `NEW … AS name`. |
| `unknown name \`x\`` | A bare identifier resolves at execution time. Bind it with `LET`, pass it as a function parameter, or use `name.field` for a registered object. |
| a magnetised body will not spin | Multiply your moment tensor by your field by hand — you probably have a zero column (example 8). |
| a quantum result is off by a fraction of a percent | Refine the grid and see whether it converges (example 15). Discretisation error converges; a bug does not. |
| `probability accumulating near the wall` warning | Your wavepacket has reached the domain edge and is reflecting. Widen the grid or add `QM ABSORB`. |
| the scene window does not open | Set `POSIM_NO_BROWSER=1` if you are headless, or open the printed `http://127.0.0.1:<port>/` yourself. |

Anything the language accepts is documented in
[`grammar.md`](grammar.md) — every command, every field, every builtin,
with the complete EBNF grammar and a further eighteen worked examples.

---

## 11. Where everything lives

| path | what |
|---|---|
| `posim/` | the command language, the graphics window, the machine protocol |
| `physical_object/` | the physics: the union type, collisions, and all time integration |
| `quantum/` | 1-D, 2-D and 3-D quantum mechanics |
| `special_functions/` | Bessel, Legendre, Hankel, Airy, gamma, Wigner, quadrature, eigenproblems |
| `sundials_rs/` | the pure-Rust SUNDIALS 7.8.0 engine (vendored, read-only) |
| `jupyter/` | the JupyterLab kernel |
| `dynamic_notebooks/` | 59 runnable sessions, incl. 34 Routh problems |
| `scripts/solveit/` | the nineteen examples in section 7 |
| `scripts/collisions/` | twelve documented collision scripts |
| `videos/` | recorded browser videos; `videos/scenes/` the scripts behind them |
| `tools/` | the index builder and the verifiers (the video recorder is its own package, `recorder/`) |
| `evidence/port-7.8.0/` | the logs behind every claim in section 9.5 |

### The documents

| document | for |
|---|---|
| **`SolveIt.md`** / `.pdf` | this one: the complete solution guide |
| `grammar.md` / `.pdf` | every command, the full EBNF, 18 more worked examples |
| `ARCHITECTURE.md` | the pinned cross-module contracts, for anyone changing the code |
| `PORT_7.8.0_PROVENANCE.md` | what the 7.8.0 upgrade changed, and the evidence it changed nothing else |
| `collision_detection.md` / `.pdf` | the collision science, with 12 example scripts |
| `scene_info.md` / `.pdf` | the graphics window, and a survey of seven other simulators |
| `physical_object_simulator.md` / `.pdf` | the predecessor user guide, with 14 further examples |
| `special_functions.md` | the special-function library in detail |
| `NOTEBOOKS.md`, `dynamic_notebooks/MANIFEST.md` | the notebook catalogue and its run record |
| `CLAUDE.md` | the working rules for anyone — human or agent — modifying this repository |
| `index_of_entities.html` | a browsable catalogue of every named entity in the repository, with definitions, `file:line` locations and runnable examples |

---

*Everything above was produced by the program in this repository.
Every command shown can be typed. Every number shown can be reproduced.*
