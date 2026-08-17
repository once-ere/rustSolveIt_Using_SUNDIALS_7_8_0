# Collision Detection Science — Research Reference, Porting Evaluation & the posim Implementation

*(`collision_detection.md` — companion to `grammar.md`, `physical_object_simulator.md`
and `scene_info.md`; the same content is typeset in `collision_detection.pdf`.)*

This document is written for a reader with **zero prior knowledge**. It contains:

1. a plain-language introduction to rigid-body collision detection,
2. the **research**: how seven established simulators detect and resolve collisions,
3. a **comparison chart** and a **concept tree** of the field — its properties,
   capabilities, techniques, implementations and hurdles,
4. an **evaluation** of every technique with an eye toward porting to high-quality
   pure Rust,
5. the **recommendation** adopted for correct, precise collision detection in ALL
   simulation scenes that involve rigid-body motion,
6. the full technical documentation of the posim implementation — including the
   **precise contact normal** (the action–reaction line) and how it is exposed to
   every other simulation entity, and
7. **twelve non-trivial worked examples**, fully documented, with their real
   captured output.

---

## 1. What is rigid-body collision detection?

Two solid bodies flying through space must not pass through each other. A simulator
therefore needs to answer three questions, every time step:

1. **Did anything touch?** (*detection*) — and, harder: did anything touch
   *during* the step, even if the step's endpoints look separated?
2. **Where and in which direction?** (*contact geometry*) — the **contact point**
   and the **contact normal** `n̂`, the unit vector along which Newton's third law
   acts: body `j` receives the impulse `+J·n̂` and body `i` receives exactly
   `−J·n̂`. Get the normal wrong and momentum leaks; get it right and the pair of
   forces is a true action–reaction pair.
3. **What happens next?** (*response*) — an impulse (instantaneous momentum
   exchange), a constraint force, or a stiff penalty spring.

The hard parts, discovered independently by every engine surveyed below:
**tunneling** (a fast body crosses a thin one between two checks), **resting
contact** (an infinite sequence of ever-smaller bounces — the Zeno problem),
**simultaneity** (three balls touching at once — Newton's cradle), and
**penetration recovery** (what to do when bodies already overlap).

---

## 2. The research: seven simulators investigated

### 2.1 Bullet / PyBullet (C++, the robotics standard)

- **Broad phase**: `btDbvtBroadphase` — *two* dynamic AABB trees (one for moving,
  one for static/sleeping bodies); alternatives `btAxisSweep3` (incremental
  sweep-and-prune) and `btSimpleBroadphase` (brute force).
- **Narrow phase**: GJK (`btGjkPairDetector`) for convex distance, **EPA**
  (`btGjkEpa2`) for penetration depth, MPR optional; special-cased analytic tests
  for sphere-sphere, sphere-box, box-box (SAT).
- **Contact manifold**: `btPersistentManifold`, up to **4 cached points** per pair,
  persisted across frames (**warm starting**), refreshed and pruned by a contact
  breaking threshold (~0.02).
- **Normal convention**: `contactNormalOnB` — unit vector **from B toward A**;
  `contactDistance < 0` means penetration.
- **CCD**: per-body swept-sphere conservative advancement + **motion clamping**
  (`setCcdMotionThreshold`, `setCcdSweptSphereRadius`) — translational only.
- **Response**: `btSequentialImpulseConstraintSolver` — projected Gauss–Seidel,
  defaults: 10 iterations, Baumgarte erp 0.2, **split impulse** below −0.04
  penetration, restitution suppressed below a resting velocity threshold,
  warm-starting factor 0.85, collision margin 0.04.
- **Optimizations**: simulation islands, sleeping, SIMD, multithreaded solver.

### 2.2 Rapier / parry (pure Rust, the modern reference)

- **Broad phase**: BVH over collider AABBs (parry's quantized, SIMD-friendly
  `Qbvh`); output feeds a contact graph.
- **Narrow phase**: parry — GJK + penetration recovery, SAT for primitive pairs;
  produces `ContactManifold`s with `local_n1`/`local_n2` (per-shape local normals)
  and a world normal; per-point `dist < 0` = penetration; solver contacts carry
  persisted impulses (warm starting).
- **Normal convention**: from **collider 1 toward collider 2**.
- **CCD**: **nonlinear time-of-impact** (translation *and* rotation) + motion
  clamping, `max_ccd_substeps`.
- **Response**: sequential impulse with stabilization iterations, SoA/SIMD contact
  batches, restitution with a velocity threshold, islands + sleeping.

### 2.3 Project Chrono / PyChrono (C++, multi-physics engineering)

- Two collision systems: an embedded customized **Bullet** (DBVT + GJK/EPA) or the
  in-house **Multicore** system (spatial binning broad phase + analytic/MPR narrow
  phase) for granular dynamics.
- Two *physics* formulations:
  **NSC** (non-smooth): contacts as hard constraints — a Differential Variational
  Inequality solved per step as a Cone Complementarity Problem by **APGD**
  (accelerated projected gradient descent); large steps allowed.
  **SMC** (smooth): penalty/compliant contact forces from interpenetration
  (Hertz-style); embarrassingly parallel but needs small steps.
- **Envelope & margin**: contacts are *anticipated* at an outward envelope
  distance before touching (an anti-tunneling and constraint-warm-up band).
- **Normal convention**: the X column of the per-contact `plane_coord` rotation
  matrix (reported via `ReportContactCallback` with reaction forces expressed in
  that contact frame).
- No dedicated CCD path — the envelope plus small steps stand in for it.

### 2.4 SOFA (C++, interactive multi-physics / biomechanics)

- Explicit multi-stage pipeline: `BruteForceBroadPhase` (AABB pair cull) →
  `BVHNarrowPhase` → pluggable intersection methods (`MinProximityIntersection`,
  `LocalMinDistance`, …) with two key distances:
  **alarmDistance** (contact created *before* overlap) and **contactDistance**
  (the separation the response aims to hold). Detection-before-contact is SOFA's
  anti-tunneling and stability mechanism.
- `DetectionOutput` carries both contact points, a **normal (outward from the
  first model)** and a **signed distance (negative = interpenetration)**.
- Response: penalty springs (`PenalityContactForceField`) or **Lagrangian
  unilateral constraints** (`FrictionContactConstraint`, Gauss–Seidel).

### 2.5 Open Source Physics / EJS (Java, the teaching classic)

- No engine pipeline: 2-body **analytic overlap tests** (center distance vs. sum
  of radii) with the response computed in closed form — velocity decomposed along
  the **line of centers** (the normal), the along-normal components transformed by
  the 1-D elastic formulas, the perpendicular components untouched.
- The teaching devices: draggable bodies, an **impact-parameter slider** for
  head-on vs. glancing, the **center-of-mass cross** shown sailing straight
  through the collision, momentum framing throughout.

### 2.6 SIMple Physics (Rust, educational; engine = nphysics/ncollide)

- ncollide (ancestor of parry): **DBVT** broad phase with loosened AABBs, GJK +
  EPA narrow phase, **persistent contact manifolds** with feature IDs and
  proximity events (Disjoint / WithinMargin / Intersecting).
- SIMple's own layer exposes a **user-editable Lua `collision_fn`** — students see
  and modify the response formula (`p1:dist(p2) <= r1 + r2`, impulse along the
  radius vector, a `cos_angle < 0` guard against re-colliding separating pairs).

### 2.7 PhysicsHub (JavaScript, student-built web sims)

- Cheap squared-distance reject, then the exact circle-overlap test; response is
  the Wikipedia two-body elastic formula along the center line, with a global
  **damping slider** as the restitution control; kinetic energy plotted live;
  velocity vectors drawn as arrows. Fixed-timestep overlap testing — the honest
  example of *how tunneling happens* when nothing guards against it.

---

## 3. The comparison chart

| | Bullet | Rapier/parry | Chrono | SOFA | OSP/EJS | ncollide (SIMple) | PhysicsHub |
|---|---|---|---|---|---|---|---|
| **Broad phase** | dual DBVT / SAP | Qbvh BVH | DBVT or grid binning | brute-force AABB | none (2-body) | DBVT (loose AABBs) | squared-distance reject |
| **Narrow phase** | GJK+EPA, SAT prims | GJK+pen., SAT prims | GJK/EPA or analytic+MPR | BVH + proximity methods | analytic circles | GJK+EPA | analytic circles |
| **Manifold** | persistent ≤4 pts | persistent + impulses | Bullet's / own | DetectionOutput list | single implicit | persistent + feature IDs | none |
| **Normal** | on B, B→A | collider1→collider2 | X col of plane_coord | outward from 1st model | line of centers | GJK/EPA geometric | line of centers |
| **Penetration sign** | dist < 0 | dist < 0 | dist < 0 | dist < 0 | overlap bool | depth ≥ 0 | overlap bool |
| **TOI / anti-tunnel** | swept-sphere + clamp | nonlinear TOI + clamp | envelope + small dt | alarmDistance band | exact 2-body algebra | swept TOI | none (tunnels!) |
| **Response** | sequential impulse PGS | sequential impulse | NSC (DVI/CCP/APGD) or SMC penalty | penalty or Lagrangian | closed-form elastic | impulse/constraint | closed-form + damping |
| **Stabilization** | split impulse / erp 0.2 | bias iterations | constraint / compliance | contactDistance | exact | slop + projection | none |
| **Restitution** | threshold ≈ resting vel | velocity threshold | material pairs | material | e in formula | material | damping slider |
| **Key numbers** | margin 0.04, break 0.02, 10 iters | substeps, margins | envelope/margin | alarm/contact dist | — | margin band | — |

### 3.1 The concept tree (properties · capabilities · techniques · hurdles → what posim adopted)

```
collision detection science
├── DETECTION
│   ├── broad phase (find candidate pairs cheaply)
│   │   ├── brute-force AABB cull            ← SOFA, PhysicsHub      → adopted (O(n²), educational scale)
│   │   ├── sweep-and-prune                  ← Bullet btAxisSweep3   → documented upgrade path
│   │   └── dynamic BVH / Qbvh / binning     ← Bullet, Rapier, Chrono→ documented upgrade path
│   ├── narrow phase (exact pair geometry)
│   │   ├── analytic primitive pairs         ← OSP, PhysicsHub, all  → adopted: sphere-sphere,
│   │   │                                                              sphere-cuboid, point-as-sphere
│   │   ├── SAT for boxes (15 axes)          ← Bullet box-box        → adopted: cuboid-cuboid
│   │   └── GJK + EPA (any convex)           ← Bullet, parry         → NOT needed (closed shape set),
│   │                                                                  justified in §4
│   ├── contact geometry
│   │   ├── contact point                    ← all                   → adopted (support vertex /
│   │   │                                                              closest point / edge-edge)
│   │   ├── UNIT NORMAL = action-reaction    ← all (conventions vary)→ adopted: from body i toward j
│   │   └── signed separation / depth        ← all (dist < 0)        → adopted: separation > 0 apart,
│   │                                                                  ALSO the sundials root function
│   └── time of impact (anti-tunneling)  ★ THE HARD PROBLEM ★
│       ├── swept shapes + motion clamping   ← Bullet, Rapier        → surveyed, not needed
│       ├── proximity band (detect early)    ← SOFA, Chrono envelope → surveyed
│       └── INTEGRATOR EVENT ROOTFINDING     ← (none of the seven!)  → ADOPTED: sundials CVodeRootInit
│                                                                      lands on the exact root — posim's
│                                                                      precision differentiator
├── RESPONSE
│   ├── impulse (sequential, Gauss-Seidel)   ← Bullet, Rapier        → adopted (K-matrix with angular
│   │                                                                  terms, ≤10 passes)
│   ├── restitution e + velocity threshold   ← Bullet, Rapier        → adopted (e=1 default, min-combine,
│   │                                                                  threshold kills resting jitter)
│   ├── positional projection with slop      ← Bullet split impulse  → adopted (depth − slop split by
│   │                                                                  inverse mass)
│   ├── Lagrangian / complementarity (APGD)  ← Chrono NSC, SOFA      → overkill (documented)
│   └── penalty springs                      ← SOFA, Chrono SMC      → rejected (stiffness ruins the
│                                                                      adaptive integrator)
└── HURDLES (and the posim answer)
    ├── tunneling            → event rootfinding + CVodeSetMaxStep cap (verified: 100 m/s vs 5 mm plate)
    ├── resting contact/Zeno → tiered guard: 64 events IN A BURST → plastic; 128 → disarm + project
    ├── simultaneity         → Gauss-Seidel passes over all pairs (Newton's cradle emerges)
    ├── deep initial overlap → end-of-interval sweep + projection (no root to find)
    └── degenerate geometry  → concentric centers etc. skipped, never a division by zero
```

---

## 4. Evaluation for a pure-Rust port (zero unsafe, zero dependencies)

**Essential — adopted:**

| Technique | Source | Why |
|---|---|---|
| AABB broad-phase cull | all | O(n²) is exact and fast at classroom scale; trees add code, not correctness |
| Analytic narrow phase | OSP lineage + Bullet primitives | with a *closed* shape set {point, sphere, cuboid} every pair has an exact closed form — faster than GJK, no termination heuristics, and the normal is **exact**, which is the entire point |
| SAT for cuboid-cuboid | Bullet | 15 axes, deepest-axis normal — exact for boxes |
| Impulse response with effective-mass matrix K | Bullet/Rapier | provably momentum-conserving action–reaction along n̂; angular terms give correct spin transfer |
| Restitution + velocity threshold | Bullet/Rapier | e ∈ [0,1] with min-combine; threshold prevents settling jitter |
| Positional projection + slop | Bullet split-impulse idea | recovers penetration without injecting energy |
| Sequential (Gauss–Seidel) multi-pair passes | Bullet | resolves simultaneous and cascading contacts |
| **Integrator event rootfinding for TOI** | *novel here* | the integration backend (SUNDIALS) already contains a root finder that checks every internal step and interpolates onto the crossing — precision no engine-side swept test can match |

**Valuable — documented as the upgrade tier:** persistent contact manifolds with
warm starting (stacking), 2–4-point face manifolds (boxes resting flat),
sweep-and-prune/BVH broad phase (hundreds of bodies), Coulomb friction (per-object
μ, tangential impulse clamped to the cone — designed, deliberately not shipped in
this delivery), islands + sleeping.

**Overkill — rejected with reasons:** GJK/EPA/MPR (no open shape set to serve);
NSC complementarity solvers (APGD) — research-grade machinery for granular
matter; penalty springs — their stiffness would fight CVODE's adaptive error
control; GPU/SIMD batching — no workload to feed it.

**Why GJK/EPA is genuinely unnecessary here** (the one decision an expert would
question): GJK exists to handle *arbitrary* convex shapes behind a support
function. posim's shape set is closed — point, sphere, cuboid — so all six pair
types have exact constant-time formulas with exact normals. Bullet itself
special-cases exactly these pairs for the same reason. If a general convex hull
shape is ever added, GJK+EPA (or parry) becomes the right tool; the `Contact`
record and solver do not change. *(The shape set has since grown — torus, disk,
cylinder, and now the compound dumbbell — and remains closed: §6's three-tier
dispatch keeps every ball contact and every face contact exact, still without
GJK; the dumbbell never even reaches the dispatch whole — it decomposes into
ball and cylinder parts first.)*

---

## 5. The recommendation (adopted): collision detection in ALL rigid-body scenes

1. **On by default, everywhere.** Every `STEP`, `RUN`, and scene-window playback
   detects collisions when at least one collidable pair exists. `COLLIDE OFF`
   opts out for pure point-mass studies. Two `Point`s cannot collide;
   two static bodies are skipped.
2. **Event-driven time of impact.** The pairwise **signed separation** is handed
   to the integrator as a root function (`CVodeRootInit` / `ARKodeRootInit`).
   CVODE/ARKODE check it after every *internal* step and interpolate onto the
   zero crossing: the reported impact time is correct to solver precision
   (measured: ~1e-15 relative). Direction filtering (approach only) plus the
   integrator's own zero-at-restart handling prevent re-triggering after the
   impulse.
3. **Anti-tunneling cap.** While armed, the internal step is capped at
   (smallest feature size)/(2 × max **reachable** relative **surface**
   speed) — center speed plus what the accelerations can add before the next
   refresh, plus each body's spin bound times R_bound, since a rotating edge
   can sweep into contact without the centers moving — refreshed after every
   event and at every interval start (§6 has the exact bounds). A root
   checked only at step ends cannot miss a thin body.
4. **Exact impulse at the root.** `J = −(1+e)(v_rel·n̂)/(n̂ᵀK n̂)` with
   `K = (1/m_i + 1/m_j)·1 − [r_i]×I⁻¹ᵢ,w[r_i]× − [r_j]×I⁻¹ⱼ,w[r_j]×`,
   applied as `±J n̂` (and `±r×J n̂`) through the canonical setters — the
   action–reaction pair by construction. Static bodies receive no writes.
5. **Guards.** Tiered Zeno guard (tier 2 plastic); **elastic** end-of-interval
   penetration sweep for deep initial overlaps and grinding approximate
   contacts; degenerate geometry skipped safely.
6. **The normal is public.** Every contact is recorded and exposed: grammar
   paths (`contact0.normal`), the `CONTACTS` command, machine-mode JSON
   (`"contacts": [...]`), and the scene window (golden arrows). Any simulation
   entity can read the action–reaction line.
7. **Structural safety.** With zero collidable pairs the rootfinding is never
   armed and the CVODE code path is **bit-identical** to the pre-collision
   implementation (verified in source at cvode.rs:912-1031 and enforced by test).

---

## 6. The posim implementation (what exactly was built)

**Modules.** `physical_object/src/boundary.rs` (the shape set: SDFs, support
extents/points/ranks, analytic inertia); `physical_object/src/collide.rs`
(geometry + impulse response, fully unit-tested); event wiring in
`physical_object/src/integrate.rs` (CVODE + SPRK event loops); grammar in
`posim` (`COLLIDE`, `CONTACTS`, `BOX`, `contactK` paths); scene arrows + shape
wireframes in `posim/src/scene/`.

**The shape set.** Seven boundaries: `Point`, `Sphere`, `Cuboid`, the
shapes-and-box release's `Torus { ring_radius, tube_radius }` (a centerline
circle of radius `c` in the local xy-plane swept by a tube of radius `a`;
inner radius = `c − a`, outer radius = `c + a`), `Disk { radius }` (an
**ideal zero-thickness** solid disk in the local xy-plane) and
`Cylinder { radius, half_height }` (full height = 2 × half-height), and —
since this release — the first **compound** shape,
`Dumbbell { r1, r2, rod_radius, z1, z2, f1, f2 }`: solid sphere 1 (radius
`r1`, mass fraction `f1`) centered at `(0, 0, z1)`, solid sphere 2 (`r2`,
`f2`) at `(0, 0, z2)`, joined by a solid rod of radius `rod_radius` — two
spheres plus a rod as **one** rigid body. The
`boundary::dumbbell(m1, m2, m_rod, r1, r2, rod_radius, length)` constructor
computes the offsets from the part masses — `z1 = −(m2 + m_rod/2)·L/M`,
`z2 = (m1 + m_rod/2)·L/M` (`L` the center-to-center length,
`M = m1 + m2 + m_rod`) — so the local origin **is** the center of mass: the
identity `m1·z1 + m2·z2 + m_rod·(z1 + z2)/2 = 0` holds exactly (pinned by
test), and the stored fractions `f1`/`f2` keep every part mass recoverable
from the total. Each non-legacy shape carries an **exact SDF** (torus:
distance to the centerline circle minus the tube radius; disk: the unsigned
distance, zero exactly on the disk — a zero-thickness body has no interior;
cylinder: the 2-D box SDF in (ρ, z); dumbbell: the exact **union** — the
min of its parts' SDFs, two spheres and the rod's capped cylinder) and the
**analytic inertia tensor**: torus `I_z = m(c² + ¾a²)`,
`I_x = I_y = m(½c² + ⅝a²)`; disk `I_x = I_y = ¼ma²`, `I_z = ½ma²`
(perpendicular-axis theorem); cylinder `I_z = ½mr²`,
`I_x = I_y = m(3r² + 4h²)/12` (h = half-height); dumbbell the exact
**composite** — `2/5·m·r²` for each sphere plus its parallel-axis term
`m·z²`, plus the rod's cylinder terms about the COM (hand-checked). Every
shape also provides its **support extent** in exact closed form — since
this release the **true directed** support `h(u) = max x·u`: the dumbbell
is the first non-centrally-symmetric shape (its off-center spheres make
`h(u) ≠ h(−u)`), and every symmetric shape's closed form is bit-identically
unchanged (for them `h(−u) = h(u)`); `world_aabb` now takes the ± directed
extents per axis, and for the torus the support is that of its convex hull
— a support function cannot see the hole. Each shape also answers a
**support point** achieving the extent — the **centroid of the supporting
set** when that set is a face, edge, circle or cap, so a flat-on contact
carries no spurious lever arm — and a **support rank** (0 = single
point/rim point, 1 = edge or circle, 2 = face or cap).

**The Contact record** (pinned convention, ARCHITECTURE.md §3.8):

```rust
pub struct Contact {
    pub i: usize,       // first body (lower index)
    pub j: usize,       // second body
    pub t: f64,         // event time (the interpolated root)
    pub point: Vec3,    // world-space contact point
    pub normal: Vec3,   // UNIT normal, from body i toward body j
    pub depth: f64,     // penetration depth ≥ 0 (≈ 0 at a root)
    pub rel_vel_n: f64, // pre-impulse (v_j − v_i)·n̂  (< 0 = approaching)
    pub impulse_n: f64, // scalar impulse magnitude J
}
```

**The three-tier dispatch.** Every pair is served in one of three exactness
tiers (still no GJK/EPA/MPR machinery):

1. **Exact ball tier** — a *ball* (sphere, or point as the zero-radius case)
   against **anything**: separation = the partner shape's exact SDF at the
   ball center (in the shape frame) minus the ball radius. Sphere-sphere
   keeps its closed form (`|Δ| − (r_a + r_b)`, normal along the line of
   centers) and sphere-cuboid keeps the box-SDF formula (continuous inside
   and out; the interior branch picks the nearest face). For
   torus/disk/cylinder the contact is the exact closest surface point, the
   normal the exact outward direction, and the contact point sits midway
   across the overlap band. Because the SDFs are exact, a small ball **can
   genuinely thread a torus hole** (its separation never reaches zero on a
   through-hole line) and a point particle **passes through the ideal
   zero-thickness disk** (a measure-zero contact — see the FAQ). Degenerate
   configurations (ball center exactly on the torus centerline, exactly on
   the disk, on the cylinder axis at side contact) are skipped safely —
   never a division by zero.
2. **Exact cuboid SAT** — cuboid-cuboid is the 15-axis SAT (3+3 face axes,
   9 edge cross products), **unchanged**: the maximum separation is the
   signed separation, its axis (oriented i→j) the normal; contact point =
   deepest support vertex (face case) or closest points of the supporting
   edges (edge case).
3. **Support-axis tier** — every remaining extended-vs-extended pair (any
   pair involving a torus/disk/cylinder not covered above; a dumbbell is
   decomposed first — see below — so only its rod, as a free cylinder,
   can arrive here). The **directed** SAT gap `d·l − h_a(l) − h_b(−l)` is
   evaluated for **both orientations** of every candidate axis — for
   centrally symmetric shapes `h(−u) = h(u)` and it reduces exactly to
   the classic `|d·l| − (e_a + e_b)`; the dumbbell's off-center spheres
   need the general form — over a finite candidate-axis
   list: each cuboid's three **face axes**, each round shape's **symmetry
   axis**, every normalized **cross product** of those primary axes
   (edge-vs-edge and edge-vs-rim separations; near-parallel pairs skipped as
   in the SAT), the **radial rejection axes** — the component of the center
   offset perpendicular to each primary axis, which is the true lateral
   separating direction when two round shapes lie (near-)parallel with an
   axial offset, exactly where the parallel-axis cross product vanishes and
   the raw center line is tilted — and the **center line**. The maximum gap
   is the separation and its axis (oriented i→j) is the normal. This tier
   is **exact for face-on contacts** — in particular every wall-slab
   contact in a `BOX` — and **exact for side-side contacts of parallel and
   near-parallel round shapes**: two parallel cylinders with overlapping
   axial ranges separate by exactly the lateral gap, with a purely lateral
   normal (test `parallel_cylinders_side_contact_is_exact`; this used to
   fire early with a tilted normal). The conservative caveat now applies
   **only to genuinely skew corner-on configurations** (contact may be
   reported slightly early, along a candidate axis). For the torus it sees
   the convex hull, so only balls can thread the hole. One face-on
   configuration stays invisible outright: two disks with **parallel**
   planes both have zero extent along the shared normal, so their
   separation is `|dz|` — it touches zero at plane coincidence without a
   sign change, exactly like the point-through-disk case, and the crossing
   produces no root for the downward-crossing rootfinder (pinned by test
   `parallel_disk_disk_separation_is_the_documented_limitation`; tilt one
   disk or model a thin cylinder — see the FAQ). **Contact point = the
   support point of the lower-support-rank body** — the incident body's
   deepest point (a tilted cylinder's rim point against a wall face, not
   the centroid of the wall face). **Equal ranks prefer the body with the
   clearly smaller flat footprint** (`support_footprint_radius`, the
   lateral circumradius of the supporting set: chosen when under half the
   partner's), so a small cap landing on a big wall face contacts at the
   **cap center**, not halfway toward the face centroid (test
   `small_cap_on_large_face_contacts_at_the_cap_center`); comparable
   footprints use the midpoint of the two support-set centroids.

**Compound decomposition (the dumbbell).** A dumbbell never reaches a
dispatch tier as a whole: the narrow phase **decomposes dumbbell-vs-anything
over its parts**. Each sphere part is a ball served by the exact ball tier —
the partner shape's exact SDF evaluated at the sphere center — which is
exact against **every** shape, *including another dumbbell*, whose union
SDF is itself exact. The rod recurses as a free-standing cylinder, served
with a cylinder's usual exactness profile (exact face-on and parallel
side-side, conservative only on skew corners); contact selection takes the
deepest part (the `ball_vs_shape_contact` consolidator plus the pose-level
`contact_geometry_at`). So in dumbbell-vs-dumbbell every part pair that
involves a sphere is exact, and only **rod-vs-rod** ever reaches the
approximate support-axis tier. This part-wise exactness is what the
conservation anchor rests on (§8, Example 12): the impulse pair acts at one
shared contact point, so two tumbling dumbbells colliding off-center
conserve E, P **and** L through real CVODE events.

**The rigid box (`BOX <size>`).** Six static `Cuboid` wall slabs enclosing an
axis-aligned cube of the given inner side length, each with
`inverse_mass = 0` **and** zero inverse inertia. This is infinite mass
*exactly as the equations of motion see it*: the state stores momenta and the
dynamics only ever multiply by the inverse — `v = p·m⁻¹`, and the impulse
denominator is `n·Kn = m_i⁻¹ + m_j⁻¹ + angular terms`. A static side
therefore contributes exactly 0 to every denominator, receives no state
writes at all, and reflects everything elastically while remaining
bit-identically at rest. No large-mass approximation appears anywhere.

**The root function IS the narrow phase.** The same `separation_at` geometry runs
inside the CVODE/ARKODE `g` function on the packed state, so detection and
response can never disagree about what "touching" means.

**Anti-tunneling with spin — and with acceleration.** The while-armed step
cap is (smallest crossable feature)/(2 × max **reachable** relative
**surface** speed), where *reachable* spans the horizon Δt to the next
refresh — one output interval (the cap is re-refreshed at every interval
start and after every event). The linear term is the relative center speed
**plus (a_i + a_j)·Δt**, each body's acceleration taken from pairwise
gravity at the current configuration, uniform gravity, qE and the external
force (the magnetic force is ⟂ v and can never grow speed) — so a ball
released **from rest** above a thin plate, whose instantaneous relative
speed is zero at arm time, still gets a finite cap and is caught (test
`ball_released_from_rest_does_not_tunnel_the_thin_plate`: TOI matches the
analytic √(2h/g) fall to 1e-6). The spin term bounds each body's rate as
**|ω| ≤ √3·‖I⁻¹‖∞·(|L| + Δt·(|τ_ext| + ‖M‖∞·|B|))**, multiplied by its
R_bound — a bound through the angular *momentum* rather than the
instantaneous |ω|, so it covers torque-free polhode motion exactly (L is
conserved, yet |ω| can spike up to |L|/I_min mid-interval with no torque
at all). The cap returns no-cap only when literally nothing can move.
Feature sizes of the new shapes: the torus's tube radius, the cylinder's
min(radius, half-height); the ideal disk has zero thickness of its own, so
its cap comes from the radius of whatever ball approaches it (the pairwise
min picks the smaller feature).

**The elastic closing sweep.** `resolve_penetrations` now takes an explicit
mode. The Zeno tier-2 guard still resolves **plastically** (kill a
chattering contact dead), but the end-of-interval safety sweep resolves
**elastically**: a slowly-grinding approximate (support-axis) contact that
never produces a clean root can no longer bleed energy sweep after sweep.

**The hmax-span fix (an honest bug report).** The after-event refresh of the
anti-tunneling cap clamps the computed `hmax` into a span, and that span used
to be `tout − t` — the remainder of the current *output interval*. A
collision landing exactly **on** an output boundary (which the box geometry
of Example 11 promptly produced) collapsed that span to zero, pinning the max
step at the 1e-12 clamp floor; the stale cap was never recomputed, so every
later interval was starved into `CV_TOO_MUCH_WORK` (500 000 no-advance
steps). The fix is twofold: the clamp span is now the remaining **run**
(`t_end − t`, exactly as at arm time), and the cap is re-refreshed at the
start of every output interval, so a cap computed near a previous `tout` can
never go stale.

**Statistics.** `RunReport` gains `nge` (root-function evaluations) and
`ncollisions`; the system tracks `collision_count` and the (capped, 1024)
`contacts` record of the last run.

**Defaults.** restitution 1.0 (elastic — energy checks stay exact),
combine = min(e_i, e_j); `restitution_threshold` 1e-3; `contact_slop` 1e-9;
`MAX_EVENTS_IN_BURST` 64 (Zeno tier 1), 128 (tier 2). A *burst* is a run of
events separated by less than `ZENO_GAP_RELATIVE * max(|t|, 1)` = `1e-9 max(|t|,1)`;
any real flight between two impacts resets it. It was formerly counted per
**output interval**, which made the physics depend on how often output was
requested — see `box_of_shapes_m32.md` §5.

---

## 7. Command reference

| Command / path | Meaning |
|---|---|
| `COLLIDE` | report status: on/off, collidable pairs, impulses so far |
| `COLLIDE ON` / `COLLIDE OFF` | enable (default) / disable detection |
| `CONTACTS` | list every contact of the last `STEP`/`RUN` |
| `NEW TORUS { … }` | new torus: `ring_radius`/`tube_radius` (defaults 1 / 0.25) **or** the `inner_radius` + `outer_radius` pair — resolved and validated once at the closing brace, so the pair is genuinely order-independent; `inner_radius = 0` (the horn torus) is valid |
| `NEW DISK { … }` | new **ideal zero-thickness** solid disk: `radius` (default 1) |
| `NEW CYLINDER { … }` | new cylinder: `radius` (default 0.5); `height` sets the **full** height (default half_height 1) |
| `NEW CUBE …` / `NEW DISC …` | aliases for `CUBOID` / `DISK` |
| `NEW DUMBBELL [AS <name>] { … }` | new rigid two-spheres-plus-rod body: `m1`, `m2`, `m_rod`, `r1`, `r2`, `rod_radius`, `length` (defaults 1, 1, 0.5, 0.25, 0.25, 0.1, 1) plus the ordinary entity values — deferred and order-independent, validated once at the closing brace via `boundary::dumbbell` (a failed `NEW` leaves no ghost); the local origin is placed at the COM; `DUMBELL` is an accepted alias spelling; `AS <name>` registers a user name usable in every path (`d.mass`, `d.vx`) |
| `GET/SET objN.m1` / `.m2` / `.m_rod` | the dumbbell part masses, read **and** write — the stored mass fractions make every part recoverable, and a member write rebuilds total mass, COM offsets and the inertia tensor in one step |
| `GET/SET objN.r1` / `.r2` / `.rod_radius` / `.length` | the dumbbell geometry (`rod_r`/`len` accepted), read **and** write — same one-step rebuild |
| `BOX <size>` / `BOX OFF` / `BOX` | create / remove / report the rigid bounding box: six static wall slabs with `inverse_mass = 0` (infinitely massive) |
| `GET/SET objN.ring_radius`, `.tube_radius` | the torus generating radii (writes recompute inertia) |
| `GET/SET objN.inner_radius`, `.outer_radius` | the derived torus pair (each write holds the other one fixed) |
| `GET/SET objN.height` / `.half_height` | cylinder height (`height` = 2 × half_height) |
| `GET/SET objN.radius` | now serves sphere, **disk and cylinder** (a write keeps the shape family); a torus write errors — set `ring_radius`/`tube_radius` or `inner_radius`/`outer_radius` instead |
| `GET objN.inverse_mass` | 0 for a wall slab — infinite mass, exactly |
| `GET system.box` | inner box size (0 = none) |
| `GET contactK.i` / `.j` | the colliding pair (indices) |
| `GET contactK.t` | event time (the interpolated root) |
| `GET contactK.point` | world contact point (vec3) |
| `GET contactK.normal` | **unit action–reaction line, i → j** (vec3) |
| `GET contactK.depth` | penetration depth at the event |
| `GET contactK.rel_vel_n` | approach speed along the normal (negative) |
| `GET contactK.impulse` | scalar impulse magnitude J |
| `SET objN.restitution = e` | bounciness 0 (plastic) … 1 (elastic, default) |
| `GET system.contacts` | number of recorded contacts |
| `GET system.collisions` | running impulse total this session |
| `SET system.restitution_threshold` | resting-contact speed threshold |
| `SET system.contact_slop` | tolerated overlap before projection |

Contact paths are ordinary expression atoms: `contact0.impulse *
contact0.normal.x` computes the x-component of the impulse vector. In machine
mode, `{"op":"state"}` includes `"contacts"`, `"collide_enabled"` and
`"collision_count"`, and — since this release — `"box"` (number or null) plus
per-object `"wall"` and `"inverse_mass"`. In the scene window the
**Contacts** button / `C` key toggles golden normal arrows at the contact
points.

`LIST` tags each wall slab `[wall: static, inverse_mass=0]` (any other
inverse-mass-0 body reads `[static, inverse_mass=0]`); `DEL` keeps the wall
index list renumbered, and deleting any wall dissolves the box
(`system.box` → 0) while the surviving slabs **stay tracked**: `LIST` keeps
their `[wall]` tag, bare `BOX` reports
`box: dissolved (a wall was deleted; N tracked slab(s) remain — BOX <size>
replaces them, BOX OFF removes them)`, and `BOX <size>` removes the
survivors before building the new box — no orphan slab is ever leaked.
Scene `init` entities carry the new shape parameters and
`"wall": true`, with a top-level `"box": <size>`; `scene.html` draws torus
(outer/inner equators + tube rings + 4 cross-sections), disk (rim +
2 diameters) and cylinder (2 rims + 4 side lines) wireframes, all
quaternion-rotated so spin is visible, plus a dashed `#5d84a8` interior
wireframe for the box — the wall slabs themselves are **not** drawn as
bodies.

---

## 8. How it was verified

- **39 unit tests** in `physical_object` (geometry: known separations, SAT face
  vs. edge, continuity across touch; response: `n·Kn = 1/m₁+1/m₂` for central
  hits, static-wall reflection with the wall bit-unchanged, separating pairs
  untouched, projection separating deep overlap). New this release: the
  torus/disk/cylinder **SDFs at hand-checked points**, support extents/points
  (the invariant `p·u = h(u)` verified across shapes and directions) and
  ranks, the new **analytic inertia tensors** (torus Iz = 2.4375 for m = 1,
  c = 1.5, a = 0.5; disk Iz = Ix + Iy), ball-vs-torus contact with the hole
  genuinely passable, disk face/rim and cylinder side/cap contacts, the
  **tilted-torus anchor** — a flat torus of outer radius 2 exactly inscribes
  a 4-wide box (separation 0 to 1e-12) while the (1,1,1)/√3 tilt clears
  every wall by exactly **2 − (1.5√(2/3) + 0.5) ≈ 0.2753** — and a static
  slab **reflecting a cylinder elastically** (v → −v along the normal, wall
  bit-unchanged). New in the review round: **exact side-side contacts of
  parallel cylinders** via the radial rejection axes
  (`parallel_cylinders_side_contact_is_exact` — the lateral gap to 1e-12
  across four offsets and a purely lateral normal), the **pinned parallel
  disk-disk limitation** (`parallel_disk_disk_separation_is_the_documented_limitation`
  — separation = |dz| exactly, touching zero without a sign change) and the
  **smaller-footprint contact rule**
  (`small_cap_on_large_face_contacts_at_the_cap_center` — a 0.25-radius cap
  pressed into a big slab face contacts at the cap center). New in the
  dumbbell release: `dumbbell_constructor_com_sdf_and_supports` — the
  constructor's COM placement (`m1·z1 + m2·z2 + m_rod·(z1+z2)/2 = 0`
  exactly), the union SDF at hand-checked points, the composite inertia,
  and the **directed** supports (`h(u) ≠ h(−u)` for the off-center
  spheres; every symmetric shape's extents unchanged) — and
  `dumbbell_wall_gaps_and_ball_contacts_are_exact` — an **asymmetric**
  dumbbell's wall gaps exact at **both** ends (the light end pokes
  farther: `|z1| > |z2|` when `m2 > m1`), plus ball-vs-pole and
  ball-vs-rod contacts exact.
- **16 integration tests** (`tests/collision.rs`), each against a closed form:
  velocity exchange with **TOI error < 1e-9**, unequal-mass formulas, restitution
  ratios and plastic KE loss ½μv², oblique normals = line of centers, off-center
  ΔL = r×J with total L conserved, 3-sphere cradle, bounce apex = e²h with impact
  at √(2h/g), **no tunneling** at 100 m/s vs a 5 mm plate (TOI to 1e-9),
  SPRK-path exchange, **bit-identical zero-pair invariance** (COLLIDE ON vs OFF),
  armed-but-quiet agreement ≤ 1e-12. New this release: a **ball in a rigid
  box** of inverse-mass-0 slabs (energy conserved to 1e-9, the ball never
  escapes, all six walls bit-identical to construction); a **point particle
  threading the torus hole** with zero events while a fat ball on a nearby
  line hits the tube at the analytic **TOI = (3 − √1.45)/4** and the free
  torus recoils (E and p conserved); a **mixed cylinder + disk + cube
  rattle** inside the box with **|dE|/E < 1e-6** through the events; and a
  ball **released from rest** above a thin static plate under uniform
  gravity, caught at the analytic free-fall TOI = √(2·4.985/10) to 1e-6
  (`ball_released_from_rest_does_not_tunnel_the_thin_plate` — the
  acceleration-aware anti-tunneling cap at work: the instantaneous relative
  speed is zero at arm time). New in the dumbbell release, the headline
  anchor: **two tumbling dumbbells colliding off-center conserve E, P AND
  L (about the origin) to 1e-8 through the real CVODE event path**
  (`colliding_dumbbells_conserve_energy_momentum_and_angular_momentum`) —
  the impulse pair acts at one shared contact point, so the net torque is
  zero.
- **Grammar tests** (39 posim tests): the whole command family end-to-end,
  including read-only and range errors and the `system.collisions` vs
  `system.collide` distinction; new: the `NEW TORUS/DISK/CYLINDER` parameter
  paths (the order-independent inner/outer pair, `HEIGHT` as the full
  height, `radius` across the shape family) and the whole `BOX` family
  (creation, status, `OFF`, `system.box`, wall tags, machine-mode
  `box`/`wall`/`inverse_mass`). New in the review round:
  `torus_pair_is_order_independent_and_new_is_transactional` (both orders
  of the inner/outer pair — including `{ outer_radius = 0.5,
  inner_radius = 0.2 }`, which used to fail in one order — the horn torus
  `inner_radius = 0` on NEW and SET, the refused `SET objN.radius` on a
  torus, and a failed `NEW` leaving **no ghost object** behind) and
  `box_recreate_after_wall_deletion_leaks_nothing` (bare `BOX` reports the
  dissolved state; `BOX <size>` after a wall deletion removes the surviving
  tracked slabs first). New in the dumbbell release:
  `def_call_named_objects_and_dumbbell_members` — the full `create_dumbell`
  flow (define / call / member reads and writes / shorthands / renumber /
  errors / redefine) — and the scene test
  `reset_restores_the_initial_state_and_start_reruns`.
- **Scene**: server frame carries the exact analytic `normal/point/impulse`
  (verified live over a real WebSocket); the golden arrow's pixels verified on
  the canvas; reverse-through-collision replays to exactly t = 0. For this
  release the box-of-shapes demo was re-verified in a real browser: the
  `init` frame carried `box: 4`, the wall flags and every new shape's
  parameters; the window's playback copy conserved E to 1e-9; the golden
  contact arrows drew (2218 gold pixels counted on the canvas) and the
  dashed box wireframe drew (5904 pixels); reverse playback was exercised.
- **Regression anchors intact**: outer solar system, Kepler, gyroradius,
  tumbling body — all still SUCCESS; **104 tests green** (40 lib +
  16 collision + 9 conservation + 39 posim); the full workspace builds with
  **zero warnings** and `Cargo.lock` still lists only the five local crates.

---

## 9. The twelve examples (run for the USER and the MANAGER)

All twelve live in `scripts/collisions/` and run with
`cargo run -p posim -- --script scripts/collisions/NN_name.posim`
(outputs below are the real captured runs). Two additional self-checking Rust
examples (`newtons_cradle`, `bouncing_ball_restitution`) print SUCCESS/FAILURE
and exit nonzero on failure.

### Example 1 — `01_head_on_exchange.posim`: the canonical action–reaction pair

Two identical spheres approach at ±1. Gap 3, closing speed 2 → impact at
t = 1.5 exactly. Velocities exchange; momentum stays 0; energy is conserved.

```
In[7]:= step 3
Out[7]= t = 3 (advanced by 3, 26 solver steps, 1 collision(s) — CONTACTS lists them)
In[8]:= contacts
Out[8]= contact0: obj0 <-> obj1 at t = 1.5
  point  = [0, 0, 0]
  normal = [1, 0, 0]  (from obj0 toward obj1)
  depth = 0, approach speed = 2, impulse = 2
In[12]:= get obj0.velocity      → [-1, 0, 0]
In[13]:= get obj1.velocity      → [1, 0, 0]
In[15]:= energy                 → 1        (unchanged)
```

The impact landed at `t = 1.5` **exactly**; the normal `[1,0,0]` is the
action–reaction line; the impulse `J = 2` matches `−(1+1)(−2)/(1+1)`.

### Example 2 — `02_unequal_masses.posim`: the light ball bounces back

m₁ = 1 at v = 1 strikes m₂ = 3 at rest. Textbook 1-D elastic formulas:
v₁′ = (m₁−m₂)/(m₁+m₂) = −0.5, v₂′ = 2m₁/(m₁+m₂) = +0.5.

```
In[5]:= get obj0.velocity.x     → -0.5
In[6]:= get obj1.velocity.x     → 0.5
In[7]:= momentum                → [1, 0, 0]   (conserved)
```

### Example 3 — `03_restitution_ladder.posim`: separation speed = e × approach speed

The same collision four times with e ∈ {1.0, 0.8, 0.5, 0.2}. Approach speed is
always 2; kinetic energy after equals e² of the initial 1.

```
e = 1.0:  separation speed 2      energy 1
e = 0.8:  separation speed 1.6    energy 0.64
e = 0.5:  separation speed 1      energy 0.25
e = 0.2:  separation speed 0.4    energy 0.04
```

Exactly `e·2` and `e²` in every rung — the restitution model is exact.

### Example 4 — `04_newtons_cradle.posim`: the impulse walks the chain

Four touching spheres, one incomer at v = 1. The solver's Gauss–Seidel passes
propagate the impulse through the simultaneous contacts:

```
Out[7]= t = 4 (…, 4 collision(s) — CONTACTS lists them)
v0..v3 → 0, 0, 0, 0        v4 → 1        momentum → [1, 0, 0]
```

Only the far ball exits, at the incomer's speed — the desk-toy result.

### Example 5 — `05_billiard_break.posim`: every outgoing direction is a normal

A cue ball at v = 2 breaks a three-ball triangle at e = 0.95. The recorded
contacts show each normal along the line of centers of its pair:

```
contact0: normal [1, 0, 0]                       (cue → apex, head on)
contact1: normal [0.835…, 0.55, 0]               (apex → upper ball)
contact2: normal [0.835…, −0.55, 0]              (apex → lower ball)
contact3: a second cue–apex touch as the cascade settles
momentum before → [2, 0, 0]     momentum after → [2, 0, 0]   (exact)
```

### Example 6 — `06_spin_up.posim`: linear momentum becomes spin

A sphere strikes a free cuboid **above its center**: contact at
`point = [−0.5, 0.5, 0]` with `J = 4.444…` along +x. The angular impulse is
`L = r × J n̂`:

```
get obj1.angular_momentum   → [0, 0, -2.2222222222222223]
```

r × Jn̂ = (−0.5, 0.5, 0) × (4.444, 0, 0) = (0, 0, −0.5·4.444) = (0, 0, −2.222) ✓
Total angular momentum about the origin: −2 before, −2 after (the incoming
sphere's orbital L) — conserved through the transfer.

### Example 7 — `07_thin_wall_toi.posim`: the tunneling killer

A 1 cm bullet at speed **100** meets a **5 mm** plate inside a single 0.05 s
output step. Analytic TOI = (2 − 0.005 − 0.01)/100 = 0.01985.

```
get contact0.t          → 0.019850000000000846     (error ≈ 8e-16!)
get obj1.velocity.x     → -100                     (reflected exactly)
get obj1.position.x     → -3.03                    (back on its own side)
```

A fixed-step engine samples positions every Δt; at 100 m/s the bullet crosses
the whole plate between two samples and passes through. The integrator's root
finder cannot miss it — this is the technique none of the seven surveyed
simulators has.

### Example 8 — `08_colliding_binary.posim`: gravity and impacts cooperate

Two spheres on a bound elliptical orbit (G = 1) whose pericenter 0.5 is smaller
than the touching distance 0.6, with e = 0.6. Gravity curves the paths; the
event fires at the exact touch on the curved trajectory:

```
contact0 at t = 2.884…   normal [-0.556, 0.831, 0]   (the line of centers at TOI)
energy  −0.4  → −0.4996  (the inelastic bounce robs orbital energy: orbit shrinks)
momentum      → [2.8e-16, −2.2e-16, 0]               (zero to rounding)
```

### Example 9 — `09_spinning_target.posim`: the surface itself is moving

The cuboid tumbles at ω_z = 2, so the struck face carries velocity ω×r. The
K-matrix's angular terms price this in:

```
contact0: normal [-0.986, 0.167, 0]     (the face normal of the ROTATED box)
angmom  before → [0, 0, 0.16]   after → [0, 0, 0.15999999999999956]
energy  before → 2.96           after → 2.9599999999999995
```

Total angular momentum and energy (elastic) conserved through a
rotating-surface impact — the full rigid-body impulse, not a particle hack.

### Example 10 — `10_billiard_box.posim`: seven bounces, zero drift

An elastic ball ping-pongs between two static walls for 20 time units:

```
Out[10]= t = 20 (356 solver steps, …, |dE/E| = 0.000e0, 7 collision(s) …)
get system.collisions → 7        energy → 1.4449999999999998 (unchanged)
get obj2.velocity.x   → -1.7     (odd number of bounces)
```

Seven wall impulses and the energy drift is **identically zero**. This script
is also the scene demo: run it interactively and `SCENE CREATE` + `SCENE START`
shows a golden normal arrow flash at every wall strike.

### Example 11 — `11_box_of_shapes.posim`: every shape in the infinitely massive box

R = 1, M = 1. `BOX 4` builds the rigid box; six bodies fly inside it: a
**torus** M (inner radius 1, outer radius 2), a **point** M at
v = (100, 200, 100) — the only mover, so E₀ = ½·1·60000 = **30000 exactly** —
a **sphere** 2M of r = 1/2 (the spec leaves the radius free; R/2 is
documented in the script), a tilted **disk** 2M/3 of r = 1, a **cube** 5M/3
of side 1, and a tilted **cylinder** 2M of r = 1/2, height 3/2.

The geometry is chosen to be sharp: an axis-aligned torus of outer radius 2
**exactly inscribes** the 4-box (its outer equator touches four walls), so
"inside and not touching" forces a tilt — with the axis along the body
diagonal (1,1,1)/√3 the extent per axis is 1.5√(2/3) + 0.5 ≈ 1.7247 < 2,
clearance 0.2753, and the torus sits at the center. The other five positions
are drawn by a **documented LCG** (x ← 1664525·x + 1013904223 mod 2³²,
seed 20260724, sequential rejection at ≥ 0.05 separation), so the script is
reproducible, not hand-tuned.

```
In[2]:= box 4
Out[2]= box: inner size 4 x 4 x 4 — six static walls obj0, obj1, obj2, obj3, obj4, obj5 with inverse_mass = 0 (infinitely massive); objects collide elastically off the inside faces
In[9]:= list
Out[9]= obj0: cuboid he=[1, 4, 4], mass=0, charge=0, pos=[3, 0, 0] [wall: static, inverse_mass=0]
  ⋮ (five more walls)
obj6: torus ring=1.5 tube=0.5, mass=1, charge=0, pos=[0, 0, 0]
obj7: point, mass=1, charge=0, pos=[1.40659, -0.995859, 0.569601]
obj8: sphere r=0.5, mass=2, charge=0, pos=[1.424704, 1.367496, -0.493612]
obj9: disk r=1, mass=0.6666666666666666, charge=0, pos=[-0.677386, -1.041493, -1.091679]
obj10: cuboid he=[0.5, 0.5, 0.5], mass=1.6666666666666667, charge=0, pos=[1.074397, 0.816223, 1.102099]
obj11: cylinder r=0.5 h=1.5, mass=2, charge=0, pos=[-1.027024, -1.40389, -0.485619]
In[10]:= collide
Out[10]= collisions ON (51 collidable pair(s); 0 impulse(s) so far)
In[11]:= get obj0.inverse_mass    → 0
In[12]:= get obj6.inertia_tensor  → [[1.28125, 0, 0], [0, 1.28125, 0], [0, 0, 2.4375]]
In[13]:= energy                   → 30000
In[14]:= momentum                 → [100, 200, 100]
In[15]:= run 0.1 steps 100
Out[15]= t = 0.1 (2121 solver steps, 100 snapshots, |dE/E| = 2.040e-10, 119 collision(s) — CONTACTS lists them)
In[16]:= energy                   → 29999.999993880127
In[17]:= momentum                 → [146.48911126803657, 102.1478121131382, 46.01601291636828]
```

**119 impulses across all three dispatch tiers in 0.1 time units, and the
energy drift is |dE/E| = 2.040e-10.** The 51 collidable pairs are the 66
pairs of 12 bodies minus the 15 static wall-wall pairs; the torus inertia
readback matches the analytic tensor exactly (m(½c² + ⅝a²) = 1.28125,
m(c² + ¾a²) = 2.4375 for c = 1.5, a = 0.5). **Momentum is NOT conserved —
(100, 200, 100) → (146.49, 102.15, 46.02) — and is not supposed to be**:
every wall impulse transfers ±Jn̂ into a body with `inverse_mass = 0`. The
infinitely massive walls absorb momentum at zero velocity, so they absorb no
energy (p²/2m → 0 as m → ∞, taken *exactly* by the inverse-mass
formulation), and after 119 collisions the six walls are **bit-identically
at rest**. Momentum non-conservation with exact energy conservation is the
physical signature of a rigid container — not a bug.

### Example 12 — `12_two_dumbbells.posim`: two dumbbells from a user-defined function — E, P **and** L conserved

`create_dumbell` is defined **in the notebook language** (`DEF`): it takes a
name plus every entity value — all with defaults — and builds a named rigid
dumbbell (two solid spheres joined by a solid rod, one rigid body whose
local origin is its center of mass). Called twice, it makes `dumbell0`
(sphere masses 1 + 2, rod 0.5) and `dumbell1` (2 + 1 + 0.4, different radii
and length); they approach off-center with spin and collide **twice**
(G = 0, restitution 1, no walls). The full captured run:

```
In[1]:= set system.g_constant = 0
In[2]:= def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) {
  new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, rod_radius = rod_radius, length = length, position = position, velocity = velocity, angular_velocity = angular_velocity }
}
Out[2]= function create_dumbell(11 parameter(s)) defined — 1 body line(s)
In[3]:= create_dumbell("dumbell0", 1, 2, 0.5, 0.25, 0.25, 0.1, 1, [-2, 0.15, 0], [1.5, 0, 0], [0, 0, 0.6])
Out[3]= obj0 as dumbell0
In[4]:= create_dumbell("dumbell1", 2, 1, 0.4, 0.3, 0.2, 0.08, 1.2, [2, -0.15, 0], [-1.5, 0, 0], [0.4, 0, 0])
Out[4]= obj1 as dumbell1
In[5]:= list
Out[5]= obj0: dumbbell r1=0.25 r2=0.25 rod_r=0.1 len=1, mass=3.5, charge=0, pos=[-2, 0.15, 0]
obj1: dumbbell r1=0.3 r2=0.2 rod_r=0.08 len=1.2, mass=3.4, charge=0, pos=[2, -0.15, 0]
In[6]:= get dumbell0.m1
Out[6]= 1
In[7]:= get dumbell1.m_rod
Out[7]= 0.3999999999999999
In[8]:= get dumbell0.vx
Out[8]= 1.5
In[9]:= energy
Out[9]= 7.865310611764706
In[10]:= momentum
Out[10]= [0.15000000000000036, 0, 0]
In[11]:= angmom
Out[11]= [0.4443030588235295, 0, -1.5059999999999998]
In[12]:= run 3 steps 60
Out[12]= t = 3 (3861 solver steps, 60 snapshots, |dE/E| = 9.463e-11, 2 collision(s) — CONTACTS lists them)
In[13]:= energy
Out[13]= 7.865310611020375
In[14]:= momentum
Out[14]= [0.15000000000000036, 0, 0]
In[15]:= angmom
Out[15]= [0.44430305882373156, -0.000000000000009547918011776346, -1.505999999999855]
In[16]:= get system.collisions
Out[16]= 2
```

The named member paths are at work in `In[6..8]` — `dumbell1.m_rod` reads
back `0.3999999999999999` because the part masses are *recovered* from the
stored mass fractions (`m_rod = (1 − f1 − f2)·M`), honest floating point.
Through the two collisions:

- **Energy**: `7.865310611764706 → 7.865310611020375` — the run line
  itself prints `|dE/E| = 9.463e-11`.
- **Momentum**: `[0.15000000000000036, 0, 0]` — **bit-identical**.
- **Angular momentum about the origin**:
  `[0.4443030588235295, 0, -1.5059999999999998]` →
  `[0.44430305882373156, -9.5e-15, -1.505999999999855]` — drift at the
  1e-13 level.

**Why is L conserved here when Example 11 lost momentum?** Nothing in this
scene is static, and each impulse pair `±J n̂` acts at **one shared contact
point**: the two angular impulses about any fixed origin are
`x_c × (+J n̂)` and `x_c × (−J n̂)` — they cancel exactly. Action–reaction
at a single point produces **zero net torque**, and with G = 0 there is no
other torque source, so total L must survive — and it does, through real
CVODE events, entirely from the part-decomposed contact geometry of §6
(this is the anchor test
`colliding_dumbbells_conserve_energy_momentum_and_angular_momentum`,
which pins E, P and L to 1e-8). Verified live in a real browser as well:
the entity labels show the user names `dumbell0`/`dumbell1`, and the
window's labeled conserved-quantities readout displayed
`E = 7.86531061, P = [0.15000, 0.00000, 0.00000] |.| = 0.15000, L =
[0.44430, 0.00000, -1.50600] |.| = 1.57017` identically before and after
the impact (after it, L's y component read `-1.31228e-13`).

---

## 10. FAQ

**Do I have to turn anything on?** No — collisions are detected in every scene
by default. `COLLIDE OFF` turns them off; `COLLIDE` reports status.

**How precise is the impact time?** The integrator interpolates onto the root:
measured errors are at the 1e-15 relative level (Examples 1 and 7).

**What happens with three bodies touching at once?** The solver sweeps all
pairs repeatedly until nothing is still approaching (Example 4).

**Can objects still tunnel?** Not on the CVODE paths (the step size is capped
against the smallest feature while armed, and the cap counts speed the
accelerations can build before the next refresh — even a from-rest drop is
caught). On SPRK the sampling bound is the
user's fixed dt — choose it smaller than (thinnest body)/(fastest approach).

**Why is momentum not conserved in the box?** Because the walls are
infinitely massive, and infinite mass absorbs momentum without absorbing
energy: a wall receiving impulse J changes its momentum by J but its kinetic
energy by **p²/2m → 0 as m → ∞**. Ordinarily that is an approximation; here
the limit is *exact*, because the equations of motion only ever use the
**inverse** mass — `v = p·m⁻¹`, impulse denominator
`n·Kn = m_i⁻¹ + m_j⁻¹ + angular terms` — and a wall with `inverse_mass = 0`
contributes zero to every denominator and receives no state writes at all.
Newton's third law still acts at every contact; the "missing" momentum went
into the box (Example 11: (100, 200, 100) → (146.49, 102.15, 46.02) while
energy is conserved to 2.040e-10 and the walls stay bit-identically at rest).

**Can anything pass through the torus hole?** Balls can. Ball-vs-anything
uses the exact SDF, and on a line through the hole a small ball's separation
dips toward (inner radius − ball radius) without ever reaching zero — no
event fires. Verified by test: a point particle threads the hole with zero
collisions while a fat ball on a nearby line hits the tube at the analytic
TOI = (3 − √1.45)/4. Extended shapes (cuboid, disk, cylinder, another torus)
are served by the support-axis tier, which sees the torus's **convex hull** —
a support function cannot see a hole — so only balls can thread it.

**Why does the point particle pass through the disk?** The disk is *ideal*:
a zero-thickness solid disk whose "SDF" is the unsigned distance to it —
continuous, and zero exactly on the disk. Against a zero-radius point the
separation touches zero for a single instant and rises again without ever
becoming negative: there is no sign change for the root finder and no
overlap to resolve — a measure-zero contact carrying no impulse, which is
the exact answer for these two idealized bodies. Give either party finite
extent **along the contact normal** — any sphere radius, a cuboid, a
cylinder — and the separation genuinely crosses zero and the contact fires
(every finite-size body in Example 11 bounces off the disk). The one
pairing with zero extent on *both* sides is **two disks with parallel
planes**: their separation is `|dz|`, which touches zero at plane
coincidence without a sign change, so a face-on disk-disk crossing is
exactly as invisible as the point-through-disk case (pinned by test
`parallel_disk_disk_separation_is_the_documented_limitation`). Workaround:
tilt one disk (a tilted disk has finite extent along the other's normal,
so the separation genuinely crosses zero) or model a thin cylinder.

**How do compound bodies collide?** The dumbbell — the one compound shape —
is **decomposed over its parts** by the narrow phase: each sphere part is a
ball tested through the other shape's exact SDF (exact against everything,
*including another dumbbell*, whose union SDF is exact), the rod recurses
as a free-standing cylinder, and the deepest part supplies the contact.
Only rod-vs-rod ever reaches the approximate support-axis tier. The parts
are geometry only — dynamically the dumbbell stays **one** rigid body (one
momentum, one angular momentum, one composite inertia tensor about its
COM-origin), which is why two tumbling dumbbells conserve E, P and L
through real events (Example 12).

**Where does friction stand?** Designed (per-object μ, tangential impulse
clamped to the Coulomb cone, two-direction upgrade documented) but deliberately
not shipped in this delivery — the manager's requirement centers on the exact
normal and the action–reaction pair, which are fully delivered; adding
tangential impulses is a bounded follow-up that changes no data structures.

**Is any of this hand-rolled integration?** No. Free flight is integrated by
SUNDIALS exclusively; an impulse is an instantaneous momentum update at an
event boundary — the one thing an ODE integrator cannot represent and the
standard treatment in every engine surveyed.
