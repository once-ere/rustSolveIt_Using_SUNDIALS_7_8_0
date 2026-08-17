//! Rigid-body collision detection and impulse response.
//!
//! Modeled on the surveyed engines (Bullet, Rapier/parry, Chrono, SOFA,
//! ncollide — see `collision_detection.md`) but specialized to this
//! workspace's closed shape set {`Point`, `Sphere`, `Cuboid`, `Torus`,
//! `Disk`, `Cylinder`, `Dumbbell`}, in three exactness tiers (no
//! GJK/EPA/MPR machinery); a dumbbell-vs-anything pair decomposes over
//! its two spheres and its rod, so only rod-vs-rod reaches the
//! approximate tier:
//!
//! 1. **exact ball-vs-anything** — ball(= sphere/point) vs ball by
//!    center distance, ball vs cuboid via the box SDF, and ball vs
//!    torus/disk/cylinder via the shapes' exact SDF closest point (a
//!    small ball can genuinely pass through a torus hole);
//! 2. **exact cuboid-cuboid** — the 15-axis SAT;
//! 3. **support-axis tests** for every remaining extended-vs-extended
//!    pair: the candidate axes are each cuboid's three face axes, each
//!    round shape's symmetry axis, their pairwise cross products, the
//!    radial rejection axes, and the center line. Exact for face-on
//!    contacts (in particular every wall-slab contact) and for
//!    side-on round-shape contacts (via the radial rejection axes);
//!    conservative (may report contact slightly early, along a
//!    candidate axis) for corner-on configurations. For the torus this
//!    tier sees the convex hull — only balls can thread the hole.
//!
//! Pinned conventions (ARCHITECTURE.md §3.8):
//! - a [`Contact`] between objects `i < j` carries a **unit normal that
//!   points from body `i` toward body `j`** — the line of the
//!   action–reaction impulse pair (`+j_n·n` on `j`, `−j_n·n` on `i`);
//! - [`pair_separation`] is **positive when separated**, negative when
//!   penetrating, and is exactly the CVODE/ARKODE root-function value;
//! - `Contact::depth = max(0, −separation)` is the penetration depth.
//!
//! Detection during integration is event-driven: `integrate::run` arms
//! sundials rootfinding on the pairwise separations, so the integrator
//! itself lands on the time of impact (see `integrate.rs`). The
//! functions here are pure geometry plus the impulse update; **no time
//! stepping happens in this module** (hard rule 1).

use crate::boundary::{self, support_extent, support_point, Boundary, Sdf};
use crate::linalg::{Mat3, Quat, Vec3};
use crate::physical_object::physical_object;
use crate::PhysicalObjectSystem;

/// Impulse events in one **burst** before the Zeno guard escalates:
/// beyond this count restitution is forced to 0 (plastic); beyond twice
/// it rootfinding is disarmed for the rest of the output interval and
/// penetrations are projected out instead.
///
/// A *burst* is a run of events with essentially no time between them —
/// see [`ZENO_GAP_RELATIVE`]. It is **not** a count per output interval,
/// which is what this used to be and which made the physics depend on
/// how often the caller asked for output: 65 ordinary elastic
/// collisions in one coarse interval tripped a chattering guard and
/// took the energy from 80000 to **0**. `PROJECT_STATUS.md` carried
/// that as a known unrepaired defect and
/// `energy_does_not_depend_on_how_often_output_is_requested` now pins
/// it.
pub const MAX_EVENTS_IN_BURST: usize = 64;

/// How close together events must be, relative to the current time, to
/// count as the same burst.
///
/// Genuine Zeno behaviour — a ball settling, a chattering contact — has
/// the interval between events collapsing towards zero, so any
/// threshold this far below the dynamics catches it. Ordinary
/// collisions are separated by a free flight, which is enormous by
/// comparison, and each one resets the burst.
///
/// Scaling by `max(|t|, 1)` keeps the rule free of the two quantities
/// it must not depend on: the output interval and the run length.
pub const ZENO_GAP_RELATIVE: f64 = 1e-9;

/// Whether `t` continues the burst that last fired at `previous`.
pub fn same_burst(t: f64, previous: f64) -> bool {
    (t - previous).abs() <= ZENO_GAP_RELATIVE * t.abs().max(1.0)
}

/// Contacts kept on `PhysicalObjectSystem::contacts` (oldest dropped).
pub const CONTACTS_CAP: usize = 1024;

/// One resolved (or detected) contact between objects `i < j`.
#[derive(Clone, Debug, PartialEq)]
pub struct Contact {
    /// Index of the first body (lower index).
    pub i: usize,
    /// Index of the second body.
    pub j: usize,
    /// Simulation time of the event.
    pub t: f64,
    /// World-space contact point.
    pub point: Vec3,
    /// Unit normal, pointing from body `i` toward body `j`.
    pub normal: Vec3,
    /// Penetration depth, `>= 0` (≈ 0 at a root-detected touch).
    pub depth: f64,
    /// Pre-impulse normal relative velocity `(v_j − v_i)·n` at the
    /// contact point (negative = approaching).
    pub rel_vel_n: f64,
    /// Scalar normal impulse magnitude that was applied.
    pub impulse_n: f64,
}

/// Pure contact geometry (no dynamics).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactGeometry {
    /// World-space contact point.
    pub point: Vec3,
    /// Unit normal from the first argument's body toward the second's.
    pub normal: Vec3,
    /// Penetration depth (`max(0, −separation)`).
    pub depth: f64,
}

/// Whether a boundary has collision extent of its own. Only `Point`
/// fails: it is a zero-radius sphere, so [`collidable_pairs`] excludes
/// a pair only when **both** sides fail (two zero-radius spheres cannot
/// overlap transversally) — a `Point` still collides with every
/// extended shape.
fn is_collidable(b: &Boundary) -> bool {
    !matches!(b, Boundary::Point)
}

/// "Ball" = a shape whose collision geometry is a center plus a radius
/// (`Sphere`, or `Point` as the zero-radius case) — the shapes that get
/// the exact SDF-based tests against every other shape.
fn is_ball(b: &Boundary) -> bool {
    matches!(b, Boundary::Point | Boundary::Sphere { .. })
}

/// Collidable pairs `(i, j)` with `i < j`, in lexicographic order (this
/// order is also the root-function component order). Excluded: pairs
/// where both bodies are `Point` (two zero-radius spheres cannot
/// overlap transversally) and pairs where both bodies are static
/// (infinite mass on both sides — no impulse is definable).
pub fn collidable_pairs(system: &PhysicalObjectSystem) -> Vec<(usize, usize)> {
    let n = system.objects.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &system.objects[i];
            let b = &system.objects[j];
            if !is_collidable(&a.get_boundary()) && !is_collidable(&b.get_boundary()) {
                continue;
            }
            if a.get_inverse_mass() == 0.0 && b.get_inverse_mass() == 0.0 {
                continue;
            }
            pairs.push((i, j));
        }
    }
    pairs
}

/// Signed separation of two boundaries at the given poses: positive
/// when separated, negative when penetrating. Continuous through the
/// touch point — this exact function (via [`pair_separation`]) is the
/// sundials root function `g` for the pair.
pub fn separation_at(
    ba: &Boundary,
    pa: Vec3,
    qa: Quat,
    bb: &Boundary,
    pb: Vec3,
    qb: Quat,
) -> f64 {
    match (ba, bb) {
        (Boundary::Cuboid { .. }, Boundary::Cuboid { .. }) => {
            sat_separation(ba, pa, qa, bb, pb, qb).0
        }
        (Boundary::Cuboid { .. }, b) if is_ball(b) => {
            // sphere/point (radius r) vs cuboid: box SDF of the center
            // in the box frame, minus r. Continuous inside and out.
            let r = radius_of(bb);
            let local = qa.inverse().rotate(pb - pa);
            ba.signed_distance(&local) - r
        }
        (a, Boundary::Cuboid { .. }) if is_ball(a) => {
            let r = radius_of(ba);
            let local = qb.inverse().rotate(pa - pb);
            bb.signed_distance(&local) - r
        }
        (a, b) if is_ball(a) && is_ball(b) => {
            // sphere/point vs sphere/point: center distance minus radii.
            (pb - pa).norm() - (radius_of(ba) + radius_of(bb))
        }
        (Boundary::Dumbbell { .. }, b) if !is_ball(b) => {
            dumbbell_vs_any_separation(ba, pa, qa, bb, pb, qb)
        }
        (a, Boundary::Dumbbell { .. }) if !is_ball(a) => {
            dumbbell_vs_any_separation(bb, pb, qb, ba, pa, qa)
        }
        (_, b) if is_ball(b) => {
            // torus/disk/cylinder vs ball: the shape's exact SDF at the
            // ball center (in the shape frame), minus the ball radius.
            // Continuous everywhere; a small ball can pass through a
            // torus hole without this ever reaching zero.
            let r = radius_of(bb);
            let local = qa.inverse().rotate(pb - pa);
            ba.signed_distance(&local) - r
        }
        (a, _) if is_ball(a) => {
            let r = radius_of(ba);
            let local = qb.inverse().rotate(pa - pb);
            bb.signed_distance(&local) - r
        }
        _ => {
            // Every remaining extended-vs-extended pair (any pair
            // involving a torus/disk/cylinder that is not covered
            // above): support-axis test — exact for face-on contacts,
            // conservative for corner-on ones (see module doc).
            support_axis_separation(ba, pa, qa, bb, pb, qb).0
        }
    }
}

/// The three parts of a dumbbell, at a given world pose: the two
/// spheres as `(world center, radius)` and the rod as a free-standing
/// cylinder `(boundary, world center)` sharing the dumbbell's
/// orientation.
fn dumbbell_parts(
    bd: &Boundary,
    pd: Vec3,
    qd: Quat,
) -> ((Vec3, f64), (Vec3, f64), (Boundary, Vec3)) {
    let Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, .. } = bd else {
        unreachable!("dumbbell_parts needs a dumbbell")
    };
    let c1 = qd.rotate(Vec3::new(0.0, 0.0, *z1)) + pd;
    let c2 = qd.rotate(Vec3::new(0.0, 0.0, *z2)) + pd;
    let zc = 0.5 * (z1 + z2);
    let rod = Boundary::Cylinder { radius: *rod_radius, half_height: 0.5 * (z2 - z1) };
    let rod_p = qd.rotate(Vec3::new(0.0, 0.0, zc)) + pd;
    ((c1, *r1), (c2, *r2), (rod, rod_p))
}

/// Dumbbell vs any extended shape, decomposed over the dumbbell's
/// parts: the two sphere parts use the OTHER shape's exact SDF at
/// their centers (exact for every shape, including another dumbbell's
/// union SDF), the rod recurses as a free-standing cylinder — so only
/// rod-vs-rod ever reaches the approximate support-axis tier.
fn dumbbell_vs_any_separation(
    bd: &Boundary,
    pd: Vec3,
    qd: Quat,
    bx: &Boundary,
    px: Vec3,
    qx: Quat,
) -> f64 {
    let ((c1, r1), (c2, r2), (rod, rod_p)) = dumbbell_parts(bd, pd, qd);
    let s1 = bx.signed_distance(&qx.inverse().rotate(c1 - px)) - r1;
    let s2 = bx.signed_distance(&qx.inverse().rotate(c2 - px)) - r2;
    let s3 = separation_at(&rod, rod_p, qd, bx, px, qx);
    s1.min(s2).min(s3)
}

/// Support-axis separation for extended-vs-extended pairs: evaluates
/// the directed SAT gap `g(±l) = ±d·l − h_a(±l) − h_b(∓l)` over the
/// candidate axes (each cuboid's face axes, each round shape's symmetry
/// axis, their pairwise cross products, the radial rejection axes, and
/// the center line) and returns the maximum gap with the world axis
/// achieving it, oriented from `a` toward `b`. The directed form
/// reduces to `|d·l| − (h_a + h_b)` for centrally symmetric shapes; the
/// dumbbell's off-center spheres break `h(−u) = h(u)`, so both signs
/// are evaluated. All candidate axes are unit vectors, so no
/// renormalization is needed.
fn support_axis_separation(
    ba: &Boundary,
    pa: Vec3,
    qa: Quat,
    bb: &Boundary,
    pb: Vec3,
    qb: Quat,
) -> (f64, Vec3) {
    let d = pb - pa;
    // Primary axes per body: a cuboid's three face axes, a round
    // shape's symmetry axis.
    let mut prim_a: Vec<Vec3> = Vec::new();
    let mut prim_b: Vec<Vec3> = Vec::new();
    let push_shape_axes = |b: &Boundary, q: Quat, out: &mut Vec<Vec3>| match b {
        Boundary::Cuboid { .. } => {
            let cols = mat_columns(&q.to_rotation_matrix());
            out.extend_from_slice(&cols);
        }
        Boundary::Torus { .. }
        | Boundary::Disk { .. }
        | Boundary::Cylinder { .. }
        | Boundary::Dumbbell { .. } => {
            out.push(q.rotate(Vec3::new(0.0, 0.0, 1.0)));
        }
        _ => {}
    };
    push_shape_axes(ba, qa, &mut prim_a);
    push_shape_axes(bb, qb, &mut prim_b);
    let mut axes: Vec<Vec3> = Vec::new();
    axes.extend_from_slice(&prim_a);
    axes.extend_from_slice(&prim_b);
    // Cross-product axes (edge-vs-edge / edge-vs-rim separations),
    // normalized; near-parallel pairs are skipped like in the SAT.
    for xa in &prim_a {
        for xb in &prim_b {
            let c = xa.cross(*xb);
            let n = c.norm();
            if n > 1e-12 {
                axes.push(c / n);
            }
        }
    }
    // Radial rejection axes: the component of the center offset
    // perpendicular to each symmetry/face axis. For two round shapes
    // with (near-)parallel axes and an axial offset this is the TRUE
    // lateral separating direction — the parallel-axis cross product
    // above vanishes and the raw center line is tilted, so without
    // these a side-side cylinder/torus contact would fire while the
    // bodies are still far apart, with a wrongly tilted normal.
    let n_prim = axes.len();
    for k in 0..n_prim {
        let u = axes[k];
        let r = d - u * d.dot(u);
        let rn = r.norm();
        if rn > 1e-12 {
            axes.push(r / rn);
        }
    }
    let dist = d.norm();
    if dist > 1e-12 {
        axes.push(d / dist);
    }

    let mut best_sep = f64::NEG_INFINITY;
    let mut best_axis = Vec3::new(1.0, 0.0, 0.0);
    for l in axes {
        // Directed SAT gap, evaluated for both orientations of the
        // axis: gap(+l) = d·l − h_a(l) − h_b(−l). For centrally
        // symmetric shapes h(−u) = h(u) and this reduces exactly to
        // the classic |d·l| − (e_a + e_b); the dumbbell's off-center
        // spheres need the general form.
        let ua = qa.inverse().rotate(l);
        let ub = qb.inverse().rotate(l);
        let g_pos = d.dot(l) - support_extent(ba, ua) - support_extent(bb, -ub);
        let g_neg = -d.dot(l) - support_extent(ba, -ua) - support_extent(bb, ub);
        let (sep, oriented) = if g_pos >= g_neg { (g_pos, l) } else { (g_neg, -l) };
        if sep > best_sep {
            best_sep = sep;
            best_axis = oriented; // points from a toward b
        }
    }
    (best_sep, best_axis)
}

/// Signed separation of a pair of objects (see [`separation_at`]).
pub fn pair_separation(a: &physical_object, b: &physical_object) -> f64 {
    separation_at(
        &a.get_boundary(),
        a.get_position(),
        a.get_orientation(),
        &b.get_boundary(),
        b.get_position(),
        b.get_orientation(),
    )
}

fn radius_of(b: &Boundary) -> f64 {
    match b {
        Boundary::Sphere { radius } => *radius,
        _ => 0.0,
    }
}

/// World-space axis-aligned bounding box `(min, max)` of an object —
/// the broad-phase primitive (brute-force pair cull; a BVH/sweep-and-
/// prune broad phase is the documented upgrade path).
pub fn world_aabb(obj: &physical_object) -> (Vec3, Vec3) {
    let p = obj.get_position();
    match obj.get_boundary() {
        Boundary::Point => (p, p),
        Boundary::Sphere { radius } => {
            let r = Vec3::new(radius, radius, radius);
            (p - r, p + r)
        }
        Boundary::Cuboid { half_extents } => {
            // Extent along each world axis: Σ_k h_k |R[axis][k]|.
            let rm = obj.get_orientation().to_rotation_matrix().0;
            let ext = |row: usize| {
                half_extents[0] * rm[row][0].abs()
                    + half_extents[1] * rm[row][1].abs()
                    + half_extents[2] * rm[row][2].abs()
            };
            let e = Vec3::new(ext(0), ext(1), ext(2));
            (p - e, p + e)
        }
        b @ (Boundary::Torus { .. }
        | Boundary::Disk { .. }
        | Boundary::Cylinder { .. }
        | Boundary::Dumbbell { .. }) => {
            // Extent along each world axis = the directed support of
            // the shape along ±axis pulled into the body frame (exact;
            // the dumbbell is not centrally symmetric, so + and − can
            // genuinely differ).
            let qi = obj.get_orientation().inverse();
            let ext = |axis: Vec3| support_extent(&b, qi.rotate(axis));
            let hi = Vec3::new(
                ext(Vec3::new(1.0, 0.0, 0.0)),
                ext(Vec3::new(0.0, 1.0, 0.0)),
                ext(Vec3::new(0.0, 0.0, 1.0)),
            );
            let lo = Vec3::new(
                ext(Vec3::new(-1.0, 0.0, 0.0)),
                ext(Vec3::new(0.0, -1.0, 0.0)),
                ext(Vec3::new(0.0, 0.0, -1.0)),
            );
            (p - lo, p + hi)
        }
    }
}

/// AABB overlap test with a symmetric margin (broad-phase cull).
pub fn aabb_overlap(a: &physical_object, b: &physical_object, margin: f64) -> bool {
    let (amin, amax) = world_aabb(a);
    let (bmin, bmax) = world_aabb(b);
    amin.x - margin <= bmax.x
        && bmin.x - margin <= amax.x
        && amin.y - margin <= bmax.y
        && bmin.y - margin <= amax.y
        && amin.z - margin <= bmax.z
        && bmin.z - margin <= amax.z
}

/// The 15 candidate SAT axes for a cuboid-cuboid pair: 3 face axes of
/// A, 3 face axes of B, 9 pairwise edge cross products. Returns the
/// maximum separation over all axes and the world-space axis achieving
/// it, oriented from A toward B.
///
/// `sep < 0` on every axis ⇔ the boxes overlap; the reported axis is
/// then the minimum-penetration (deepest) axis, whose direction is the
/// contact normal.
fn sat_separation(
    ba: &Boundary,
    pa: Vec3,
    qa: Quat,
    bb: &Boundary,
    pb: Vec3,
    qb: Quat,
) -> (f64, Vec3) {
    let (ha, hb) = match (ba, bb) {
        (Boundary::Cuboid { half_extents: a }, Boundary::Cuboid { half_extents: b }) => (*a, *b),
        _ => unreachable!("sat_separation is only called for cuboid pairs"),
    };
    let ra = qa.to_rotation_matrix();
    let rb = qb.to_rotation_matrix();
    let acols = mat_columns(&ra);
    let bcols = mat_columns(&rb);
    let d = pb - pa;

    // Projected extent of a box onto a unit axis.
    let ext = |cols: &[Vec3; 3], h: &[f64; 3], l: Vec3| -> f64 {
        h[0] * cols[0].dot(l).abs() + h[1] * cols[1].dot(l).abs() + h[2] * cols[2].dot(l).abs()
    };

    let mut best_sep = f64::NEG_INFINITY;
    let mut best_axis = Vec3::new(1.0, 0.0, 0.0);
    let mut consider = |axis: Vec3| {
        let n = axis.norm();
        if n < 1e-12 {
            return; // near-parallel edges: degenerate cross product
        }
        let l = axis / n;
        let sep = d.dot(l).abs() - (ext(&acols, &ha, l) + ext(&bcols, &hb, l));
        if sep > best_sep {
            best_sep = sep;
            // Orient the axis from A toward B.
            best_axis = if d.dot(l) >= 0.0 { l } else { -l };
        }
    };

    for c in &acols {
        consider(*c);
    }
    for c in &bcols {
        consider(*c);
    }
    for ca in &acols {
        for cb in &bcols {
            consider(ca.cross(*cb));
        }
    }
    (best_sep, best_axis)
}

fn mat_columns(m: &Mat3) -> [Vec3; 3] {
    let a = m.0;
    [
        Vec3::new(a[0][0], a[1][0], a[2][0]),
        Vec3::new(a[0][1], a[1][1], a[2][1]),
        Vec3::new(a[0][2], a[1][2], a[2][2]),
    ]
}

/// Support vertex of a cuboid in world space: the vertex farthest along
/// the world direction `dir`.
fn support_vertex(h: &[f64; 3], p: Vec3, cols: &[Vec3; 3], dir: Vec3) -> Vec3 {
    let mut v = p;
    for k in 0..3 {
        let s = if cols[k].dot(dir) >= 0.0 { h[k] } else { -h[k] };
        v += cols[k] * s;
    }
    v
}

/// Full narrow-phase contact geometry for a pair of objects; `None`
/// when the pair is separated by more than `tol` or the configuration
/// is degenerate (e.g. concentric spheres — no definable normal).
///
/// The returned normal points from `a` toward `b`.
pub fn contact_geometry(
    a: &physical_object,
    b: &physical_object,
    tol: f64,
) -> Option<ContactGeometry> {
    contact_geometry_at(
        &a.get_boundary(),
        a.get_position(),
        a.get_orientation(),
        &b.get_boundary(),
        b.get_position(),
        b.get_orientation(),
        tol,
    )
}

/// Pose-level narrow phase (the dumbbell decomposition recurses into
/// this with synthesized part boundaries). The returned normal points
/// from the first shape toward the second.
fn contact_geometry_at(
    ba: &Boundary,
    pa: Vec3,
    qa: Quat,
    bb: &Boundary,
    pb: Vec3,
    qb: Quat,
    tol: f64,
) -> Option<ContactGeometry> {
    let (ba, bb) = (*ba, *bb);
    match (&ba, &bb) {
        (Boundary::Cuboid { .. }, Boundary::Cuboid { .. }) => {
            cuboid_cuboid_contact(&ba, pa, qa, &bb, pb, qb, tol)
        }
        (Boundary::Cuboid { .. }, b) if is_ball(b) => {
            // a = box, b = sphere/point: compute with the sphere first,
            // then flip the normal (helper returns sphere→box).
            let g = sphere_cuboid_contact(radius_of(&bb), pb, &ba, pa, qa, tol)?;
            Some(ContactGeometry { point: g.point, normal: -g.normal, depth: g.depth })
        }
        (a, Boundary::Cuboid { .. }) if is_ball(a) => {
            sphere_cuboid_contact(radius_of(&ba), pa, &bb, pb, qb, tol)
        }
        (a, b) if is_ball(a) && is_ball(b) => {
            sphere_sphere_contact(radius_of(&ba), pa, radius_of(&bb), pb, tol)
        }
        (Boundary::Dumbbell { .. }, b) if !is_ball(b) => {
            dumbbell_vs_any_contact(&ba, pa, qa, &bb, pb, qb, tol)
        }
        (a, Boundary::Dumbbell { .. }) if !is_ball(a) => {
            let g = dumbbell_vs_any_contact(&bb, pb, qb, &ba, pa, qa, tol)?;
            Some(ContactGeometry { point: g.point, normal: -g.normal, depth: g.depth })
        }
        (_, b) if is_ball(b) => {
            // a = torus/disk/cylinder, b = ball: helper returns the
            // normal shape→ball, which is already a→b here.
            round_ball_contact(&ba, pa, qa, radius_of(&bb), pb, tol)
        }
        (a, _) if is_ball(a) => {
            // a = ball, b = torus/disk/cylinder: flip the helper's
            // shape→ball normal to get a→b.
            let g = round_ball_contact(&bb, pb, qb, radius_of(&ba), pa, tol)?;
            Some(ContactGeometry { point: g.point, normal: -g.normal, depth: g.depth })
        }
        _ => support_axis_contact(&ba, pa, qa, &bb, pb, qb, tol),
    }
}

/// Contact of a ball (radius `r`, center `p_ball`) against ANY shape,
/// consolidating the per-shape helpers. The returned normal points
/// from the **ball toward the shape**.
fn ball_vs_shape_contact(
    r: f64,
    p_ball: Vec3,
    bx: &Boundary,
    px: Vec3,
    qx: Quat,
    tol: f64,
) -> Option<ContactGeometry> {
    match bx {
        Boundary::Point | Boundary::Sphere { .. } => {
            sphere_sphere_contact(r, p_ball, radius_of(bx), px, tol)
        }
        Boundary::Cuboid { .. } => sphere_cuboid_contact(r, p_ball, bx, px, qx, tol),
        _ => {
            // torus/disk/cylinder/dumbbell: helper returns shape→ball.
            let g = round_ball_contact(bx, px, qx, r, p_ball, tol)?;
            Some(ContactGeometry { point: g.point, normal: -g.normal, depth: g.depth })
        }
    }
}

/// Contact of a dumbbell against any extended shape, via the deepest
/// of its three parts (the same decomposition as
/// [`dumbbell_vs_any_separation`]). The returned normal points from
/// the **dumbbell toward the other shape**.
fn dumbbell_vs_any_contact(
    bd: &Boundary,
    pd: Vec3,
    qd: Quat,
    bx: &Boundary,
    px: Vec3,
    qx: Quat,
    tol: f64,
) -> Option<ContactGeometry> {
    let ((c1, r1), (c2, r2), (rod, rod_p)) = dumbbell_parts(bd, pd, qd);
    let s1 = bx.signed_distance(&qx.inverse().rotate(c1 - px)) - r1;
    let s2 = bx.signed_distance(&qx.inverse().rotate(c2 - px)) - r2;
    let s3 = separation_at(&rod, rod_p, qd, bx, px, qx);
    // Deepest part first; fall through on degenerate geometry.
    let mut order = [(s1, 0usize), (s2, 1), (s3, 2)];
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, which) in order {
        let g = match which {
            0 => ball_vs_shape_contact(r1, c1, bx, px, qx, tol),
            1 => ball_vs_shape_contact(r2, c2, bx, px, qx, tol),
            _ => contact_geometry_at(&rod, rod_p, qd, bx, px, qx, tol),
        };
        if g.is_some() {
            return g; // ball→shape and rod→shape are both dumbbell→shape
        }
    }
    None
}

/// Closest point of a solid cylinder (radius, half-height, centered at
/// `(0, 0, z_off)` in the local frame) to the point `c`: returns
/// `(surface point, outward unit normal, signed distance)`, or `None`
/// in the direction-free degenerate case.
fn cylinder_closest(c: Vec3, radius: f64, half_height: f64, z_off: f64) -> Option<(Vec3, Vec3, f64)> {
    let xy = Vec3::new(c.x, c.y, 0.0);
    let s_xy = xy.norm();
    let z = c.z - z_off;
    let inside = s_xy <= radius && z.abs() <= half_height;
    if inside {
        // Point inside the solid: nearest feature (side wall or cap),
        // like the sphere-in-cuboid branch.
        let radial = if s_xy > 1e-12 { xy / s_xy } else { Vec3::new(1.0, 0.0, 0.0) };
        let side_gap = radius - s_xy;
        let cap_gap = half_height - z.abs();
        if side_gap <= cap_gap {
            let surface = radial * radius + Vec3::new(0.0, 0.0, c.z);
            Some((surface, radial, -side_gap))
        } else {
            let sign = if z >= 0.0 { 1.0 } else { -1.0 };
            let surface = Vec3::new(c.x, c.y, z_off + sign * half_height);
            Some((surface, Vec3::new(0.0, 0.0, sign), -cap_gap))
        }
    } else {
        // Outside: clamp to the solid cylinder.
        let radial = if s_xy > 1e-12 { xy / s_xy } else { Vec3::zeros() };
        let surface = radial * s_xy.min(radius)
            + Vec3::new(0.0, 0.0, z_off + z.clamp(-half_height, half_height));
        let delta = c - surface;
        let dist = delta.norm();
        if dist < 1e-12 {
            return None; // exactly on the surface with no direction
        }
        Some((surface, delta / dist, dist))
    }
}

/// Contact between a torus/disk/cylinder/dumbbell (`shape` at `ps`,
/// `qs`) and a ball of radius `r` centered at `pball`, via the exact
/// closest point on the shape's surface. The returned normal points
/// from the **shape toward the ball**; `None` when separated by more
/// than `tol` or geometrically degenerate (ball center on the torus
/// centerline, exactly on the disk, on the cylinder axis at side
/// contact, or on a dumbbell sphere center).
fn round_ball_contact(
    shape: &Boundary,
    ps: Vec3,
    qs: Quat,
    r: f64,
    pball: Vec3,
    tol: f64,
) -> Option<ContactGeometry> {
    let c = qs.inverse().rotate(pball - ps); // ball center, shape frame
    let xy = Vec3::new(c.x, c.y, 0.0);
    let s_xy = xy.norm();

    // (closest surface point, outward unit normal, signed distance of
    // the ball center to the shape surface) in the shape frame.
    let (surface, out, sdf) = match shape {
        Boundary::Torus { ring_radius, tube_radius } => {
            let radial = if s_xy > 1e-12 { xy / s_xy } else { Vec3::new(1.0, 0.0, 0.0) };
            let center_pt = radial * *ring_radius; // nearest centerline point
            let delta = c - center_pt;
            let dist = delta.norm();
            if dist < 1e-12 {
                return None; // on the centerline circle: no definable normal
            }
            let out = delta / dist;
            (center_pt + out * *tube_radius, out, dist - tube_radius)
        }
        Boundary::Disk { radius } => {
            let radial = if s_xy > 1e-12 { xy / s_xy } else { Vec3::zeros() };
            let surface = radial * s_xy.min(*radius); // closest point of the disk
            let delta = c - surface;
            let dist = delta.norm();
            if dist < 1e-12 {
                return None; // ball center exactly on the disk: degenerate
            }
            (surface, delta / dist, dist)
        }
        Boundary::Cylinder { radius, half_height } => {
            cylinder_closest(c, *radius, *half_height, 0.0)?
        }
        Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, .. } => {
            // Union of three parts: the closest surface belongs to the
            // part with the smallest signed distance.
            let sphere_closest = |zc: f64, r: f64| -> Option<(Vec3, Vec3, f64)> {
                let delta = c - Vec3::new(0.0, 0.0, zc);
                let dist = delta.norm();
                if dist < 1e-12 {
                    return None; // on a sphere center: degenerate
                }
                let out = delta / dist;
                Some((Vec3::new(0.0, 0.0, zc) + out * r, out, dist - r))
            };
            let zc = 0.5 * (z1 + z2);
            let hh = 0.5 * (z2 - z1);
            let mut best: Option<(Vec3, Vec3, f64)> = None;
            for cand in [
                sphere_closest(*z1, *r1),
                sphere_closest(*z2, *r2),
                cylinder_closest(c, *rod_radius, hh, zc),
            ]
            .into_iter()
            .flatten()
            {
                if best.as_ref().is_none_or(|b| cand.2 < b.2) {
                    best = Some(cand);
                }
            }
            best?
        }
        _ => unreachable!("round_ball_contact needs a torus/disk/cylinder/dumbbell"),
    };

    let sep = sdf - r;
    if sep > tol {
        return None;
    }
    let depth = (-sep).max(0.0);
    let n_world = qs.rotate(out); // shape → ball
    let shape_surface_world = qs.rotate(surface) + ps;
    let ball_surface_world = pball - n_world * r;
    // Midway across the overlap band, as in sphere_sphere_contact.
    let point = (shape_surface_world + ball_surface_world) * 0.5;
    Some(ContactGeometry { point, normal: n_world, depth })
}

/// Contact for extended-vs-extended pairs via the support-axis test:
/// normal = the maximum-gap axis (oriented `a`→`b`). The contact point
/// is the **deepest point of the incident body** — the support point of
/// whichever body's supporting set has lower dimension (a tilted
/// cylinder's rim point against a wall face, not the centroid of the
/// wall). Equal ranks (face-on-face, edge-on-edge) use the midpoint of
/// the two support-set centroids.
fn support_axis_contact(
    ba: &Boundary,
    pa: Vec3,
    qa: Quat,
    bb: &Boundary,
    pb: Vec3,
    qb: Quat,
    tol: f64,
) -> Option<ContactGeometry> {
    let (sep, axis) = support_axis_separation(ba, pa, qa, bb, pb, qb);
    if sep > tol {
        return None;
    }
    let ua = qa.inverse().rotate(axis);
    let ub = qb.inverse().rotate(-axis);
    let sa = qa.rotate(support_point(ba, ua)) + pa;
    let sb = qb.rotate(support_point(bb, ub)) + pb;
    let ra = boundary::support_rank(ba, ua);
    let rb = boundary::support_rank(bb, ub);
    let point = if ra < rb {
        sa
    } else if rb < ra {
        sb
    } else {
        // Equal rank (face-on-face, edge-on-edge): a small face landing
        // on a much larger one contacts around the SMALL face's center,
        // not halfway to the big face's centroid — prefer the clearly
        // smaller footprint, midpoint only when they are comparable.
        let fa = boundary::support_footprint_radius(ba, ua);
        let fb = boundary::support_footprint_radius(bb, ub);
        if fa < 0.5 * fb {
            sa
        } else if fb < 0.5 * fa {
            sb
        } else {
            (sa + sb) * 0.5
        }
    };
    Some(ContactGeometry { point, normal: axis, depth: (-sep).max(0.0) })
}

/// Sphere-sphere (Point = zero radius): normal along the line of
/// centers, from the first sphere toward the second.
fn sphere_sphere_contact(
    ra: f64,
    pa: Vec3,
    rb: f64,
    pb: Vec3,
    tol: f64,
) -> Option<ContactGeometry> {
    let d = pb - pa;
    let dist = d.norm();
    let sep = dist - (ra + rb);
    if sep > tol || dist == 0.0 {
        return None; // separated, or concentric (degenerate normal)
    }
    let normal = d / dist;
    let depth = (-sep).max(0.0);
    // Contact point: midway across the overlap band, on the i→j line.
    let point = pa + normal * (ra - depth * 0.5);
    Some(ContactGeometry { point, normal, depth })
}

/// Sphere (radius `r`, center `ps`) vs cuboid: closest-point query in
/// the box frame; handles the sphere-center-inside-box branch via the
/// nearest face. The returned normal points from the **sphere toward
/// the box**.
fn sphere_cuboid_contact(
    r: f64,
    ps: Vec3,
    bbox: &Boundary,
    pbox: Vec3,
    qbox: Quat,
    tol: f64,
) -> Option<ContactGeometry> {
    let h = match bbox {
        Boundary::Cuboid { half_extents } => *half_extents,
        _ => unreachable!("sphere_cuboid_contact needs a cuboid"),
    };
    let c = qbox.inverse().rotate(ps - pbox); // sphere center, box frame
    let q = Vec3::new(
        c.x.clamp(-h[0], h[0]),
        c.y.clamp(-h[1], h[1]),
        c.z.clamp(-h[2], h[2]),
    );
    let delta = c - q; // from closest box point toward sphere center
    let dist = delta.norm();

    if dist > 0.0 {
        // Center outside (or on) the box surface.
        let sep = dist - r;
        if sep > tol {
            return None;
        }
        let n_local = delta / dist; // box → sphere
        let normal = qbox.rotate(-n_local); // sphere → box
        let point = qbox.rotate(q) + pbox;
        Some(ContactGeometry { point, normal, depth: (-sep).max(0.0) })
    } else {
        // Center strictly inside the box: nearest face by interior SDF.
        let dx = h[0] - c.x.abs();
        let dy = h[1] - c.y.abs();
        let dz = h[2] - c.z.abs();
        let (k, dmin, ck, hk) = if dx <= dy && dx <= dz {
            (0usize, dx, c.x, h[0])
        } else if dy <= dz {
            (1usize, dy, c.y, h[1])
        } else {
            (2usize, dz, c.z, h[2])
        };
        let sign = if ck >= 0.0 { 1.0 } else { -1.0 };
        // Outward face normal (box → sphere side); flip for sphere → box.
        let mut out = Vec3::zeros();
        match k {
            0 => out.x = sign,
            1 => out.y = sign,
            _ => out.z = sign,
        }
        let normal = qbox.rotate(-out);
        let mut face_pt = c;
        match k {
            0 => face_pt.x = sign * hk,
            1 => face_pt.y = sign * hk,
            _ => face_pt.z = sign * hk,
        }
        let point = qbox.rotate(face_pt) + pbox;
        // separation = interior sdf (−dmin) − r  →  depth = dmin + r
        Some(ContactGeometry { point, normal, depth: dmin + r })
    }
}

/// Cuboid-cuboid via SAT: the minimum-penetration axis is the contact
/// normal; the contact point is the deepest support vertex of the
/// incident box (face axes) or the midpoint of the closest points of
/// the two supporting edges (edge-edge axes).
fn cuboid_cuboid_contact(
    ba: &Boundary,
    pa: Vec3,
    qa: Quat,
    bb: &Boundary,
    pb: Vec3,
    qb: Quat,
    tol: f64,
) -> Option<ContactGeometry> {
    let (sep, normal) = sat_separation(ba, pa, qa, bb, pb, qb);
    if sep > tol {
        return None;
    }
    let (ha, hb) = match (ba, bb) {
        (Boundary::Cuboid { half_extents: a }, Boundary::Cuboid { half_extents: b }) => (*a, *b),
        _ => unreachable!(),
    };
    let acols = mat_columns(&qa.to_rotation_matrix());
    let bcols = mat_columns(&qb.to_rotation_matrix());
    let depth = (-sep).max(0.0);

    // Decide whether the best axis is (numerically) a face axis of A or
    // B, or an edge-edge axis: a face axis is parallel to a box column.
    let is_face_of = |cols: &[Vec3; 3]| -> bool {
        cols.iter().any(|c| c.dot(normal).abs() > 1.0 - 1e-9)
    };

    let point = if is_face_of(&acols) {
        // Reference face on A: deepest vertex of B along −normal.
        support_vertex(&hb, pb, &bcols, -normal)
    } else if is_face_of(&bcols) {
        // Reference face on B: deepest vertex of A along +normal.
        support_vertex(&ha, pa, &acols, normal)
    } else {
        // Edge-edge: closest points between the two supporting edges.
        // The supporting edge of a box for a separating axis runs along
        // the box column most orthogonal to the axis, anchored at the
        // support vertex in the axis direction.
        let edge_dir = |cols: &[Vec3; 3]| -> Vec3 {
            let mut best = cols[0];
            let mut best_dot = f64::INFINITY;
            for c in cols {
                let d = c.dot(normal).abs();
                if d < best_dot {
                    best_dot = d;
                    best = *c;
                }
            }
            best
        };
        let da = edge_dir(&acols);
        let db = edge_dir(&bcols);
        let va = support_vertex(&ha, pa, &acols, normal);
        let vb = support_vertex(&hb, pb, &bcols, -normal);
        closest_point_between_lines(va, da, vb, db)
    };
    Some(ContactGeometry { point, normal, depth })
}

// --------------------------------------------------------------------
// Impulse response (sequential-impulse building blocks).
// --------------------------------------------------------------------

/// Resolves one pair by an impulse at the contact point, if the pair is
/// touching/penetrating **and approaching**. Returns `Ok(None)` when
/// the pair is separated, separating, or geometrically degenerate —
/// in all three cases the system state is untouched.
///
/// Physics (the standard rigid-body contact impulse, cf. Bullet's
/// sequential-impulse solver and Rapier's solver contacts):
///
/// ```text
/// v_rel = (v_j + ω_j×r_j) − (v_i + ω_i×r_i)        contact-point velocity
/// K     = (m_i⁻¹ + m_j⁻¹)·1  −  [r_i]× I⁻¹ᵢ,w [r_i]×  −  [r_j]× I⁻¹ⱼ,w [r_j]×
/// j_n   = −(1 + e) (v_rel·n) / (n·K n)
/// ```
///
/// The action–reaction pair acts along the contact normal: `+j_n·n` on
/// body `j`, `−j_n·n` on body `i` (with the matching angular impulses
/// `±r × j_n·n`), applied through the canonical setters. Static bodies
/// (`inverse_mass = 0`) contribute nothing to `K` and receive no
/// state writes. `e = min(e_i, e_j)`, forced to 0 below the system's
/// `restitution_threshold` or when `force_plastic` is set (Zeno guard).
pub fn resolve_pair(
    system: &mut PhysicalObjectSystem,
    i: usize,
    j: usize,
    force_plastic: bool,
) -> Result<Option<Contact>, String> {
    let n_obj = system.objects.len();
    if i >= n_obj || j >= n_obj || i == j {
        return Err(format!(
            "collision: invalid pair (obj{i}, obj{j}) — system has {n_obj} object(s)"
        ));
    }
    let geometry = {
        let (a, b) = (&system.objects[i], &system.objects[j]);
        contact_geometry(a, b, system.contact_slop)
    };
    let Some(geom) = geometry else {
        return Ok(None);
    };
    let n = geom.normal;
    if n.norm_squared() == 0.0 {
        return Ok(None); // degenerate (e.g. concentric centers)
    }

    let (r_i, r_j, v_rel, inv_m_i, inv_m_j, iinv_i, iinv_j, e_pair) = {
        let a = &system.objects[i];
        let b = &system.objects[j];
        let r_i = geom.point - a.get_position();
        let r_j = geom.point - b.get_position();
        let v_i = a.get_velocity() + a.get_angular_velocity().cross(r_i);
        let v_j = b.get_velocity() + b.get_angular_velocity().cross(r_j);
        (
            r_i,
            r_j,
            v_j - v_i,
            a.get_inverse_mass(),
            b.get_inverse_mass(),
            a.world_inverse_inertia(),
            b.world_inverse_inertia(),
            a.get_restitution().min(b.get_restitution()),
        )
    };

    let rel_vel_n = v_rel.dot(n);
    if rel_vel_n >= 0.0 {
        return Ok(None); // separating (or resting exactly): no impulse
    }

    let e = if force_plastic || rel_vel_n.abs() < system.restitution_threshold {
        0.0
    } else {
        e_pair
    };

    let k = Mat3::identity() * (inv_m_i + inv_m_j)
        - Mat3::skew(r_i) * iinv_i * Mat3::skew(r_i)
        - Mat3::skew(r_j) * iinv_j * Mat3::skew(r_j);
    let denom = n.dot(k * n);
    if !(denom.is_finite() && denom > 0.0) {
        return Err(format!(
            "collision: singular effective mass for pair (obj{i}, obj{j}) — n·Kn = {denom}"
        ));
    }
    let j_n = -(1.0 + e) * rel_vel_n / denom;
    let impulse = n * j_n;

    // Action–reaction along the normal, through the canonical setters.
    // Static sides (inverse_mass = 0) receive no state writes.
    if inv_m_i > 0.0 {
        let a = &mut system.objects[i];
        a.set_momentum(a.get_momentum() - impulse);
        a.set_angular_momentum(a.get_angular_momentum() - r_i.cross(impulse));
    }
    if inv_m_j > 0.0 {
        let b = &mut system.objects[j];
        b.set_momentum(b.get_momentum() + impulse);
        b.set_angular_momentum(b.get_angular_momentum() + r_j.cross(impulse));
    }

    Ok(Some(Contact {
        i,
        j,
        t: system.time,
        point: geom.point,
        normal: n,
        depth: geom.depth,
        rel_vel_n,
        impulse_n: j_n,
    }))
}

/// Gauss–Seidel passes over the given pairs until no flagged pair is
/// still approaching (at most 10 passes, Bullet's default iteration
/// count). `flagged[k]` marks pairs whose root fired; unflagged pairs
/// are still checked after the first pass (an impulse on one pair can
/// push a body into another — Newton's cradle propagation).
pub fn resolve_impulses(
    system: &mut PhysicalObjectSystem,
    pairs: &[(usize, usize)],
    flagged: &[bool],
    force_plastic: bool,
) -> Result<Vec<Contact>, String> {
    let mut contacts = Vec::new();
    for pass in 0..10 {
        let mut any = false;
        for (k, &(i, j)) in pairs.iter().enumerate() {
            // First pass: only root-flagged pairs. Later passes: every
            // pair (impulses may have created new approaches).
            if pass == 0 && !flagged.get(k).copied().unwrap_or(true) {
                continue;
            }
            if let Some(c) = resolve_pair(system, i, j, force_plastic)? {
                contacts.push(c);
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    Ok(contacts)
}

/// Safety-net penetration sweep: AABB-culled. Applies impulses to
/// approaching penetrating pairs and positional projection (split by
/// inverse mass, honoring `contact_slop`) to every penetrating pair.
/// `force_plastic` selects the impulse mode: the Zeno tier-2 guard
/// passes `true` (kill chattering dead), while the end-of-interval
/// sweep passes `false` so a slowly-grinding support-axis contact is
/// resolved **elastically** and total energy stays conserved.
pub fn resolve_penetrations(
    system: &mut PhysicalObjectSystem,
    force_plastic: bool,
) -> Result<Vec<Contact>, String> {
    let pairs = collidable_pairs(system);
    let mut contacts = Vec::new();
    for &(i, j) in &pairs {
        {
            let (a, b) = (&system.objects[i], &system.objects[j]);
            if !aabb_overlap(a, b, system.contact_slop) {
                continue;
            }
        }
        if let Some(c) = resolve_pair(system, i, j, force_plastic)? {
            contacts.push(c);
        }
        // Positional projection of any remaining penetration.
        let geometry = {
            let (a, b) = (&system.objects[i], &system.objects[j]);
            contact_geometry(a, b, 0.0)
        };
        if let Some(g) = geometry {
            let overlap = g.depth - system.contact_slop;
            if overlap > 0.0 {
                let (inv_i, inv_j) = (
                    system.objects[i].get_inverse_mass(),
                    system.objects[j].get_inverse_mass(),
                );
                let total = inv_i + inv_j;
                if total > 0.0 {
                    let n = g.normal;
                    if inv_i > 0.0 {
                        let a = &mut system.objects[i];
                        a.set_position(a.get_position() - n * (overlap * inv_i / total));
                    }
                    if inv_j > 0.0 {
                        let b = &mut system.objects[j];
                        b.set_position(b.get_position() + n * (overlap * inv_j / total));
                    }
                }
            }
        }
    }
    Ok(contacts)
}

/// Appends contacts to the system's record, dropping the oldest past
/// [`CONTACTS_CAP`].
pub fn record_contacts(system: &mut PhysicalObjectSystem, contacts: Vec<Contact>) {
    for c in contacts {
        if system.contacts.len() >= CONTACTS_CAP {
            system.contacts.remove(0);
        }
        system.contacts.push(c);
    }
}

/// Midpoint of the closest points of two (infinite) lines `p1 + s·d1`
/// and `p2 + t·d2`; falls back to the midpoint of the anchors when the
/// lines are near-parallel.
fn closest_point_between_lines(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> Vec3 {
    let r = p1 - p2;
    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d2.dot(d2);
    let denom = a * c - b * b;
    if denom.abs() < 1e-12 {
        return (p1 + p2) * 0.5;
    }
    let e = d1.dot(r);
    let f = d2.dot(r);
    let s = (b * f - c * e) / denom;
    let t = (a * f - b * e) / denom;
    let q1 = p1 + d1 * s;
    let q2 = p2 + d2 * t;
    (q1 + q2) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Vec3;

    fn sphere(id: usize, mass: f64, r: f64, pos: Vec3) -> physical_object {
        physical_object::new_from_shape(
            id,
            mass,
            0.0,
            pos,
            Vec3::zeros(),
            Vec3::zeros(),
            Boundary::Sphere { radius: r },
        )
    }

    fn cuboid(id: usize, mass: f64, h: [f64; 3], pos: Vec3) -> physical_object {
        physical_object::new_from_shape(
            id,
            mass,
            0.0,
            pos,
            Vec3::zeros(),
            Vec3::zeros(),
            Boundary::Cuboid { half_extents: h },
        )
    }

    #[test]
    fn sphere_sphere_separation_and_contact() {
        let a = sphere(0, 1.0, 1.0, Vec3::zeros());
        let b = sphere(1, 1.0, 1.0, Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(pair_separation(&a, &b), 1.0);
        assert!(contact_geometry(&a, &b, 0.0).is_none());

        let c = sphere(2, 1.0, 1.0, Vec3::new(1.5, 0.0, 0.0));
        assert!((pair_separation(&a, &c) - (-0.5)).abs() < 1e-15);
        let g = contact_geometry(&a, &c, 0.0).expect("overlapping");
        assert_eq!(g.normal, Vec3::new(1.0, 0.0, 0.0));
        assert!((g.depth - 0.5).abs() < 1e-15);
        assert!((g.point.x - 0.75).abs() < 1e-15, "midway across the overlap");

        // Concentric: degenerate, no contact reported.
        let d = sphere(3, 1.0, 0.5, Vec3::zeros());
        assert!(contact_geometry(&a, &d, 0.0).is_none());
    }

    #[test]
    fn sphere_cuboid_face_edge_corner_and_interior() {
        let bx = cuboid(0, 1.0, [1.0, 1.0, 1.0], Vec3::zeros());

        // Face region: sphere left of the +x face.
        let s = sphere(1, 1.0, 0.5, Vec3::new(1.3, 0.0, 0.0));
        assert!((pair_separation(&s, &bx) - (0.3 - 0.5)).abs() < 1e-15);
        let g = contact_geometry(&s, &bx, 0.0).expect("touching face");
        // normal from sphere toward box = -x
        assert!((g.normal.x + 1.0).abs() < 1e-12);
        assert!((g.depth - 0.2).abs() < 1e-12);
        assert!((g.point.x - 1.0).abs() < 1e-12);

        // Corner region.
        let sc = sphere(2, 1.0, 0.5, Vec3::new(1.2, 1.2, 1.2));
        let dist = (3.0f64 * 0.2 * 0.2).sqrt();
        assert!((pair_separation(&sc, &bx) - (dist - 0.5)).abs() < 1e-12);
        let gc = contact_geometry(&sc, &bx, 0.0).expect("corner overlap");
        let expect_n = Vec3::new(-1.0, -1.0, -1.0).normalize();
        assert!((gc.normal - expect_n).norm() < 1e-12);

        // Sphere center inside the box: deepest along the nearest face.
        let si = sphere(3, 1.0, 0.25, Vec3::new(0.8, 0.0, 0.0));
        let gi = contact_geometry(&si, &bx, 0.0).expect("interior");
        assert!((gi.normal.x + 1.0).abs() < 1e-12, "nearest face is +x");
        assert!((gi.depth - (0.2 + 0.25)).abs() < 1e-12);

        // Reversed argument order flips the normal.
        let gf = contact_geometry(&bx, &s, 0.0).expect("flipped");
        assert!((gf.normal.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cuboid_cuboid_sat_face_and_edge_cases() {
        // Axis-aligned overlap along x: face axis, exact depth.
        let a = cuboid(0, 1.0, [1.0, 1.0, 1.0], Vec3::zeros());
        let b = cuboid(1, 1.0, [1.0, 1.0, 1.0], Vec3::new(1.8, 0.0, 0.0));
        assert!((pair_separation(&a, &b) - (-0.2)).abs() < 1e-15);
        let g = contact_geometry(&a, &b, 0.0).expect("face overlap");
        assert!((g.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((g.depth - 0.2).abs() < 1e-12);

        // Separated boxes.
        let c = cuboid(2, 1.0, [1.0, 1.0, 1.0], Vec3::new(4.0, 0.0, 0.0));
        assert!(pair_separation(&a, &c) > 0.0);
        assert!(contact_geometry(&a, &c, 0.0).is_none());

        // One box rotated 45° about z, corner-on: the SAT still finds
        // the deepest axis and a unit normal from a toward b.
        let mut d = cuboid(3, 1.0, [1.0, 1.0, 1.0], Vec3::new(2.2, 0.0, 0.0));
        d.set_orientation(Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_4));
        let sep = pair_separation(&a, &d);
        // Corner of d reaches x = 2.2 - sqrt(2) ≈ 0.786 < 1 → penetrating.
        assert!(sep < 0.0, "rotated corner penetrates: sep = {sep}");
        let gd = contact_geometry(&a, &d, 0.0).expect("rotated overlap");
        assert!((gd.normal.norm() - 1.0).abs() < 1e-12);
        assert!(gd.normal.x > 0.9, "normal along +x from a toward d");
    }

    #[test]
    fn separation_is_continuous_across_touch() {
        // Slide a sphere along x past a cuboid corner; sample separation
        // densely and require no jump larger than the step bound.
        let bx = cuboid(0, 1.0, [1.0, 1.0, 1.0], Vec3::zeros());
        let mut prev = None;
        let mut x = -3.0;
        while x <= 3.0 {
            let s = sphere(1, 1.0, 0.5, Vec3::new(x, 1.45, 1.45));
            let sep = pair_separation(&s, &bx);
            if let Some(p) = prev {
                let jump: f64 = sep - p;
                assert!(jump.abs() < 0.02, "separation jump {jump} at x = {x}");
            }
            prev = Some(sep);
            x += 0.01;
        }
    }

    #[test]
    fn collidable_pairs_exclude_point_point_and_static_static() {
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 0.0);
        sys.add_object(physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros()));
        sys.add_object(physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros()));
        sys.add_object(sphere(2, 1.0, 0.5, Vec3::new(2.0, 0.0, 0.0)));
        // point-point excluded; point-sphere and sphere pairs included.
        assert_eq!(collidable_pairs(&sys), vec![(0, 2), (1, 2)]);

        // Two static spheres: excluded.
        let mut sys2 = PhysicalObjectSystem::new(Vec::new(), 0.0);
        let mut s0 = sphere(0, 1.0, 0.5, Vec3::zeros());
        s0.set_inverse_mass(0.0);
        let mut s1 = sphere(1, 1.0, 0.5, Vec3::new(0.6, 0.0, 0.0));
        s1.set_inverse_mass(0.0);
        sys2.add_object(s0);
        sys2.add_object(s1);
        assert!(collidable_pairs(&sys2).is_empty());
    }

    fn moving_sphere(id: usize, mass: f64, r: f64, pos: Vec3, vel: Vec3) -> physical_object {
        physical_object::new_from_shape(
            id,
            mass,
            0.0,
            pos,
            vel,
            Vec3::zeros(),
            Boundary::Sphere { radius: r },
        )
    }

    #[test]
    fn effective_mass_central_sphere_hit_is_inverse_mass_sum() {
        // Two free spheres in central contact: r ∥ n, so the angular
        // terms vanish and n·Kn must equal 1/m_i + 1/m_j exactly.
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 0.0);
        sys.add_object(moving_sphere(0, 2.0, 0.5, Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)));
        sys.add_object(moving_sphere(1, 4.0, 0.5, Vec3::new(0.99, 0.0, 0.0), Vec3::zeros()));
        let c = resolve_pair(&mut sys, 0, 1, false).unwrap().expect("contact");
        // j_n = -(1+e) v_rel·n / (1/m0 + 1/m1); v_rel·n = -1, e = 1
        let expect = 2.0 / (0.5 + 0.25);
        assert!((c.impulse_n - expect).abs() < 1e-12, "j_n = {}", c.impulse_n);
        assert!((c.rel_vel_n + 1.0).abs() < 1e-12);
        // Momentum is exchanged, total conserved.
        let p_total = sys.objects[0].get_momentum() + sys.objects[1].get_momentum();
        assert!((p_total - Vec3::new(2.0, 0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn static_wall_reflects_and_stays_bit_unchanged() {
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 0.0);
        // Static cuboid wall at the origin (inverse mass 0).
        let mut wall = cuboid(0, 1.0, [0.1, 5.0, 5.0], Vec3::zeros());
        wall.set_inverse_mass(0.0);
        wall.set_inverse_inertia_tensor(Mat3::zeros());
        let wall_p = wall.get_momentum();
        let wall_l = wall.get_angular_momentum();
        let wall_x = wall.get_position();
        sys.add_object(wall);
        // Sphere flying into the wall along −x.
        sys.add_object(moving_sphere(1, 1.0, 0.5, Vec3::new(0.55, 0.0, 0.0), Vec3::new(-3.0, 0.0, 0.0)));
        let c = resolve_pair(&mut sys, 0, 1, false).unwrap().expect("hit");
        // e = 1 wall bounce: v → v − 2(v·n)n, i.e. reflected exactly.
        let v = sys.objects[1].get_velocity();
        assert!((v - Vec3::new(3.0, 0.0, 0.0)).norm() < 1e-12, "v = {v:?}");
        // Static side bit-unchanged.
        assert_eq!(sys.objects[0].get_momentum(), wall_p);
        assert_eq!(sys.objects[0].get_angular_momentum(), wall_l);
        assert_eq!(sys.objects[0].get_position(), wall_x);
        // Normal points from wall (i=0) toward sphere (j=1): +x.
        assert!((c.normal.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn separating_pair_is_untouched_and_plastic_flag_kills_bounce() {
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 0.0);
        sys.add_object(moving_sphere(0, 1.0, 0.5, Vec3::zeros(), Vec3::new(-1.0, 0.0, 0.0)));
        sys.add_object(moving_sphere(1, 1.0, 0.5, Vec3::new(0.9, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)));
        // Overlapping but separating: no impulse, state untouched.
        let before0 = sys.objects[0].get_momentum();
        assert!(resolve_pair(&mut sys, 0, 1, false).unwrap().is_none());
        assert_eq!(sys.objects[0].get_momentum(), before0);

        // Same pair approaching with force_plastic: e treated as 0 →
        // common velocity afterwards (momentum split equally).
        sys.objects[0].set_velocity(Vec3::new(1.0, 0.0, 0.0));
        sys.objects[1].set_velocity(Vec3::new(-1.0, 0.0, 0.0));
        let c = resolve_pair(&mut sys, 0, 1, true).unwrap().expect("plastic hit");
        assert!((sys.objects[0].get_velocity().x).abs() < 1e-12);
        assert!((sys.objects[1].get_velocity().x).abs() < 1e-12);
        assert!((c.impulse_n - 1.0).abs() < 1e-12, "j_n = -(1+0)(-2)/2 = 1");
    }

    #[test]
    fn penetration_projection_separates_overlap() {
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 0.0);
        // Deeply overlapping, at rest → impulse does nothing (not
        // approaching), projection must push them apart.
        sys.add_object(moving_sphere(0, 1.0, 1.0, Vec3::zeros(), Vec3::zeros()));
        sys.add_object(moving_sphere(1, 1.0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros()));
        resolve_penetrations(&mut sys, true).unwrap();
        let gap = pair_separation(&sys.objects[0], &sys.objects[1]);
        // Projection resolves down to the slop band (residual ≤ slop).
        assert!(gap >= -(sys.contact_slop + 1e-12), "post-projection separation = {gap}");
        // Equal masses: symmetric displacement about the midpoint.
        assert!((sys.objects[0].get_position().x + sys.objects[1].get_position().x - 1.0).abs() < 1e-12);
    }

    fn shaped(id: usize, mass: f64, b: Boundary, pos: Vec3) -> physical_object {
        physical_object::new_from_shape(id, mass, 0.0, pos, Vec3::zeros(), Vec3::zeros(), b)
    }

    #[test]
    fn ball_vs_torus_exact_contact_and_hole_passage() {
        let torus = shaped(0, 1.0, Boundary::Torus { ring_radius: 1.5, tube_radius: 0.5 }, Vec3::zeros());

        // Ball outside the tube, on the +x side: sdf(2.2,0,0) = 0.2,
        // so a r=0.3 ball penetrates by 0.1.
        let ball = sphere(1, 1.0, 0.3, Vec3::new(2.2, 0.0, 0.0));
        assert!((pair_separation(&torus, &ball) - (-0.1)).abs() < 1e-12);
        let g = contact_geometry(&torus, &ball, 0.0).expect("touching tube");
        assert!((g.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12, "normal torus→ball");
        assert!((g.depth - 0.1).abs() < 1e-12);
        // Contact point midway across the overlap band on the x axis.
        assert!((g.point - Vec3::new(1.95, 0.0, 0.0)).norm() < 1e-12);
        // Reversed argument order flips the normal.
        let gf = contact_geometry(&ball, &torus, 0.0).expect("flipped");
        assert!((gf.normal - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-12);

        // A ball in the hole: sdf(0,0,0) = 1.0 → a r=0.3 ball at the
        // center is separated by 0.7 and reports NO contact.
        let inhole = sphere(2, 1.0, 0.3, Vec3::zeros());
        assert!((pair_separation(&torus, &inhole) - 0.7).abs() < 1e-12);
        assert!(contact_geometry(&torus, &inhole, 0.0).is_none());
    }

    #[test]
    fn ball_vs_disk_and_cylinder_contacts() {
        // Disk r=1 in the xy-plane; ball above its face.
        let disk = shaped(0, 1.0, Boundary::Disk { radius: 1.0 }, Vec3::zeros());
        let ball = sphere(1, 1.0, 0.5, Vec3::new(0.3, 0.0, 0.4));
        assert!((pair_separation(&disk, &ball) - (-0.1)).abs() < 1e-12);
        let g = contact_geometry(&disk, &ball, 0.0).expect("face contact");
        assert!((g.normal - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
        assert!((g.depth - 0.1).abs() < 1e-12);
        // Rim contact from beyond the edge.
        let rim = sphere(2, 1.0, 0.5, Vec3::new(1.3, 0.0, 0.4));
        let dist = (0.3f64 * 0.3 + 0.4 * 0.4).sqrt();
        assert!((pair_separation(&disk, &rim) - (dist - 0.5)).abs() < 1e-12);

        // Cylinder r=0.25 h=0.75: side contact.
        let cyl = shaped(3, 1.0, Boundary::Cylinder { radius: 0.25, half_height: 0.75 }, Vec3::zeros());
        let side = sphere(4, 1.0, 0.2, Vec3::new(0.4, 0.0, 0.1));
        assert!((pair_separation(&cyl, &side) - (0.15 - 0.2)).abs() < 1e-12);
        let gs = contact_geometry(&cyl, &side, 0.0).expect("side contact");
        assert!((gs.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
        // Cap contact.
        let cap = sphere(5, 1.0, 0.2, Vec3::new(0.0, 0.1, 0.9));
        assert!((pair_separation(&cyl, &cap) - (0.15 - 0.2)).abs() < 1e-12);
        let gc = contact_geometry(&cyl, &cap, 0.0).expect("cap contact");
        assert!((gc.normal - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn tilted_torus_fits_a_4x4_box_where_flat_does_not() {
        // The bounding-box demo fact: a torus with outer radius 2 in a
        // 4-wide box (slab inner faces at x = ±2 …) touches when its
        // axis is along z, but clears every wall by 2 − (1.5·√(2/3)+0.5)
        // ≈ 0.2753 when the axis points along the body diagonal.
        let slab = cuboid(0, 1.0, [1.0, 4.0, 4.0], Vec3::new(3.0, 0.0, 0.0)); // inner face x = 2
        let mut torus = shaped(1, 1.0, Boundary::Torus { ring_radius: 1.5, tube_radius: 0.5 }, Vec3::zeros());

        // Axis-aligned: outer equator reaches x = 2 → separation 0.
        assert!(pair_separation(&torus, &slab).abs() < 1e-12, "flat torus exactly inscribes");

        // Tilted: axis along (1,1,1)/√3.
        let axis = Vec3::new(0.0, 0.0, 1.0).cross(Vec3::new(1.0, 1.0, 1.0).normalize());
        let angle = (1.0f64 / 3.0f64.sqrt()).acos();
        torus.set_orientation(Quat::from_axis_angle(axis.normalize(), angle));
        let clearance = 2.0 - (1.5 * (2.0f64 / 3.0).sqrt() + 0.5);
        assert!((pair_separation(&torus, &slab) - clearance).abs() < 1e-9,
            "tilted torus clears the wall by {clearance}");

        // And the support-axis contact reports the wall's face normal
        // once pushed into the slab.
        torus.set_position(Vec3::new(clearance + 0.01, 0.0, 0.0));
        let g = contact_geometry(&torus, &slab, 0.0).expect("pressed into wall");
        assert!((g.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((g.depth - 0.01).abs() < 1e-9);
    }

    #[test]
    fn static_slab_reflects_a_cylinder_elastically() {
        // The inverse-mass wall bounce for a NEW shape: an infinitely
        // massive slab (inverse_mass = 0) reflects a cylinder's normal
        // velocity exactly (e = 1) and is itself bit-unchanged.
        let mut sys = PhysicalObjectSystem::new(Vec::new(), 0.0);
        let mut wall = cuboid(0, 1.0, [0.5, 4.0, 4.0], Vec3::new(2.5, 0.0, 0.0)); // inner face x = 2
        wall.set_inverse_mass(0.0);
        wall.set_inverse_inertia_tensor(Mat3::zeros());
        sys.add_object(wall);
        let mut cyl = shaped(1, 2.0, Boundary::Cylinder { radius: 0.25, half_height: 0.75 }, Vec3::new(1.76, 0.0, 0.0));
        cyl.set_velocity(Vec3::new(3.0, 0.0, 0.0)); // side extent 0.25 → touches at x = 1.75
        sys.add_object(cyl);
        let c = resolve_pair(&mut sys, 0, 1, false).unwrap().expect("wall hit");
        let v = sys.objects[1].get_velocity();
        assert!((v - Vec3::new(-3.0, 0.0, 0.0)).norm() < 1e-12, "elastic reflection, v = {v:?}");
        assert_eq!(sys.objects[0].get_momentum(), Vec3::zeros());
        assert_eq!(sys.objects[0].get_position(), Vec3::new(2.5, 0.0, 0.0));
        // Normal from wall (i=0) toward cylinder (j=1): −x.
        assert!((c.normal.x + 1.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_cylinders_side_contact_is_exact() {
        // Two z-aligned cylinders (r = 0.1, h = 1) with an axial offset
        // of 1: their z-intervals overlap, so the true separation is
        // purely lateral, dx − 0.2. The radial rejection axis makes the
        // support-axis tier report exactly that (this used to fire at a
        // lateral gap of ~9 radii with a tilted normal).
        let cyl = |id: usize, pos: Vec3| {
            shaped(id, 1.0, Boundary::Cylinder { radius: 0.1, half_height: 1.0 }, pos)
        };
        let a = cyl(0, Vec3::zeros());
        for dx in [1.5, 1.0, 0.5, 0.25] {
            let b = cyl(1, Vec3::new(dx, 0.0, 1.0));
            let sep = pair_separation(&a, &b);
            assert!(
                (sep - (dx - 0.2)).abs() < 1e-12,
                "parallel side separation at dx = {dx}: {sep} (true {})",
                dx - 0.2
            );
        }
        // Pressed side-on: the normal is the lateral direction, not a
        // tilted axis mix.
        let b = cyl(1, Vec3::new(0.19, 0.0, 1.0));
        let g = contact_geometry(&a, &b, 0.0).expect("side contact");
        assert!((g.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9, "normal = {:?}", g.normal);
        assert!((g.depth - 0.01).abs() < 1e-9);
    }

    #[test]
    fn parallel_disk_disk_separation_is_the_documented_limitation() {
        // KNOWN ideal-body limitation, pinned deliberately: two disks
        // with PARALLEL planes have zero extent along the shared
        // normal, so their separation equals |dz| — it touches zero at
        // plane coincidence without a sign change, and face-on
        // crossings are invisible to downward-crossing rootfinding
        // (documented in collision_detection.md; tilt a disk or use a
        // thin cylinder for a detectable version).
        let disk = |id: usize, pos: Vec3| shaped(id, 1.0, Boundary::Disk { radius: 1.0 }, pos);
        let a = disk(0, Vec3::zeros());
        for dz in [0.1, 0.0, -0.1] {
            let b = disk(1, Vec3::new(0.2, 0.0, dz));
            let sep = pair_separation(&a, &b);
            assert!(
                (sep - dz.abs()).abs() < 1e-12,
                "parallel disk-disk separation at dz = {dz}: {sep}"
            );
        }
    }

    #[test]
    fn small_cap_on_large_face_contacts_at_the_cap_center() {
        // A cylinder cap (r = 0.25) pressed axially into a big slab
        // face: both supporting sets are rank-2 faces, but the cap is
        // far smaller — the contact point must sit at the CAP center,
        // not halfway toward the slab-face centroid (which would inject
        // a spurious lever arm).
        let slab = cuboid(0, 1.0, [1.0, 4.0, 4.0], Vec3::new(3.0, 0.0, 0.0)); // face x = 2
        let mut cyl = shaped(
            1,
            1.0,
            Boundary::Cylinder { radius: 0.25, half_height: 0.75 },
            Vec3::new(1.26, 0.5, 0.5),
        );
        // Axis along +x: rotate local z onto x.
        cyl.set_orientation(Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::FRAC_PI_2));
        let g = contact_geometry(&cyl, &slab, 0.0).expect("cap pressed into face");
        assert!((g.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((g.depth - 0.01).abs() < 1e-9);
        assert!((g.point.y - 0.5).abs() < 1e-9 && (g.point.z - 0.5).abs() < 1e-9,
            "contact at the cap center, got {:?}", g.point);
    }

    #[test]
    fn dumbbell_wall_gaps_and_ball_contacts_are_exact() {
        // Asymmetric dumbbell (m1 = 1, m2 = 2): z1 = −9/14, z2 = 5/14.
        let (mass, db) = boundary::dumbbell(1.0, 2.0, 0.5, 0.25, 0.25, 0.1, 1.0).unwrap();
        let mk = |pos: Vec3, q: Quat| {
            let mut o = physical_object::new_from_shape(
                0,
                mass,
                0.0,
                pos,
                Vec3::zeros(),
                Vec3::zeros(),
                db,
            );
            o.set_orientation(q);
            o
        };
        let (z1, z2, r1, r2) = match db {
            Boundary::Dumbbell { r1, r2, z1, z2, .. } => (z1, z2, r1, r2),
            _ => unreachable!(),
        };

        // Wall slab with inner face at x = 2; dumbbell axis along +x.
        let slab = cuboid(9, 1.0, [1.0, 4.0, 4.0], Vec3::new(3.0, 0.0, 0.0));
        let q_x = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::FRAC_PI_2);
        // Heavy-end-forward (local +z → world +x): gap = 2 − (z2 + r2).
        let heavy = mk(Vec3::zeros(), q_x);
        let sep = pair_separation(&heavy, &slab);
        assert!((sep - (2.0 - (z2 + r2))).abs() < 1e-12, "heavy end gap {sep}");
        // Light-end-forward (axis flipped): gap = 2 − (|z1| + r1).
        let q_flip = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), -std::f64::consts::FRAC_PI_2);
        let light = mk(Vec3::zeros(), q_flip);
        let sep2 = pair_separation(&light, &slab);
        assert!((sep2 - (2.0 - (-z1 + r1))).abs() < 1e-12, "light end gap {sep2}");
        // The COM sits nearer the HEAVY sphere, so the light end pokes
        // out farther (|z1| > |z2|) and leaves the SMALLER wall gap.
        assert!(sep2 < sep, "light-end-forward closes the gap: {sep2} vs {sep}");

        // Ball against sphere-2's pole (exact SDF tier).
        let upright = mk(Vec3::zeros(), Quat::identity());
        let ball = sphere(1, 1.0, 0.2, Vec3::new(0.0, 0.0, z2 + r2 + 0.15));
        assert!((pair_separation(&upright, &ball) + 0.05).abs() < 1e-12, "penetrating 0.05");
        let g = contact_geometry(&upright, &ball, 0.0).expect("pole contact");
        assert!((g.normal - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
        assert!((g.depth - 0.05).abs() < 1e-12);
        // Ball against the rod's side.
        let mid = 0.5 * (z1 + z2);
        let side = sphere(2, 1.0, 0.2, Vec3::new(0.25, 0.0, mid));
        assert!((pair_separation(&upright, &side) - (0.15 - 0.2)).abs() < 1e-12);
        let gs = contact_geometry(&upright, &side, 0.0).expect("rod side contact");
        assert!((gs.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn aabb_overlap_and_support_vertex() {
        let a = cuboid(0, 1.0, [1.0, 2.0, 0.5], Vec3::zeros());
        let b = cuboid(1, 1.0, [1.0, 1.0, 1.0], Vec3::new(2.5, 0.0, 0.0));
        assert!(!aabb_overlap(&a, &b, 0.0));
        assert!(aabb_overlap(&a, &b, 0.6));
        let cols = mat_columns(&a.get_orientation().to_rotation_matrix());
        let v = support_vertex(&[1.0, 2.0, 0.5], Vec3::zeros(), &cols, Vec3::new(1.0, -1.0, 1.0));
        assert_eq!(v, Vec3::new(1.0, -2.0, 0.5));
    }
}
