//! Isosurface extraction from a 3-D scalar field, by **marching
//! tetrahedra**.
//!
//! # Why tetrahedra and not marching cubes
//!
//! Marching cubes is the famous algorithm, and it needs a 256-entry
//! lookup table mapping corner-sign patterns to triangle lists. That
//! table is the algorithm: transcribing it is the work, and getting one
//! entry wrong produces a surface with a hole in it that looks fine from
//! most angles.
//!
//! Marching tetrahedra needs **no table at all**. Split each cube into
//! six tetrahedra; a tetrahedron has four corners, so only
//! `2^4 = 16` sign patterns, and up to symmetry there are just three
//! cases — none inside, one inside (one triangle), or two inside (two
//! triangles). Each is derivable in a line of reasoning rather than
//! looked up, and the classic marching-cubes ambiguity, where a face
//! shared between two cubes can be triangulated two incompatible ways
//! and leaves a crack, cannot arise: a face of a tetrahedron is a
//! triangle, and a triangle's crossing pattern is unique.
//!
//! The cost is more triangles for the same surface. That is a good trade
//! for code whose correctness you want to be able to argue rather than
//! trust.
//!
//! # What is verified
//!
//! The tests do not eyeball a picture. For a sphere, whose area and
//! volume are known exactly, they check:
//!
//! * the enclosed volume, by the divergence theorem over the mesh;
//! * the surface area;
//! * that both converge at the expected rate as the grid refines;
//! * that the mesh is **watertight and consistently oriented** — every
//!   directed edge appears exactly once and its reverse exactly once.
//!   A single missing or flipped triangle breaks that immediately.

/// A triangle mesh: positions and triangle indices.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// `[x, y, z]` per vertex.
    pub vertices: Vec<[f64; 3]>,
    /// Three vertex indices per triangle, wound counter-clockwise seen
    /// from OUTSIDE the enclosed region (the region where the field
    /// exceeds the isolevel).
    pub triangles: Vec<[usize; 3]>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Total surface area.
    pub fn area(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let (a, b, c) = (
                    self.vertices[t[0]],
                    self.vertices[t[1]],
                    self.vertices[t[2]],
                );
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
            })
            .sum()
    }

    /// Volume enclosed by the mesh, via the divergence theorem: the sum
    /// of signed tetrahedron volumes from the origin to each triangle.
    ///
    /// Only meaningful for a closed, consistently oriented mesh — which
    /// is exactly what [`Mesh::is_watertight`] checks.
    pub fn enclosed_volume(&self) -> f64 {
        let v: f64 = self
            .triangles
            .iter()
            .map(|t| {
                let (a, b, c) = (
                    self.vertices[t[0]],
                    self.vertices[t[1]],
                    self.vertices[t[2]],
                );
                (a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    / 6.0
            })
            .sum();
        v.abs()
    }

    /// Whether the mesh is closed AND consistently oriented.
    ///
    /// Every interior edge of such a mesh is traversed once in each
    /// direction. So collecting directed edges `(i, j)` from every
    /// triangle, each must appear exactly once, and `(j, i)` must too.
    /// A hole leaves an unmatched edge; a flipped triangle leaves a
    /// duplicate. Both are caught.
    pub fn is_watertight(&self) -> bool {
        use std::collections::HashMap;
        let mut seen: HashMap<(usize, usize), i32> = HashMap::new();
        for t in &self.triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *seen.entry((a, b)).or_insert(0) += 1;
            }
        }
        for (&(a, b), &count) in &seen {
            if count != 1 {
                return false;
            }
            if seen.get(&(b, a)) != Some(&1) {
                return false;
            }
        }
        true
    }
}

/// A cube's eight corners, as `(dx, dy, dz)` offsets.
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// Six tetrahedra tiling the cube, all sharing the main diagonal 0–6.
///
/// This decomposition is the reason no table is needed: it is a fixed
/// tiling, and every tetrahedron is then handled by the same three-case
/// analysis. Sharing one diagonal guarantees adjacent cubes agree on how
/// their common face is split, which is what keeps the surface crack-free.
const TETS: [[usize; 4]; 6] = [
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
];

/// Extract the surface where `field == level`, enclosing the region
/// where `field > level`.
///
/// The field is sampled on an `nx × ny × nz` grid with index
/// `(iz*ny + iy)*nx + ix` — the same layout as
/// [`crate::qm3d::Wavefunction3::density`]. `origin` and `spacing` place
/// it in space.
///
/// # Errors
/// A length mismatch, a non-finite level, or a grid too small to contain
/// a cell (every axis needs at least two points).
pub fn marching_tetrahedra(
    field: &[f64],
    dims: (usize, usize, usize),
    origin: (f64, f64, f64),
    spacing: (f64, f64, f64),
    level: f64,
) -> Result<Mesh, String> {
    let (nx, ny, nz) = dims;
    if field.len() != nx * ny * nz {
        return Err(format!(
            "marching_tetrahedra: field has {} values but the grid is {nx}x{ny}x{nz}",
            field.len()
        ));
    }
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(format!(
            "marching_tetrahedra: every axis needs at least 2 points, got {nx}x{ny}x{nz}"
        ));
    }
    if !level.is_finite() {
        return Err(format!("marching_tetrahedra: the level must be finite, got {level}"));
    }

    // A grid sample lying EXACTLY on the isolevel is a degeneracy, and
    // a common one: on a symmetric grid the surface often passes through
    // sample points precisely. The crossing then interpolates to t = 0,
    // placing a vertex exactly on that corner — so several distinct grid
    // edges produce coincident-but-separately-indexed vertices, the
    // triangles between them collapse to zero area and get dropped, and
    // the surface has pinholes at exactly those points.
    //
    // (Found by the non-cubic sphere test: hy and hz landed on y = 0 and
    // z = 0, hx landed on x = 1.5, and the r = 1.5 sphere ran straight
    // through the sample there. Ten broken edges out of ~31 800, all at
    // the six axis points.)
    //
    // The fix is to move the level, not the samples: a relative nudge
    // far below any physical significance, applied only when an exact
    // hit actually occurs.
    let level = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for v in field {
            if v.is_finite() {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        let scale = (hi - lo).abs().max(level.abs()).max(1.0);
        let eps = scale * 1e-12;
        if field.iter().any(|v| (v - level).abs() <= eps) {
            level + eps
        } else {
            level
        }
    };

    let idx = |ix: usize, iy: usize, iz: usize| (iz * ny + iy) * nx + ix;
    let pos = |ix: usize, iy: usize, iz: usize| {
        [
            origin.0 + ix as f64 * spacing.0,
            origin.1 + iy as f64 * spacing.1,
            origin.2 + iz as f64 * spacing.2,
        ]
    };

    let mut mesh = Mesh::default();
    // Vertices are shared between triangles via this map, keyed by the
    // grid edge they lie on. Without it the mesh would be triangle soup
    // and the watertightness check could not run.
    let mut edge_vertex: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();

    let mut vertex_on_edge =
        |mesh: &mut Mesh,
         ka: usize,
         kb: usize,
         pa: [f64; 3],
         pb: [f64; 3],
         va: f64,
         vb: f64| -> usize {
            let key = if ka < kb { (ka, kb) } else { (kb, ka) };
            if let Some(&i) = edge_vertex.get(&key) {
                return i;
            }
            // linear interpolation to the crossing
            let denom = vb - va;
            let t = if denom.abs() < f64::EPSILON {
                0.5
            } else {
                ((level - va) / denom).clamp(0.0, 1.0)
            };
            // interpolate in the SAME order the key was built from, so
            // both cubes sharing this edge compute the same point
            let (p0, p1, t) = if ka < kb { (pa, pb, t) } else { (pb, pa, 1.0 - t) };
            let p = [
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]),
            ];
            let i = mesh.vertices.len();
            mesh.vertices.push(p);
            edge_vertex.insert(key, i);
            i
        };

    for iz in 0..nz - 1 {
        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                // gather the cube's eight corners
                let mut ck = [0usize; 8];
                let mut cv = [0.0f64; 8];
                let mut cp = [[0.0f64; 3]; 8];
                for (c, off) in CORNERS.iter().enumerate() {
                    let (jx, jy, jz) = (ix + off[0], iy + off[1], iz + off[2]);
                    let k = idx(jx, jy, jz);
                    ck[c] = k;
                    cv[c] = field[k];
                    cp[c] = pos(jx, jy, jz);
                }

                for tet in TETS.iter() {
                    // "inside" is where the field EXCEEDS the level, so
                    // the surface encloses the high-density region.
                    let inside: Vec<usize> =
                        (0..4).filter(|&i| cv[tet[i]] > level).collect();
                    let outside: Vec<usize> =
                        (0..4).filter(|&i| cv[tet[i]] <= level).collect();

                    match inside.len() {
                        0 | 4 => {}
                        // One corner inside: a single triangle separates
                        // it from the other three.
                        1 => {
                            let a = inside[0];
                            let mut tri = [0usize; 3];
                            for (n, &b) in outside.iter().enumerate() {
                                tri[n] = vertex_on_edge(
                                    &mut mesh,
                                    ck[tet[a]],
                                    ck[tet[b]],
                                    cp[tet[a]],
                                    cp[tet[b]],
                                    cv[tet[a]],
                                    cv[tet[b]],
                                );
                            }
                            push_oriented(&mut mesh, tri, cp[tet[a]]);
                        }
                        // Three inside: the same shape around the single
                        // outside corner, wound the other way.
                        3 => {
                            let a = outside[0];
                            let mut tri = [0usize; 3];
                            for (n, &b) in inside.iter().enumerate() {
                                tri[n] = vertex_on_edge(
                                    &mut mesh,
                                    ck[tet[a]],
                                    ck[tet[b]],
                                    cp[tet[a]],
                                    cp[tet[b]],
                                    cv[tet[a]],
                                    cv[tet[b]],
                                );
                            }
                            // the interior is on the far side from `a`,
                            // so orient away from the OUTSIDE corner
                            push_oriented_away(&mut mesh, tri, cp[tet[a]]);
                        }
                        // Two inside, two outside: a quadrilateral,
                        // split into two triangles.
                        2 => {
                            let (i0, i1) = (inside[0], inside[1]);
                            let (o0, o1) = (outside[0], outside[1]);
                            let ends = [(i0, o0), (i0, o1), (i1, o1), (i1, o0)];
                            let mut q = [0usize; 4];
                            for (n, &(a, b)) in ends.iter().enumerate() {
                                q[n] = vertex_on_edge(
                                    &mut mesh,
                                    ck[tet[a]],
                                    ck[tet[b]],
                                    cp[tet[a]],
                                    cp[tet[b]],
                                    cv[tet[a]],
                                    cv[tet[b]],
                                );
                            }
                            // The quad q[0] -> q[1] -> q[2] -> q[3] splits
                            // along the diagonal q[0]-q[2]. BOTH halves
                            // must take the SAME winding: orienting them
                            // independently lets a near-degenerate one
                            // flip while its partner does not, and the
                            // shared diagonal is then traversed twice in
                            // the same direction — a hole.
                            let inside_pt = cp[tet[i0]];
                            let flip = needs_flip(&mesh, [q[0], q[1], q[2]], inside_pt);
                            push_with_winding(&mut mesh, [q[0], q[1], q[2]], flip);
                            push_with_winding(&mut mesh, [q[0], q[2], q[3]], flip);
                        }
                        _ => unreachable!("a tetrahedron has four corners"),
                    }
                }
            }
        }
    }
    Ok(mesh)
}

/// Whether `tri` must be reversed for its normal to point away from
/// `interior`. Split out so a quad's two halves can share one decision.
fn needs_flip(mesh: &Mesh, tri: [usize; 3], interior: [f64; 3]) -> bool {
    let (a, b, c) = (mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let d = [a[0] - interior[0], a[1] - interior[1], a[2] - interior[2]];
    n[0] * d[0] + n[1] * d[1] + n[2] * d[2] < 0.0
}

/// Append a triangle with a winding already decided.
fn push_with_winding(mesh: &mut Mesh, tri: [usize; 3], flip: bool) {
    if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
        return;
    }
    if flip {
        mesh.triangles.push([tri[0], tri[2], tri[1]]);
    } else {
        mesh.triangles.push(tri);
    }
}

/// Append a triangle wound so its normal points AWAY from `interior`.
fn push_oriented(mesh: &mut Mesh, tri: [usize; 3], interior: [f64; 3]) {
    if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
        return; // degenerate: the crossing collapsed onto a grid point
    }
    let (a, b, c) = (mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let d = [a[0] - interior[0], a[1] - interior[1], a[2] - interior[2]];
    if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] >= 0.0 {
        mesh.triangles.push(tri);
    } else {
        mesh.triangles.push([tri[0], tri[2], tri[1]]);
    }
}

/// As [`push_oriented`], but `exterior` is on the OUTSIDE, so the normal
/// should point towards it.
fn push_oriented_away(mesh: &mut Mesh, tri: [usize; 3], exterior: [f64; 3]) {
    if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
        return;
    }
    let (a, b, c) = (mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]);
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let d = [exterior[0] - a[0], exterior[1] - a[1], exterior[2] - a[2]];
    if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] >= 0.0 {
        mesh.triangles.push(tri);
    } else {
        mesh.triangles.push([tri[0], tri[2], tri[1]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Sample `f` on a cubic grid spanning `[-a, a]` on every axis.
    fn sample<F: Fn(f64, f64, f64) -> f64>(n: usize, a: f64, f: F) -> (Vec<f64>, f64) {
        let h = 2.0 * a / (n - 1) as f64;
        let mut v = Vec::with_capacity(n * n * n);
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    v.push(f(
                        -a + ix as f64 * h,
                        -a + iy as f64 * h,
                        -a + iz as f64 * h,
                    ));
                }
            }
        }
        (v, h)
    }

    fn sphere_mesh(n: usize, a: f64, r: f64) -> Mesh {
        // field = -(x^2+y^2+z^2), so "inside" (field > level) is the
        // ball of radius r when level = -r^2
        let (v, h) = sample(n, a, |x, y, z| -(x * x + y * y + z * z));
        marching_tetrahedra(&v, (n, n, n), (-a, -a, -a), (h, h, h), -r * r).unwrap()
    }

    /// The volume and area of a sphere are known exactly, so the mesh
    /// can be checked against arithmetic rather than against a picture.
    #[test]
    fn a_sphere_has_the_right_volume_and_area() {
        let r = 2.0_f64;
        let m = sphere_mesh(61, 3.0, r);
        assert!(!m.is_empty(), "no surface was produced");
        let v_exact = 4.0 / 3.0 * PI * r * r * r;
        let a_exact = 4.0 * PI * r * r;
        let v = m.enclosed_volume();
        let ar = m.area();
        assert!(
            (v - v_exact).abs() / v_exact < 0.01,
            "volume {v} vs exact {v_exact}"
        );
        // A faceted surface is systematically a little larger than the
        // smooth one it approximates; 3 % at this resolution.
        assert!(
            (ar - a_exact).abs() / a_exact < 0.03,
            "area {ar} vs exact {a_exact}"
        );
    }

    /// **Watertight and consistently oriented**: every directed edge
    /// exactly once, its reverse exactly once. A hole leaves an
    /// unmatched edge; a flipped triangle leaves a duplicate. This is
    /// the test that would catch a wrong case in the analysis.
    #[test]
    fn the_mesh_is_closed_and_consistently_wound() {
        for n in [21usize, 32, 45] {
            let m = sphere_mesh(n, 3.0, 2.0);
            assert!(!m.is_empty());
            assert!(
                m.is_watertight(),
                "n = {n}: the mesh is not closed or not consistently oriented"
            );
        }
    }

    /// Refining the grid must converge on the true volume.
    #[test]
    fn refinement_converges_on_the_exact_volume() {
        let r = 2.0_f64;
        let exact = 4.0 / 3.0 * PI * r * r * r;
        let mut prev = f64::INFINITY;
        for n in [21usize, 41, 81] {
            let err = (sphere_mesh(n, 3.0, r).enclosed_volume() - exact).abs();
            assert!(err < prev, "error did not fall at n = {n}: {err} vs {prev}");
            prev = err;
        }
        assert!(prev / exact < 2e-3, "finest grid still {} off", prev / exact);
    }

    /// An off-centre, non-spherical surface: an ellipsoid, whose volume
    /// is also exact. A sphere would hide an axis mix-up.
    #[test]
    fn an_offset_ellipsoid_has_the_right_volume() {
        let (ax, by, cz) = (2.0_f64, 1.2_f64, 0.7_f64);
        let (x0, y0, z0) = (0.4_f64, -0.3_f64, 0.6_f64);
        let n = 71;
        let a = 3.0;
        let (v, h) = sample(n, a, |x, y, z| {
            -(((x - x0) / ax).powi(2) + ((y - y0) / by).powi(2) + ((z - z0) / cz).powi(2))
        });
        let m = marching_tetrahedra(&v, (n, n, n), (-a, -a, -a), (h, h, h), -1.0).unwrap();
        assert!(m.is_watertight(), "the ellipsoid mesh is not closed");
        let exact = 4.0 / 3.0 * PI * ax * by * cz;
        let got = m.enclosed_volume();
        assert!((got - exact).abs() / exact < 0.02, "volume {got} vs exact {exact}");
    }

    /// A NON-CUBIC grid with different spacings per axis, where an
    /// index or spacing mix-up would distort the result measurably.
    #[test]
    fn a_non_cubic_grid_gives_the_right_volume() {
        let (nx, ny, nz) = (41usize, 33usize, 27usize);
        let (ax, ay, az) = (3.0_f64, 2.5_f64, 2.0_f64);
        let (hx, hy, hz) = (
            2.0 * ax / (nx - 1) as f64,
            2.0 * ay / (ny - 1) as f64,
            2.0 * az / (nz - 1) as f64,
        );
        let r = 1.5_f64;
        let mut v = Vec::with_capacity(nx * ny * nz);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let (x, y, z) = (
                        -ax + ix as f64 * hx,
                        -ay + iy as f64 * hy,
                        -az + iz as f64 * hz,
                    );
                    v.push(-(x * x + y * y + z * z));
                }
            }
        }
        let m = marching_tetrahedra(
            &v,
            (nx, ny, nz),
            (-ax, -ay, -az),
            (hx, hy, hz),
            -r * r,
        )
        .unwrap();
        assert!(m.is_watertight());
        let exact = 4.0 / 3.0 * PI * r * r * r;
        let got = m.enclosed_volume();
        assert!((got - exact).abs() / exact < 0.02, "volume {got} vs exact {exact}");
    }

    /// A level above the field's maximum encloses nothing; below its
    /// minimum would enclose everything, but the surface then runs into
    /// the domain boundary and is not closed — so the empty case is the
    /// one worth pinning.
    #[test]
    fn a_level_outside_the_range_gives_no_surface() {
        let (v, h) = sample(15, 2.0, |x, y, z| -(x * x + y * y + z * z));
        let m = marching_tetrahedra(&v, (15, 15, 15), (-2.0, -2.0, -2.0), (h, h, h), 1.0)
            .unwrap();
        assert!(m.is_empty(), "a level above the maximum produced {} triangles", m.triangle_count());
    }

    #[test]
    fn invalid_input_is_reported() {
        let v = vec![0.0; 27];
        assert!(
            marching_tetrahedra(&v, (3, 3, 3), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), f64::NAN)
                .is_err(),
            "non-finite level"
        );
        assert!(
            marching_tetrahedra(&v, (2, 3, 3), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5).is_err(),
            "length mismatch"
        );
        let flat = vec![0.0; 9];
        assert!(
            marching_tetrahedra(&flat, (9, 1, 1), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5)
                .is_err(),
            "an axis with one point has no cells"
        );
    }
}
