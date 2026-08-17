//! The `QM3` command family: three-dimensional quantum mechanics.
//!
//! A third family alongside `QM` and `QM2`, for the same reason `QM2` is
//! separate from `QM`: the argument lists differ throughout, and a
//! hidden dimensionality mode that silently reinterprets your commands
//! would be worse than a third word.
//!
//! ```text
//! def v(x, y, z) { 0.5 * (x * x + y * y + z * z) }
//! qm3 grid -6 6 24, -6 6 24, -6 6 24
//! qm3 potential v
//! qm3 states 4            # 1.5, then 2.5 three times
//! qm3 packet -2 0 0, 1 1 1, 2 0 0
//! qm3 run 1 steps 100
//! ```
//!
//! # What is affordable
//!
//! Propagation is `O(nx*ny*nz)` per step and comfortable well past 64³.
//! `QM3 STATES` is not: the Lanczos solver reorthogonalises fully and
//! stores its whole Krylov basis, so it is practical to roughly 40³.
//! Asking for more is refused up front rather than left to exhaust
//! memory.

use quantum::isosurface::marching_tetrahedra;
use quantum::qm3d::{
    Axis, BoundStates3, DrivenPropagator3, Grid3, Hamiltonian3, Propagator3, Wavefunction3,
};

use crate::vm::{SimState, Value};

/// Every `QM3` subcommand word the parser accepts, canonical spelling only.
///
/// Input to `every_qm3_subcommand_is_documented_in_lockstep`; see
/// [`crate::qm2::QM2_SUBCOMMANDS`] for why the check runs in both
/// directions.
pub const QM3_SUBCOMMANDS: &[&str] = &[
    "status", "grid", "potential", "packet", "states", "state", "step", "run", "norm",
    "energy", "centroid", "prob", "absorb", "drive", "animate", "iso", "reset",
];

/// A `QM3` subcommand.
#[derive(Clone, Debug, PartialEq)]
pub enum Qm3Cmd {
    Status,
    /// Pops nz, z_max, z_min, ny, y_max, y_min, nx, x_max, x_min.
    Grid,
    /// `zero`, or a `DEF`ined function of three arguments.
    Potential(String),
    /// Pops kz, ky, kx, sz, sy, sx, z0, y0, x0.
    Packet,
    Step,
    Run,
    Norm,
    Energy,
    Centroid,
    /// Pops zb, za, yb, ya, xb, xa.
    Prob,
    States,
    LoadState,
    /// `f(t) g(x, y, z)`: two DEF'd function names.
    Drive(String, String),
    DriveOff,
    Absorb,
    AbsorbOff,
    Reset,
    /// Pops frames, then total time; writes an HTML page showing the
    /// three marginal densities.
    Animate(String),
    /// Pops the level fraction, the frame count, then the total time;
    /// writes an HTML page with a rotatable isosurface.
    Iso(String),
}

/// The 3-D problem carried by a session.
#[derive(Clone, Debug, Default)]
pub struct Qm3State {
    pub grid: Option<Grid3>,
    pub potential: Option<Vec<f64>>,
    pub potential_name: Option<String>,
    pub mass: f64,
    pub hbar: f64,
    pub psi: Option<Wavefunction3>,
    pub time: f64,
    pub absorber: Option<(f64, f64, f64)>,
    pub drive: Option<(Vec<f64>, String, String)>,
    pub states: Option<BoundStates3>,
}

impl Qm3State {
    pub fn fresh() -> Self {
        Self { mass: 1.0, hbar: 1.0, ..Default::default() }
    }

    fn hamiltonian(&self) -> Result<Hamiltonian3, String> {
        let grid = self.grid.clone().ok_or(
            "QM3: no grid — use `QM3 GRID <x0> <x1> <nx>, <y0> <y1> <ny>, <z0> <z1> <nz>`",
        )?;
        let v = self
            .potential
            .clone()
            .ok_or("QM3: no potential — use `QM3 POTENTIAL <function of x,y,z>` or `zero`")?;
        let h = Hamiltonian3::new(grid, v, self.mass, self.hbar)?;
        match self.absorber {
            Some((w, s, p)) => h.with_absorber(w, s, p),
            None => Ok(h),
        }
    }

    fn wavefunction(&self) -> Result<&Wavefunction3, String> {
        self.psi
            .as_ref()
            .ok_or_else(|| "QM3: no wavefunction — use `QM3 PACKET ...`".to_string())
    }
}

fn pop_num(stack: &mut Vec<Value>) -> Result<f64, String> {
    match stack.pop() {
        Some(Value::Num(n)) => Ok(n),
        Some(other) => Err(format!("QM3: expected a number, got {other}")),
        None => Err("QM3: missing an argument".to_string()),
    }
}

fn pop_count(stack: &mut Vec<Value>, what: &str) -> Result<usize, String> {
    let v = pop_num(stack)?;
    if !v.is_finite() || v.fract() != 0.0 || v < 0.0 {
        return Err(format!("QM3: {what} must be a whole number >= 0, got {v}"));
    }
    Ok(v as usize)
}

/// Above this the Lanczos basis alone runs to gigabytes, so the command
/// says so instead of trying and failing slowly.
const EIGEN_POINT_LIMIT: usize = 70_000;

/// Execute one `QM3` subcommand.
pub fn exec_qm3(
    cmd: &Qm3Cmd,
    state: &mut SimState,
    stack: &mut Vec<Value>,
) -> Result<String, String> {
    match cmd {
        Qm3Cmd::Status => {
            let q = &state.qm3;
            let mut s = String::from("quantum (3-D, ADI):\n");
            match &q.grid {
                Some(g) => s.push_str(&format!(
                    "  grid      x [{}, {}] x {}, y [{}, {}] x {}, z [{}, {}] x {}\n            \
                     {} points, h = ({:.4}, {:.4}, {:.4})\n",
                    g.x_min, g.x_max, g.nx, g.y_min, g.y_max, g.ny, g.z_min, g.z_max, g.nz,
                    g.len(), g.hx(), g.hy(), g.hz()
                )),
                None => s.push_str("  grid      (unset)\n"),
            }
            s.push_str(&match &q.potential_name {
                Some(n) => format!("  potential {n} (sampled when the command ran)\n"),
                None => "  potential (unset)\n".to_string(),
            });
            s.push_str(&format!("  mass      {}\n  hbar      {}\n", q.mass, q.hbar));
            s.push_str(&match q.absorber {
                Some((w, st, p)) => format!("  absorber  width {w}, strength {st}, power {p}\n"),
                None => "  absorber  off — all six faces REFLECT\n".to_string(),
            });
            s.push_str(&match &q.psi {
                Some(w) => {
                    let (cx, cy, cz) = w.centroid();
                    format!(
                        "  psi       set, norm {:.12}, centroid ({cx:.4}, {cy:.4}, {cz:.4}), \
                         t = {}\n",
                        w.norm(),
                        q.time
                    )
                }
                None => "  psi       (unset — QM3 PACKET)\n".to_string(),
            });
            Ok(s.trim_end().to_string())
        }

        Qm3Cmd::Grid => {
            let nz = pop_count(stack, "nz")?;
            let z_max = pop_num(stack)?;
            let z_min = pop_num(stack)?;
            let ny = pop_count(stack, "ny")?;
            let y_max = pop_num(stack)?;
            let y_min = pop_num(stack)?;
            let nx = pop_count(stack, "nx")?;
            let x_max = pop_num(stack)?;
            let x_min = pop_num(stack)?;
            let g = Grid3::new(x_min, x_max, nx, y_min, y_max, ny, z_min, z_max, nz)?;
            let n = g.len();
            let (hx, hy, hz) = (g.hx(), g.hy(), g.hz());
            state.qm3.grid = Some(g);
            state.qm3.potential = None;
            state.qm3.potential_name = None;
            state.qm3.psi = None;
            state.qm3.time = 0.0;
            state.qm3.states = None;
            Ok(format!(
                "grid {nx} x {ny} x {nz} = {n} points, h = ({hx:.6}, {hy:.6}, {hz:.6}) \
                 (potential and psi cleared)"
            ))
        }

        Qm3Cmd::Potential(name) => {
            let grid = state.qm3.grid.clone().ok_or("QM3 POTENTIAL: set a grid first")?;
            let v: Vec<f64> = if name == "zero" || name == "free" {
                vec![0.0; grid.len()]
            } else {
                if !state.functions.contains_key(name) {
                    return Err(format!(
                        "QM3 POTENTIAL: no function `{name}` — define one with \
                         `DEF {name}(x, y, z) {{ ... }}`, or use `QM3 POTENTIAL zero`"
                    ));
                }
                let mut out = Vec::with_capacity(grid.len());
                for iz in 0..grid.nz {
                    for iy in 0..grid.ny {
                        for ix in 0..grid.nx {
                            let val = crate::vm::call_user_function_public(
                                name,
                                vec![
                                    Value::Num(grid.x(ix)),
                                    Value::Num(grid.y(iy)),
                                    Value::Num(grid.z(iz)),
                                ],
                                state,
                            )?;
                            match val {
                                Value::Num(y) => out.push(y),
                                other => {
                                    return Err(format!(
                                        "QM3 POTENTIAL: `{name}(x, y, z)` must return a number, \
                                         got {other}"
                                    ))
                                }
                            }
                        }
                    }
                }
                out
            };
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let n = grid.len();
            state.qm3.potential = Some(v);
            state.qm3.potential_name = Some(name.clone());
            state.qm3.psi = None;
            state.qm3.time = 0.0;
            state.qm3.states = None;
            Ok(format!(
                "potential `{name}` sampled at {n} points, V in [{lo}, {hi}] (psi cleared)"
            ))
        }

        Qm3Cmd::Packet => {
            let kz = pop_num(stack)?;
            let ky = pop_num(stack)?;
            let kx = pop_num(stack)?;
            let sz = pop_num(stack)?;
            let sy = pop_num(stack)?;
            let sx = pop_num(stack)?;
            let z0 = pop_num(stack)?;
            let y0 = pop_num(stack)?;
            let x0 = pop_num(stack)?;
            let grid = state.qm3.grid.clone().ok_or("QM3 PACKET: set a grid first")?;
            let w = Wavefunction3::gaussian(grid, (x0, y0, z0), (sx, sy, sz), (kx, ky, kz))?;
            let edge = w.edge_probability(0.05);
            state.qm3.psi = Some(w);
            state.qm3.time = 0.0;
            let warn = if edge > 1e-6 {
                format!("\n  warning: {edge:.3e} already sits within 5% of a face")
            } else {
                String::new()
            };
            Ok(format!(
                "psi = 3-D Gaussian at ({x0}, {y0}, {z0}), sigma ({sx}, {sy}, {sz}), \
                 k ({kx}, {ky}, {kz}), t = 0{warn}"
            ))
        }

        Qm3Cmd::Step | Qm3Cmd::Run => {
            let (dt, steps) = if matches!(cmd, Qm3Cmd::Step) {
                (pop_num(stack)?, 1usize)
            } else {
                let n = pop_count(stack, "the step count")?;
                let t = pop_num(stack)?;
                if n == 0 {
                    return Err("QM3 RUN: the step count must be at least 1".to_string());
                }
                (t / n as f64, n)
            };
            if !dt.is_finite() || dt == 0.0 {
                return Err(format!("QM3: dt must be finite and non-zero, got {dt}"));
            }
            let ham = state.qm3.hamiltonian()?;
            let mut w = state.qm3.wavefunction()?.clone();
            let n0 = w.norm();
            match state.qm3.drive.clone() {
                None => {
                    Propagator3::new(ham.clone(), dt)?.run(&mut w, steps)?;
                }
                Some((shape, _, time_name)) => {
                    let mut prop = DrivenPropagator3::new(ham.clone(), shape, dt)?;
                    let t0 = state.qm3.time;
                    for k in 0..steps {
                        let mid = t0 + dt * (k as f64 + 0.5);
                        let v = crate::vm::call_user_function_public(
                            &time_name,
                            vec![Value::Num(mid)],
                            state,
                        )?;
                        let amp = match v {
                            Value::Num(y) => y,
                            other => {
                                return Err(format!(
                                    "QM3 RUN: `{time_name}(t)` must return a number, got {other}"
                                ))
                            }
                        };
                        prop.step(&mut w, |_| amp)?;
                    }
                }
            }
            let n1 = w.norm();
            let drift = (n1 / n0 - 1.0).abs();
            let edge = w.edge_probability(0.05);
            state.qm3.time += dt * steps as f64;
            let t = state.qm3.time;
            let e = w.energy(&ham);
            let absorbing = ham.is_absorbing();
            state.qm3.psi = Some(w);
            let warn = if edge > 1e-4 && !absorbing {
                format!("\n  warning: {edge:.3e} is within 5% of a face — the faces REFLECT")
            } else {
                String::new()
            };
            Ok(format!(
                "t = {t} ({steps} ADI step(s) of dt = {dt}), <E> = {e:.12}, \
                 norm drift = {drift:.3e}{warn}"
            ))
        }

        Qm3Cmd::Norm => Ok(format!("{:.15}", state.qm3.wavefunction()?.norm())),
        Qm3Cmd::Centroid => {
            let (x, y, z) = state.qm3.wavefunction()?.centroid();
            Ok(format!("[{x}, {y}, {z}]"))
        }
        Qm3Cmd::Energy => {
            let ham = state.qm3.hamiltonian()?;
            Ok(format!("{:.15}", state.qm3.wavefunction()?.energy(&ham)))
        }
        Qm3Cmd::Prob => {
            let zb = pop_num(stack)?;
            let za = pop_num(stack)?;
            let yb = pop_num(stack)?;
            let ya = pop_num(stack)?;
            let xb = pop_num(stack)?;
            let xa = pop_num(stack)?;
            Ok(format!(
                "{:.15}",
                state
                    .qm3
                    .wavefunction()?
                    .probability_in((xa, xb), (ya, yb), (za, zb))
            ))
        }

        Qm3Cmd::States => {
            let k = pop_count(stack, "the state count")?;
            if k == 0 {
                return Err("QM3 STATES: ask for at least one state".to_string());
            }
            let ham = state.qm3.hamiltonian()?;
            let n = ham.grid.len();
            if n > EIGEN_POINT_LIMIT {
                return Err(format!(
                    "QM3 STATES: {n} grid points is beyond what the eigensolver can do. Lanczos \
                     reorthogonalises fully and stores its whole Krylov basis, so cost grows as \
                     O(m^2 n) in time and O(m n) in memory — about {EIGEN_POINT_LIMIT} points \
                     (roughly 40^3) is the practical ceiling. Propagation has no such limit; use \
                     a coarser grid for the spectrum."
                ));
            }
            let b = ham.bound_states(k, 0)?;
            let mut s = format!(
                "{k} lowest bound state(s) — Lanczos, {} iterations{}:\n",
                b.iterations,
                if b.converged { "" } else { " (NOT converged)" }
            );
            for (i, e) in b.energies.iter().enumerate() {
                s.push_str(&format!("  E[{i}] = {e:.10}   residual {:.2e}\n", b.residuals[i]));
            }
            state.qm3.states = Some(b);
            Ok(s.trim_end().to_string())
        }

        Qm3Cmd::LoadState => {
            let n = pop_count(stack, "the state index")?;
            let ham = state.qm3.hamiltonian()?;
            let need = n + 1;
            let have = state.qm3.states.as_ref().map(|b| b.energies.len()).unwrap_or(0);
            if have < need {
                if ham.grid.len() > EIGEN_POINT_LIMIT {
                    return Err(format!(
                        "QM3 STATE: {} grid points is beyond the eigensolver's practical ceiling \
                         of {EIGEN_POINT_LIMIT}",
                        ham.grid.len()
                    ));
                }
                state.qm3.states = Some(ham.bound_states(need, 0)?);
            }
            let b = state.qm3.states.as_ref().expect("just filled");
            let e = b.energies[n];
            let mut w = Wavefunction3::new(
                ham.grid.clone(),
                b.states[n].iter().map(quantum::qm3d::real_to_complex).collect(),
            )?;
            w.normalise()?;
            state.qm3.psi = Some(w);
            state.qm3.time = 0.0;
            Ok(format!("psi = 3-D bound state {n}, E = {e:.10}, t reset to 0"))
        }

        Qm3Cmd::Drive(shape_name, time_name) => {
            let grid = state.qm3.grid.clone().ok_or("QM3 DRIVE: set a grid first")?;
            for nm in [shape_name, time_name] {
                if !state.functions.contains_key(nm) {
                    return Err(format!(
                        "QM3 DRIVE: no function `{nm}` — the shape is \
                         `DEF {nm}(x, y, z) {{ ... }}` and the modulation is `DEF f(t) {{ ... }}`"
                    ));
                }
            }
            let mut shape = Vec::with_capacity(grid.len());
            for iz in 0..grid.nz {
                for iy in 0..grid.ny {
                    for ix in 0..grid.nx {
                        let v = crate::vm::call_user_function_public(
                            shape_name,
                            vec![
                                Value::Num(grid.x(ix)),
                                Value::Num(grid.y(iy)),
                                Value::Num(grid.z(iz)),
                            ],
                            state,
                        )?;
                        match v {
                            Value::Num(y) => shape.push(y),
                            other => {
                                return Err(format!(
                                    "QM3 DRIVE: `{shape_name}(x, y, z)` must return a number, \
                                     got {other}"
                                ))
                            }
                        }
                    }
                }
            }
            state.qm3.drive = Some((shape, shape_name.clone(), time_name.clone()));
            state.qm3.states = None;
            Ok(format!(
                "drive V(x,y,z,t) += {time_name}(t) * {shape_name}(x,y,z). Energy is NO LONGER \
                 conserved; propagation stays unitary. QM3 STATES uses the STATIC potential."
            ))
        }

        Qm3Cmd::DriveOff => {
            state.qm3.drive = None;
            state.qm3.states = None;
            Ok("drive removed".to_string())
        }

        Qm3Cmd::Absorb => {
            let power = pop_num(stack)?;
            let strength = pop_num(stack)?;
            let width = pop_num(stack)?;
            if let Some(g) = state.qm3.grid.clone() {
                let probe = Hamiltonian3::new(g.clone(), vec![0.0; g.len()], 1.0, 1.0)?;
                probe.with_absorber(width, strength, power)?;
            }
            state.qm3.absorber = Some((width, strength, power));
            state.qm3.states = None;
            Ok(format!(
                "absorbing faces on all six sides: width {width}, strength {strength}, \
                 power {power}. Propagation is no longer unitary — the norm decays by design."
            ))
        }
        Qm3Cmd::AbsorbOff => {
            state.qm3.absorber = None;
            state.qm3.states = None;
            Ok("absorbing faces removed — all six faces reflect again".to_string())
        }

        Qm3Cmd::Animate(path) => {
            let frames = pop_count(stack, "the frame count")?;
            let total = pop_num(stack)?;
            if frames < 2 {
                return Err("QM3 ANIMATE: ask for at least 2 frames".to_string());
            }
            if !total.is_finite() || total <= 0.0 {
                return Err(format!("QM3 ANIMATE: total time must be positive, got {total}"));
            }
            let ham = state.qm3.hamiltonian()?;
            let mut w = state.qm3.wavefunction()?.clone();
            let g = ham.grid.clone();
            let per_frame = 8usize;
            let dt = total / (frames * per_frame) as f64;
            // See the note in qm.rs: ignoring the drive here made every
            // frame identical, which is a silently wrong picture.
            let drive = state.qm3.drive.clone();
            let prop = Propagator3::new(ham.clone(), dt)?;
            let mut driven = match &drive {
                Some((shape, _, _)) => {
                    Some(DrivenPropagator3::new(ham.clone(), shape.clone(), dt)?)
                }
                None => None,
            };
            let t_start = state.qm3.time;

            // Three MARGINALS per frame rather than the volume: a volume
            // cannot be drawn on a 2-D canvas without an isosurface or
            // ray-caster, and P(x,y) = integral |psi|^2 dz is a genuine
            // observable rather than a rendering convention. Three of
            // them determine a great deal about where the packet is.
            let n0 = w.norm();
            let mut worst = 0.0_f64;
            let mut xy = Vec::with_capacity(frames);
            let mut xz = Vec::with_capacity(frames);
            let mut yz = Vec::with_capacity(frames);
            let mut times = Vec::with_capacity(frames);
            let enc = |v: &[f64]| {
                v.iter().map(|x| format!("{x:.5e}")).collect::<Vec<_>>().join(",")
            };
            for f in 0..frames {
                if f > 0 {
                    advance_3d(
                        &mut w,
                        &prop,
                        driven.as_mut(),
                        drive.as_ref().map(|(_, _, t)| t.as_str()),
                        state,
                        per_frame,
                        dt,
                        t_start + dt * ((f - 1) * per_frame) as f64,
                    )?;
                }
                worst = worst.max((w.norm() / n0 - 1.0).abs());
                xy.push(format!("[{}]", enc(&w.marginal(Axis::Z))));
                xz.push(format!("[{}]", enc(&w.marginal(Axis::Y))));
                yz.push(format!("[{}]", enc(&w.marginal(Axis::X))));
                times.push(format!("{:.4}", state.qm3.time + dt * (f * per_frame) as f64));
            }
            state.qm3.time += total;
            state.qm3.psi = Some(w);

            let label = state
                .qm3
                .potential_name
                .clone()
                .unwrap_or_else(|| "unnamed".to_string());
            let html = render_html_3d(
                &xy.join(",\n"),
                &xz.join(",\n"),
                &yz.join(",\n"),
                &times.join(","),
                &label,
                (g.nx, g.ny, g.nz),
                (g.x_min, g.x_max, g.y_min, g.y_max, g.z_min, g.z_max),
            );
            std::fs::write(path, &html)
                .map_err(|e| format!("QM3 ANIMATE: cannot write `{path}`: {e}"))?;
            Ok(format!(
                "wrote {path} — {frames} frames over t = {total} (dt = {dt:.6}), three marginal \
                 densities per frame, worst norm drift {worst:.3e}. Open it in a browser."
            ))
        }

        Qm3Cmd::Iso(path) => {
            let level_frac = pop_num(stack)?;
            let frames = pop_count(stack, "the frame count")?;
            let total = pop_num(stack)?;
            if frames == 0 {
                return Err("QM3 ISO: ask for at least one frame".to_string());
            }
            if !total.is_finite() || total <= 0.0 {
                return Err(format!("QM3 ISO: total time must be positive, got {total}"));
            }
            if !level_frac.is_finite() || level_frac <= 0.0 || level_frac >= 1.0 {
                return Err(format!(
                    "QM3 ISO: the level must be a fraction of the peak density, strictly \
                     between 0 and 1, got {level_frac}"
                ));
            }
            let ham = state.qm3.hamiltonian()?;
            let mut w = state.qm3.wavefunction()?.clone();
            let g = ham.grid.clone();
            let per_frame = 8usize;
            let dt = total / (frames * per_frame).max(1) as f64;
            // See the note in qm.rs: ignoring the drive here made every
            // frame identical, which is a silently wrong picture.
            let drive = state.qm3.drive.clone();
            let prop = Propagator3::new(ham.clone(), dt)?;
            let mut driven = match &drive {
                Some((shape, _, _)) => {
                    Some(DrivenPropagator3::new(ham.clone(), shape.clone(), dt)?)
                }
                None => None,
            };
            let t_start = state.qm3.time;

            // Meshing every grid point would put megabytes of triangles
            // in the page. Subsample to at most this many per axis; the
            // eye cannot use more through an orthographic projection.
            const MAX_AXIS: usize = 28;
            let sx = g.nx.div_ceil(MAX_AXIS).max(1);
            let sy = g.ny.div_ceil(MAX_AXIS).max(1);
            let sz = g.nz.div_ceil(MAX_AXIS).max(1);
            let cols: Vec<usize> = (0..g.nx).step_by(sx).collect();
            let rows: Vec<usize> = (0..g.ny).step_by(sy).collect();
            let laps: Vec<usize> = (0..g.nz).step_by(sz).collect();
            let dims = (cols.len(), rows.len(), laps.len());
            if dims.0 < 2 || dims.1 < 2 || dims.2 < 2 {
                return Err("QM3 ISO: the grid is too small to build a surface".to_string());
            }

            let mut meshes = Vec::with_capacity(frames);
            let mut times = Vec::with_capacity(frames);
            let mut tri_total = 0usize;
            for f in 0..frames {
                if f > 0 {
                    advance_3d(
                        &mut w,
                        &prop,
                        driven.as_mut(),
                        drive.as_ref().map(|(_, _, t)| t.as_str()),
                        state,
                        per_frame,
                        dt,
                        t_start + dt * ((f - 1) * per_frame) as f64,
                    )?;
                }
                let d = w.density();
                let mut sub = Vec::with_capacity(dims.0 * dims.1 * dims.2);
                let mut peak = 0.0_f64;
                for &iz in &laps {
                    for &iy in &rows {
                        for &ix in &cols {
                            let v = d[g.idx(ix, iy, iz)];
                            peak = peak.max(v);
                            sub.push(v);
                        }
                    }
                }
                let m = marching_tetrahedra(
                    &sub,
                    dims,
                    (g.x(cols[0]), g.y(rows[0]), g.z(laps[0])),
                    (
                        g.hx() * sx as f64,
                        g.hy() * sy as f64,
                        g.hz() * sz as f64,
                    ),
                    peak * level_frac,
                )?;
                tri_total += m.triangle_count();
                let verts = m
                    .vertices
                    .iter()
                    .map(|p| format!("{:.3},{:.3},{:.3}", p[0], p[1], p[2]))
                    .collect::<Vec<_>>()
                    .join(",");
                let tris = m
                    .triangles
                    .iter()
                    .map(|t| format!("{},{},{}", t[0], t[1], t[2]))
                    .collect::<Vec<_>>()
                    .join(",");
                meshes.push(format!("{{v:[{verts}],t:[{tris}]}}"));
                times.push(format!("{:.4}", state.qm3.time + dt * (f * per_frame) as f64));
            }
            state.qm3.time += total;
            state.qm3.psi = Some(w);

            let span = (g.x_max - g.x_min)
                .max(g.y_max - g.y_min)
                .max(g.z_max - g.z_min);
            let label = state
                .qm3
                .potential_name
                .clone()
                .unwrap_or_else(|| "unnamed".to_string());
            let html = render_html_iso(
                &meshes.join(",\n"),
                &times.join(","),
                &label,
                level_frac,
                span,
                dims,
            );
            std::fs::write(path, &html)
                .map_err(|e| format!("QM3 ISO: cannot write `{path}`: {e}"))?;
            Ok(format!(
                "wrote {path} — {frames} isosurface(s) at {:.0}% of peak density over t = {total}, \
                 meshed on {}x{}x{}, {tri_total} triangles total. Drag to rotate.",
                level_frac * 100.0,
                dims.0,
                dims.1,
                dims.2
            ))
        }

        Qm3Cmd::Reset => {
            state.qm3 = Qm3State::fresh();
            Ok("3-D quantum state cleared".to_string())
        }
    }
}

/// Advance a 3-D wavefunction by `steps`, honouring a drive if set.
#[allow(clippy::too_many_arguments)]
fn advance_3d(
    w: &mut Wavefunction3,
    prop: &Propagator3,
    driven: Option<&mut DrivenPropagator3>,
    time_name: Option<&str>,
    state: &mut SimState,
    steps: usize,
    dt: f64,
    t0: f64,
) -> Result<(), String> {
    match (driven, time_name) {
        (Some(dp), Some(name)) => {
            for k in 0..steps {
                let mid = t0 + dt * (k as f64 + 0.5);
                let v = crate::vm::call_user_function_public(
                    name,
                    vec![Value::Num(mid)],
                    state,
                )?;
                let amp = match v {
                    Value::Num(y) => y,
                    other => {
                        return Err(format!("`{name}(t)` must return a number, got {other}"))
                    }
                };
                dp.step(w, |_| amp)?;
            }
            Ok(())
        }
        _ => prop.run(w, steps),
    }
}

/// The three-marginal animation page, self-contained.
///
/// A volume cannot be drawn on a 2-D canvas without an isosurface mesh
/// or a ray-caster, both of which would mean shipping a WebGL pipeline
/// inside a file that must work from `file://`. Three marginal
/// densities are the honest alternative: each is a real observable, and
/// together they locate the packet on every axis.
#[allow(clippy::too_many_arguments)]
fn render_html_3d(
    xy: &str,
    xz: &str,
    yz: &str,
    times: &str,
    label: &str,
    dims: (usize, usize, usize),
    bounds: (f64, f64, f64, f64, f64, f64),
) -> String {
    let (nx, ny, nz) = dims;
    let (x0, x1, y0, y1, z0, z1) = bounds;
    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>posim — 3-D quantum, marginal densities</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
        background:#0e1116; color:#e6e6e6; }}
 header {{ padding:14px 18px; border-bottom:1px solid #263042; }}
 h1 {{ margin:0; font-size:16px; font-weight:600; }}
 .sub {{ color:#8b98ad; font-size:12px; margin-top:3px; }}
 #wrap {{ padding:14px 18px; }}
 .panels {{ display:flex; gap:16px; flex-wrap:wrap; }}
 .panel {{ flex:1 1 260px; min-width:220px; }}
 .cap {{ font-size:12px; color:#9fb3d0; margin-bottom:5px;
         font-variant-numeric:tabular-nums; }}
 canvas {{ width:100%; height:auto; display:block; image-rendering:pixelated;
           background:#11151c; border:1px solid #263042; border-radius:6px; }}
 .row {{ display:flex; gap:12px; align-items:center; margin-top:14px; flex-wrap:wrap; }}
 button {{ background:#1b2330; color:#e6e6e6; border:1px solid #33405a;
           border-radius:5px; padding:6px 14px; cursor:pointer; font:inherit; }}
 button:hover {{ background:#243149; }}
 input[type=range] {{ flex:1; min-width:200px; }}
 .stat {{ font-variant-numeric:tabular-nums; color:#9fb3d0; }}
 .note {{ margin-top:10px; font-size:12px; color:#8b98ad; max-width:60em; }}
</style></head><body>
<header>
  <h1>3-D quantum — marginal probability densities</h1>
  <div class="sub">potential: <b>{label}</b> &middot; grid {nx}&times;{ny}&times;{nz}
  &middot; ADI (Strang-split Cayley, exactly unitary) &middot; generated by posim</div>
</header>
<div id="wrap">
  <div class="panels">
    <div class="panel"><div class="cap" id="cxy"></div><canvas id="a" width="{nx}" height="{ny}"></canvas></div>
    <div class="panel"><div class="cap" id="cxz"></div><canvas id="b" width="{nx}" height="{nz}"></canvas></div>
    <div class="panel"><div class="cap" id="cyz"></div><canvas id="c" width="{ny}" height="{nz}"></canvas></div>
  </div>
  <div class="row">
    <button id="play">Pause</button>
    <button id="rew">Restart</button>
    <input type="range" id="scrub" min="0" value="0">
    <span class="stat" id="stat"></span>
  </div>
  <div class="note">
    Each panel is a genuine observable, not a projection trick:
    P(x,y) = &int;|&psi;|&sup2; dz is the probability of finding the particle at
    (x,y) whatever its z. Each integrates to the total norm. Brightness is
    scaled per panel per frame, so panels show <em>shape</em>, not relative
    weight — the norms in the captions carry that.
  </div>
</div>
<script>
const XY=[
{xy}
], XZ=[
{xz}
], YZ=[
{yz}
], T=[{times}];
const NX={nx}, NY={ny}, NZ={nz};
const B={{x0:{x0}, x1:{x1}, y0:{y0}, y1:{y1}, z0:{z0}, z1:{z1}}};
const panels = [
  {{cv:'a', cap:'cxy', data:XY, w:NX, h:NY, name:'P(x, y)  = int |psi|^2 dz'}},
  {{cv:'b', cap:'cxz', data:XZ, w:NX, h:NZ, name:'P(x, z)  = int |psi|^2 dy'}},
  {{cv:'c', cap:'cyz', data:YZ, w:NY, h:NZ, name:'P(y, z)  = int |psi|^2 dx'}},
];
const dx=(B.x1-B.x0)/(NX+1), dy=(B.y1-B.y0)/(NY+1), dz=(B.z1-B.z0)/(NZ+1);
const cells=[dx*dy, dx*dz, dy*dz];
const scrub=document.getElementById('scrub'); scrub.max=T.length-1;
let i=0, playing=true;

function drawPanel(p, k) {{
  const cv=document.getElementById(p.cv), g=cv.getContext('2d');
  const f=p.data[i];
  let m=0, tot=0;
  for (const v of f) {{ if (v>m) m=v; tot+=v; }}
  if (m<=0) m=1;
  const img=g.createImageData(p.w, p.h);
  for (let r=0;r<p.h;r++) for (let c=0;c<p.w;c++) {{
    const src=r*p.w+c;
    const dst=((p.h-1-r)*p.w+c)*4;      // flip so the second axis points up
    const t=Math.sqrt(f[src]/m);        // sqrt makes the tails visible
    img.data[dst  ]=Math.min(255, 40*t + 215*t*t*t);
    img.data[dst+1]=Math.min(255, 90*t + 165*t*t*t);
    img.data[dst+2]=Math.min(255, 200*t + 55*t*t*t);
    img.data[dst+3]=255;
  }}
  g.putImageData(img,0,0);
  document.getElementById(p.cap).textContent =
    `${{p.name}}   norm ${{(tot*cells[k]).toFixed(6)}}`;
}}
function draw() {{
  panels.forEach(drawPanel);
  document.getElementById('stat').textContent =
    `t = ${{(+T[i]).toFixed(2)}}   frame ${{i+1}}/${{T.length}}`;
  scrub.value=i;
}}
let last=0;
function loop(ts) {{
  if (playing && ts-last>60) {{ i=(i+1)%T.length; last=ts; draw(); }}
  requestAnimationFrame(loop);
}}
document.getElementById('play').onclick=e=>{{playing=!playing;e.target.textContent=playing?'Pause':'Play';}};
document.getElementById('rew').onclick=()=>{{i=0;draw();}};
scrub.oninput=e=>{{i=+e.target.value;playing=false;
  document.getElementById('play').textContent='Play';draw();}};
draw(); requestAnimationFrame(loop);
</script></body></html>
"##
    )
}

/// The rotatable isosurface page, self-contained.
///
/// Rendered by a **software rasteriser on a 2-D canvas**, not WebGL.
/// WebGL would be faster and is technically self-contained, but it can
/// fail silently on a machine with no GPU or a blocked context, and then
/// the page shows nothing at all. A painter's-algorithm rasteriser over
/// a few thousand triangles is fast enough here, works everywhere, and
/// can be checked by reading pixels back.
#[allow(clippy::too_many_arguments)]
fn render_html_iso(
    meshes: &str,
    times: &str,
    label: &str,
    level_frac: f64,
    span: f64,
    dims: (usize, usize, usize),
) -> String {
    let (dx, dy, dz) = dims;
    let pct = level_frac * 100.0;
    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>posim — 3-D isosurface</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
        background:#0e1116; color:#e6e6e6; }}
 header {{ padding:14px 18px; border-bottom:1px solid #263042; }}
 h1 {{ margin:0; font-size:16px; font-weight:600; }}
 .sub {{ color:#8b98ad; font-size:12px; margin-top:3px; }}
 #wrap {{ padding:14px 18px; }}
 canvas {{ width:100%; max-width:680px; height:auto; display:block; cursor:grab;
           background:#11151c; border:1px solid #263042; border-radius:6px; }}
 canvas:active {{ cursor:grabbing; }}
 .row {{ display:flex; gap:12px; align-items:center; margin-top:12px; flex-wrap:wrap; }}
 button {{ background:#1b2330; color:#e6e6e6; border:1px solid #33405a;
           border-radius:5px; padding:6px 14px; cursor:pointer; font:inherit; }}
 button:hover {{ background:#243149; }}
 input[type=range] {{ flex:1; min-width:180px; }}
 .stat {{ font-variant-numeric:tabular-nums; color:#9fb3d0; }}
 .note {{ margin-top:10px; font-size:12px; color:#8b98ad; max-width:60em; }}
</style></head><body>
<header>
  <h1>3-D isosurface — |&psi;|&sup2; at {pct:.0}% of peak</h1>
  <div class="sub">potential: <b>{label}</b> &middot; meshed on {dx}&times;{dy}&times;{dz}
  by marching tetrahedra &middot; generated by posim</div>
</header>
<div id="wrap">
  <canvas id="c" width="680" height="520"></canvas>
  <div class="row">
    <button id="play">Pause</button>
    <button id="reset">Reset view</button>
    <input type="range" id="scrub" min="0" value="0">
    <span class="stat" id="stat"></span>
  </div>
  <div class="note">
    Drag to rotate. The surface encloses the region where the probability
    density exceeds {pct:.0}&nbsp;% of its peak in that frame, so it tracks
    the packet's <em>shape</em> rather than its absolute weight. Meshes are
    watertight and consistently oriented; shading is Lambertian from the
    triangle normals.
  </div>
</div>
<script>
const M=[
{meshes}
];
const T=[{times}];
const SPAN={span};
const c=document.getElementById('c'), g=c.getContext('2d');
const scrub=document.getElementById('scrub'); scrub.max=M.length-1;
let i=0, playing=M.length>1, yaw=0.6, pitch=-0.35, drag=null;

function rot(p) {{
  const [x,y,z]=p;
  const cy=Math.cos(yaw), sy=Math.sin(yaw);
  const x1=x*cy - z*sy, z1=x*sy + z*cy;
  const cp=Math.cos(pitch), sp=Math.sin(pitch);
  const y2=y*cp - z1*sp, z2=y*sp + z1*cp;
  return [x1, y2, z2];
}}

function draw() {{
  const m=M[i];
  g.fillStyle='#11151c'; g.fillRect(0,0,c.width,c.height);
  const s=Math.min(c.width, c.height) / (SPAN*1.15);
  const ox=c.width/2, oy=c.height/2;
  const nv=m.v.length/3;
  const px=new Float64Array(nv), py=new Float64Array(nv), pz=new Float64Array(nv);
  for (let k=0;k<nv;k++) {{
    const r=rot([m.v[3*k], m.v[3*k+1], m.v[3*k+2]]);
    px[k]=ox + r[0]*s; py[k]=oy - r[1]*s; pz[k]=r[2];
  }}
  const nt=m.t.length/3;
  const order=new Array(nt);
  for (let k=0;k<nt;k++) {{
    const a=m.t[3*k], b=m.t[3*k+1], d=m.t[3*k+2];
    order[k]=[k, (pz[a]+pz[b]+pz[d])/3];
  }}
  order.sort((p,q)=>p[1]-q[1]);           // painter's algorithm: far first
  const L=[0.4,0.6,0.7];                  // light direction, normalised below
  const ln=Math.hypot(L[0],L[1],L[2]);
  let drawn=0;
  for (const [k] of order) {{
    const a=m.t[3*k], b=m.t[3*k+1], d=m.t[3*k+2];
    const ux=px[b]-px[a], uy=py[b]-py[a], uz=pz[b]-pz[a];
    const vx=px[d]-px[a], vy=py[d]-py[a], vz=pz[d]-pz[a];
    let nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    const nl=Math.hypot(nx,ny,nz); if (nl===0) continue;
    nx/=nl; ny/=nl; nz/=nl;
    if (nz>0) continue;                   // back-face cull (screen y is flipped)
    const lam=Math.abs((nx*L[0]+ny*L[1]+nz*L[2])/ln);
    const t=0.18+0.82*lam;
    g.fillStyle=`rgb(${{Math.round(50+150*t)}},${{Math.round(90+140*t)}},${{Math.round(150+105*t)}})`;
    g.beginPath();
    g.moveTo(px[a],py[a]); g.lineTo(px[b],py[b]); g.lineTo(px[d],py[d]); g.closePath();
    g.fill();
    drawn++;
  }}
  document.getElementById('stat').textContent =
    `t = ${{(+T[i]).toFixed(2)}}   frame ${{i+1}}/${{M.length}}   ` +
    `${{nt}} triangles (${{drawn}} front-facing)`;
  scrub.value=i;
}}

c.addEventListener('pointerdown', e => {{ drag={{x:e.clientX, y:e.clientY}}; c.setPointerCapture(e.pointerId); }});
c.addEventListener('pointermove', e => {{
  if (!drag) return;
  yaw += (e.clientX-drag.x)*0.01; pitch += (e.clientY-drag.y)*0.01;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  drag={{x:e.clientX, y:e.clientY}}; draw();
}});
c.addEventListener('pointerup', e => {{ drag=null; c.releasePointerCapture(e.pointerId); }});

let last=0;
function loop(ts) {{
  if (playing && M.length>1 && ts-last>90) {{ i=(i+1)%M.length; last=ts; draw(); }}
  requestAnimationFrame(loop);
}}
document.getElementById('play').onclick=e=>{{playing=!playing;e.target.textContent=playing?'Pause':'Play';}};
document.getElementById('reset').onclick=()=>{{yaw=0.6;pitch=-0.35;draw();}};
scrub.oninput=e=>{{i=+e.target.value;playing=false;
  document.getElementById('play').textContent='Play';draw();}};
draw(); requestAnimationFrame(loop);
</script></body></html>
"##
    )
}

#[cfg(test)]
mod tests {
    /// Words the `qm3cmd` EBNF quotes that are arguments, not subcommands.
    const QM3_ARGUMENT_WORDS: &[&str] = &["ZERO", "OFF", "STEPS", "FRAMES", "LEVEL", "STATUS"];


    /// Every subcommand of this family must appear in the parser, in
    /// `HELP_TEXT`, and in both grammar documents — **and the EBNF must
    /// promise nothing the parser does not implement.**
    ///
    /// That second direction is why this test exists. `QM` has had the
    /// forward check since Stage 2A and has stayed clean; `QM2` and
    /// `QM3` had no check at all, and the `qm2cmd` EBNF advertised an
    /// `ISO` production for which `qm2_command` has no arm. Typing it
    /// errors. A forward-only check could never have found that: it
    /// asks whether the documents mention what the code does, never
    /// whether the code does what the documents promise.
    #[test]
    fn every_qm3_subcommand_is_documented_in_lockstep() {
        // `\_` in LaTeX, `\` nowhere else; case differs between the
        // parser (lowercase) and the documents (upper).
        let prep = |s: &str| s.replace('\\', "").to_ascii_uppercase();
        let help = prep(crate::vm::HELP_TEXT);
        // Only this family's production: the three qm*cmd blocks quote
        // the same words, so searching the whole file would pass on a
        // word this family never declared.
        let all = prep(include_str!("parser.rs"));
        let from = all.find("QM3CMD").expect("parser.rs must declare the qm3cmd production");
        let to = all[from..].find("UNARY ").map_or(all.len(), |k| from + k);
        let ebnf = all[from..to].to_string();
        let md = prep(include_str!("../../grammar.md"));
        let tex = prep(include_str!("../../grammar.tex"));

        let mut missing = Vec::new();

        // Forward: everything the parser accepts is documented.
        for w in super::QM3_SUBCOMMANDS {
            let up = w.to_ascii_uppercase();
            let needle = format!("QM3 {up}");
            let ebnf_needle = format!("\"{up}\"");
            for (what, hay) in [
                ("HELP_TEXT", &help),
                ("parser.rs EBNF", &ebnf),
                ("grammar.md", &md),
                ("grammar.tex", &tex),
            ] {
                // `QM3 STATUS` is spelled as bare `QM3` in the
                // documents; the EBNF writes the optional `[ "STATUS" ]`.
                if *w == "status" && what != "parser.rs EBNF" {
                    continue;
                }
                let want = if what == "parser.rs EBNF" { &ebnf_needle } else { &needle };
                if !hay.contains(want.as_str()) {
                    missing.push(format!("{want} is missing from {what}"));
                }
            }
        }

        // Converse: the EBNF promises nothing the parser lacks. Every
        // quoted word is either a declared subcommand or a declared
        // argument word — a new quoted word must be classified as one or
        // the other, deliberately.
        let mut phantom = Vec::new();
        let mut rest = ebnf.as_str();
        while let Some(i) = rest.find('"') {
            rest = &rest[i + 1..];
            let Some(j) = rest.find('"') else { break };
            let word = &rest[..j];
            rest = &rest[j + 1..];
            if word.is_empty() || !word.chars().all(|c| c.is_ascii_uppercase() || c == '_') {
                continue; // punctuation like "," or a metavariable
            }
            let known = super::QM3_SUBCOMMANDS.iter().any(|s| s.eq_ignore_ascii_case(word))
                || QM3_ARGUMENT_WORDS.contains(&word);
            if !known {
                phantom.push(format!(
                    "the qm3cmd EBNF quotes `{word}`, which is neither a declared \
                     subcommand nor a declared argument word — either the parser is \
                     missing an arm or the comment is promising a command that does \
                     not exist"
                ));
            }
        }

        missing.extend(phantom);
        assert!(
            missing.is_empty(),
            "QM3 grammar lockstep is broken:\n  {}",
            missing.join("\n  ")
        );
    }

    use crate::vm::{execute_line, SimState};

    fn run(lines: &[&str]) -> (SimState, Vec<String>) {
        let mut st = SimState::default();
        let mut out = Vec::new();
        for l in lines {
            let v = execute_line(l, &mut st).unwrap_or_else(|e| panic!("`{l}` failed: {e}"));
            out.push(v.to_string());
        }
        (st, out)
    }

    /// A potential of three arguments, sampled with the right axes. A
    /// NON-CUBIC grid, because an axis mix-up is invisible on a cube.
    #[test]
    fn a_three_argument_potential_is_sampled_with_correct_axes() {
        let (st, _) = run(&[
            "def v(x, y, z) { x + 10 * y + 100 * z }",
            "qm3 grid -3 3 9, -2 2 6, -1 1 4",
            "qm3 potential v",
        ]);
        let g = st.qm3.grid.as_ref().unwrap();
        assert_eq!((g.nx, g.ny, g.nz), (9, 6, 4));
        let p = st.qm3.potential.as_ref().unwrap();
        assert_eq!(p.len(), 9 * 6 * 4);
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let want = g.x(ix) + 10.0 * g.y(iy) + 100.0 * g.z(iz);
                    let got = p[g.idx(ix, iy, iz)];
                    assert!(
                        (got - want).abs() < 1e-10,
                        "V({ix},{iy},{iz}) = {got}, want {want}"
                    );
                }
            }
        }
    }

    /// ADI in 3-D is exactly unitary.
    #[test]
    fn propagation_conserves_the_norm() {
        let (_, out) = run(&[
            "qm3 grid -6 6 20, -6 6 20, -6 6 20",
            "qm3 potential zero",
            "qm3 packet -2 0 0, 1 1 1, 1 0 0",
            "qm3 run 0.5 steps 50",
            "qm3 norm",
        ]);
        let norm: f64 = out[4].trim().parse().unwrap();
        assert!((norm - 1.0).abs() < 1e-10, "norm = {norm}");
    }

    /// The packet moves along the axis it was given momentum on, and
    /// only that one.
    #[test]
    fn a_packet_drifts_along_its_momentum() {
        let (st, _) = run(&[
            "qm3 grid -12 12 32, -12 12 32, -12 12 32",
            "qm3 potential zero",
            "qm3 packet -4 0 0, 1.5 1.5 1.5, 2 0 0",
            "qm3 run 1.5 steps 150",
        ]);
        let (x, y, z) = st.qm3.psi.as_ref().unwrap().centroid();
        assert!(x > -4.0 + 1.0, "<x> = {x}, should have moved");
        assert!(y.abs() < 1e-9, "<y> = {y}, must not move");
        assert!(z.abs() < 1e-9, "<z> = {z}, must not move");
    }

    /// The 3-D oscillator spectrum, with its three-fold degenerate first
    /// excited level, through the language.
    #[test]
    fn the_oscillator_spectrum_through_the_language() {
        let (_, out) = run(&[
            "def v(x, y, z) { 0.5 * (x * x + y * y + z * z) }",
            "qm3 grid -5.5 5.5 20, -5.5 5.5 20, -5.5 5.5 20",
            "qm3 potential v",
            "qm3 states 4",
        ]);
        let vals: Vec<f64> = out[3]
            .lines()
            .filter_map(|l| l.split('=').nth(1))
            .filter_map(|v| v.split_whitespace().next())
            .filter_map(|v| v.parse().ok())
            .collect();
        assert_eq!(vals.len(), 4, "expected four energies in:\n{}", out[3]);
        // E = 3/2 then 5/2 three times, up to the grid's discretisation
        assert!((vals[0] - 1.5).abs() < 0.06, "E0 = {}", vals[0]);
        for v in &vals[1..] {
            assert!((v - 2.5).abs() < 0.08, "excited level {v}, want ~2.5");
        }
        // the triplet must be degenerate with ITSELF far more tightly
        assert!(
            (vals[3] - vals[1]).abs() < 1e-8,
            "the triplet split by {}",
            vals[3] - vals[1]
        );
    }

    /// Past the eigensolver's ceiling the command must refuse up front
    /// rather than exhaust memory slowly.
    #[test]
    fn a_too_large_grid_is_refused_for_eigenstates() {
        let mut st = SimState::default();
        execute_line("qm3 grid -6 6 60, -6 6 60, -6 6 60", &mut st).unwrap();
        execute_line("qm3 potential zero", &mut st).unwrap();
        let e = execute_line("qm3 states 2", &mut st).unwrap_err();
        assert!(e.contains("ceiling") || e.contains("beyond"), "got: {e}");
        // ...but propagation on the same grid is fine
        execute_line("qm3 packet 0 0 0, 1 1 1, 1 0 0", &mut st).unwrap();
        assert!(execute_line("qm3 run 0.05 steps 2", &mut st).is_ok());
    }

    /// The 3-D drive must move only the axis it acts on.
    #[test]
    fn a_drive_moves_only_its_own_axis() {
        let (st, out) = run(&[
            "def v(x, y, z) { 0.5 * (x * x + y * y + z * z) }",
            "def dip(x, y, z) { x }",
            "def f(t) { 0.6 * cos(0.7 * t) }",
            "qm3 grid -7 7 26, -7 7 26, -7 7 26",
            "qm3 potential v",
            "qm3 state 0",
            "qm3 drive dip, f",
            "qm3 run 5 steps 250",
            "qm3 norm",
        ]);
        let (x, y, z) = st.qm3.psi.as_ref().unwrap().centroid();
        assert!(x > 0.5, "<x> = {x}, the drive should have displaced it");
        assert!(y.abs() < 1e-8, "<y> = {y}, must not move");
        assert!(z.abs() < 1e-8, "<z> = {z}, must not move");
        let norm: f64 = out[8].trim().parse().unwrap();
        assert!((norm - 1.0).abs() < 1e-10, "driven but must stay unitary: {norm}");
    }

    /// `QM3 ANIMATE` writes a self-contained page whose marginals are
    /// probability densities. Checked by parsing the numbers back out
    /// of the file rather than trusting that it was written.
    #[test]
    fn animate_writes_marginals_that_integrate_to_one() {
        let dir = std::env::temp_dir().join("posim_qm3_anim_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v.html");
        let p = path.to_string_lossy().to_string();
        let (_, out) = run(&[
            "qm3 grid -6 6 16, -6 6 16, -6 6 16",
            "qm3 potential zero",
            "qm3 packet -1 0 0, 1 1 1, 1 0 0",
            &format!("qm3 animate \"{p}\" 0.4 frames 3"),
        ]);
        assert!(out[3].contains("marginal densities"), "got: {}", out[3]);
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.starts_with("<!DOCTYPE html>"));
        // nothing fetched from the network
        assert!(!html.contains("http://") && !html.contains("https://"), "external reference");
        // the first XY marginal must sum to 1 / (dx dy)
        let body = html.split("const XY=[").nth(1).unwrap();
        let first = body.split(']').next().unwrap().trim_start_matches('\n').trim_start_matches('[');
        let vals: Vec<f64> = first
            .split(',')
            .filter_map(|t| t.trim().parse::<f64>().ok())
            .collect();
        assert_eq!(vals.len(), 16 * 16, "wrong marginal size: {}", vals.len());
        let h = 12.0 / 17.0;
        let total: f64 = vals.iter().sum::<f64>() * h * h;
        assert!((total - 1.0).abs() < 1e-4, "marginal integrates to {total}, want 1");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **An animation must honour the drive.** The first version of
    /// ISO and ANIMATE used the static propagator, so with a drive set
    /// every frame came out identical — a silently wrong picture, which
    /// is worse than an error. Caught by checking that the isosurface
    /// centroid actually moved; it did not.
    #[test]
    fn iso_and_animate_honour_the_drive() {
        let dir = std::env::temp_dir().join("posim_qm3_drive_anim");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("i.html").to_string_lossy().to_string();
        run(&[
            "def v(x, y, z) { 0.5 * (x * x + y * y + z * z) }",
            "def dip(x, y, z) { x }",
            "def f(t) { 0.8 * cos(0.7 * t) }",
            "qm3 grid -7 7 24, -7 7 24, -7 7 24",
            "qm3 potential v",
            "qm3 state 0",
            "qm3 drive dip, f",
            &format!("qm3 iso \"{p}\" 6 frames 6 level 0.2"),
        ]);
        let html = std::fs::read_to_string(&p).unwrap();
        // pull each frame's vertex list and compare mean x
        let mut xs = Vec::new();
        for chunk in html.split("{v:[").skip(1) {
            let verts = chunk.split(']').next().unwrap();
            let n: Vec<f64> = verts
                .split(',')
                .filter_map(|t| t.trim().parse::<f64>().ok())
                .collect();
            if n.len() < 9 {
                continue;
            }
            let mean_x: f64 = n.iter().step_by(3).sum::<f64>() / (n.len() / 3) as f64;
            xs.push(mean_x);
        }
        assert!(xs.len() >= 4, "expected several frames, parsed {}", xs.len());
        let spread = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - xs.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            spread > 0.2,
            "the isosurface did not move under a drive: mean x spread {spread} over {:?}",
            xs
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_prerequisites_are_reported() {
        let mut st = SimState::default();
        assert!(execute_line("qm3 energy", &mut st).unwrap_err().contains("no grid"));
        execute_line("qm3 grid -4 4 10, -4 4 10, -4 4 10", &mut st).unwrap();
        assert!(execute_line("qm3 energy", &mut st).unwrap_err().contains("no potential"));
        execute_line("qm3 potential zero", &mut st).unwrap();
        assert!(execute_line("qm3 norm", &mut st).unwrap_err().contains("no wavefunction"));
        assert!(execute_line("qm3 potential nosuch", &mut st).unwrap_err().contains("DEF"));
        assert!(execute_line("qm3 grid 4 -4 10, -4 4 10, -4 4 10", &mut st).is_err());
    }
}
