//! The `QM2` command family: two-dimensional quantum mechanics.
//!
//! Deliberately a *separate* family from `QM` rather than a mode switch
//! on it. A 2-D problem has a different grid, a potential of two
//! arguments, a two-component packet and a rectangular probability
//! region — almost every argument list differs. Overloading `QM` would
//! have meant either arity guessing or a hidden mode that silently
//! reinterprets your commands, and both are worse than a second word.
//!
//! ```text
//! def v(x, y) { 0.5 * (x * x + y * y) }
//! qm2 grid -8 8 80, -8 8 80      # x range and count, then y
//! qm2 potential v
//! qm2 packet -3 0, 1 1, 2 0      # (x0,y0), (sigma_x,sigma_y), (kx,ky)
//! qm2 run 2 steps 200            # ADI, exactly unitary
//! qm2 energy
//! qm2 animate "slit.html" 20 frames 80
//! ```

use quantum::qm2d::{
    BoundStates2, DrivenPropagator2, Grid2, Hamiltonian2, Propagator2, Wavefunction2,
};

use crate::vm::{SimState, Value};

/// Every `QM2` subcommand word the parser accepts, canonical spelling only.
///
/// This list is not documentation — it is the input to
/// `every_qm2_subcommand_is_documented_in_lockstep`, which checks each
/// word against the parser, `HELP_TEXT`, the EBNF comment and both
/// grammar documents, **and checks the converse**: that every word the
/// EBNF quotes is one this list declares.
///
/// The converse direction is the one that earns its keep. The `QM`
/// family has had the forward check since Stage 2A; `QM2` and `QM3` had
/// neither, and the EBNF comment for `qm2cmd` quietly advertised an
/// `ISO` production that `qm2_command` never implemented. Nothing
/// caught it, because a forward check only ever asks whether the
/// documents mention what the code does — never whether the code does
/// what the documents promise.
pub const QM2_SUBCOMMANDS: &[&str] = &[
    "status", "grid", "potential", "packet", "step", "run", "norm", "energy", "centroid",
    "prob", "absorb", "drive", "states", "state", "reset", "animate",
];

/// A `QM2` subcommand.
#[derive(Clone, Debug, PartialEq)]
pub enum Qm2Cmd {
    Status,
    /// Pops ny, y_max, y_min, nx, x_max, x_min.
    Grid,
    /// `zero`, or a `DEF`ined function of two arguments.
    Potential(String),
    /// Pops ky, kx, sigma_y, sigma_x, y0, x0.
    Packet,
    /// Pops dt.
    Step,
    /// Pops steps, then total time.
    Run,
    Norm,
    Energy,
    /// `(<x>, <y>)`.
    Centroid,
    /// Pops yb, ya, xb, xa.
    Prob,
    /// Pops power, strength, width.
    /// `f(t) g(x, y)`: two DEF'd function names.
    Drive(String, String),
    DriveOff,
    Absorb,
    AbsorbOff,
    Reset,
    /// Pops k: the k lowest bound-state energies, by Lanczos.
    States,
    /// Pops n: load bound state n as psi.
    LoadState,
    /// Pops frames, then total time; writes an HTML heat-map animation.
    Animate(String),
}

/// The 2-D problem carried by a session.
#[derive(Clone, Debug, Default)]
pub struct Qm2State {
    pub grid: Option<Grid2>,
    pub potential: Option<Vec<f64>>,
    pub potential_name: Option<String>,
    pub mass: f64,
    pub hbar: f64,
    pub psi: Option<Wavefunction2>,
    pub time: f64,
    pub absorber: Option<(f64, f64, f64)>,
    /// A drive: sampled spatial shape, its name, and the modulation's.
    pub drive: Option<(Vec<f64>, String, String)>,
    /// Cached bound states, so `QM2 STATE n` after `QM2 STATES k` does
    /// not pay for a second Lanczos run.
    pub states: Option<BoundStates2>,
}

impl Qm2State {
    /// `Default` gives 0 for `mass`/`hbar`, which are invalid, so the
    /// session builds one through here instead.
    pub fn fresh() -> Self {
        Self { mass: 1.0, hbar: 1.0, ..Default::default() }
    }

    fn hamiltonian(&self) -> Result<Hamiltonian2, String> {
        let grid = self
            .grid
            .clone()
            .ok_or("QM2: no grid — use `QM2 GRID <x_min> <x_max> <nx>, <y_min> <y_max> <ny>`")?;
        let v = self
            .potential
            .clone()
            .ok_or("QM2: no potential — use `QM2 POTENTIAL <function of x,y>` or `zero`")?;
        let h = Hamiltonian2::new(grid, v, self.mass, self.hbar)?;
        match self.absorber {
            Some((w, s, p)) => h.with_absorber(w, s, p),
            None => Ok(h),
        }
    }

    fn wavefunction(&self) -> Result<&Wavefunction2, String> {
        self.psi
            .as_ref()
            .ok_or_else(|| "QM2: no wavefunction — use `QM2 PACKET ...`".to_string())
    }
}

fn pop_num(stack: &mut Vec<Value>) -> Result<f64, String> {
    match stack.pop() {
        Some(Value::Num(n)) => Ok(n),
        Some(other) => Err(format!("QM2: expected a number, got {other}")),
        None => Err("QM2: missing an argument".to_string()),
    }
}

fn pop_count(stack: &mut Vec<Value>, what: &str) -> Result<usize, String> {
    let v = pop_num(stack)?;
    if !v.is_finite() || v.fract() != 0.0 || v < 0.0 {
        return Err(format!("QM2: {what} must be a whole number >= 0, got {v}"));
    }
    Ok(v as usize)
}

/// Execute one `QM2` subcommand.
pub fn exec_qm2(
    cmd: &Qm2Cmd,
    state: &mut SimState,
    stack: &mut Vec<Value>,
) -> Result<String, String> {
    match cmd {
        Qm2Cmd::Status => {
            let q = &state.qm2;
            let mut s = String::from("quantum (2-D, ADI):\n");
            match &q.grid {
                Some(g) => s.push_str(&format!(
                    "  grid      x [{}, {}] x {}, y [{}, {}] x {}  ({} points, hx {:.5}, hy {:.5})\n",
                    g.x_min, g.x_max, g.nx, g.y_min, g.y_max, g.ny, g.len(), g.hx(), g.hy()
                )),
                None => s.push_str("  grid      (unset)\n"),
            }
            s.push_str(&match &q.potential_name {
                Some(n) => format!("  potential {n} (sampled when the command ran)\n"),
                None => "  potential (unset)\n".to_string(),
            });
            s.push_str(&format!("  mass      {}\n  hbar      {}\n", q.mass, q.hbar));
            s.push_str(&match q.absorber {
                Some((w, st, p)) => {
                    format!("  absorber  width {w}, strength {st}, power {p}\n")
                }
                None => "  absorber  off — all four walls REFLECT\n".to_string(),
            });
            s.push_str(&match &q.psi {
                Some(w) => {
                    let (cx, cy) = w.centroid();
                    format!(
                        "  psi       set, norm {:.12}, <x> {cx:.4}, <y> {cy:.4}, t = {}\n",
                        w.norm(),
                        q.time
                    )
                }
                None => "  psi       (unset — QM2 PACKET)\n".to_string(),
            });
            Ok(s.trim_end().to_string())
        }

        Qm2Cmd::Grid => {
            let ny = pop_count(stack, "ny")?;
            let y_max = pop_num(stack)?;
            let y_min = pop_num(stack)?;
            let nx = pop_count(stack, "nx")?;
            let x_max = pop_num(stack)?;
            let x_min = pop_num(stack)?;
            let g = Grid2::new(x_min, x_max, nx, y_min, y_max, ny)?;
            let (hx, hy, n) = (g.hx(), g.hy(), g.len());
            state.qm2.grid = Some(g);
            state.qm2.potential = None;
            state.qm2.potential_name = None;
            state.qm2.psi = None;
            state.qm2.time = 0.0;
            state.qm2.states = None;
            Ok(format!(
                "grid x [{x_min}, {x_max}] x {nx}, y [{y_min}, {y_max}] x {ny} — {n} points, \
                 hx = {hx:.6}, hy = {hy:.6} (potential and psi cleared)"
            ))
        }

        Qm2Cmd::Potential(name) => {
            let grid = state
                .qm2
                .grid
                .clone()
                .ok_or("QM2 POTENTIAL: set a grid first")?;
            let v: Vec<f64> = if name == "zero" || name == "free" {
                vec![0.0; grid.len()]
            } else {
                if !state.functions.contains_key(name) {
                    return Err(format!(
                        "QM2 POTENTIAL: no function `{name}` — define one with \
                         `DEF {name}(x, y) {{ ... }}`, or use `QM2 POTENTIAL zero`"
                    ));
                }
                let mut out = Vec::with_capacity(grid.len());
                for iy in 0..grid.ny {
                    for ix in 0..grid.nx {
                        let val = crate::vm::call_user_function_public(
                            name,
                            vec![Value::Num(grid.x(ix)), Value::Num(grid.y(iy))],
                            state,
                        )?;
                        match val {
                            Value::Num(y) => out.push(y),
                            other => {
                                return Err(format!(
                                    "QM2 POTENTIAL: `{name}(x, y)` must return a number, got {other}"
                                ))
                            }
                        }
                    }
                }
                out
            };
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let n = grid.len();
            state.qm2.potential = Some(v);
            state.qm2.potential_name = Some(name.clone());
            state.qm2.psi = None;
            state.qm2.time = 0.0;
            state.qm2.states = None;
            Ok(format!(
                "potential `{name}` sampled at {n} points, V in [{lo}, {hi}] (psi cleared)"
            ))
        }

        Qm2Cmd::Packet => {
            let ky = pop_num(stack)?;
            let kx = pop_num(stack)?;
            let sy = pop_num(stack)?;
            let sx = pop_num(stack)?;
            let y0 = pop_num(stack)?;
            let x0 = pop_num(stack)?;
            let grid = state.qm2.grid.clone().ok_or("QM2 PACKET: set a grid first")?;
            let w = Wavefunction2::gaussian(grid, x0, y0, sx, sy, kx, ky)?;
            let edge = w.edge_probability(0.05);
            state.qm2.psi = Some(w);
            state.qm2.time = 0.0;
            let warn = if edge > 1e-6 {
                format!("\n  warning: {edge:.3e} already sits within 5% of a wall")
            } else {
                String::new()
            };
            Ok(format!(
                "psi = 2-D Gaussian at ({x0}, {y0}), sigma ({sx}, {sy}), k ({kx}, {ky}), t = 0{warn}"
            ))
        }

        Qm2Cmd::Step | Qm2Cmd::Run => {
            let (dt, steps) = if matches!(cmd, Qm2Cmd::Step) {
                (pop_num(stack)?, 1usize)
            } else {
                let n = pop_count(stack, "the step count")?;
                let t = pop_num(stack)?;
                if n == 0 {
                    return Err("QM2 RUN: the step count must be at least 1".to_string());
                }
                (t / n as f64, n)
            };
            if !dt.is_finite() || dt == 0.0 {
                return Err(format!("QM2: dt must be finite and non-zero, got {dt}"));
            }
            let ham = state.qm2.hamiltonian()?;
            let mut w = state.qm2.wavefunction()?.clone();
            let n0 = w.norm();
            match state.qm2.drive.clone() {
                None => {
                    Propagator2::new(ham.clone(), dt)?.run(&mut w, steps)?;
                }
                Some((shape, _, time_name)) => {
                    let mut prop = DrivenPropagator2::new(ham.clone(), shape, dt)?;
                    let t0 = state.qm2.time;
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
                                    "QM2 RUN: `{time_name}(t)` must return a number, got {other}"
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
            state.qm2.time += dt * steps as f64;
            let t = state.qm2.time;
            let e = w.energy(&ham);
            state.qm2.psi = Some(w);
            let warn = if edge > 1e-4 && !ham.is_absorbing() {
                format!("\n  warning: {edge:.3e} is within 5% of a wall — the walls REFLECT")
            } else {
                String::new()
            };
            Ok(format!(
                "t = {t} ({steps} ADI step(s) of dt = {dt}), <E> = {e:.12}, \
                 norm drift = {drift:.3e}{warn}"
            ))
        }

        Qm2Cmd::Norm => Ok(format!("{:.15}", state.qm2.wavefunction()?.norm())),
        Qm2Cmd::Centroid => {
            let (x, y) = state.qm2.wavefunction()?.centroid();
            Ok(format!("[{x}, {y}]"))
        }
        Qm2Cmd::Energy => {
            let ham = state.qm2.hamiltonian()?;
            Ok(format!("{:.15}", state.qm2.wavefunction()?.energy(&ham)))
        }
        Qm2Cmd::Prob => {
            let yb = pop_num(stack)?;
            let ya = pop_num(stack)?;
            let xb = pop_num(stack)?;
            let xa = pop_num(stack)?;
            Ok(format!(
                "{:.15}",
                state.qm2.wavefunction()?.probability_in(xa, xb, ya, yb)
            ))
        }

        Qm2Cmd::Drive(shape_name, time_name) => {
            let grid = state.qm2.grid.clone().ok_or("QM2 DRIVE: set a grid first")?;
            for nm in [shape_name, time_name] {
                if !state.functions.contains_key(nm) {
                    return Err(format!(
                        "QM2 DRIVE: no function `{nm}` — the shape is `DEF {nm}(x, y) {{ ... }}` \
                         and the modulation is `DEF f(t) {{ ... }}`"
                    ));
                }
            }
            let mut shape = Vec::with_capacity(grid.len());
            for iy in 0..grid.ny {
                for ix in 0..grid.nx {
                    let v = crate::vm::call_user_function_public(
                        shape_name,
                        vec![Value::Num(grid.x(ix)), Value::Num(grid.y(iy))],
                        state,
                    )?;
                    match v {
                        Value::Num(y) => shape.push(y),
                        other => {
                            return Err(format!(
                                "QM2 DRIVE: `{shape_name}(x, y)` must return a number, got {other}"
                            ))
                        }
                    }
                }
            }
            state.qm2.drive = Some((shape, shape_name.clone(), time_name.clone()));
            state.qm2.states = None;
            Ok(format!(
                "drive V(x,y,t) += {time_name}(t) * {shape_name}(x,y). Energy is NO LONGER \
                 conserved; propagation stays unitary. QM2 STATES uses the STATIC potential."
            ))
        }

        Qm2Cmd::DriveOff => {
            state.qm2.drive = None;
            state.qm2.states = None;
            Ok("drive removed".to_string())
        }

        Qm2Cmd::Absorb => {
            let power = pop_num(stack)?;
            let strength = pop_num(stack)?;
            let width = pop_num(stack)?;
            if let Some(g) = state.qm2.grid.clone() {
                let probe = Hamiltonian2::new(g.clone(), vec![0.0; g.len()], 1.0, 1.0)?;
                probe.with_absorber(width, strength, power)?;
            }
            state.qm2.absorber = Some((width, strength, power));
            state.qm2.states = None;
            Ok(format!(
                "absorbing edges on all four walls: width {width}, strength {strength}, \
                 power {power}. Propagation is no longer unitary — the norm decays by design."
            ))
        }
        Qm2Cmd::AbsorbOff => {
            state.qm2.absorber = None;
            state.qm2.states = None;
            Ok("absorbing edges removed — all four walls reflect again".to_string())
        }

        Qm2Cmd::States => {
            let k = pop_count(stack, "the state count")?;
            if k == 0 {
                return Err("QM2 STATES: ask for at least one state".to_string());
            }
            let ham = state.qm2.hamiltonian()?;
            let b = ham.bound_states(k, 0)?;
            let mut s = format!(
                "{k} lowest bound state(s) — Lanczos, {} iterations{}:\n",
                b.iterations,
                if b.converged { "" } else { " (NOT converged)" }
            );
            for (i, e) in b.energies.iter().enumerate() {
                s.push_str(&format!("  E[{i}] = {e:.10}   residual {:.2e}\n", b.residuals[i]));
            }
            if !b.converged {
                s.push_str(
                    "  warning: the iteration limit was reached — the residuals above say \
                     how far off these are\n",
                );
            }
            state.qm2.states = Some(b);
            Ok(s.trim_end().to_string())
        }

        Qm2Cmd::LoadState => {
            let n = pop_count(stack, "the state index")?;
            let ham = state.qm2.hamiltonian()?;
            let need = n + 1;
            let have = state.qm2.states.as_ref().map(|b| b.energies.len()).unwrap_or(0);
            if have < need {
                state.qm2.states = Some(ham.bound_states(need, 0)?);
            }
            let b = state.qm2.states.as_ref().expect("just filled");
            let e = b.energies[n];
            let mut w = Wavefunction2::new(
                ham.grid.clone(),
                b.states[n].iter().map(quantum::qm2d::real_to_complex).collect(),
            )?;
            w.normalise()?;
            state.qm2.psi = Some(w);
            state.qm2.time = 0.0;
            Ok(format!("psi = 2-D bound state {n}, E = {e:.10}, t reset to 0"))
        }

        Qm2Cmd::Reset => {
            state.qm2 = Qm2State::fresh();
            Ok("2-D quantum state cleared".to_string())
        }

        Qm2Cmd::Animate(path) => {
            let frames = pop_count(stack, "the frame count")?;
            let total = pop_num(stack)?;
            if frames < 2 {
                return Err("QM2 ANIMATE: ask for at least 2 frames".to_string());
            }
            if !total.is_finite() || total <= 0.0 {
                return Err(format!("QM2 ANIMATE: total time must be positive, got {total}"));
            }
            let ham = state.qm2.hamiltonian()?;
            let mut w = state.qm2.wavefunction()?.clone();
            let g = ham.grid.clone();
            let per_frame = 10usize;
            let dt = total / (frames * per_frame) as f64;
            // See the note in qm.rs: an animation that ignores the
            // drive produces a silently wrong picture.
            let drive = state.qm2.drive.clone();
            let prop = Propagator2::new(ham.clone(), dt)?;
            let mut driven = match &drive {
                Some((shape, _, _)) => {
                    Some(DrivenPropagator2::new(ham.clone(), shape.clone(), dt)?)
                }
                None => None,
            };
            let t_start = state.qm2.time;

            // Downsample both axes: a browser cannot use more than a few
            // hundred cells per side, and the file grows as their product.
            let sx = (g.nx / 200).max(1);
            let sy = (g.ny / 200).max(1);
            let cols: Vec<usize> = (0..g.nx).step_by(sx).collect();
            let rows: Vec<usize> = (0..g.ny).step_by(sy).collect();

            let mut frames_js: Vec<String> = Vec::with_capacity(frames);
            let mut times: Vec<f64> = Vec::with_capacity(frames);
            let mut worst = 0.0_f64;
            let n0 = w.norm();
            for f in 0..frames {
                if f > 0 {
                    advance_2d(
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
                let d = w.density();
                let mut buf = String::from("[");
                for (ri, &iy) in rows.iter().enumerate() {
                    if ri > 0 {
                        buf.push(',');
                    }
                    for (ci, &ix) in cols.iter().enumerate() {
                        if ci > 0 {
                            buf.push(',');
                        }
                        buf.push_str(&format!("{:.5e}", d[g.idx(ix, iy)]));
                    }
                }
                buf.push(']');
                frames_js.push(buf);
                times.push(state.qm2.time + dt * (f * per_frame) as f64);
            }
            state.qm2.time += total;
            state.qm2.psi = Some(w);

            let flat = |v: &[usize], f: &dyn Fn(usize) -> f64| {
                v.iter().map(|&i| format!("{:.4}", f(i))).collect::<Vec<_>>().join(",")
            };
            let pot: Vec<String> = rows
                .iter()
                .map(|&iy| {
                    cols.iter()
                        .map(|&ix| format!("{:.4}", ham.potential[g.idx(ix, iy)]))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect();
            let label = state
                .qm2
                .potential_name
                .clone()
                .unwrap_or_else(|| "unnamed".to_string());
            let html = render_html_2d(
                &flat(&cols, &|i| g.x(i)),
                &flat(&rows, &|i| g.y(i)),
                &pot.join(","),
                &frames_js.join(",\n"),
                &times.iter().map(|t| format!("{t:.4}")).collect::<Vec<_>>().join(","),
                &label,
                cols.len(),
                rows.len(),
            );
            std::fs::write(path, &html)
                .map_err(|e| format!("QM2 ANIMATE: cannot write `{path}`: {e}"))?;
            Ok(format!(
                "wrote {path} — {frames} frames over t = {total} (dt = {dt:.6}, \
                 {} x {} cells), worst norm drift {worst:.3e}. Open it in a browser.",
                cols.len(),
                rows.len()
            ))
        }
    }
}

/// Advance a 2-D wavefunction by `steps`, honouring a drive if set.
#[allow(clippy::too_many_arguments)]
fn advance_2d(
    w: &mut Wavefunction2,
    prop: &Propagator2,
    driven: Option<&mut DrivenPropagator2>,
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

/// A self-contained heat-map animation page. Everything is inlined; the
/// page fetches nothing.
#[allow(clippy::too_many_arguments)]
fn render_html_2d(
    xs: &str,
    ys: &str,
    pot: &str,
    frames: &str,
    times: &str,
    label: &str,
    nx: usize,
    ny: usize,
) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>posim — 2-D quantum scattering (ADI)</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
        background:#0e1116; color:#e6e6e6; }}
 header {{ padding:14px 18px; border-bottom:1px solid #263042; }}
 h1 {{ margin:0; font-size:16px; font-weight:600; }}
 .sub {{ color:#8b98ad; font-size:12px; margin-top:3px; }}
 #wrap {{ padding:14px 18px; }}
 canvas {{ width:100%; max-width:820px; height:auto; display:block; image-rendering:pixelated;
           background:#11151c; border:1px solid #263042; border-radius:6px; }}
 .row {{ display:flex; gap:12px; align-items:center; margin-top:12px; flex-wrap:wrap; }}
 button {{ background:#1b2330; color:#e6e6e6; border:1px solid #33405a;
           border-radius:5px; padding:6px 14px; cursor:pointer; font:inherit; }}
 button:hover {{ background:#243149; }}
 input[type=range] {{ flex:1; min-width:200px; }}
 .stat {{ font-variant-numeric:tabular-nums; color:#9fb3d0; }}
 .key {{ margin-top:8px; font-size:12px; color:#8b98ad; }}
</style></head><body>
<header>
  <h1>2-D quantum scattering — |&psi;(x,y,t)|&sup2;</h1>
  <div class="sub">potential: <b>{label}</b> &middot; {nx}&times;{ny} cells &middot;
  ADI (Strang-split Cayley, exactly unitary) &middot; generated by posim</div>
</header>
<div id="wrap">
  <canvas id="c" width="{nx}" height="{ny}"></canvas>
  <div class="row">
    <button id="play">Pause</button>
    <button id="rew">Restart</button>
    <input type="range" id="scrub" min="0" value="0">
    <span class="stat" id="stat"></span>
  </div>
  <div class="key">brightness = probability density (per-frame scaling); red overlay = potential;
  x increases rightwards, y upwards</div>
</div>
<script>
const X=[{xs}], Y=[{ys}], T=[{times}];
const NX={nx}, NY={ny};
const POT=[{pot}];
const F=[
{frames}
];
const c=document.getElementById('c'), g=c.getContext('2d');
const scrub=document.getElementById('scrub'); scrub.max=F.length-1;
let i=0, playing=true;
const img=g.createImageData(NX,NY);
const potMax=Math.max(...POT.map(Math.abs))||1;

function draw() {{
  const f=F[i];
  let m=0; for (const v of f) if (v>m) m=v;
  if (m<=0) m=1;
  for (let r=0;r<NY;r++) for (let col=0;col<NX;col++) {{
    const k=r*NX+col;
    // flip vertically so +y points up on screen
    const p=((NY-1-r)*NX+col)*4;
    const t=Math.sqrt(f[k]/m);              // sqrt makes the tails visible
    // blue-white ramp for |psi|^2
    img.data[p  ]=Math.min(255, 40*t + 215*t*t*t);
    img.data[p+1]=Math.min(255, 90*t + 165*t*t*t);
    img.data[p+2]=Math.min(255, 200*t + 55*t*t*t);
    // red wash where the potential is
    const vv=Math.abs(POT[k])/potMax;
    img.data[p  ]=Math.min(255, img.data[p]+150*vv);
    img.data[p+3]=255;
  }}
  g.putImageData(img,0,0);
  let tot=0; for (const v of f) tot+=v;
  document.getElementById('stat').textContent =
    `t = ${{T[i].toFixed(2)}}   frame ${{i+1}}/${{F.length}}   peak = ${{m.toExponential(2)}}`;
  scrub.value=i;
}}
let last=0;
function loop(ts) {{
  if (playing && ts-last>50) {{ i=(i+1)%F.length; last=ts; draw(); }}
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

#[cfg(test)]
mod tests {
    /// Words the `qm2cmd` EBNF quotes that are arguments, not subcommands.
    const QM2_ARGUMENT_WORDS: &[&str] = &["ZERO", "OFF", "STEPS", "FRAMES", "STATUS"];


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
    fn every_qm2_subcommand_is_documented_in_lockstep() {
        // `\_` in LaTeX, `\` nowhere else; case differs between the
        // parser (lowercase) and the documents (upper).
        let prep = |s: &str| s.replace('\\', "").to_ascii_uppercase();
        let help = prep(crate::vm::HELP_TEXT);
        // Only this family's production: the three qm*cmd blocks quote
        // the same words, so searching the whole file would pass on a
        // word this family never declared.
        let all = prep(include_str!("parser.rs"));
        let from = all.find("QM2CMD").expect("parser.rs must declare the qm2cmd production");
        let to = all[from..].find("QM3CMD").map_or(all.len(), |k| from + k);
        let ebnf = all[from..to].to_string();
        let md = prep(include_str!("../../grammar.md"));
        let tex = prep(include_str!("../../grammar.tex"));

        let mut missing = Vec::new();

        // Forward: everything the parser accepts is documented.
        for w in super::QM2_SUBCOMMANDS {
            let up = w.to_ascii_uppercase();
            let needle = format!("QM2 {up}");
            let ebnf_needle = format!("\"{up}\"");
            for (what, hay) in [
                ("HELP_TEXT", &help),
                ("parser.rs EBNF", &ebnf),
                ("grammar.md", &md),
                ("grammar.tex", &tex),
            ] {
                // `QM2 STATUS` is spelled as bare `QM2` in the
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
            let known = super::QM2_SUBCOMMANDS.iter().any(|s| s.eq_ignore_ascii_case(word))
                || QM2_ARGUMENT_WORDS.contains(&word);
            if !known {
                phantom.push(format!(
                    "the qm2cmd EBNF quotes `{word}`, which is neither a declared \
                     subcommand nor a declared argument word — either the parser is \
                     missing an arm or the comment is promising a command that does \
                     not exist"
                ));
            }
        }

        missing.extend(phantom);
        assert!(
            missing.is_empty(),
            "QM2 grammar lockstep is broken:\n  {}",
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

    /// The 2-D oscillator through the language, with a potential of two
    /// arguments.
    #[test]
    fn a_two_argument_potential_is_sampled() {
        let (st, _) = run(&[
            "def v(x, y) { 0.5 * (x * x + y * y) }",
            "qm2 grid -6 6 40, -6 6 40",
            "qm2 potential v",
        ]);
        let g = st.qm2.grid.as_ref().unwrap();
        let p = st.qm2.potential.as_ref().unwrap();
        assert_eq!(p.len(), 1600);
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let want = 0.5 * (g.x(ix).powi(2) + g.y(iy).powi(2));
                let got = p[g.idx(ix, iy)];
                assert!((got - want).abs() < 1e-10, "V({ix},{iy}) = {got}, want {want}");
            }
        }
    }

    /// ADI is exactly unitary, so the reported drift must be tiny.
    #[test]
    fn adi_propagation_conserves_the_norm() {
        let (_, out) = run(&[
            "qm2 grid -10 10 60, -10 10 60",
            "qm2 potential zero",
            "qm2 packet -3 0, 1 1, 2 0",
            "qm2 run 1 steps 100",
            "qm2 norm",
        ]);
        let norm: f64 = out[4].trim().parse().unwrap();
        assert!((norm - 1.0).abs() < 1e-10, "norm = {norm}");
    }

    /// The packet must actually move, and in the direction it was given.
    #[test]
    fn a_packet_drifts_along_its_momentum() {
        let (st, _) = run(&[
            "qm2 grid -20 20 100, -20 20 100",
            "qm2 potential zero",
            "qm2 packet -8, -4, 1.5 1.5, 2 1",
            "qm2 run 2 steps 200",
        ]);
        let (x, y) = st.qm2.psi.as_ref().unwrap().centroid();
        assert!(x > -8.0 + 2.0, "<x> = {x}, should have moved right");
        assert!(y > -4.0 + 1.0, "<y> = {y}, should have moved up");
    }

    /// A rectangular grid catches an axis swap that a square one hides.
    #[test]
    fn a_rectangular_grid_keeps_its_axes_straight() {
        let (st, _) = run(&[
            "def v(x, y) { x }",
            "qm2 grid -4 4 30, -1 1 12",
            "qm2 potential v",
        ]);
        let g = st.qm2.grid.as_ref().unwrap();
        assert_eq!((g.nx, g.ny), (30, 12));
        let p = st.qm2.potential.as_ref().unwrap();
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                assert!((p[g.idx(ix, iy)] - g.x(ix)).abs() < 1e-12);
            }
        }
    }

    /// Negative arguments need commas here too: `-8 -4` is subtraction,
    /// so it would silently supply five arguments where six were
    /// wanted. The failure must be loud.
    #[test]
    fn negative_arguments_need_commas() {
        let mut st = SimState::default();
        execute_line("qm2 grid -20 20 40, -20 20 40", &mut st).unwrap();
        execute_line("qm2 potential zero", &mut st).unwrap();
        assert!(execute_line("qm2 packet -8, -4, 1 1, 1 0", &mut st).is_ok());
        assert!(
            execute_line("qm2 packet -8 -4, 1 1, 1 0", &mut st).is_err(),
            "the ambiguous spelling must be a parse error"
        );
    }

    #[test]
    fn missing_prerequisites_are_reported() {
        let mut st = SimState::default();
        assert!(execute_line("qm2 norm", &mut st).unwrap_err().contains("no grid")
            || execute_line("qm2 energy", &mut st).unwrap_err().contains("no grid"));
        execute_line("qm2 grid -5 5 20, -5 5 20", &mut st).unwrap();
        assert!(execute_line("qm2 energy", &mut st).unwrap_err().contains("no potential"));
        execute_line("qm2 potential zero", &mut st).unwrap();
        assert!(execute_line("qm2 norm", &mut st).unwrap_err().contains("no wavefunction"));
        assert!(execute_line("qm2 potential nosuch", &mut st).unwrap_err().contains("DEF"));
        assert!(execute_line("qm2 grid 5 -5 10, -5 5 10", &mut st).is_err(), "reversed x");
    }
}
