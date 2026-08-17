//! The `QM` command family: one-dimensional quantum mechanics in the
//! notebook.
//!
//! This is the language front end for the `quantum` crate. The session
//! carries one quantum problem at a time — a grid, a potential sampled
//! on it, and a current wavefunction — built up command by command:
//!
//! ```text
//! def v(x) { 0.5 * x * x }     # the potential, as an ordinary function
//! qm grid -8 8 300             # domain and resolution
//! qm potential v               # sample v(x) onto the grid
//! qm states 5                  # the five lowest bound states
//! qm state 0                   # load the ground state as psi
//! qm packet -3 0.7 4           # ...or launch a Gaussian wavepacket
//! qm run 2 steps 400           # propagate (Crank-Nicolson, unitary)
//! qm energy                    # observables
//! ```
//!
//! # Why the potential is an ordinary user function
//!
//! `DEF` already gives the language first-class functions of one
//! argument, which is exactly what a potential is. `QM POTENTIAL v`
//! evaluates `v(x)` once at every grid point and stores the samples, so
//! there is no need for the expression evaluator to be re-entered during
//! propagation, and no need for a new kind of deferred-expression value.
//! Sampling happens **at command time**, so if you redefine `v` you must
//! re-issue `QM POTENTIAL v` — which is stated in the status output
//! rather than left to be discovered.

pub use quantum::nash::Splitting;
use quantum::transfer::{scan as tm_scan, scatter as tm_scatter, EnergyRange};
use special_functions::complex::Complex64 as Cx;
use quantum::nash::{NashPropagator, PeriodicGrid};
use quantum::qm1d::{DrivenPropagator, Grid, Hamiltonian, Propagator, Wavefunction};

use crate::vm::{SimState, Value};

/// A `QM` subcommand. Numeric arguments are compiled as ordinary
/// expressions and popped from the stack, so `qm grid -8 8 2*150` works.
#[derive(Clone, Debug, PartialEq)]
pub enum QmCmd {
    /// Bare `QM`: report what is set up.
    Status,
    /// Pops n, x_max, x_min.
    Grid,
    /// Set the potential. See [`PotentialSpec`].
    Potential(PotentialSpec),
    /// Pops the particle mass.
    Mass,
    /// Pops hbar.
    Hbar,
    /// Pops k: report the k lowest bound-state energies.
    States,
    /// Pops n: load bound state n as the current wavefunction.
    LoadState,
    /// Pops k0, sigma, x0.
    Packet,
    /// Pops dt.
    Step,
    /// Pops the number of steps, then the elapsed time.
    Run,
    Norm,
    Energy,
    Position,
    Momentum,
    /// Pops b, a: probability in [a, b].
    Prob,
    /// The probability density, as a list.
    Density,
    /// A time-dependent drive `f(t) g(x)`: two DEF'd function names,
    /// the spatial shape and the time modulation.
    Drive(String, String),
    /// Remove the drive.
    DriveOff,
    /// Attach absorbing edges. Pops power, strength, width.
    Absorb,
    /// Remove them.
    AbsorbOff,
    /// Choose the propagator. See [`EvolveMethod`].
    Method(EvolveMethod),
    /// Pops the energy: `T(E)` and `R(E)` by transfer matrix.
    Transmission,
    /// Pops the point count, then the high and low energies: scan
    /// `T(E)` and report the resonances.
    Scan,
    /// Forget the whole quantum problem.
    Reset,
    /// Propagate while capturing |psi|^2, and write a self-contained
    /// HTML animation to the given path. Pops the frame count, then the
    /// total time.
    Animate(String),
}

/// Every `QM` subcommand word the parser accepts.
///
/// This list is not documentation — it is the input to
/// `every_qm_subcommand_is_documented_in_lockstep`, which checks each
/// word against the parser, `HELP_TEXT`, the EBNF comment and both
/// grammar documents. The project rule is that a command is not "added"
/// until all five agree, and until Stage 2A that rule was enforced by
/// discipline for the `QM` family, which is to say not enforced.
pub const QM_SUBCOMMANDS: &[&str] = &[
    "status", "grid", "potential", "mass", "hbar", "method", "states", "state", "packet",
    "step", "run", "norm", "energy", "position", "momentum", "prob", "density", "drive",
    "absorb", "animate", "reset", "transmission", "scan",
];

/// Which propagator `QM RUN` and `QM STEP` use.
///
/// This is **not** only a choice of algorithm: it changes the boundary
/// condition. Crank–Nicolson runs on the Dirichlet grid, whose walls
/// reflect. The Nash propagator is periodic — the two ends are
/// identified — because that is what the original C++ index wrap means
/// and there is no honest way to pretend otherwise.
///
/// The `n` interior points of the Dirichlet grid are exactly `n`
/// periodic points at the same spacing `h`, so switching methods
/// re-uses the potential samples verbatim and moves no grid point. Only
/// the interpretation of the two ends changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvolveMethod {
    /// Crank–Nicolson on the Dirichlet grid. The default.
    Cayley,
    /// The Bessel-stencil split-operator scheme, periodic.
    Nash(Splitting),
}

impl std::fmt::Display for EvolveMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cayley => write!(f, "cayley (Crank-Nicolson, Dirichlet walls)"),
            Self::Nash(Splitting::Lie) => {
                write!(f, "nash (Bessel stencil, periodic, Lie - as the original C++)")
            }
            Self::Nash(Splitting::Strang) => {
                write!(f, "nash strang (Bessel stencil, periodic, 2nd order)")
            }
        }
    }
}

/// How a potential is specified.
///
/// A user function is the general case, but this language has **no
/// comparison operators**, so a piecewise potential — a square barrier,
/// a finite well — cannot be written as a `DEF` at all. Those are
/// exactly the canonical 1-D problems, so they are provided as named
/// shapes rather than left unreachable. This is a deliberate
/// workaround for a language limitation, not a preference for built-ins.
#[derive(Clone, Debug, PartialEq)]
pub enum PotentialSpec {
    /// Free particle.
    Zero,
    /// `v0` on `[x1, x2]`, zero elsewhere. Pops x2, x1, v0.
    Barrier,
    /// `-depth` on `[x1, x2]`, zero elsewhere. Pops x2, x1, depth.
    Well,
    /// A `DEF`ined function of one argument, sampled onto the grid.
    Named(String),
}

/// Everything the `QM` family needs to remember between commands.
#[derive(Clone, Debug)]
pub struct QmState {
    pub grid: Option<Grid>,
    pub potential: Option<Vec<f64>>,
    /// Absorbing edges: (width, strength, power), if enabled.
    pub absorber: Option<(f64, f64, f64)>,
    /// A drive: the sampled spatial shape and the modulation's name.
    pub drive: Option<(Vec<f64>, String, String)>,
    /// The function name the potential came from, for the status line.
    pub potential_name: Option<String>,
    pub mass: f64,
    pub hbar: f64,
    pub psi: Option<Wavefunction>,
    /// Elapsed propagation time, so observables can be reported against
    /// a clock rather than a step count.
    pub time: f64,
    /// Cached bound states, so `QM STATE n` after `QM STATES k` does not
    /// pay for a second diagonalisation.
    pub states: Option<(Vec<f64>, Vec<Vec<f64>>)>,
    /// Which propagator to use, and therefore which boundary condition.
    pub method: EvolveMethod,
}

impl Default for QmState {
    fn default() -> Self {
        Self {
            grid: None,
            potential: None,
            absorber: None,
            drive: None,
            potential_name: None,
            mass: 1.0,
            hbar: 1.0,
            psi: None,
            time: 0.0,
            states: None,
            method: EvolveMethod::Cayley,
        }
    }
}

impl QmState {
    /// The Hamiltonian, if a grid and potential have both been set.
    fn hamiltonian(&self) -> Result<Hamiltonian, String> {
        let grid = self
            .grid
            .clone()
            .ok_or("QM: no grid — use `QM GRID <x_min> <x_max> <n>` first")?;
        let v = self
            .potential
            .clone()
            .ok_or("QM: no potential — use `QM POTENTIAL <function>` (or `QM POTENTIAL zero`)")?;
        let ham = Hamiltonian::new(grid, v, self.mass, self.hbar)?;
        match self.absorber {
            Some((w, st, p)) => ham.with_absorber(w, st, p),
            None => Ok(ham),
        }
    }

    /// The Nash propagator for this session, on the periodic reading of
    /// the same grid.
    ///
    /// The Dirichlet grid's `n` interior points span `[x_min + h,
    /// x_max - h]` at spacing `h`; a periodic grid over
    /// `[x_min + h, x_max]` with `n` points has the same spacing and the
    /// same points, so nothing is re-sampled and the potential carries
    /// over unchanged.
    fn nash(&self, dt: f64, splitting: Splitting) -> Result<NashPropagator, String> {
        let grid = self
            .grid
            .clone()
            .ok_or("QM: no grid — use `QM GRID <x_min> <x_max> <n>` first")?;
        let v = self
            .potential
            .clone()
            .ok_or("QM: no potential — use `QM POTENTIAL <function>` (or `QM POTENTIAL zero`)")?;
        // Both of these are Crank-Nicolson features and neither has a
        // meaning for this propagator yet. Refusing beats quietly
        // ignoring them, which would produce a plausible wrong picture.
        if self.absorber.is_some() {
            return Err("QM: NASH has no absorbing edges — the propagator takes a real \
                        potential, and an absorber is a complex one. Use `QM ABSORB OFF`, \
                        or `QM METHOD CAYLEY`."
                .to_string());
        }
        if self.drive.is_some() {
            return Err("QM: NASH has no drive — its potential phase is built once, so a \
                        time-dependent V(x,t) is not expressible. Use `QM DRIVE OFF`, or \
                        `QM METHOD CAYLEY`."
                .to_string());
        }
        let h = grid.h();
        let pg = PeriodicGrid::new(grid.x(0), grid.x(0) + grid.n as f64 * h, grid.n)?;
        Ok(NashPropagator::new(pg, &v, self.hbar, self.mass, dt, None)?
            .with_splitting(splitting))
    }

    /// The splitting in force, if the method is Nash.
    fn nash_splitting(&self) -> Option<Splitting> {
        match self.method {
            EvolveMethod::Nash(s) => Some(s),
            EvolveMethod::Cayley => None,
        }
    }

    /// The potential as transfer-matrix **cells**.
    ///
    /// The grid samples `V` at `n` interior points spaced `h` apart, so
    /// reading those as the midpoints of `n` cells of width `h` spans
    /// `[x(0) - h/2, x(n-1) + h/2]` exactly. That is the natural
    /// reading and it is second-order accurate, which is what
    /// [`quantum::transfer`] wants.
    fn cells(&self) -> Result<(Vec<Cx>, f64, f64), String> {
        let grid = self
            .grid
            .clone()
            .ok_or("QM: no grid — use `QM GRID <x_min> <x_max> <n>` first")?;
        let v = self
            .potential
            .clone()
            .ok_or("QM: no potential — use `QM POTENTIAL <function>` (or `QM POTENTIAL zero`)")?;
        if self.absorber.is_some() {
            return Err("QM: a transfer matrix needs a real potential at the boundary, and \
                        an absorber is complex there — the incident flux would be \
                        undefined. Use `QM ABSORB OFF`."
                .to_string());
        }
        let h = grid.h();
        Ok((v.into_iter().map(Cx::real).collect(), grid.x(0) - 0.5 * h, grid.x(grid.n - 1) + 0.5 * h))
    }

    fn wavefunction(&self) -> Result<&Wavefunction, String> {
        self.psi.as_ref().ok_or_else(|| {
            "QM: no wavefunction — use `QM PACKET <x0> <sigma> <k0>` or `QM STATE <n>`".to_string()
        })
    }

    /// Anything that changes the Hamiltonian invalidates cached results.
    fn invalidate(&mut self) {
        self.states = None;
    }
}

fn pop_num(stack: &mut Vec<Value>) -> Result<f64, String> {
    match stack.pop() {
        Some(Value::Num(n)) => Ok(n),
        Some(other) => Err(format!("QM: expected a number, got {other}")),
        None => Err("QM: missing an argument".to_string()),
    }
}

/// A count argument: must be a whole, non-negative number.
fn pop_count(stack: &mut Vec<Value>, what: &str) -> Result<usize, String> {
    let v = pop_num(stack)?;
    if !v.is_finite() || v.fract() != 0.0 || v < 0.0 {
        return Err(format!("QM: {what} must be a whole number >= 0, got {v}"));
    }
    Ok(v as usize)
}

/// Execute one `QM` subcommand, returning the text to display.
pub fn exec_qm(
    cmd: &QmCmd,
    state: &mut SimState,
    stack: &mut Vec<Value>,
) -> Result<String, String> {
    match cmd {
        QmCmd::Status => {
            let q = &state.qm;
            let mut s = String::from("quantum (1-D):\n");
            match &q.grid {
                Some(g) => s.push_str(&format!(
                    "  grid      [{}, {}], {} interior points, h = {:.6}\n",
                    g.x_min,
                    g.x_max,
                    g.n,
                    g.h()
                )),
                None => s.push_str("  grid      (unset — QM GRID <x_min> <x_max> <n>)\n"),
            }
            match &q.potential_name {
                Some(n) => s.push_str(&format!(
                    "  potential {n}  (sampled when the command ran; re-issue \
                     QM POTENTIAL after editing it)\n"
                )),
                None => s.push_str("  potential (unset — QM POTENTIAL <function>)\n"),
            }
            s.push_str(&format!("  mass      {}\n  hbar      {}\n", q.mass, q.hbar));
            s.push_str(&format!("  method    {}\n", q.method));
            // The boundary is a property of the METHOD, so saying "the
            // walls reflect" under NASH would be a plain falsehood.
            match (q.method, q.absorber) {
                (EvolveMethod::Nash(_), _) => s.push_str(
                    "  boundary  PERIODIC — the two ends are identified; a packet leaving \
                     one re-enters at the other\n",
                ),
                (EvolveMethod::Cayley, Some((w, st, p))) => s.push_str(&format!(
                    "  boundary  Dirichlet, absorber width {w}, strength {st}, power {p} \
                     (norm decays by design)\n"
                )),
                (EvolveMethod::Cayley, None) => {
                    s.push_str("  boundary  Dirichlet — the walls REFLECT (absorber off)\n");
                }
            }
            match &q.psi {
                Some(w) => s.push_str(&format!(
                    "  psi       set, norm = {:.12}, t = {}\n",
                    w.norm(),
                    q.time
                )),
                None => s.push_str("  psi       (unset — QM PACKET or QM STATE)\n"),
            }
            Ok(s.trim_end().to_string())
        }

        QmCmd::Grid => {
            let n = pop_count(stack, "the point count")?;
            let x_max = pop_num(stack)?;
            let x_min = pop_num(stack)?;
            let g = Grid::new(x_min, x_max, n)?;
            let h = g.h();
            state.qm.grid = Some(g);
            // The grid defines the sampling, so everything downstream is
            // stale: drop it rather than leave a potential of the wrong
            // length to fail confusingly later.
            state.qm.potential = None;
            state.qm.potential_name = None;
            state.qm.psi = None;
            state.qm.time = 0.0;
            state.qm.invalidate();
            Ok(format!(
                "grid [{x_min}, {x_max}] with {n} interior points, h = {h:.6} \
                 (potential and psi cleared)"
            ))
        }

        QmCmd::Potential(spec) => {
            let grid = state
                .qm
                .grid
                .clone()
                .ok_or("QM POTENTIAL: set a grid first (QM GRID <x_min> <x_max> <n>)")?;
            let (v, label): (Vec<f64>, String) = match spec {
                PotentialSpec::Zero => (vec![0.0; grid.n], "zero".to_string()),
                PotentialSpec::Barrier | PotentialSpec::Well => {
                    let x2 = pop_num(stack)?;
                    let x1 = pop_num(stack)?;
                    let amp = pop_num(stack)?;
                    if !amp.is_finite() || !x1.is_finite() || !x2.is_finite() {
                        return Err("QM POTENTIAL: arguments must be finite".to_string());
                    }
                    let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
                    let sign = if matches!(spec, PotentialSpec::Well) { -1.0 } else { 1.0 };
                    let vv = (0..grid.n)
                        .map(|i| {
                            let x = grid.x(i);
                            if x >= lo && x <= hi {
                                sign * amp
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let kind = if sign < 0.0 { "well" } else { "barrier" };
                    // A feature narrower than the grid spacing is not
                    // represented at all, and would silently behave as
                    // if it were absent.
                    if hi - lo < grid.h() {
                        return Err(format!(
                            "QM POTENTIAL {kind}: the region [{lo}, {hi}] is narrower than the \
                             grid spacing h = {:.6}, so it would fall between points — use a \
                             finer grid",
                            grid.h()
                        ));
                    }
                    (vv, format!("{kind} {amp} on [{lo}, {hi}]"))
                }
                PotentialSpec::Named(name) => {
                    if !state.functions.contains_key(name) {
                        return Err(format!(
                            "QM POTENTIAL: no function `{name}` — define one with \
                             `DEF {name}(x) {{ ... }}`, or use one of the built-in shapes \
                             (zero, barrier, well)"
                        ));
                    }
                    let mut out = Vec::with_capacity(grid.n);
                    for i in 0..grid.n {
                        let val = crate::vm::call_user_function_public(
                            name,
                            vec![Value::Num(grid.x(i))],
                            state,
                        )?;
                        match val {
                            Value::Num(y) => out.push(y),
                            other => {
                                return Err(format!(
                                    "QM POTENTIAL: `{name}(x)` must return a number, got {other}"
                                ))
                            }
                        }
                    }
                    (out, name.clone())
                }
            };
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let n = grid.n;
            state.qm.potential = Some(v);
            state.qm.potential_name = Some(label.clone());
            state.qm.psi = None;
            state.qm.time = 0.0;
            state.qm.invalidate();
            Ok(format!(
                "potential `{label}` sampled at {n} points, V in [{lo}, {hi}] (psi cleared)"
            ))
        }

        QmCmd::Mass => {
            let m = pop_num(stack)?;
            if !m.is_finite() || m <= 0.0 {
                return Err(format!("QM MASS: must be finite and positive, got {m}"));
            }
            state.qm.mass = m;
            state.qm.invalidate();
            Ok(format!("mass = {m}"))
        }

        QmCmd::Hbar => {
            let h = pop_num(stack)?;
            if !h.is_finite() || h <= 0.0 {
                return Err(format!("QM HBAR: must be finite and positive, got {h}"));
            }
            state.qm.hbar = h;
            state.qm.invalidate();
            Ok(format!("hbar = {h}"))
        }

        QmCmd::States => {
            // Bound states come from the DIRICHLET Hamiltonian, whose
            // walls pin psi to zero. Under NASH the domain is periodic
            // and those are simply different eigenproblems — a periodic
            // box has Bloch states, not box states. Loading one into a
            // periodic run would be a silent boundary-condition mix, so
            // it is refused. `quantum::qm1d` has no periodic
            // eigensolver; that is a stated gap, not an oversight.
            if state.qm.nash_splitting().is_some() {
                return Err("QM: bound states are computed with Dirichlet walls, and the \
                            NASH method is periodic — the two are different eigenproblems. \
                            Use `QM METHOD CAYLEY` for bound states."
                    .to_string());
            }
            let k = pop_count(stack, "the state count")?;
            if k == 0 {
                return Err("QM STATES: ask for at least one state".to_string());
            }
            let ham = state.qm.hamiltonian()?;
            let (e, v) = ham.bound_states(k)?;
            let mut s = format!("{k} lowest bound state(s):\n");
            for (i, en) in e.iter().enumerate() {
                s.push_str(&format!("  E[{i}] = {en:.12}\n"));
            }
            state.qm.states = Some((e, v));
            Ok(s.trim_end().to_string())
        }

        QmCmd::LoadState => {
            // Bound states come from the DIRICHLET Hamiltonian, whose
            // walls pin psi to zero. Under NASH the domain is periodic
            // and those are simply different eigenproblems — a periodic
            // box has Bloch states, not box states. Loading one into a
            // periodic run would be a silent boundary-condition mix, so
            // it is refused. `quantum::qm1d` has no periodic
            // eigensolver; that is a stated gap, not an oversight.
            if state.qm.nash_splitting().is_some() {
                return Err("QM: bound states are computed with Dirichlet walls, and the \
                            NASH method is periodic — the two are different eigenproblems. \
                            Use `QM METHOD CAYLEY` for bound states."
                    .to_string());
            }
            let n = pop_count(stack, "the state index")?;
            let ham = state.qm.hamiltonian()?;
            // reuse the cached diagonalisation when it is deep enough
            let need = n + 1;
            let have = state.qm.states.as_ref().map(|(e, _)| e.len()).unwrap_or(0);
            if have < need {
                let (e, v) = ham.bound_states(need)?;
                state.qm.states = Some((e, v));
            }
            let (e, v) = state.qm.states.as_ref().expect("just filled");
            let w = Wavefunction::from_real(ham.grid.clone(), &v[n])?;
            let en = e[n];
            state.qm.psi = Some(w);
            state.qm.time = 0.0;
            Ok(format!("psi = bound state {n}, E = {en:.12}, t reset to 0"))
        }

        QmCmd::Packet => {
            let k0 = pop_num(stack)?;
            let sigma = pop_num(stack)?;
            let x0 = pop_num(stack)?;
            let grid = state
                .qm
                .grid
                .clone()
                .ok_or("QM PACKET: set a grid first (QM GRID <x_min> <x_max> <n>)")?;
            let w = Wavefunction::gaussian(grid, x0, sigma, k0)?;
            let edge = w.edge_probability(0.05);
            state.qm.psi = Some(w);
            state.qm.time = 0.0;
            let warn = if edge > 1e-6 {
                format!(
                    "\n  warning: {edge:.3e} of the packet already sits within 5% of a wall; \
                     the walls REFLECT, so widen the domain"
                )
            } else {
                String::new()
            };
            Ok(format!(
                "psi = Gaussian packet at x0 = {x0}, sigma = {sigma}, k0 = {k0}, t = 0{warn}"
            ))
        }

        QmCmd::Step | QmCmd::Run => {
            let (dt, steps) = if matches!(cmd, QmCmd::Step) {
                (pop_num(stack)?, 1usize)
            } else {
                let n = pop_count(stack, "the step count")?;
                let t = pop_num(stack)?;
                if n == 0 {
                    return Err("QM RUN: the step count must be at least 1".to_string());
                }
                (t / n as f64, n)
            };
            if !dt.is_finite() || dt == 0.0 {
                return Err(format!("QM: the time step must be finite and non-zero, got {dt}"));
            }
            if let Some(sp) = state.qm.nash_splitting() {
                let prop = state.qm.nash(dt, sp)?;
                let mut w = state.qm.wavefunction()?.clone();
                let n0 = w.norm();
                prop.run(&mut w.psi, steps)?;
                let n1 = w.norm();
                state.qm.psi = Some(w);
                state.qm.time += dt * steps as f64;
                return Ok(format!(
                    "t = {:.6}, {steps} step(s) of dt = {dt:.6e} by {}; norm {n0:.12} -> {n1:.12}",
                    state.qm.time, state.qm.method
                ));
            }
            let ham = state.qm.hamiltonian()?;
            let mut w = state.qm.wavefunction()?.clone();
            let n0 = w.norm();

            match state.qm.drive.clone() {
                None => {
                    Propagator::new(ham.clone(), dt)?.run(&mut w, steps)?;
                }
                Some((shape, _, time_name)) => {
                    // The modulation is a user function, so it must be
                    // evaluated through the VM — once per step, at the
                    // midpoint. Stepping manually rather than using
                    // `DrivenPropagator::run` because that takes a plain
                    // closure and this one needs `&mut SimState`.
                    let mut prop = DrivenPropagator::new(ham.clone(), shape, dt)?;
                    // continue the clock where the session left off
                    let t0 = state.qm.time;
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
                                    "QM RUN: `{time_name}(t)` must return a number, got {other}"
                                ))
                            }
                        };
                        // `step` samples its closure at ITS own midpoint;
                        // a constant closure hands it the value already
                        // computed at the correct absolute time.
                        prop.step(&mut w, |_| amp)?;
                    }
                }
            }
            let n1 = w.norm();
            let drift = (n1 / n0 - 1.0).abs();
            let edge = w.edge_probability(0.05);
            state.qm.time += dt * steps as f64;
            let t = state.qm.time;
            let e = w.energy(&ham);
            let driven = state.qm.drive.is_some();
            state.qm.psi = Some(w);
            let warn = if edge > 1e-4 {
                format!(
                    "\n  warning: {edge:.3e} of the probability is within 5% of a wall — \
                     the walls REFLECT, so results past this point are suspect"
                )
            } else {
                String::new()
            };
            let note = if driven {
                "  (<E> is of the STATIC potential; a driven system does not conserve it)\n"
            } else {
                ""
            };
            Ok(format!(
                "t = {t} ({steps} step(s) of dt = {dt}), <E> = {e:.12}, \
                 norm drift = {drift:.3e}\n{note}{warn}"
            )
            .trim_end()
            .to_string())
        }

        QmCmd::Norm => Ok(format!("{:.15}", state.qm.wavefunction()?.norm())),
        QmCmd::Position => Ok(format!("{:.15}", state.qm.wavefunction()?.position())),
        QmCmd::Momentum => {
            let hbar = state.qm.hbar;
            Ok(format!("{:.15}", state.qm.wavefunction()?.momentum(hbar)))
        }
        QmCmd::Method(m) => {
            state.qm.method = *m;
            // Bound states are Dirichlet and the cache would outlive the
            // boundary condition that produced it.
            state.qm.invalidate();
            Ok(format!("method {m}"))
        }

        QmCmd::Transmission => {
            let e = pop_num(stack)?;
            let (v, x0, x1) = state.qm.cells()?;
            let s = tm_scatter(&v, x0, x1, e, state.qm.mass, state.qm.hbar)?;
            stack.push(Value::Num(s.transmission));
            Ok(format!(
                "E = {e}: T = {:.12}, R = {:.12}, T + R = {:.12}",
                s.transmission,
                s.reflection,
                s.transmission + s.reflection
            ))
        }

        QmCmd::Scan => {
            let points = pop_count(stack, "the point count")?;
            let e_hi = pop_num(stack)?;
            let e_lo = pop_num(stack)?;
            let (v, x0, x1) = state.qm.cells()?;
            let range = EnergyRange { lo: e_lo, hi: e_hi, points };
            let curve = tm_scan(&v, x0, x1, range, state.qm.mass, state.qm.hbar)?;
            if curve.is_empty() {
                return Err(format!(
                    "QM SCAN: no energy in [{e_lo}, {e_hi}] could be solved — all of them \
                     are at or below the potential at the left edge"
                ));
            }
            // Interior maxima are the resonances, which are the reason
            // this command exists: a wavepacket averages them away.
            let mut peaks: Vec<(f64, f64)> = Vec::new();
            for w in curve.windows(3) {
                if w[1].transmission > w[0].transmission && w[1].transmission > w[2].transmission {
                    peaks.push((w[1].energy, w[1].transmission));
                }
            }
            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let hi = curve.iter().map(|s| s.transmission).fold(0.0, f64::max);
            let lo = curve.iter().map(|s| s.transmission).fold(f64::INFINITY, f64::min);
            let mut out = format!(
                "{} energies in [{e_lo}, {e_hi}] ({} refused), T from {lo:.3e} to {hi:.3e}",
                curve.len(),
                points - curve.len()
            );
            if peaks.is_empty() {
                out.push_str("\n  no interior maxima — T is monotone over this range");
            } else {
                out.push_str(&format!("\n  {} resonance(s), strongest first:", peaks.len()));
                for (e, t) in peaks.iter().take(8) {
                    out.push_str(&format!("\n    E = {e:.9}   T = {t:.9}"));
                }
            }
            stack.push(Value::List(curve.iter().map(|s| Value::Num(s.transmission)).collect()));
            Ok(out)
        }

        QmCmd::Energy => match state.qm.nash_splitting() {
            // The Dirichlet Hamiltonian drops the two wrap terms, which
            // are exactly the ones that matter when the run is periodic.
            Some(sp) => {
                let psi = state.qm.wavefunction()?.psi.clone();
                let e = state.qm.nash(state.qm.hbar.abs().max(1e-300), sp)?.energy(&psi)?;
                Ok(format!("{e:.15}"))
            }
            None => {
                let ham = state.qm.hamiltonian()?;
                Ok(format!("{:.15}", state.qm.wavefunction()?.energy(&ham)))
            }
        },
        QmCmd::Prob => {
            let b = pop_num(stack)?;
            let a = pop_num(stack)?;
            Ok(format!("{:.15}", state.qm.wavefunction()?.probability_in(a, b)))
        }
        QmCmd::Density => {
            let d = state.qm.wavefunction()?.density();
            stack.push(Value::List(d.into_iter().map(Value::Num).collect()));
            Ok(String::new())
        }

        QmCmd::Animate(path) => {
            let frames = pop_count(stack, "the frame count")?;
            let total = pop_num(stack)?;
            if frames < 2 {
                return Err("QM ANIMATE: ask for at least 2 frames".to_string());
            }
            if !total.is_finite() || total <= 0.0 {
                return Err(format!("QM ANIMATE: the total time must be positive, got {total}"));
            }
            let ham = state.qm.hamiltonian()?;
            let mut w = state.qm.wavefunction()?.clone();
            let grid = ham.grid.clone();

            // A frame per capture; each capture advances by total/frames.
            // The inner step count keeps dt small enough that the
            // animation is smooth AND the physics is resolved.
            let per_frame = 20usize;
            let dt = total / (frames * per_frame) as f64;
            // An animation MUST honour the drive. Using the static
            // propagator here made every frame identical when a drive
            // was set — a silently wrong picture, which is worse than an
            // error. Found by checking that the isosurface centroid
            // actually moved between frames; it did not.
            let drive = state.qm.drive.clone();
            let prop = Propagator::new(ham.clone(), dt)?;
            let mut driven = match &drive {
                Some((shape, _, _)) => {
                    Some(DrivenPropagator::new(ham.clone(), shape.clone(), dt)?)
                }
                None => None,
            };
            let t_start = state.qm.time;

            // Downsample along x so the file stays small; the eye cannot
            // use 2000 points across a plot anyway.
            let stride = (grid.n / 600).max(1);
            let xs: Vec<f64> = (0..grid.n).step_by(stride).map(|i| grid.x(i)).collect();
            let vs: Vec<f64> = ham
                .potential
                .iter()
                .enumerate()
                .filter(|(i, _)| i % stride == 0)
                .map(|(_, v)| *v)
                .collect();

            let mut series: Vec<Vec<f64>> = Vec::with_capacity(frames);
            let mut times: Vec<f64> = Vec::with_capacity(frames);
            let mut n0 = w.norm();
            let mut worst_drift = 0.0_f64;
            for f in 0..frames {
                if f > 0 {
                    advance_1d(
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
                let nn = w.norm();
                worst_drift = worst_drift.max((nn / n0 - 1.0).abs());
                n0 = if f == 0 { nn } else { n0 };
                let d = w.density();
                series.push((0..grid.n).step_by(stride).map(|i| d[i]).collect());
                times.push(state.qm.time + dt * (f * per_frame) as f64);
            }
            state.qm.time += total;
            state.qm.psi = Some(w);

            let num = |v: &[f64]| {
                v.iter()
                    .map(|x| format!("{x:.6}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let frames_js = series
                .iter()
                .map(|f| format!("[{}]", num(f)))
                .collect::<Vec<_>>()
                .join(",\n");
            let label = state
                .qm
                .potential_name
                .clone()
                .unwrap_or_else(|| "unnamed".to_string());
            let html = render_html(
                &num(&xs),
                &num(&vs),
                &frames_js,
                &num(&times),
                &label,
                grid.x_min,
                grid.x_max,
            );
            std::fs::write(path, &html)
                .map_err(|e| format!("QM ANIMATE: cannot write `{path}`: {e}"))?;
            Ok(format!(
                "wrote {path} — {frames} frames over t = {total} (dt = {dt:.6}, \
                 {} points per frame), worst norm drift {worst_drift:.3e}. \
                 Open it in a browser.",
                xs.len()
            ))
        }

        QmCmd::Drive(shape_name, time_name) => {
            let grid = state
                .qm
                .grid
                .clone()
                .ok_or("QM DRIVE: set a grid first (QM GRID <x_min> <x_max> <n>)")?;
            for nm in [shape_name, time_name] {
                if !state.functions.contains_key(nm) {
                    return Err(format!(
                        "QM DRIVE: no function `{nm}` — define the spatial shape as \
                         `DEF {nm}(x) {{ ... }}` and the modulation as `DEF f(t) {{ ... }}`"
                    ));
                }
            }
            // The SHAPE is sampled once; the MODULATION is evaluated per
            // step. A general V(x,t) would need n user-function calls
            // every step, which would cost more than the solve it feeds.
            let mut shape = Vec::with_capacity(grid.n);
            for i in 0..grid.n {
                let v = crate::vm::call_user_function_public(
                    shape_name,
                    vec![Value::Num(grid.x(i))],
                    state,
                )?;
                match v {
                    Value::Num(y) => shape.push(y),
                    other => {
                        return Err(format!(
                            "QM DRIVE: `{shape_name}(x)` must return a number, got {other}"
                        ))
                    }
                }
            }
            let lo = shape.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = shape.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            state.qm.drive = Some((shape, shape_name.clone(), time_name.clone()));
            state.qm.invalidate();
            Ok(format!(
                "drive V(x,t) += {time_name}(t) * {shape_name}(x), shape in [{lo}, {hi}]. \
                 The energy is NO LONGER conserved — a driven system exchanges energy with \
                 its drive — but propagation stays unitary. QM STATES uses the STATIC \
                 potential only."
            ))
        }

        QmCmd::DriveOff => {
            state.qm.drive = None;
            state.qm.invalidate();
            Ok("drive removed".to_string())
        }

        QmCmd::Absorb => {
            let power = pop_num(stack)?;
            let strength = pop_num(stack)?;
            let width = pop_num(stack)?;
            // Validate NOW against the current grid rather than at the
            // next propagation, so a bad number is reported where it
            // was typed.
            if let Some(g) = state.qm.grid.clone() {
                let probe = Hamiltonian::new(g, vec![0.0; state.qm.grid.as_ref().unwrap().n], 1.0, 1.0)?;
                probe.with_absorber(width, strength, power)?;
            }
            state.qm.absorber = Some((width, strength, power));
            state.qm.invalidate();
            Ok(format!(
                "absorbing edges: width {width}, strength {strength}, power {power}. \
                 Propagation is NO LONGER unitary — the norm decays, which is the absorber \
                 working. QM STATES is unavailable while this is on."
            ))
        }

        QmCmd::AbsorbOff => {
            state.qm.absorber = None;
            state.qm.invalidate();
            Ok("absorbing edges removed — the walls reflect again".to_string())
        }

        QmCmd::Reset => {
            state.qm = QmState::default();
            Ok("quantum state cleared".to_string())
        }
    }
}

/// Advance a 1-D wavefunction by `steps`, honouring a drive if one is
/// set. Split out because three call sites need it and each must not
/// quietly fall back to the static propagator.
#[allow(clippy::too_many_arguments)]
fn advance_1d(
    w: &mut Wavefunction,
    prop: &Propagator,
    driven: Option<&mut DrivenPropagator>,
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

/// Render the self-contained animation page.
///
/// Everything is inlined — no scripts, styles or fonts fetched from
/// anywhere — so the file works from `file://`, survives being emailed,
/// and cannot phone home.
#[allow(clippy::too_many_arguments)]
fn render_html(
    xs: &str,
    vs: &str,
    frames: &str,
    times: &str,
    label: &str,
    x_min: f64,
    x_max: f64,
) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>posim — 1-D quantum scattering</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
        background:#0e1116; color:#e6e6e6; }}
 header {{ padding:14px 18px; border-bottom:1px solid #263042; }}
 h1 {{ margin:0; font-size:16px; font-weight:600; }}
 .sub {{ color:#8b98ad; font-size:12px; margin-top:3px; }}
 #wrap {{ padding:14px 18px; }}
 canvas {{ width:100%; height:auto; display:block; background:#11151c;
           border:1px solid #263042; border-radius:6px; }}
 .row {{ display:flex; gap:12px; align-items:center; margin-top:12px; flex-wrap:wrap; }}
 button {{ background:#1b2330; color:#e6e6e6; border:1px solid #33405a;
           border-radius:5px; padding:6px 14px; cursor:pointer; font:inherit; }}
 button:hover {{ background:#243149; }}
 input[type=range] {{ flex:1; min-width:200px; }}
 .stat {{ font-variant-numeric:tabular-nums; color:#9fb3d0; }}
 .key {{ display:flex; gap:16px; margin-top:8px; font-size:12px; color:#8b98ad; }}
 .sw {{ display:inline-block; width:22px; height:3px; vertical-align:middle;
        margin-right:5px; border-radius:2px; }}
</style></head><body>
<header>
  <h1>1-D quantum scattering — |&psi;(x,t)|&sup2;</h1>
  <div class="sub">potential: <b>{label}</b> &middot; domain [{x_min}, {x_max}] &middot;
  generated by posim</div>
</header>
<div id="wrap">
  <canvas id="c" width="1200" height="460"></canvas>
  <div class="row">
    <button id="play">Pause</button>
    <button id="rew">Restart</button>
    <input type="range" id="scrub" min="0" value="0">
    <span class="stat" id="stat"></span>
  </div>
  <div class="key">
    <span><i class="sw" style="background:#4ea3ff"></i>|&psi;|&sup2;</span>
    <span><i class="sw" style="background:#ff7a59"></i>V(x), scaled</span>
    <span><i class="sw" style="background:#3ddc97"></i>transmitted region</span>
  </div>
</div>
<script>
const X = [{xs}], V = [{vs}], T = [{times}];
const F = [
{frames}
];
const c = document.getElementById('c'), g = c.getContext('2d');
const scrub = document.getElementById('scrub');
scrub.max = F.length - 1;
let i = 0, playing = true;

const xmin = Math.min(...X), xmax = Math.max(...X);
let ymax = 0; for (const f of F) for (const v of f) if (v > ymax) ymax = v;
ymax *= 1.08;
const vmax = Math.max(...V.map(Math.abs)) || 1;
// The barrier sits where V > 0; anything to its right counts as
// transmitted. Found from the potential itself so the readout stays
// correct for whatever potential was used.
let vRight = xmin; for (let k = 0; k < X.length; k++) if (V[k] > 0) vRight = Math.max(vRight, X[k]);

const PAD = {{l: 58, r: 16, t: 14, b: 34}};
function px(x) {{ return PAD.l + (x - xmin) / (xmax - xmin) * (c.width - PAD.l - PAD.r); }}
function py(y) {{ return c.height - PAD.b - y / ymax * (c.height - PAD.t - PAD.b); }}

function frameStats(f) {{
  // trapezoid over the transmitted half
  let tot = 0, tr = 0;
  for (let k = 1; k < X.length; k++) {{
    const dx = X[k] - X[k-1], a = (f[k] + f[k-1]) / 2;
    tot += a * dx;
    if (X[k] > vRight) tr += a * dx;
  }}
  return {{tot, tr}};
}}

function draw() {{
  const f = F[i];
  g.clearRect(0, 0, c.width, c.height);
  // axes
  g.strokeStyle = '#263042'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(PAD.l, py(0)); g.lineTo(c.width - PAD.r, py(0)); g.stroke();
  // transmitted region shading
  g.fillStyle = 'rgba(61,220,151,.07)';
  g.fillRect(px(vRight), PAD.t, c.width - PAD.r - px(vRight), c.height - PAD.t - PAD.b);
  // potential, scaled to the top third
  g.strokeStyle = '#ff7a59'; g.lineWidth = 1.6; g.beginPath();
  for (let k = 0; k < X.length; k++) {{
    const y = c.height - PAD.b - (V[k] / vmax) * (c.height - PAD.t - PAD.b) * 0.30;
    k ? g.lineTo(px(X[k]), y) : g.moveTo(px(X[k]), y);
  }}
  g.stroke();
  // |psi|^2, filled
  g.beginPath(); g.moveTo(px(X[0]), py(0));
  for (let k = 0; k < X.length; k++) g.lineTo(px(X[k]), py(f[k]));
  g.lineTo(px(X[X.length-1]), py(0)); g.closePath();
  g.fillStyle = 'rgba(78,163,255,.22)'; g.fill();
  g.strokeStyle = '#4ea3ff'; g.lineWidth = 1.8; g.stroke();
  // x labels
  g.fillStyle = '#8b98ad'; g.font = '12px ui-sans-serif,system-ui,sans-serif';
  g.textAlign = 'center';
  for (let n = 0; n <= 4; n++) {{
    const x = xmin + (xmax - xmin) * n / 4;
    g.fillText(x.toFixed(0), px(x), c.height - 12);
  }}
  g.textAlign = 'left'; g.fillText('|psi|^2', 8, PAD.t + 10);
  const s = frameStats(f);
  document.getElementById('stat').textContent =
    `t = ${{T[i].toFixed(2)}}   frame ${{i+1}}/${{F.length}}   ` +
    `norm = ${{s.tot.toFixed(6)}}   transmitted = ${{s.tr.toFixed(6)}}`;
  scrub.value = i;
}}

let last = 0;
function loop(ts) {{
  if (playing && ts - last > 45) {{ i = (i + 1) % F.length; last = ts; draw(); }}
  requestAnimationFrame(loop);
}}
document.getElementById('play').onclick = e => {{
  playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';
}};
document.getElementById('rew').onclick = () => {{ i = 0; draw(); }};
scrub.oninput = e => {{ i = +e.target.value; playing = false;
  document.getElementById('play').textContent = 'Play'; draw(); }};
draw(); requestAnimationFrame(loop);
</script></body></html>
"##
    )
}

#[cfg(test)]
mod tests {
    use super::{EvolveMethod, QmCmd, Splitting, QM_SUBCOMMANDS};
    use crate::vm::{execute_line, SimState};

    /// The parser must own every word in [`QM_SUBCOMMANDS`].
    ///
    /// Most of them need arguments, so parsing `qm <word>` alone usually
    /// fails — but it must not fail with *unknown subcommand*. That
    /// distinction is what makes the list authoritative rather than
    /// decorative: add a word here that the parser does not handle and
    /// this fails.
    #[test]
    fn the_parser_owns_every_listed_subcommand() {
        for w in QM_SUBCOMMANDS {
            let err = match crate::parser::compile_line(&format!("qm {w}")) {
                Ok(_) => continue,
                Err(e) => e,
            };
            assert!(
                !err.contains("unknown subcommand"),
                "`qm {w}` is listed but the parser does not know it: {err}"
            );
        }
        // ...and the check can fail, which is the point.
        let err = crate::parser::compile_line("qm wibble").unwrap_err();
        assert!(err.contains("unknown subcommand"), "got: {err}");
    }

    /// The five-way lockstep, mechanically.
    ///
    /// A `QM` subcommand is not added until it is parseable, in
    /// `HELP_TEXT`, in the parser's EBNF comment, and in **both** grammar
    /// documents. Forget one and the build fails here rather than in a
    /// reader's hands.
    #[test]
    fn every_qm_subcommand_is_documented_in_lockstep() {
        // `\_` in LaTeX, `\` nowhere else; case differs between the
        // parser (lowercase) and the documents (upper).
        let prep = |s: &str| s.replace('\\', "").to_ascii_uppercase();
        let help = prep(crate::vm::HELP_TEXT);
        // Only the `qmcmd` production, not the whole file: qm2cmd and
        // qm3cmd quote the same words, so searching the file would pass
        // on a word this family never declared.
        let all = prep(include_str!("parser.rs"));
        let from = all.find("QMCMD").expect("parser.rs must declare the qmcmd production");
        let to = all[from..].find("QM2CMD").map_or(all.len(), |k| from + k);
        let ebnf = all[from..to].to_string();
        let md = prep(include_str!("../../grammar.md"));
        let tex = prep(include_str!("../../grammar.tex"));

        let mut missing = Vec::new();
        for w in QM_SUBCOMMANDS {
            let up = w.to_ascii_uppercase();
            // Each document spells it its own way: the EBNF quotes the
            // bare word, everything else writes `QM <WORD>`. Quoting in
            // the EBNF needle is what keeps "STATE" from matching
            // inside "STATES".
            let needle = format!("QM {up}");
            let ebnf_needle = format!("\"{up}\"");
            for (what, hay) in [
                ("HELP_TEXT", &help),
                ("parser.rs EBNF", &ebnf),
                ("grammar.md", &md),
                ("grammar.tex", &tex),
            ] {
                // `QM STATUS` is spelled as bare `QM` in the documents,
                // and the EBNF writes the optional form `[ "STATUS" ]`.
                if *w == "status" && what != "parser.rs EBNF" {
                    continue;
                }
                let want = if what == "parser.rs EBNF" { &ebnf_needle } else { &needle };
                if !hay.contains(want.as_str()) {
                    missing.push(format!("{want} is missing from {what}"));
                }
            }
        }
        assert!(missing.is_empty(), "QM grammar lockstep is broken:\n  {}", missing.join("\n  "));
    }

    /// The two methods, and the boundary condition each implies.
    #[test]
    fn qm_method_parses_every_form() {
        for (src, want) in [
            ("qm method cayley", EvolveMethod::Cayley),
            ("qm method nash", EvolveMethod::Nash(Splitting::Lie)),
            ("qm method nash lie", EvolveMethod::Nash(Splitting::Lie)),
            ("qm method nash strang", EvolveMethod::Nash(Splitting::Strang)),
            ("QM METHOD NASH STRANG", EvolveMethod::Nash(Splitting::Strang)),
        ] {
            let prog = crate::parser::compile_line(src)
                .unwrap_or_else(|e| panic!("`{src}` did not parse: {e}"));
            let got = prog.iter().find_map(|i| match i {
                crate::vm::Instr::Qm(QmCmd::Method(m)) => Some(*m),
                _ => None,
            });
            assert_eq!(got, Some(want), "`{src}`");
        }
        assert!(crate::parser::compile_line("qm method fourier").is_err());
        assert!(crate::parser::compile_line("qm method").is_err());
    }


    fn run(lines: &[&str]) -> (SimState, Vec<String>) {
        let mut st = SimState::default();
        let mut out = Vec::new();
        for l in lines {
            let v = execute_line(l, &mut st).unwrap_or_else(|e| panic!("`{l}` failed: {e}"));
            out.push(v.to_string());
        }
        (st, out)
    }

    /// **The result `TUNNELING_RESULTS.md` §5 recorded as unreachable.**
    ///
    /// That section swept a double barrier with wavepackets, found a
    /// monotone rise with no peak, tried a four-times narrower packet
    /// in `k`, still found none, and recorded the negative result
    /// rather than tuning until something appeared. The diagnosis was
    /// that the resonances are narrower than any affordable packet's
    /// momentum spread.
    ///
    /// At fixed energy they are simply there, and this asserts it
    /// through the language: two peaks at essentially unit
    /// transmission, standing three to four orders of magnitude above
    /// the background between them.
    #[test]
    fn qm_scan_finds_the_resonances_wavepackets_could_not() {
        let (st, out) = run(&[
            "def dbl(x) { 3 * ((abs(x) > 1.5) * (abs(x) < 2)) }",
            "qm grid -12 12 4800",
            "qm potential dbl",
            "qm scan 0.05 2.9 1200",
            "qm transmission 0.5",
        ]);
        let report = &out[3];
        assert!(report.contains("resonance"), "no resonances reported:\n{report}");
        // Two quasi-bound states in this well, both at ~unit T.
        assert!(report.contains("E = 0.31"), "missing the lower resonance:\n{report}");
        assert!(report.contains("E = 1.28"), "missing the upper resonance:\n{report}");
        assert!(report.contains("T = 0.99"), "a resonance should reach ~1:\n{report}");

        // Off resonance the same barrier is nearly opaque — the
        // contrast is the whole point.
        let off = &out[4];
        assert!(off.contains("T = 0.03"), "off-resonance transmission:\n{off}");
        assert!(off.contains("T + R = 1.000000000000"), "flux must balance:\n{off}");

        // SCAN pushes the curve so it can be plotted or reduced; the
        // `run` helper stringifies whatever the line returned.
        let _ = &st;
        assert!(report.starts_with("1200 energies"), "point count:\n{report}");
    }

    /// A transfer matrix needs a real potential where the wave comes
    /// in, so an absorber is refused rather than silently ignored.
    #[test]
    fn the_transfer_matrix_refuses_an_absorbing_boundary() {
        let mut st = SimState::default();
        for l in ["qm grid -8 8 400", "qm potential zero", "qm absorb 2 0.5"] {
            execute_line(l, &mut st).unwrap();
        }
        let e = execute_line("qm transmission 1.0", &mut st).unwrap_err();
        assert!(e.contains("absorber") || e.contains("real potential"), "got: {e}");
        assert!(e.contains("ABSORB OFF"), "the refusal must name the way out: {e}");
    }

    /// The Nash method, end to end through the language, judged by
    /// something the language can check: a free packet's energy is a
    /// constant of the motion, and the norm is conserved.
    #[test]
    fn nash_propagates_through_the_language() {
        for method in ["qm method nash", "qm method nash strang"] {
            let (_, out) = run(&[
                "qm grid -8 8 200",
                "qm potential zero",
                "qm packet -2 0.7 4",
                method,
                "qm energy",
                "qm run 1 steps 200",
                "qm energy",
                "qm norm",
            ]);
            let e0: f64 = out[4].parse().unwrap();
            let e1: f64 = out[6].parse().unwrap();
            let n: f64 = out[7].parse().unwrap();
            // A free particle's energy is conserved exactly by this
            // scheme: with V constant the two factors commute, so there
            // is no splitting error to leak into it.
            assert!((e1 - e0).abs() < 1e-9, "{method}: energy {e0} -> {e1}");
            assert!((n - 1.0).abs() < 1e-9, "{method}: norm {n}");
            assert!(e0 > 7.0 && e0 < 9.0, "{method}: E = {e0} is not a k0 = 4 packet");
        }
    }

    /// The boundary condition, demonstrated rather than asserted.
    ///
    /// The same packet, the same grid, the same time — launched at the
    /// right-hand edge. Under `NASH` it **wraps** and arrives at the far
    /// left; under `CAYLEY` it **reflects** and stays. If the two
    /// methods ever quietly shared a boundary condition, this is the
    /// test that would notice.
    #[test]
    fn nash_wraps_where_cayley_reflects() {
        let script = |method: &str| {
            let mut lines = vec!["qm grid 0 20 400", "qm potential zero", "qm packet 17 0.8 6"];
            if !method.is_empty() {
                lines.push(method);
            }
            lines.push("qm run 0.85 steps 400");
            lines.push("qm prob 0 5");
            lines.push("qm prob 15 20");
            let (_, out) = run(&lines);
            let n = out.len();
            let left: f64 = out[n - 2].parse().unwrap();
            let right: f64 = out[n - 1].parse().unwrap();
            (left, right)
        };

        let (left, right) = script("qm method nash strang");
        assert!(left > 0.9, "NASH: only {left} of the packet came round the seam");
        assert!(right < 0.1, "NASH: {right} stayed at the right edge");

        let (left, right) = script("");
        assert!(left < 0.01, "CAYLEY: {left} leaked through a reflecting wall");
        assert!(right > 0.9, "CAYLEY: only {right} bounced back");
    }

    /// The combinations that have no meaning are refused, with the way
    /// out named in the message.
    ///
    /// Each of these could have been implemented as a silent no-op, and
    /// each would then have produced a plausible, wrong picture — a
    /// packet that never absorbs, a drive that never drives, bound
    /// states from the wrong boundary condition.
    #[test]
    fn nash_refuses_what_it_cannot_do() {
        let base = ["qm grid -8 8 100", "qm potential zero", "qm packet -2 0.7 4"];
        let cases: [(&[&str], &str); 3] = [
            (&["qm method nash", "qm states 3"], "Dirichlet"),
            (&["qm absorb 2 0.5", "qm method nash", "qm run 1 steps 10"], "absorbing"),
            (
                &[
                    "def g(x) { x }",
                    "def f(t) { t }",
                    "qm drive g f",
                    "qm method nash",
                    "qm run 1 steps 10",
                ],
                "drive",
            ),
        ];
        for (extra, needle) in cases {
            let mut st = SimState::default();
            let mut refused: Option<String> = None;
            for l in base.iter().chain(extra.iter()) {
                if let Err(e) = execute_line(l, &mut st) {
                    refused = Some(e);
                    break;
                }
            }
            // The whole point is that it REFUSES. Without this the test
            // passes when nothing goes wrong, which is the failure mode
            // it exists to catch.
            let e = refused
                .unwrap_or_else(|| panic!("`{needle}` case was accepted; it must be refused"));
            assert!(e.contains(needle), "expected a message mentioning `{needle}`, got: {e}");
            assert!(
                e.contains("CAYLEY") || e.contains("cayley"),
                "the refusal must name the way out: {e}"
            );
        }
    }

    /// Switching to NASH does not move a single grid point.
    ///
    /// The Dirichlet grid's `n` interior points are `n` periodic points
    /// at the same spacing, which is why the potential samples carry
    /// over untouched. If that ever stopped being true the potential
    /// would be silently misaligned, so it is checked rather than
    /// assumed.
    #[test]
    fn switching_method_moves_no_grid_point() {
        let (st, _) = run(&["qm grid -3.5 2.25 64", "qm potential zero"]);
        let g = st.qm.grid.clone().unwrap();
        let p = st.qm.nash(1e-3, Splitting::Lie).unwrap();
        let pg = p.grid();
        assert_eq!(pg.n, g.n);
        assert!((pg.h() - g.h()).abs() < 1e-15, "spacing changed");
        for i in 0..g.n {
            assert!((pg.x(i) - g.x(i)).abs() < 1e-12, "point {i} moved");
        }
    }

    /// The status line must state the boundary condition in force, and
    /// it must CHANGE with the method — saying "the walls REFLECT"
    /// during a periodic run would be a plain falsehood.
    #[test]
    fn the_status_line_reports_the_boundary() {
        let (_, out) = run(&["qm grid -8 8 64", "qm potential zero", "qm", "qm method nash", "qm"]);
        assert!(out[2].contains("REFLECT"), "cayley status: {}", out[2]);
        assert!(out[4].contains("PERIODIC"), "nash status: {}", out[4]);
        assert!(!out[4].contains("REFLECT"), "nash must not claim reflection: {}", out[4]);
    }

    /// The harmonic oscillator, entirely through the language: define a
    /// potential as an ordinary function, sample it, diagonalise.
    #[test]
    fn harmonic_oscillator_through_the_language() {
        let (_, out) = run(&[
            "def v(x) { 0.5 * x * x }",
            "qm grid -8 8 250",
            "qm potential v",
            "qm states 4",
        ]);
        let text = out.last().unwrap();
        // E_n = n + 1/2
        for (n, want) in [(0, 0.5), (1, 1.5), (2, 2.5), (3, 3.5)] {
            let line = text
                .lines()
                .find(|l| l.trim_start().starts_with(&format!("E[{n}]")))
                .unwrap_or_else(|| panic!("no E[{n}] in:\n{text}"));
            let got: f64 = line.split('=').nth(1).unwrap().trim().parse().unwrap();
            assert!((got - want).abs() < 5e-3, "E[{n}] = {got}, want {want}");
        }
    }

    /// A bound state loaded into psi must have the eigenvalue as its
    /// energy, and stay put under propagation.
    #[test]
    fn a_loaded_eigenstate_is_stationary() {
        let (_, out) = run(&[
            "def v(x) { 0.5 * x * x }",
            "qm grid -8 8 120",
            "qm potential v",
            "qm state 1",
            "qm energy",
            "qm run 2 steps 200",
            "qm energy",
        ]);
        let e_before: f64 = out[4].trim().parse().unwrap();
        let e_after: f64 = out[6].trim().parse().unwrap();
        assert!((e_before - 1.5).abs() < 5e-3, "E = {e_before}, want ~1.5");
        assert!(
            (e_after - e_before).abs() < 1e-9,
            "energy moved: {e_before} -> {e_after}"
        );
    }

    /// Propagation is unitary, so the reported norm drift must be tiny.
    #[test]
    fn propagation_conserves_the_norm() {
        let (_, out) = run(&[
            "qm grid -40 40 400",
            "qm potential zero",
            "qm packet -10 2 2",
            "qm run 3 steps 300",
            "qm norm",
        ]);
        let norm: f64 = out[4].trim().parse().unwrap();
        assert!((norm - 1.0).abs() < 1e-10, "norm = {norm}");
        assert!(out[3].contains("norm drift"), "run should report drift");
    }

    /// The walls reflect. A packet aimed at one must SAY so rather than
    /// quietly returning a wrong scattering answer.
    #[test]
    fn approaching_a_wall_is_reported() {
        let (_, out) = run(&[
            "qm grid -10 10 200",
            "qm potential zero",
            "qm packet 0 1 8",
            "qm run 2 steps 200",
        ]);
        assert!(
            out[3].contains("warning") && out[3].contains("wall"),
            "expected a wall warning, got: {}",
            out[3]
        );
    }

    /// Changing the grid invalidates the potential: a stale potential of
    /// the wrong length would otherwise fail much later and confusingly.
    #[test]
    fn changing_the_grid_clears_downstream_state() {
        let mut st = SimState::default();
        for l in [
            "def v(x) { 0.5 * x * x }",
            "qm grid -8 8 100",
            "qm potential v",
            "qm state 0",
        ] {
            execute_line(l, &mut st).unwrap();
        }
        assert!(st.qm.psi.is_some());
        execute_line("qm grid -8 8 150", &mut st).unwrap();
        assert!(st.qm.potential.is_none(), "potential should be cleared");
        assert!(st.qm.psi.is_none(), "psi should be cleared");
        let e = execute_line("qm states 2", &mut st).unwrap_err();
        assert!(e.contains("no potential"), "got: {e}");
    }

    /// The built-in shapes exist because the language has no comparison
    /// operators, so a piecewise potential cannot be written as a DEF.
    /// Tunnelling through a square barrier is the canonical 1-D problem,
    /// so it must be reachable.
    #[test]
    fn a_square_barrier_transmits_and_reflects() {
        let (_, out) = run(&[
            "qm grid -60 60 1200",
            "qm potential barrier 2.5 0 1",
            "qm packet -20 2 2",
            "qm run 20 steps 2000",
            "qm prob 1 60",
            "qm prob -60 0",
        ]);
        let t: f64 = out[4].trim().parse().unwrap();
        let r: f64 = out[5].trim().parse().unwrap();
        // E0 = 2 against a 2.5 barrier: substantial tunnelling, and the
        // two channels must exhaust the probability.
        assert!((0.25..0.40).contains(&t), "T = {t}");
        assert!((t + r - 1.0).abs() < 1e-4, "T + R = {}", t + r);
    }

    /// A well binds states below zero; a barrier of the same footprint
    /// binds none.
    #[test]
    fn a_well_binds_states_and_a_barrier_does_not() {
        let (_, out) = run(&[
            "qm grid -20 20 300",
            "qm potential well 5, -2, 2",
            "qm states 3",
        ]);
        let bound = out[2]
            .lines()
            .filter_map(|l| l.split('=').nth(1))
            .filter_map(|v| v.trim().parse::<f64>().ok())
            .filter(|&e| e < 0.0)
            .count();
        assert!(bound >= 2, "expected bound states below 0, got {bound} in:\n{}", out[2]);

        let (_, out2) = run(&[
            "qm grid -20 20 300",
            "qm potential barrier 5, -2, 2",
            "qm states 3",
        ]);
        let neg = out2[2]
            .lines()
            .filter_map(|l| l.split('=').nth(1))
            .filter_map(|v| v.trim().parse::<f64>().ok())
            .filter(|&e| e < 0.0)
            .count();
        assert_eq!(neg, 0, "a barrier must bind nothing below zero");
    }

    /// Negative arguments need commas, because `5 -2` is subtraction.
    /// Both spellings must behave, and the failure must be a parse
    /// error rather than a silently wrong potential.
    #[test]
    fn negative_arguments_need_comma_separation() {
        let mut st = SimState::default();
        execute_line("qm grid -20 20 100", &mut st).unwrap();
        // commas: unambiguous
        assert!(execute_line("qm potential well 5, -2, 2", &mut st).is_ok());
        // spaces with a negative: `5 -2` is subtraction, so an argument
        // goes missing and this MUST fail loudly
        assert!(
            execute_line("qm potential well 5 -2 2", &mut st).is_err(),
            "the ambiguous spelling must be a parse error, not a wrong potential"
        );
        // parentheses work too
        assert!(execute_line("qm potential well 5 (-2) 2", &mut st).is_ok());
        // all-positive space separation still reads fine
        assert!(execute_line("qm potential barrier 2.5 0 1", &mut st).is_ok());
    }

    /// A feature narrower than the grid spacing falls between points and
    /// would silently act as if absent. Refuse instead.
    #[test]
    fn a_subgrid_feature_is_refused() {
        let mut st = SimState::default();
        execute_line("qm grid -10 10 50", &mut st).unwrap();
        let e = execute_line("qm potential barrier 5 0 0.01", &mut st).unwrap_err();
        assert!(e.contains("narrower than the grid spacing"), "got: {e}");
    }

    /// Comparison operators make a piecewise potential writable as an
    /// ordinary user function — the thing that was impossible before.
    #[test]
    fn a_barrier_can_be_written_as_a_user_function() {
        let mut st = SimState::default();
        for l in [
            "def barrier(x) { 2.5 * (x > 0) * (x < 1) }",
            "qm grid -40 40 800",
            "qm potential barrier",
        ] {
            execute_line(l, &mut st).unwrap_or_else(|e| panic!("`{l}`: {e}"));
        }
        let v = st.qm.potential.as_ref().unwrap();
        let g = st.qm.grid.as_ref().unwrap();
        for (i, &vi) in v.iter().enumerate() {
            let x = g.x(i);
            let want = if x > 0.0 && x < 1.0 { 2.5 } else { 0.0 };
            assert!((vi - want).abs() < 1e-12, "V({x}) = {vi}, want {want}");
        }
    }

    /// A bare name is the USER's function; a name with arguments is the
    /// built-in shape. Without this rule a user's `barrier` would be
    /// shadowed by the built-in of the same name.
    #[test]
    fn a_user_function_is_not_shadowed_by_the_builtin_shape() {
        let mut st = SimState::default();
        execute_line("def barrier(x) { 7 * (x > 2) * (x < 3) }", &mut st).unwrap();
        execute_line("qm grid -10 10 400", &mut st).unwrap();
        execute_line("qm potential barrier", &mut st).unwrap();
        let hi = st.qm.potential.as_ref().unwrap().iter().cloned().fold(0.0_f64, f64::max);
        assert!((hi - 7.0).abs() < 1e-12, "the user's barrier should peak at 7, got {hi}");
        // with arguments it is the built-in shape instead
        execute_line("qm potential barrier 1.5, -1, 1", &mut st).unwrap();
        let hi = st.qm.potential.as_ref().unwrap().iter().cloned().fold(0.0_f64, f64::max);
        assert!((hi - 1.5).abs() < 1e-12, "the built-in should peak at 1.5, got {hi}");
    }

    /// The absorber must buy domain size without changing the answer:
    /// a short absorbing domain and a long reflection-free one must
    /// agree on the transmission.
    #[test]
    fn an_absorbing_short_domain_agrees_with_a_long_one() {
        let long = run(&[
            "def barrier(x) { 2.5 * (x > 0) * (x < 1) }",
            "qm grid -100 100 2000",
            "qm potential barrier",
            "qm packet -25 2 2",
            "qm run 20 steps 2000",
            "qm prob 1 30",
        ]);
        let short = run(&[
            "def barrier(x) { 2.5 * (x > 0) * (x < 1) }",
            "qm grid -45 45 1350",
            "qm potential barrier",
            "qm absorb 15 3",
            "qm packet -25 2 2",
            "qm run 20 steps 2000",
            "qm prob 1 30",
        ]);
        let t_long: f64 = long.1[5].trim().parse().unwrap();
        let t_short: f64 = short.1[6].trim().parse().unwrap();
        let rel = (t_short - t_long).abs() / t_long;
        assert!(rel < 0.01, "T short {t_short} vs long {t_long} ({:.3}% apart)", 100.0 * rel);
    }

    /// An absorbing Hamiltonian is non-Hermitian, so bound states must
    /// be refused; turning the absorber off restores them.
    #[test]
    fn absorber_blocks_bound_states_and_off_restores_them() {
        let mut st = SimState::default();
        for l in ["qm grid -10 10 150", "qm potential zero", "qm absorb 3 2"] {
            execute_line(l, &mut st).unwrap();
        }
        let e = execute_line("qm states 2", &mut st).unwrap_err();
        assert!(e.contains("Hermitian"), "got: {e}");
        execute_line("qm absorb off", &mut st).unwrap();
        assert!(execute_line("qm states 2", &mut st).is_ok());
    }

    /// A driven oscillator through the language must trace the
    /// classical trajectory, which for a quadratic potential with a
    /// linear drive is exact by Ehrenfest.
    #[test]
    fn a_drive_reproduces_the_classical_trajectory() {
        let (_, out) = run(&[
            "def v(x) { 0.5 * x * x }",
            "def dipole(x) { x }",
            "def f(t) { 0.3 * cos(0.7 * t) }",
            "qm grid -12 12 400",
            "qm potential v",
            "qm state 0",
            "qm drive dipole, f",
            "qm run 5 steps 500",
            "qm position",
            "qm norm",
        ]);
        let x: f64 = out[8].trim().parse().unwrap();
        let norm: f64 = out[9].trim().parse().unwrap();
        // x(t) = -F0/(1-w^2) [cos(w t) - cos t]
        let (f0, om, t) = (0.3_f64, 0.7_f64, 5.0_f64);
        let want = -f0 / (1.0 - om * om) * ((om * t).cos() - t.cos());
        assert!((x - want).abs() < 0.01, "<x> = {x}, classical {want}");
        // driven but still unitary
        assert!((norm - 1.0).abs() < 1e-10, "norm = {norm}");
        // and the run reports that <E> is of the static potential
        assert!(out[7].contains("STATIC"), "expected a note about <E>: {}", out[7]);
    }

    /// Turning the drive off restores the undriven behaviour: a bound
    /// state stops moving.
    #[test]
    fn drive_off_restores_a_stationary_state() {
        let mut st = SimState::default();
        for l in [
            "def v(x) { 0.5 * x * x }",
            "def dipole(x) { x }",
            "def f(t) { 0.3 }",
            "qm grid -10 10 200",
            "qm potential v",
            "qm state 0",
            "qm drive dipole, f",
        ] {
            execute_line(l, &mut st).unwrap();
        }
        execute_line("qm run 2 steps 200", &mut st).unwrap();
        let moved: f64 = execute_line("qm position", &mut st).unwrap().to_string().trim().parse().unwrap();
        assert!(moved.abs() > 0.05, "a constant drive should displace it, got {moved}");

        execute_line("qm drive off", &mut st).unwrap();
        execute_line("qm state 0", &mut st).unwrap();
        execute_line("qm run 2 steps 200", &mut st).unwrap();
        let still: f64 = execute_line("qm position", &mut st).unwrap().to_string().trim().parse().unwrap();
        assert!(still.abs() < 1e-9, "undriven ground state should not move, got {still}");
    }

    #[test]
    fn drive_reports_missing_functions() {
        let mut st = SimState::default();
        execute_line("qm grid -5 5 50", &mut st).unwrap();
        assert!(execute_line("qm drive nosuch, alsonot", &mut st).unwrap_err().contains("DEF"));
        execute_line("def g(x) { x }", &mut st).unwrap();
        assert!(execute_line("qm drive g, nosuch", &mut st).unwrap_err().contains("nosuch"));
    }

    /// Every ordering mistake gets a message naming the fix.
    #[test]
    fn missing_prerequisites_are_reported_helpfully() {
        let mut st = SimState::default();
        assert!(execute_line("qm states 3", &mut st).unwrap_err().contains("no grid"));
        execute_line("qm grid -5 5 50", &mut st).unwrap();
        assert!(execute_line("qm states 3", &mut st).unwrap_err().contains("no potential"));
        execute_line("qm potential zero", &mut st).unwrap();
        assert!(execute_line("qm norm", &mut st).unwrap_err().contains("no wavefunction"));
        assert!(execute_line("qm potential nosuch", &mut st).unwrap_err().contains("DEF"));
        assert!(execute_line("qm grid 5 -5 10", &mut st).is_err(), "reversed bounds");
        assert!(execute_line("qm grid -5 5 0", &mut st).is_err(), "n = 0");
        assert!(execute_line("qm mass -1", &mut st).is_err());
        assert!(execute_line("qm hbar 0", &mut st).is_err());
    }

    /// Arguments are full expressions, not just literals.
    #[test]
    fn arguments_are_expressions() {
        let (st, _) = run(&[
            "let w = 4",
            "qm grid 0 - w w 2 * 50",
            "qm potential zero",
        ]);
        let g = st.qm.grid.as_ref().unwrap();
        assert_eq!(g.n, 100);
        assert_eq!(g.x_max, 4.0);
    }
}
