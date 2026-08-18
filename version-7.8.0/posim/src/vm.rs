//! The stack machine: executes the postfix instruction programs emitted
//! by [`crate::parser`] against a [`SimState`] holding the
//! [`PhysicalObjectSystem`]. All field access goes through the
//! `physical_object` get/set API — the VM never pokes raw state.

use std::collections::BTreeMap;
use std::fmt;

use ::physical_object::boundary::Boundary;
use ::physical_object::integrate::{run as sundials_run, step as sundials_step, Method};
use ::physical_object::linalg::{Mat3, Quat, Vec3};
use ::special_functions::complex::Complex64;
use ::physical_object::physical_object::physical_object;
use ::physical_object::PhysicalObjectSystem;

/// The six comparison operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    pub fn symbol(self) -> &'static str {
        match self {
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        }
    }
}

/// Runtime values on the operand stack.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Num(f64),
    /// A complex number. Wavefunctions are complex, so the propagator
    /// routines need this; see `special_functions::complex`.
    Complex(Complex64),
    Vec3(Vec3),
    Quat(Quat),
    Mat3(Mat3),
    List(Vec<Value>),
    Str(String),
    Unit,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Num(n) => write!(f, "{n}"),
            Value::Complex(z) => {
                if z.im < 0.0 {
                    write!(f, "{} - {}i", z.re, -z.im)
                } else {
                    write!(f, "{} + {}i", z.re, z.im)
                }
            }
            Value::Vec3(v) => write!(f, "[{}, {}, {}]", v.x, v.y, v.z),
            Value::Quat(q) => write!(f, "quat[w={}, x={}, y={}, z={}]", q.w, q.x, q.y, q.z),
            Value::Mat3(m) => write!(
                f,
                "[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]",
                m.0[0][0], m.0[0][1], m.0[0][2],
                m.0[1][0], m.0[1][1], m.0[1][2],
                m.0[2][0], m.0[2][1], m.0[2][2]
            ),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{it}")?;
                }
                write!(f, "]")
            }
            Value::Str(s) => write!(f, "{s}"),
            Value::Unit => Ok(()),
        }
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Num(_) => "number",
        Value::Complex(_) => "complex",
        Value::Vec3(_) => "vec3",
        Value::Quat(_) => "quaternion",
        Value::Mat3(_) => "mat3",
        Value::List(_) => "list",
        Value::Str(_) => "string",
        Value::Unit => "unit",
    }
}

/// Root of a dotted path.
#[derive(Clone, Debug, PartialEq)]
pub enum PathRoot {
    Object(usize),
    System,
    /// A recorded contact of the last STEP/RUN (read-only).
    Contact(usize),
    /// A user name registered with `NEW ... AS name` (resolved through
    /// the name registry at execution; a function parameter or LET
    /// variable holding a string indirects to that name first).
    Named(String),
}

/// A dotted access path: `obj0.position.x`, `system.g_constant`,
/// `contact0.normal`.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub root: PathRoot,
    pub field: String,
    /// Component index: 0=x, 1=y, 2=z, 3=w.
    pub comp: Option<usize>,
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.root {
            PathRoot::Object(i) => write!(f, "obj{i}.{}", self.field)?,
            PathRoot::System => write!(f, "system.{}", self.field)?,
            PathRoot::Contact(k) => write!(f, "contact{k}.{}", self.field)?,
            PathRoot::Named(n) => write!(f, "{n}.{}", self.field)?,
        }
        if let Some(c) = self.comp {
            write!(f, ".{}", ["x", "y", "z", "w"][c.min(3)])?;
        }
        Ok(())
    }
}

/// Shapes creatable via `NEW`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShapeKind {
    Point,
    Sphere,
    Cuboid,
    Torus,
    Disk,
    Cylinder,
    Dumbbell,
}

/// The `AS <name>` argument of NEW: a bare identifier (resolved
/// against a parameter/LET string binding at execution, else taken
/// literally) or an exact string literal.
#[derive(Clone, Debug, PartialEq)]
pub enum NameArg {
    Ident(String),
    Str(String),
}

/// A user-defined function: `DEF name(params) { body }`. The body is
/// kept as SOURCE lines (recompiled per call), which is also what
/// `SHOW` prints and what `%save` round-trips.
#[derive(Clone, Debug, PartialEq)]
pub struct FuncDef {
    /// Parameter names with optional default values (evaluated once,
    /// at definition time).
    pub params: Vec<(String, Option<Value>)>,
    /// Newline/semicolon-separated body commands.
    pub body: Vec<String>,
    /// The verbatim definition source (for SHOW / editing).
    pub source: String,
}

/// Parsed `BOX` command mode (`Create` pops the inner side length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoxMode {
    Status,
    Off,
    Create,
}

/// Parsed `METHOD` argument.
#[derive(Clone, Debug, PartialEq)]
pub enum MethodSpec {
    Adams,
    Bdf,
    Sprk { table: String, dt: f64 },
    /// IDA on the GGL index-2 DAE — the only method that can hold a
    /// `CONSTRAIN` rod. Translational only.
    Ida,
}

/// Parsed `SCENE` sub-command. Numeric arguments (translate deltas,
/// rotation angles, zoom factor, dt) arrive on the operand stack.
#[derive(Clone, Debug, PartialEq)]
pub enum SceneCmd {
    /// Open the scene window server (port 0 = OS-assigned).
    Create { port: u16 },
    /// Shut the scene window server down.
    Close,
    /// Pop dz, dy, dx: move the camera target.
    Translate,
    /// Pop dpitch, dyaw (degrees): orbit the camera.
    Rotate,
    /// Pop a factor: > 1 zooms in, < 1 zooms out.
    Zoom,
    ZoomIn,
    ZoomOut,
    /// `None` = all objects.
    Hide(Option<usize>),
    Show(Option<usize>),
    /// Re-sync the scene copy from the notebook system.
    Refresh,
    /// Re-send the full scene description to every window.
    Redraw,
    /// Playback control of the time-stepped evolution.
    Start,
    Stop,
    Pause,
    Reverse,
    /// Re-initialize the playback: every mutable value and the time
    /// return to their initial (last-synced) values; Start re-starts.
    ResetPlayback,
    /// Pop dt: set the playback time step.
    SetTimeStep,
    Status,
    /// Drain asynchronous window -> notebook events.
    Events,
}

/// Stack-machine instructions (the compiled program is pure postfix).
#[derive(Clone, Debug, PartialEq)]
pub enum Instr {
    Push(Value),
    Load(Path),
    Store(Path),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    /// Pop `n` values, pack: 3 numbers → vec3, 4 numbers → quaternion,
    /// 3 vec3 → mat3 (rows), anything else → list.
    PackList(usize),
    /// Pop `argc` arguments, call a builtin, push the result.
    Call(String, usize),
    NewObject(ShapeKind),
    /// Pop a value, apply it as a `NEW { field = ... }` initializer.
    InitField(String),
    FinishNew {
        recompute_inertia: bool,
        name: Option<NameArg>,
    },
    Delete(usize),
    ListObjects,
    /// Pop dt, advance the system by dt via sundials.
    Step,
    /// Pop a duration, advance via sundials with `outputs` snapshots.
    Run {
        outputs: usize,
    },
    SetMethod(MethodSpec),
    Energy,
    CenterOfMass,
    TotalMomentum,
    TotalAngularMomentum,
    Laplace(usize),
    Reset,
    Help,
    /// Graphical scene window command (see [`SceneCmd`]).
    Scene(SceneCmd),
    Qm(crate::qm::QmCmd),
    Qm2(crate::qm2::Qm2Cmd),
    Qm3(crate::qm3::Qm3Cmd),
    /// Comparisons. Each pops two values and pushes 1.0 or 0.0 — the
    /// language has numbers and no boolean type, and 1/0 is what makes
    /// an indicator function like `(x > a) * (x < b)` work as a
    /// piecewise potential.
    Cmp(CmpOp),
    /// `COLLIDE [ON|OFF]` — `None` reports the current status.
    Collide(Option<bool>),
    /// `CONTACTS` — list the contacts of the last STEP/RUN.
    Contacts,
    /// `CONSTRAIN <a> <b> [len]` — add a rigid rod. `None` length means
    /// "freeze the separation they already have". The optional length
    /// arrives on the operand stack.
    Constrain { a: String, b: String, has_len: bool },
    /// `BALL <a> <b>` — a spherical joint at the bodies' midpoint.
    Ball { a: String, b: String },
    /// `HINGE <a> <b> <axis>` — one rotational freedom about `axis`
    /// (a vec3 on the operand stack).
    Hinge { a: String, b: String },
    /// `UNIVERSAL <a> <b> <axis_a> <axis_b>` — a Cardan joint; the two
    /// axes arrive on the operand stack.
    Universal { a: String, b: String },
    /// `GEAR <a> <b> <axis> <ratio>` — the two bodies' turns about
    /// `axis` locked in proportion. Axis then ratio on the stack.
    Gear { a: String, b: String },
    /// `RACK <pinion> <rack> <axis> <direction> <radius>` — a rotation
    /// coupled to a translation. Axis, direction, radius on the stack.
    Rack { a: String, b: String },
    /// `PRISMATIC <a> <b> <axis>` — a slider: one translation left.
    Prismatic { a: String, b: String },
    /// `CONSTRAIN OFF` — drop every rod.
    ConstrainOff,
    /// `CONSTRAINTS` — list the rods and how well they are being held.
    Constraints,
    /// `EQUILIBRIUM` — KINSOL: move the system to a static rest state.
    Equilibrium,
    /// `SENSITIVITY <t> "<param>" ...` — CVODES (or IDAS when
    /// constrained): integrate, and report d(state)/d(param). The
    /// duration arrives on the operand stack.
    Sensitivity(Vec<String>),
    /// `BOX [OFF | <size>]` — the rigid, infinitely massive bounding
    /// box (six static wall slabs with inverse mass 0).
    Box(BoxMode),
    /// Push a function parameter / `LET` variable.
    LoadIdent(String),
    /// `LET name = expr` — pop and store a session variable.
    StoreGlobal(String),
    /// `FUNCS` — list the user-defined functions.
    ListFns,
    /// `SHOW name` — print a user function's definition source.
    ShowFn(String),
}

/// The mutable simulator state driven by the notebook / machine modes.
pub struct SimState {
    pub system: PhysicalObjectSystem,
    /// The scene window server, when `SCENE CREATE` has run.
    pub scene: Option<crate::scene::SceneHandle>,
    /// True under `--machine`: scene events are also pushed as
    /// unsolicited JSON lines for the Jupyter kernel.
    pub machine_mode: bool,
    /// Inner side length of the rigid bounding box, when `BOX <size>`
    /// created one (`None` = no box). The box is realized as six static
    /// wall-slab objects whose indices are tracked in `wall_indices`.
    pub box_size: Option<f64>,
    /// Object indices of the six wall slabs (kept in sync by DEL).
    pub wall_indices: Vec<usize>,
    last_new: Option<usize>,
    /// NEW contexts stashed by `Instr::Call` while a user function runs
    /// inside an initializer. A stack, because calls nest. Held HERE
    /// (not in Call-frame locals) so deletions during the call — `DEL`
    /// or `BOX` on a lower index — can renumber every live rollback
    /// target (`shift_new_targets`), exactly as the wall indices are
    /// kept in sync.
    stashed_last_new: Vec<Option<usize>>,
    pending_velocity: Option<Vec3>,
    pending_angular_velocity: Option<Vec3>,
    /// Deferred torus geometry from a `NEW TORUS { ... }` initializer
    /// list: `[ring, tube, inner, outer]`, resolved and validated ONCE
    /// in `FinishNew` so the inner/outer pair is order-independent.
    pending_torus: Option<[Option<f64>; 4]>,
    /// Deferred dumbbell parts from a `NEW DUMBBELL { ... }` list:
    /// `[m1, m2, m_rod, r1, r2, rod_radius, length]`, resolved and
    /// validated once in `FinishNew`.
    pending_dumbbell: Option<[Option<f64>; 7]>,
    /// `LET` session variables.
    pub globals: BTreeMap<String, Value>,
    /// User-defined functions (`DEF name(...) { ... }`).
    pub functions: BTreeMap<String, FuncDef>,
    /// The one-dimensional quantum problem, if any.
    pub qm: crate::qm::QmState,
    /// The two-dimensional quantum problem, if any.
    pub qm2: crate::qm2::Qm2State,
    /// The three-dimensional quantum problem, if any.
    pub qm3: crate::qm3::Qm3State,
    /// User names registered with `NEW ... AS name` → object index
    /// (kept renumbered by DEL / BOX OFF).
    pub names: BTreeMap<String, usize>,
    /// Call frames of executing user functions (parameter bindings).
    env_stack: Vec<BTreeMap<String, Value>>,
}

impl Default for SimState {
    fn default() -> Self {
        Self {
            system: PhysicalObjectSystem::new(Vec::new(), 1.0),
            scene: None,
            machine_mode: false,
            box_size: None,
            wall_indices: Vec::new(),
            last_new: None,
            stashed_last_new: Vec::new(),
            pending_velocity: None,
            pending_angular_velocity: None,
            pending_torus: None,
            pending_dumbbell: None,
            globals: BTreeMap::new(),
            functions: BTreeMap::new(),
            qm: crate::qm::QmState::default(),
            qm2: crate::qm2::Qm2State::fresh(),
            qm3: crate::qm3::Qm3State::fresh(),
            names: BTreeMap::new(),
            env_stack: Vec::new(),
        }
    }
}

pub const HELP_TEXT: &str = "\
posim command language (case-insensitive keywords):
  NEW POINT|SPHERE|CUBOID|TORUS|DISK|CYLINDER { field = expr, ... }
                            create an object (-> objN)
      fields: mass, charge, position, velocity, momentum, orientation,
              angular_velocity, angular_momentum, radius, half_extents,
              ring_radius, tube_radius (torus; or inner_radius +
              outer_radius), height (cylinder), inertia_tensor,
              magnetic_moment_tensor, force, torque, restitution
  BOX <size> | BOX OFF | BOX
                            rigid, infinitely massive bounding box:
                            six static wall slabs (inverse_mass = 0 —
                            the equations of motion only ever use the
                            INVERSE mass, so infinite mass is exactly
                            representable); objects collide elastically
                            off the inside faces; bare BOX = status;
                            GET system.box reads the size (0 = none)
  NEW <shape> AS <name> { ... }
                            register a user NAME for the object: paths
                            then use the name (ball.mass, dumbell0.m1);
                            inside a function a parameter holding a
                            string names the object it creates
  NEW DUMBBELL AS d { m1 = ..., m2 = ..., m_rod = ..., r1 = ..., r2 = ...,
                      rod_radius = ..., length = ..., position = ..., ... }
                            ONE rigid body: two solid spheres joined by
                            a rigid rod; the local origin is the COM;
                            d.m1 d.m2 d.m_rod d.r1 d.r2 d.rod_radius
                            d.length read AND write the parts (mass,
                            COM offsets and inertia recompute); every
                            object also has the shorthands .x .y .z and
                            .vx .vy .vz for position/velocity components
  DEF name(p1, p2 = default, ...) { <body lines> }
                            define a user function: the body is
                            newline/;-separated commands using the
                            parameters as variables; every line is
                            syntax-checked at definition; trailing
                            arguments take their defaults; re-DEF
                            replaces (that is how you edit); the
                            notebook keeps reading `...` lines until
                            the closing brace
  name(arg, ...)            call it (returns the last body line's value)
  LET name = <expr>         session variable (visible in expressions
                            and as defaults)
  FUNCS | SHOW <name>       list functions / print one's source
  \"text\"                  string literal (names, function arguments)
  SET <path> = <expr>       write a field   (e.g. SET obj0.mass = 2)
  GET <path>                read a field    (e.g. GET obj0.position.x)
      paths: objN.<field>[.x|y|z|w], system.<field>, contactK.<field>
      system fields: g_constant, softening, uniform_gravity, e_field,
                     b_field, rtol, atol, time, method, collide,
                     contacts, collisions, restitution_threshold,
                     contact_slop, box
  DEL <n>                   delete object n (later objects renumber)
  LIST                      list all objects
  STEP <dt>                 advance time by dt (sundials solver)
  RUN <t> [STEPS <n>]       advance by t with n output snapshots
  METHOD ADAMS | BDF | SPRK <table> [dt] | IDA
                            choose the sundials integrator
  ENERGY | COM | MOMENTUM | ANGMOM | LAPLACE <n>
                            system observables
  <expr>                    evaluate: numbers, [x,y,z], paths, + - * /,
                            dot() cross() norm() normalize() sqrt() abs()
                            sin() cos() exp() log(), pi, tau
                            complex: 3i is imaginary, so 2 + 3i is a
                            complex number; + - * / all accept them
                            comparisons: < <= > >= == != give 1 or 0,
                            so (x > a) * (x < b) is an indicator
                            function — that is how you write a
                            piecewise potential
  RESET                     clear the system
  HELP                      this text
one-dimensional quantum mechanics (see grammar.md):
  QM                        report the current quantum setup
  QM GRID <x_min> <x_max> <n>
                            the domain and its interior point count.
                            The walls are infinite: psi = 0 outside
  QM POTENTIAL ZERO         free particle
  QM POTENTIAL BARRIER <v0>, <x1>, <x2>
  QM POTENTIAL WELL <depth>, <x1>, <x2>
  QM POTENTIAL <function>   sample a DEF'd function of one argument
  QM MASS <m> | QM HBAR <h> default 1 each
  QM METHOD CAYLEY          Crank-Nicolson, Dirichlet walls (default)
  QM METHOD NASH [STRANG]   Bessel-stencil split-operator, PERIODIC.
                            LIE (default) is the original scheme, first
                            order in dt; STRANG is second order for the
                            same cost. Bound states need CAYLEY.
  QM STATES <k>             the k lowest bound-state energies
  QM STATE <n>              load bound state n as psi
  QM PACKET <x0> <sigma> <k0>
                            a normalised Gaussian wavepacket
  QM STEP <dt> | QM RUN <t> [STEPS <n>]
                            propagate with the current QM METHOD; both
                            methods are unitary at any step size
  QM TRANSMISSION <e>       T(E) and R(E) by transfer matrix — exact at
                            one energy, no packet, no time stepping
  QM SCAN <e1> <e2> <n>     scan T(E) and report the resonances; pushes
                            the transmission list. Resolves peaks that
                            are narrower than any affordable wavepacket
  QM NORM | QM ENERGY | QM POSITION | QM MOMENTUM
  QM PROB <a> <b>           probability in [a, b]
  QM DRIVE <shape> <modulation>
                            time-dependent potential
                            V(x,t) += modulation(t) * shape(x), from two
                            DEF'd functions. The shape is sampled once,
                            the modulation once per step at the midpoint.
                            Energy is then NOT conserved — a driven
                            system trades energy with its drive — but
                            propagation stays unitary
  QM DRIVE OFF              back to a static potential
  QM ABSORB <width> <strength> [<power>]
                            absorbing edges (complex absorbing
                            potential): the walls stop reflecting, so a
                            much shorter domain gives the same answer.
                            Propagation is then NOT unitary — the norm
                            decays by design — and QM STATES is refused
                            because H is no longer Hermitian.
                            power defaults to 2, the measured optimum
  QM ABSORB OFF             back to reflecting walls
  QM DENSITY                |psi|^2 as a list
  QM ANIMATE \"<file>\" <t> [FRAMES <n>]
                            propagate and write a self-contained HTML
                            animation of |psi|^2; open it in a browser
two-dimensional quantum mechanics (ADI; see grammar.md):
  QM2                       report the current 2-D setup
  QM2 GRID <x0> <x1> <nx>, <y0> <y1> <ny>
  QM2 POTENTIAL ZERO | <function of x,y>
  QM2 PACKET <x0> <y0>, <sx> <sy>, <kx> <ky>
  QM2 RUN <t> [STEPS <n>] | QM2 STEP <dt>
                            Strang-split Cayley ADI: each direction is
                            its own Cayley transform, so the propagator
                            is EXACTLY unitary for any dt, with the
                            splitting error confined to the dynamics
  QM2 DRIVE <shape> <modulation> | QM2 DRIVE OFF
                            time-dependent V(x,y,t), as in 1-D
  QM2 STATES <k>            the k lowest bound-state energies, by
                            matrix-free Lanczos with deflation. The 2-D
                            Hamiltonian is (nx*ny)^2, far too big to
                            form, so no dense solver is possible here.
                            Degeneracies ARE resolved. Residuals are
                            printed: an iterative solver has no exact
                            stopping point
  QM2 STATE <n>             load bound state n as psi
  QM2 NORM | QM2 ENERGY | QM2 CENTROID
  QM2 PROB <xa> <xb>, <ya> <yb>
  QM2 ABSORB <width> <strength> [<power>] | QM2 ABSORB OFF
  QM2 ANIMATE \"<file>\" <t> [FRAMES <n>]
                            heat-map animation of |psi(x,y)|^2
  QM2 RESET
three-dimensional quantum mechanics (ADI; see grammar.md):
  QM3                       report the current 3-D setup
  QM3 GRID <x0> <x1> <nx>, <y0> <y1> <ny>, <z0> <z1> <nz>
  QM3 POTENTIAL ZERO | <function of x,y,z>
  QM3 PACKET <x0> <y0> <z0>, <sx> <sy> <sz>, <kx> <ky> <kz>
  QM3 RUN <t> [STEPS <n>] | QM3 STEP <dt>
                            five directional sweeps per step:
                            U_x(dt/2) U_y(dt/2) U_z(dt) U_y(dt/2) U_x(dt/2)
                            — still exactly unitary
  QM3 STATES <k> | QM3 STATE <n>
                            bound states. Practical to about 40^3: the
                            eigensolver stores its whole Krylov basis,
                            so it is REFUSED on larger grids rather than
                            left to exhaust memory. Propagation has no
                            such limit
  QM3 NORM | QM3 ENERGY | QM3 CENTROID
  QM3 PROB <xa> <xb>, <ya> <yb>, <za> <zb>
  QM3 DRIVE <shape> <modulation> | QM3 DRIVE OFF
                            time-dependent V(x,y,z,t), as in 1-D and 2-D
  QM3 ABSORB <width> <strength> [<power>] | QM3 ABSORB OFF
  QM3 ANIMATE \"<file>\" <t> [FRAMES <n>]
                            three MARGINAL densities per frame:
                            P(x,y), P(x,z), P(y,z). A volume cannot be
                            drawn flat without an isosurface or a
                            ray-caster; a marginal is a real observable
                            and each integrates to the norm
  QM3 ISO \"<file>\" <t> [FRAMES <n>] [LEVEL <frac>]
                            a rotatable ISOSURFACE of |psi|^2 at LEVEL
                            times the peak density (default 0.25).
                            Marching tetrahedra, meshed in Rust;
                            software-rasterised in the browser, so no
                            WebGL is needed. Drag the canvas to rotate
  QM3 RESET
  QM RESET                  forget the quantum problem
                            NOTE: separate negative arguments with
                            commas — `well 5 -2 2` reads `5 - 2` as
                            subtraction; write `well 5, -2, 2`
special functions (see grammar.md; orders must be WHOLE numbers):
  spherical Bessel          sph_j(n,x) sph_y(n,x)
                            sph_j_prime(n,x) sph_y_prime(n,x)
  Legendre                  legendre_p(n,x) legendre_p_prime(n,x)
                            assoc_legendre_p(l,m,x)
                            norm_assoc_legendre_p(l,m,x)
  spherical harmonics       sph_harm(l,m,theta,phi)      -> [re, im]
                            sph_harm_real(l,m,theta,phi)
  orthogonal polynomials    hermite_h(n,x) hermite_he(n,x)
                            laguerre_l(n,x) laguerre_l_assoc(n,alpha,x)
                            chebyshev_t(n,x) chebyshev_u(n,x)
                            gegenbauer_c(n,alpha,x)
                            jacobi_p(n,alpha,beta,x)
  cylindrical Bessel        bessel_j(n,x)
                            bessel_j_array(n_max,x)      -> list
                            bessel_j_z(n,z)  COMPLEX argument
                            bessel_i_z(n,z)  modified, complex argument
                            bessel_y_z(n,z)  second kind, complex
                            bessel_k_z(n,z)  modified second kind
                            (Y and K have a BRANCH CUT on the negative
                            real axis and are singular at z = 0)
                            (accuracy falls with |Im z|: ~1e-13 at 8,
                            ~1e-8 at 18, ~1e-6 at 25 — see grammar.md)
                            bessel_j_nu(nu,z)  REAL non-integer order
                            bessel_i_nu(nu,z)  modified, any real order
                            bessel_y_nu(nu,z)  second kind, any order
                            bessel_k_nu(nu,z)  modified second kind
                            (the _nu forms accept whole orders too and
                            hand those to the _z forms; they use an
                            ascending series, so keep |z| <~ 15 for J
                            and Y, and Re z <~ 12 for I and K)
  (abs() of a complex value is its modulus; the other scalar builtins
   are real-only, since complex sqrt and log need a branch-cut choice
   this language has not made)
  (the _nu forms take a COMPLEX order as well as a real one:
   bessel_j_nu(1 + 2i, 3) is J_(1+2i)(3). Complex order costs nothing
   on the positive real axis and ~exp(Im nu * arg z) off it)
  Airy                      airy_z(z)  -> [Ai, Ai', Bi, Bi']
                            (complex argument; Ai Bi' - Ai' Bi = 1/pi
                            exactly, which is how it is verified)
  gamma                     gamma_z(z)      complex argument
                            ln_gamma_z(z)   defined past Gamma's overflow
                            rgamma_z(z)     1/Gamma, ZERO at the poles
  Hankel (travelling wave)  hankel_h1_z(n,z)   H1 = J + iY, outgoing
                            hankel_h2_z(n,z)   H2 = J - iY, incoming
                            hankel_h1_nu(nu,z) any real order
                            hankel_h2_nu(nu,z)
                            hankel_h1_prime_z(n,z)   derivatives
                            hankel_h2_prime_z(n,z)
                            hankel_h1_prime_nu(nu,z)
                            hankel_h2_prime_nu(nu,z)
                            (H1 is accurate for Im z <= 0 and H2 for
                            Im z >= 0; each loses ~3|Im z| nepers on
                            its bad side, so use the other one)
  spherical Hankel          sph_hankel_h1(n,x)  REAL x, complex result
                            sph_hankel_h2(n,x)
                            sph_hankel_h1_prime(n,x)
                            sph_hankel_h2_prime(n,x)
                            (the outgoing/incoming spherical waves;
                            |x*h1| -> 1 exactly as it should)
  scaled forms              bessel_j_scaled(nu,z)   e^-|Im z| J
                            bessel_y_scaled(nu,z)   e^-|Im z| Y
                            bessel_i_scaled(nu,z)   e^-|Re z| I
                            bessel_k_scaled(nu,z)   e^z K
                            hankel_h1_scaled(nu,z)  e^-iz H1
                            hankel_h2_scaled(nu,z)  e^iz H2
                            (the exponential factored out, by a method
                            that never forms it — accurate where the
                            plain forms are not, and DEFINED where they
                            overflow or underflow out of f64. Y_0(5000)
                            and e^x K_0(2000) both work. When no method
                            reaches a point these return an error rather
                            than a wrong first digit)
  quadrature                gauss_legendre(n)  -> [nodes, weights]
  eigenproblems             eigenvalues(matrix)          -> list
                            jacobi_eigen(matrix) -> [values, vectors]
  angular momentum          wigner_3j(j1,j2,j3,m1,m2,m3)
                            wigner_6j(j1,j2,j3,j4,j5,j6)
                            wigner_9j(a,b,c,d,e,f,g,h,i)  row by row
                            clebsch_gordan(j1,m1,j2,m2,j3,m3)
                            (spins may be HALF-integers; a forbidden
                            coupling is 0, not an error)
  linear algebra            solve_tridiag(sub,diag,sup,rhs) -> list
                            solve_tridiag_c(sub,diag,sup,rhs) -> list
                            solve_cyclic_tridiag_c(sub,diag,sup,rhs,
                                                   bl,tr) -> list
  utility                   rel_err(a,b)
                            a 3-element bracket is a VECTOR, so it is
                            also accepted wherever a list is wanted; a
                            3x3 matrix value works as a matrix argument
rigid-body collisions (event-detected at the exact time of impact):
  CONSTRAIN <a> <b> [len]   rigid rod between two objects (objN or a
                            registered name). No length = freeze the
                            separation they already have, which is always
                            consistent. A body with inverse_mass = 0 is a
                            fixed ANCHOR: anchor + bob + rod = pendulum.
                            Requires METHOD IDA to integrate.
  BALL <a> <b>              spherical joint at the two bodies' midpoint:
                            they share a point and may turn any way about
                            it (3 rows, 3 freedoms left)
  HINGE <a> <b> <axis>      like BALL, and the hinge axis is held too, so
                            ONE freedom is left -- a door, a knee
                            (5 rows). <axis> is a vec3 in world
                            coordinates as things stand now.
  UNIVERSAL <a> <b> <u> <w> Cardan joint: a shared point plus two shafts
                            held square to each other (4 rows, 2 left).
                            <u> is carried by <a>, <w> by <b>, and they
                            must start 90 degrees apart.
  CONSTRAIN OFF             drop every joint
  CONSTRAINTS               list the joints, and how well they are held
                            (worst |g| and |g_dot|; both stay at roundoff)
  EQUILIBRIUM               KINSOL: move the system to a static rest
                            state and stop there. Reports the largest net
                            force left on any free body. Says nothing
                            about STABILITY - perturb and run to find out.
  SENSITIVITY <t> \"<p>\" ... run for <t> and also report d(state)/d(p) for
                            each parameter, via CVODES (or IDAS when the
                            system is constrained). Parameter names are
                            QUOTED: \"g_constant\", \"mass 0\", \"charge 1\",
                            \"gravity.y\", \"e_field.x\", \"b_field.z\".
                            e.g.  sensitivity 3 \"gravity.y\" \"mass 0\"
  COLLIDE [ON|OFF]          enable/disable (default ON; bare = status)
  CONTACTS                  list contacts of the last STEP/RUN:
                            pair, time, point, normal (i->j, the
                            action-reaction line), depth, impulse
  GET contactK.<field>      read one contact: i, j, t, point, normal,
                            depth, rel_vel_n, impulse (read-only)
  SET objN.restitution = e  bounciness 0..1 (default 1 = elastic;
                            a pair uses min(e_i, e_j))
  SET system.restitution_threshold | system.contact_slop
                            resting-contact and overlap tolerances
graphical scene window (opens in your web browser):
  SCENE CREATE [port]       open the scene window (all entities shown)
  SCENE CLOSE               close it (alias: SCENE DESTROY)
  SCENE TRANSLATE dx dy [dz]  move the view; SCENE ROTATE dyaw dpitch
  SCENE ZOOM IN | OUT | <f> zoom the view (f > 1 zooms in)
  SCENE HIDE [n|ALL] / SCENE SHOW [n|ALL]   hide / show entities
  SCENE REFRESH             re-sync the window from the notebook state
  SCENE REDRAW              force a full redraw of every window
  SCENE START | STOP | PAUSE | REVERSE
                            control the time-stepped evolution
                            (REVERSE replays recorded history backward)
  SCENE RESET               re-initialize the playback: every value and
                            the time return to their initial values;
                            START then re-starts the simulation (the
                            window's Reset button does the same)
  SCENE SET_TIME_STEP dt    set the playback time step
  SCENE STATUS              connection / camera / playback report
  SCENE EVENTS              read async events sent by the window
  in the window: arrow keys translate, left-drag rotates,
                 wheel or +/- zooms, H shows all controls
notebook magics: %history  %edit <n> <text>  %rerun <n>  %save <file>
                 %load <file>  %reset  %quit";

/// Executes a compiled program; returns the resulting top-of-stack
/// value (or `Unit` for pure side-effect commands).
pub fn execute(prog: &[Instr], state: &mut SimState) -> Result<Value, String> {
    let mut stack: Vec<Value> = Vec::new();
    for instr in prog {
        if let Err(e) = exec_one(instr, state, &mut stack) {
            /* NEW is transactional: a failing initializer (or a failing
             * final validation) must not leave a half-built object in
             * the system. A nested call inside the initializer may have
             * appended objects AFTER this one, so removing it renumbers
             * them — do exactly the bookkeeping DEL does. */
            if let Some(idx) = state.last_new.take() {
                state.system.remove_object(idx);
                state.wall_indices.retain(|w| *w != idx);
                for w in &mut state.wall_indices {
                    if *w > idx {
                        *w -= 1;
                    }
                }
                unregister_index(&mut state.names, idx);
                state.pending_velocity = None;
                state.pending_angular_velocity = None;
                state.pending_torus = None;
                state.pending_dumbbell = None;
            }
            return Err(e);
        }
    }
    Ok(stack.pop().unwrap_or(Value::Unit))
}

fn pop(stack: &mut Vec<Value>) -> Result<Value, String> {
    stack.pop().ok_or_else(|| "stack underflow (internal error)".to_string())
}

fn pop_num(stack: &mut Vec<Value>) -> Result<f64, String> {
    match pop(stack)? {
        Value::Num(n) => Ok(n),
        v => Err(format!("expected a number, got {} `{v}`", type_name(&v))),
    }
}

fn as_vec3(v: Value) -> Result<Vec3, String> {
    match v {
        Value::Vec3(x) => Ok(x),
        Value::List(items) if items.len() == 3 => {
            let mut a = [0.0; 3];
            for (i, it) in items.into_iter().enumerate() {
                match it {
                    Value::Num(n) => a[i] = n,
                    other => return Err(format!("vector component {i} is {}", type_name(&other))),
                }
            }
            Ok(Vec3::from_array(a))
        }
        other => Err(format!("expected a vec3 like [x, y, z], got {} `{other}`", type_name(&other))),
    }
}

fn as_quat(v: Value) -> Result<Quat, String> {
    match v {
        Value::Quat(q) => Ok(q),
        Value::Vec3(_) => Err("expected a quaternion [w, x, y, z] (4 components)".to_string()),
        other => Err(format!("expected a quaternion [w, x, y, z], got {} `{other}`", type_name(&other))),
    }
}

fn as_mat3(v: Value) -> Result<Mat3, String> {
    match v {
        Value::Mat3(m) => Ok(m),
        Value::Num(n) => Ok(Mat3::from_diagonal(Vec3::new(n, n, n))),
        other => Err(format!(
            "expected a mat3 like [[a,b,c],[d,e,f],[g,h,i]] (or a number for a diagonal), got {} `{other}`",
            type_name(&other)
        )),
    }
}

fn exec_one(instr: &Instr, state: &mut SimState, stack: &mut Vec<Value>) -> Result<(), String> {
    match instr {
        Instr::Push(v) => stack.push(v.clone()),
        Instr::Load(path) => {
            let v = load_path(state, path)?;
            stack.push(v);
        }
        Instr::Store(path) => {
            let v = pop(stack)?;
            store_path(state, path, v)?;
            stack.push(Value::Unit);
        }
        Instr::Add => {
            let b = pop(stack)?;
            let a = pop(stack)?;
            stack.push(binary_add(a, b, "+")?);
        }
        Instr::Sub => {
            let b = pop(stack)?;
            let a = pop(stack)?;
            let neg = binary_mul(Value::Num(-1.0), b)?;
            stack.push(binary_add(a, neg, "-")?);
        }
        Instr::Mul => {
            let b = pop(stack)?;
            let a = pop(stack)?;
            stack.push(binary_mul(a, b)?);
        }
        Instr::Div => {
            let b = pop(stack)?;
            let a = pop(stack)?;
            if matches!(a, Value::Complex(_)) || matches!(b, Value::Complex(_)) {
                if let (Some(x), Some(y)) = (as_c(&a), as_c(&b)) {
                    stack.push(Value::Complex(x / y));
                    return Ok(());
                }
            }
            match (a, b) {
                (Value::Num(x), Value::Num(y)) => stack.push(Value::Num(x / y)),
                (Value::Vec3(v), Value::Num(y)) => stack.push(Value::Vec3(v / y)),
                (a, b) => {
                    return Err(format!("cannot divide {} by {}", type_name(&a), type_name(&b)))
                }
            }
        }
        Instr::Neg => {
            let a = pop(stack)?;
            stack.push(binary_mul(Value::Num(-1.0), a)?);
        }
        Instr::PackList(n) => {
            let mut items = Vec::with_capacity(*n);
            for _ in 0..*n {
                items.push(pop(stack)?);
            }
            items.reverse();
            let all_num = items.iter().all(|v| matches!(v, Value::Num(_)));
            let all_vec3 = items.iter().all(|v| matches!(v, Value::Vec3(_)));
            let packed = if all_num && items.len() == 3 {
                let nums: Vec<f64> = items
                    .iter()
                    .map(|v| if let Value::Num(x) = v { *x } else { 0.0 })
                    .collect();
                Value::Vec3(Vec3::new(nums[0], nums[1], nums[2]))
            } else if all_num && items.len() == 4 {
                let nums: Vec<f64> = items
                    .iter()
                    .map(|v| if let Value::Num(x) = v { *x } else { 0.0 })
                    .collect();
                Value::Quat(Quat::new(nums[0], nums[1], nums[2], nums[3]))
            } else if all_vec3 && items.len() == 3 {
                let rows: Vec<Vec3> = items
                    .iter()
                    .map(|v| if let Value::Vec3(x) = v { *x } else { Vec3::zeros() })
                    .collect();
                Value::Mat3(Mat3([
                    rows[0].to_array(),
                    rows[1].to_array(),
                    rows[2].to_array(),
                ]))
            } else {
                Value::List(items)
            };
            stack.push(packed);
        }
        Instr::Call(name, argc) => {
            let mut args = Vec::with_capacity(*argc);
            for _ in 0..*argc {
                args.push(pop(stack)?);
            }
            args.reverse();
            if state.functions.contains_key(name) {
                /* a function called from INSIDE a NEW initializer list
                 * (e.g. `new sphere { mass = f() }`) must not clobber
                 * the in-progress NEW context: stash it, call, restore
                 * on BOTH paths so the outer initializers and the
                 * outer rollback still see their own object. The
                 * `last_new` INDEX is stashed in SimState (not a
                 * local) so a DEL/BOX during the call can renumber it
                 * with the objects (`shift_new_targets`); the four
                 * pending_* are plain values no deletion can
                 * invalidate, so a local tuple is fine for them. */
                state.stashed_last_new.push(state.last_new.take());
                let stash = (
                    state.pending_velocity.take(),
                    state.pending_angular_velocity.take(),
                    state.pending_torus.take(),
                    state.pending_dumbbell.take(),
                );
                let result = call_user_function(name, args, state);
                state.last_new = state.stashed_last_new.pop().unwrap_or(None);
                state.pending_velocity = stash.0;
                state.pending_angular_velocity = stash.1;
                state.pending_torus = stash.2;
                state.pending_dumbbell = stash.3;
                stack.push(result?);
            } else {
                stack.push(call_builtin(name, args)?);
            }
        }
        Instr::NewObject(shape) => {
            let id = state.system.objects.len();
            let obj = match shape {
                ShapeKind::Point => {
                    physical_object::new_point(id, 1.0, Vec3::zeros(), Vec3::zeros())
                }
                ShapeKind::Sphere => physical_object::new_from_shape(
                    id,
                    1.0,
                    0.0,
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Boundary::Sphere { radius: 1.0 },
                ),
                ShapeKind::Cuboid => physical_object::new_from_shape(
                    id,
                    1.0,
                    0.0,
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Boundary::Cuboid { half_extents: [1.0, 1.0, 1.0] },
                ),
                ShapeKind::Torus => physical_object::new_from_shape(
                    id,
                    1.0,
                    0.0,
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Boundary::Torus { ring_radius: 1.0, tube_radius: 0.25 },
                ),
                ShapeKind::Disk => physical_object::new_from_shape(
                    id,
                    1.0,
                    0.0,
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Boundary::Disk { radius: 1.0 },
                ),
                ShapeKind::Cylinder => physical_object::new_from_shape(
                    id,
                    1.0,
                    0.0,
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Boundary::Cylinder { radius: 0.5, half_height: 1.0 },
                ),
                ShapeKind::Dumbbell => {
                    /* defaults: m1 = m2 = 1, m_rod = 0.5, r1 = r2 = 0.25,
                     * rod_radius = 0.1, length = 1 (overridden by the
                     * deferred initializers in FinishNew) */
                    let (mass, b) = ::physical_object::boundary::dumbbell(
                        1.0, 1.0, 0.5, 0.25, 0.25, 0.1, 1.0,
                    )
                    .expect("default dumbbell parameters are valid");
                    physical_object::new_from_shape(
                        id,
                        mass,
                        0.0,
                        Vec3::zeros(),
                        Vec3::zeros(),
                        Vec3::zeros(),
                        b,
                    )
                }
            };
            let idx = state.system.add_object(obj);
            state.last_new = Some(idx);
            state.pending_velocity = None;
            state.pending_angular_velocity = None;
            state.pending_torus = None;
            state.pending_dumbbell = None;
        }
        Instr::InitField(field) => {
            let v = pop(stack)?;
            let idx = state
                .last_new
                .ok_or_else(|| "internal error: initializer outside NEW".to_string())?;
            match field.as_str() {
                /* velocity-like inits are deferred until mass/inertia
                 * are final (initializer order must not matter) */
                "velocity" => state.pending_velocity = Some(as_vec3(v)?),
                "angular_velocity" => state.pending_angular_velocity = Some(as_vec3(v)?),
                /* torus geometry is deferred and resolved ONCE in
                 * FinishNew, so the inner_radius + outer_radius pair is
                 * genuinely order-independent (validating each write
                 * against a half-updated default would make one order
                 * fail where the other succeeds) */
                "ring_radius" | "tube_radius" | "inner_radius" | "outer_radius"
                    if matches!(
                        state.system.objects.get(idx).map(|o| o.get_boundary()),
                        Some(Boundary::Torus { .. })
                    ) =>
                {
                    let n = match v {
                        Value::Num(n) => n,
                        v => {
                            return Err(format!(
                                "{field} expects a number, got {}",
                                type_name(&v)
                            ))
                        }
                    };
                    let lo_ok = if field == "inner_radius" { n >= 0.0 } else { n > 0.0 };
                    if !(n.is_finite() && lo_ok) {
                        return Err(format!(
                            "{field} must be a finite number {} 0, got {n}",
                            if field == "inner_radius" { ">=" } else { ">" }
                        ));
                    }
                    let slot = match field.as_str() {
                        "ring_radius" => 0,
                        "tube_radius" => 1,
                        "inner_radius" => 2,
                        _ => 3,
                    };
                    state.pending_torus.get_or_insert([None; 4])[slot] = Some(n);
                }
                /* dumbbell parts are deferred the same way and
                 * validated once in FinishNew via boundary::dumbbell */
                "m1" | "m2" | "m_rod" | "r1" | "r2" | "rod_radius" | "rod_r" | "length" | "len"
                    if matches!(
                        state.system.objects.get(idx).map(|o| o.get_boundary()),
                        Some(Boundary::Dumbbell { .. })
                    ) =>
                {
                    let n = match v {
                        Value::Num(n) => n,
                        v => {
                            return Err(format!(
                                "{field} expects a number, got {}",
                                type_name(&v)
                            ))
                        }
                    };
                    let slot = match field.as_str() {
                        "m1" => 0,
                        "m2" => 1,
                        "m_rod" => 2,
                        "r1" => 3,
                        "r2" => 4,
                        "rod_radius" | "rod_r" => 5,
                        _ => 6,
                    };
                    state.pending_dumbbell.get_or_insert([None; 7])[slot] = Some(n);
                }
                _ => {
                    let path = Path {
                        root: PathRoot::Object(idx),
                        field: field.clone(),
                        comp: None,
                    };
                    store_path(state, &path, v)?;
                }
            }
        }
        Instr::FinishNew { recompute_inertia, name } => {
            let idx = state
                .last_new
                .ok_or_else(|| "internal error: FinishNew outside NEW".to_string())?;
            /* Resolve deferred torus geometry: apply ring/tube first,
             * then let inner/outer override the derived pair — all
             * validated ONCE against the FINAL values (an error here
             * rolls the whole NEW back in `execute`). */
            if let Some(p) = state.pending_torus.take() {
                if let Some(o) = state.system.objects.get_mut(idx) {
                    let (ring0, tube0) = match o.get_boundary() {
                        Boundary::Torus { ring_radius, tube_radius } => (ring_radius, tube_radius),
                        _ => (1.0, 0.25),
                    };
                    let ring1 = p[0].unwrap_or(ring0);
                    let tube1 = p[1].unwrap_or(tube0);
                    let inner = p[2].unwrap_or(ring1 - tube1);
                    let outer = p[3].unwrap_or(ring1 + tube1);
                    if !(0.0 <= inner && inner < outer) {
                        return Err(format!(
                            "torus needs 0 <= inner < outer (got inner = {inner}, outer = {outer})"
                        ));
                    }
                    o.set_boundary(Boundary::Torus {
                        ring_radius: 0.5 * (inner + outer),
                        tube_radius: 0.5 * (outer - inner),
                    });
                }
            }
            /* Resolve deferred dumbbell parts: start from the current
             * (default) parts and override with whatever was given —
             * boundary::dumbbell validates once against final values. */
            if let Some(pd) = state.pending_dumbbell.take() {
                if let Some(o) = state.system.objects.get_mut(idx) {
                    let (m1, m2, m_rod, r1, r2, rod_r, len) =
                        dumbbell_members(o).ok_or_else(|| {
                            "internal error: pending dumbbell on a non-dumbbell".to_string()
                        })?;
                    let (mass, b) = ::physical_object::boundary::dumbbell(
                        pd[0].unwrap_or(m1),
                        pd[1].unwrap_or(m2),
                        pd[2].unwrap_or(m_rod),
                        pd[3].unwrap_or(r1),
                        pd[4].unwrap_or(r2),
                        pd[5].unwrap_or(rod_r),
                        pd[6].unwrap_or(len),
                    )?;
                    o.set_mass(mass);
                    o.set_boundary(b);
                }
            }
            /* Resolve and validate the AS name BEFORE committing (an
             * error here rolls the whole NEW back in `execute`). */
            let resolved_name = match name {
                None => None,
                Some(arg) => Some(resolve_name_arg(state, arg)?),
            };
            state.last_new = None;
            if *recompute_inertia {
                if let Some(o) = state.system.objects.get_mut(idx) {
                    if o.get_boundary() != Boundary::Point {
                        o.recompute_inertia_from_boundary();
                    }
                }
            }
            let pv = state.pending_velocity.take();
            let pw = state.pending_angular_velocity.take();
            if let Some(o) = state.system.objects.get_mut(idx) {
                if let Some(v) = pv {
                    o.set_velocity(v);
                }
                if let Some(w) = pw {
                    o.set_angular_velocity(w);
                }
            }
            match resolved_name {
                Some(n) => {
                    state.names.insert(n.clone(), idx);
                    stack.push(Value::Str(format!("obj{idx} as {n}")));
                }
                None => stack.push(Value::Str(format!("obj{idx}"))),
            }
        }
        Instr::Delete(i) => {
            if state.system.remove_object(*i).is_none() {
                return Err(format!("no object obj{i}"));
            }
            /* keep the wall-slab index list in sync with renumbering */
            state.wall_indices.retain(|w| w != i);
            for w in &mut state.wall_indices {
                if *w > *i {
                    *w -= 1;
                }
            }
            /* user names renumber the same way */
            unregister_index(&mut state.names, *i);
            /* ... and so does any NEW under construction (a DEL can
             * run mid-initializer through a user-function call) */
            shift_new_targets(state, *i);
            if state.wall_indices.len() < 6 && state.box_size.is_some() {
                state.box_size = None; // a wall was deleted: no closed box anymore
            }
            stack.push(Value::Str(format!(
                "deleted obj{i}; {} object(s) remain (indices renumbered)",
                state.system.objects.len()
            )));
        }
        Instr::ListObjects => {
            let mut out = String::new();
            if state.system.objects.is_empty() {
                out.push_str("(no objects)");
            }
            for (i, o) in state.system.objects.iter().enumerate() {
                let shape = match o.get_boundary() {
                    Boundary::Point => "point".to_string(),
                    Boundary::Sphere { radius } => format!("sphere r={radius}"),
                    Boundary::Cuboid { half_extents } => format!(
                        "cuboid he=[{}, {}, {}]",
                        half_extents[0], half_extents[1], half_extents[2]
                    ),
                    Boundary::Torus { ring_radius, tube_radius } => {
                        format!("torus ring={ring_radius} tube={tube_radius}")
                    }
                    Boundary::Disk { radius } => format!("disk r={radius}"),
                    Boundary::Cylinder { radius, half_height } => {
                        format!("cylinder r={radius} h={}", 2.0 * half_height)
                    }
                    Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, .. } => {
                        format!("dumbbell r1={r1} r2={r2} rod_r={rod_radius} len={}", z2 - z1)
                    }
                };
                let p = o.get_position();
                if i > 0 {
                    out.push('\n');
                }
                /* static bodies (inverse mass 0) are flagged; the six
                 * BOX slabs additionally read "[wall]" */
                let tag = if state.wall_indices.contains(&i) {
                    " [wall: static, inverse_mass=0]"
                } else if o.get_inverse_mass() == 0.0 {
                    " [static, inverse_mass=0]"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "obj{i}: {shape}, mass={}, charge={}, pos=[{}, {}, {}]{tag}",
                    o.get_mass(),
                    o.get_charge(),
                    p.x,
                    p.y,
                    p.z
                ));
            }
            stack.push(Value::Str(out));
        }
        Instr::Step => {
            let dt = pop_num(stack)?;
            let report = sundials_step(&mut state.system, dt)?;
            /* the collision suffix appears only when something collided,
             * so collision-free output keeps its historical shape */
            let coll = if report.ncollisions > 0 {
                format!(", {} collision(s) — CONTACTS lists them", report.ncollisions)
            } else {
                String::new()
            };
            stack.push(Value::Str(format!(
                "t = {} (advanced by {dt}, {} solver steps{coll})",
                state.system.time, report.nst
            )));
        }
        Instr::Run { outputs } => {
            let duration = pop_num(stack)?;
            let e0 = state.system.total_energy();
            let t_end = state.system.time + duration;
            let report = sundials_run(&mut state.system, t_end, *outputs)?;
            let e1 = state.system.total_energy();
            let drift = if e0 != 0.0 { ((e1 - e0) / e0).abs() } else { (e1 - e0).abs() };
            let coll = if report.ncollisions > 0 {
                format!(", {} collision(s) — CONTACTS lists them", report.ncollisions)
            } else {
                String::new()
            };
            stack.push(Value::Str(format!(
                "t = {} ({} solver steps, {} snapshots, |dE/E| = {:.3e}{coll})",
                state.system.time,
                report.nst,
                report.snapshots.len(),
                drift
            )));
        }
        Instr::SetMethod(spec) => {
            let (method, desc) = match spec {
                MethodSpec::Adams => (Method::Adams, "CVODE Adams".to_string()),
                MethodSpec::Bdf => (Method::Bdf, "CVODE BDF".to_string()),
                MethodSpec::Sprk { table, dt } => (
                    Method::Sprk { table: table.clone(), dt: *dt },
                    format!("ARKODE SPRK {table}, fixed dt = {dt}"),
                ),
                MethodSpec::Ida => (
                    Method::Ida,
                    "IDA (constrained DAE, GGL index-2)".to_string(),
                ),
            };
            state.system.method = method;
            stack.push(Value::Str(format!("method = {desc}")));
        }
        Instr::Energy => stack.push(Value::Num(state.system.total_energy())),
        Instr::CenterOfMass => stack.push(Value::Vec3(state.system.center_of_mass())),
        Instr::TotalMomentum => stack.push(Value::Vec3(state.system.total_momentum())),
        Instr::TotalAngularMomentum => {
            stack.push(Value::Vec3(state.system.total_angular_momentum(Vec3::zeros())))
        }
        Instr::Laplace(i) => match state.system.laplace_vector(*i) {
            Some(v) => stack.push(Value::Vec3(v)),
            None => return Err(format!("no object obj{i}")),
        },
        Instr::Reset => {
            /* the scene window survives a RESET: it re-syncs to the
             * (now empty) system instead of closing */
            let scene = state.scene.take();
            let machine_mode = state.machine_mode;
            *state = SimState::default();
            state.machine_mode = machine_mode;
            if let Some(s) = scene {
                s.sync(&state.system)?;
                /* the fresh state has no box: clear the window's box
                 * wireframe and wall flags too */
                s.set_box(state.box_size, &state.wall_indices, &state.names)?;
                state.scene = Some(s);
            }
            stack.push(Value::Str("system reset".to_string()));
        }
        Instr::Help => stack.push(Value::Str(HELP_TEXT.to_string())),
        Instr::Scene(cmd) => {
            let out = exec_scene(cmd, state, stack)?;
            stack.push(Value::Str(out));
        }
        Instr::Cmp(op) => {
            let b = pop(stack)?;
            let a = pop(stack)?;
            stack.push(Value::Num(compare(*op, a, b)?));
        }
        Instr::Qm3(cmd) => {
            let out = crate::qm3::exec_qm3(cmd, state, stack)?;
            if !out.is_empty() {
                stack.push(Value::Str(out));
            }
        }
        Instr::Qm2(cmd) => {
            let out = crate::qm2::exec_qm2(cmd, state, stack)?;
            if !out.is_empty() {
                stack.push(Value::Str(out));
            }
        }
        Instr::Qm(cmd) => {
            let out = crate::qm::exec_qm(cmd, state, stack)?;
            /* QM DENSITY pushes its own list value; everything else
             * reports text */
            if !out.is_empty() {
                stack.push(Value::Str(out));
            }
        }
        Instr::Collide(mode) => {
            if let Some(on) = mode {
                state.system.collide_enabled = *on;
            }
            let pairs = ::physical_object::collide::collidable_pairs(&state.system);
            stack.push(Value::Str(format!(
                "collisions {} ({} collidable pair(s); {} impulse(s) so far)",
                if state.system.collide_enabled { "ON" } else { "OFF" },
                pairs.len(),
                state.system.collision_count
            )));
        }
        Instr::Constrain { a, b, has_len } => {
            let len = if *has_len {
                Some(pop_num(stack)?)
            } else {
                None
            };
            let i = resolve_object_ref(state, a)?;
            let j = resolve_object_ref(state, b)?;
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_distance(&snapshot, i, j, len)?;
            let c = state.system.constraints.joints[k];
            let (i, j) = c.bodies();
            stack.push(Value::Str(format!(
                "constraint{k}: {} obj{i} <-> obj{j}, {} row(s) \
                 (METHOD IDA is required to integrate it)",
                c.kind(),
                c.rows()
            )));
        }
        Instr::Ball { a, b } => {
            let (i, j) = (resolve_object_ref(state, a)?, resolve_object_ref(state, b)?);
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_ball(&snapshot, i, j)?;
            stack.push(Value::Str(format!(
                "constraint{k}: ball obj{i} <-> obj{j}, 3 row(s) — they share a point and may \
                 turn any way about it (METHOD IDA is required to integrate it)"
            )));
        }
        Instr::Hinge { a, b } => {
            let axis = as_vec3(pop(stack)?)?;
            let (i, j) = (resolve_object_ref(state, a)?, resolve_object_ref(state, b)?);
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_hinge(&snapshot, i, j, axis)?;
            stack.push(Value::Str(format!(
                "constraint{k}: hinge obj{i} <-> obj{j} about [{}, {}, {}], 5 row(s) — one \
                 freedom left (METHOD IDA is required to integrate it)",
                axis.x, axis.y, axis.z
            )));
        }
        Instr::Gear { a, b } => {
            /* axis pushed first, ratio second, so the ratio is on top */
            let ratio = pop_num(stack)?;
            let axis = as_vec3(pop(stack)?)?;
            let (i, j) = (resolve_object_ref(state, a)?, resolve_object_ref(state, b)?);
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_gear(&snapshot, i, j, axis, ratio)?;
            stack.push(Value::Str(format!(
                "constraint{k}: gear obj{i} <-> obj{j} about [{}, {}, {}], ratio {ratio}, \
                 1 row — their turns are locked in proportion (METHOD IDA is required to \
                 integrate it)",
                axis.x, axis.y, axis.z
            )));
        }
        Instr::Prismatic { a, b } => {
            let axis = as_vec3(pop(stack)?)?;
            let (i, j) = (resolve_object_ref(state, a)?, resolve_object_ref(state, b)?);
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_prismatic(&snapshot, i, j, axis)?;
            stack.push(Value::Str(format!(
                "constraint{k}: prismatic obj{i} <-> obj{j} along [{}, {}, {}], 5 row(s) — one \
                 freedom left, the slide (METHOD IDA is required to integrate it)",
                axis.x, axis.y, axis.z
            )));
        }
        Instr::Rack { a, b } => {
            /* pushed axis, direction, radius — so radius is on top */
            let radius = pop_num(stack)?;
            let dir = as_vec3(pop(stack)?)?;
            let axis = as_vec3(pop(stack)?)?;
            let (i, j) = (resolve_object_ref(state, a)?, resolve_object_ref(state, b)?);
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_rack(&snapshot, i, j, axis, dir, radius)?;
            stack.push(Value::Str(format!(
                "constraint{k}: rack obj{i} (pinion) <-> obj{j} (rack), pitch radius {radius}, \
                 1 row — the rack travels {radius} per radian of pinion (METHOD IDA is \
                 required to integrate it)"
            )));
        }
        Instr::Universal { a, b } => {
            /* two vec3 arguments: the SECOND is on top of the stack */
            let axis_b = as_vec3(pop(stack)?)?;
            let axis_a = as_vec3(pop(stack)?)?;
            let (i, j) = (resolve_object_ref(state, a)?, resolve_object_ref(state, b)?);
            let snapshot = state.system.clone();
            let k = state.system.constraints.add_universal(&snapshot, i, j, axis_a, axis_b)?;
            stack.push(Value::Str(format!(
                "constraint{k}: universal obj{i} <-> obj{j}, 4 row(s) — two freedoms left \
                 (METHOD IDA is required to integrate it)"
            )));
        }
        Instr::ConstrainOff => {
            let n = state.system.constraints.len();
            state.system.constraints.joints.clear();
            stack.push(Value::Str(format!("removed {n} constraint(s)")));
        }
        Instr::Constraints => {
            if state.system.constraints.is_empty() {
                stack.push(Value::Str(
                    "(no constraints — CONSTRAIN <a> <b> [length] adds a rigid rod)".to_string(),
                ));
            } else {
                let (g, gdot) = state.system.constraints.drift(&state.system);
                let mut out = String::new();
                for (k, c) in state.system.constraints.joints.iter().enumerate() {
                    let (i, j) = c.bodies();
                    out.push_str(&format!(
                        "constraint{k}: {} obj{i} <-> obj{j}, {} row(s)\n",
                        c.kind(),
                        c.rows()
                    ));
                }
                out.push_str(&format!(
                    "worst |g| = {g:e}, worst |g_dot| = {gdot:e}"
                ));
                stack.push(Value::Str(out));
            }
        }
        Instr::Equilibrium => {
            let report = ::physical_object::equilibrium::solve(&mut state.system)?;
            state.system.contacts.clear();
            stack.push(Value::Str(format!(
                "equilibrium found in {} Newton iteration(s), {} residual evaluation(s); \
                 largest net force on any free body = {:e}{}",
                report.iterations,
                report.func_evals,
                report.max_net_force,
                if state.system.constraints.is_empty() {
                    String::new()
                } else {
                    format!(", worst |g| = {:e}", report.constraint_error)
                }
            )));
        }
        Instr::Sensitivity(names) => {
            let dt = pop_num(stack)?;
            let n = state.system.objects.len();
            let mut params = Vec::with_capacity(names.len());
            for name in names {
                params.push(::physical_object::sensitivity::SensParam::parse(name, n)?);
            }
            let t_end = state.system.time + dt;
            let report =
                ::physical_object::sensitivity::run(&mut state.system, t_end, &params)?;
            let mut out = format!(
                "t = {} ({}, {} solver steps)\n",
                report.t, report.solver, report.nst
            );
            for pp in &report.per_param {
                out.push_str(&format!("d/d({}):\n", pp.param));
                for (k, d) in pp.d_position.iter().enumerate() {
                    out.push_str(&format!(
                        "  obj{k} position [{}, {}, {}]\n",
                        d.x, d.y, d.z
                    ));
                }
            }
            out.pop();
            stack.push(Value::Str(out));
        }
        Instr::Contacts => {
            if state.system.contacts.is_empty() {
                stack.push(Value::Str(
                    "(no contacts recorded in the last STEP/RUN)".to_string(),
                ));
            } else {
                let mut out = String::new();
                for (k, c) in state.system.contacts.iter().enumerate() {
                    if k > 0 {
                        out.push('\n');
                    }
                    out.push_str(&format!(
                        "contact{k}: obj{} <-> obj{} at t = {}\n  point  = [{}, {}, {}]\n  \
                         normal = [{}, {}, {}]  (from obj{} toward obj{})\n  \
                         depth = {}, approach speed = {}, impulse = {}",
                        c.i, c.j, c.t,
                        c.point.x, c.point.y, c.point.z,
                        c.normal.x, c.normal.y, c.normal.z, c.i, c.j,
                        c.depth, -c.rel_vel_n, c.impulse_n
                    ));
                }
                stack.push(Value::Str(out));
            }
        }
        Instr::Box(mode) => {
            let out = exec_box(*mode, state, stack)?;
            stack.push(Value::Str(out));
        }
        Instr::LoadIdent(name) => {
            let v = state
                .env_stack
                .last()
                .and_then(|env| env.get(name))
                .or_else(|| state.globals.get(name))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "unknown name `{name}` (define it with LET, pass it as a function \
                         parameter, or use `{name}.field` for a registered object)"
                    )
                })?;
            stack.push(v);
        }
        Instr::StoreGlobal(name) => {
            let v = pop(stack)?;
            state.globals.insert(name.clone(), v);
            stack.push(Value::Str(format!("{name} set")));
        }
        Instr::ListFns => {
            if state.functions.is_empty() {
                stack.push(Value::Str(
                    "(no user functions — DEF name(params) { body } creates one)".to_string(),
                ));
            } else {
                let mut out = String::new();
                for (i, (name, f)) in state.functions.iter().enumerate() {
                    if i > 0 {
                        out.push('\n');
                    }
                    let sig: Vec<String> = f
                        .params
                        .iter()
                        .map(|(p, d)| match d {
                            Some(v) => format!("{p} = {v}"),
                            None => p.clone(),
                        })
                        .collect();
                    out.push_str(&format!(
                        "{name}({}) — {} body line(s); SHOW {name} prints it",
                        sig.join(", "),
                        f.body.len()
                    ));
                }
                stack.push(Value::Str(out));
            }
        }
        Instr::ShowFn(name) => {
            let f = state.functions.get(name).ok_or_else(|| {
                format!(
                    "no user function `{name}` — FUNCS lists the defined ones"
                )
            })?;
            stack.push(Value::Str(f.source.clone()));
        }
    }
    Ok(())
}

/// Keeps every live NEW rollback target in step with a deletion,
/// exactly like the wall indices: a target above the removed index
/// shifts down; the removed index itself disarms that rollback (the
/// half-built object is already gone, so there is nothing to remove —
/// a later initializer field then errors honestly instead of writing
/// to a neighbour). Covers the active `last_new` and every frame the
/// Call instruction has stashed.
fn shift_new_targets(state: &mut SimState, removed: usize) {
    fn shift(slot: &mut Option<usize>, removed: usize) {
        match *slot {
            Some(ln) if ln == removed => *slot = None,
            Some(ln) if ln > removed => *slot = Some(ln - 1),
            _ => {}
        }
    }
    shift(&mut state.last_new, removed);
    for s in &mut state.stashed_last_new {
        shift(s, removed);
    }
}

/// Removes a deleted object index from the name registry and shifts
/// every higher index down by one (Vec renumbering).
fn unregister_index(names: &mut BTreeMap<String, usize>, removed: usize) {
    names.retain(|_, i| *i != removed);
    for i in names.values_mut() {
        if *i > removed {
            *i -= 1;
        }
    }
}

/// Resolves an `AS` name argument: a string literal is taken exactly;
/// a bare identifier first looks for a parameter / LET string binding
/// (so a function can name objects from its arguments), else names the
/// identifier itself. The result is validated against the reserved
/// path roots and existing names.
fn resolve_name_arg(state: &SimState, arg: &NameArg) -> Result<String, String> {
    let raw = match arg {
        NameArg::Str(st) => st.clone(),
        NameArg::Ident(id) => match state
            .env_stack
            .last()
            .and_then(|env| env.get(id))
            .or_else(|| state.globals.get(id))
        {
            Some(Value::Str(st)) => st.clone(),
            Some(other) => {
                return Err(format!(
                    "`{id}` is bound to a {}, not a string name — pass a string \
                     (e.g. \"{id}0\") or bind `{id}` to a string",
                    type_name(other)
                ))
            }
            None => id.clone(),
        },
    };
    let name = raw.to_ascii_lowercase();
    let valid = !name.is_empty()
        && name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(format!(
            "`{raw}` is not a valid object name (letters, digits and _ only, starting \
             with a letter)"
        ));
    }
    if name == "system" || name == "sys" {
        return Err(format!("`{name}` is reserved"));
    }
    let digits_after = |prefix: &str| {
        name.strip_prefix(prefix).is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
    };
    if digits_after("obj") || digits_after("contact") {
        return Err(format!("`{name}` is reserved for positional paths"));
    }
    if state.names.contains_key(&name) {
        return Err(format!(
            "the name `{name}` already refers to obj{} — DEL it or pick another name",
            state.names[&name]
        ));
    }
    Ok(name)
}

/// Resolves a named path root to an object index: a parameter / LET
/// string binding indirects first, then the name registry.
/// Resolves an object written either positionally (`obj3`) or by a name
/// registered with `NEW ... AS`. `CONSTRAIN` takes bodies this way so
/// that `constrain anchor bob 1.5` reads the way it sounds.
fn resolve_object_ref(state: &SimState, text: &str) -> Result<usize, String> {
    let lower = text.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("obj") {
        if let Ok(k) = rest.parse::<usize>() {
            if k < state.system.objects.len() {
                return Ok(k);
            }
            return Err(format!(
                "no object obj{k} ({} object(s) exist)",
                state.system.objects.len()
            ));
        }
    }
    resolve_name(state, &lower)
}

fn resolve_name(state: &SimState, name: &str) -> Result<usize, String> {
    let target = match state
        .env_stack
        .last()
        .and_then(|env| env.get(name))
        .or_else(|| state.globals.get(name))
    {
        Some(Value::Str(st)) => st.to_ascii_lowercase(),
        _ => name.to_string(),
    };
    state.names.get(&target).copied().ok_or_else(|| {
        let known: Vec<&String> = state.names.keys().collect();
        format!(
            "no object named `{target}` (registered names: {}; NEW ... AS <name> creates \
             one; positional paths are objN.field, contactK.field, system.field)",
            if known.is_empty() { "none".to_string() } else { format!("{known:?}") }
        )
    })
}

/// Extracts a dumbbell's user-facing members from `(mass, boundary)`:
/// `(m1, m2, m_rod, r1, r2, rod_radius, length)` — the mass fractions
/// stored in the boundary make every part recoverable.
fn dumbbell_members(o: &physical_object) -> Option<(f64, f64, f64, f64, f64, f64, f64)> {
    match o.get_boundary() {
        Boundary::Dumbbell { r1, r2, rod_radius, z1, z2, f1, f2 } => {
            let m = o.get_mass();
            Some((f1 * m, f2 * m, (1.0 - f1 - f2) * m, r1, r2, rod_radius, z2 - z1))
        }
        _ => None,
    }
}

/// Rewrites one dumbbell member and rebuilds the body: total mass, COM
/// offsets and the inertia tensor all follow. The object's position
/// (its COM) and orientation are untouched; MOMENTUM and ANGULAR
/// MOMENTUM are preserved (the setters are momentum-canonical), so the
/// velocity and angular velocity rescale with the new mass/inertia.
fn dumbbell_member_write(
    o: &mut physical_object,
    field: &str,
    v: f64,
) -> Result<(), String> {
    let (mut m1, mut m2, mut m_rod, mut r1, mut r2, mut rod_r, mut len) =
        dumbbell_members(o).ok_or_else(|| "not a dumbbell".to_string())?;
    match field {
        "m1" => m1 = v,
        "m2" => m2 = v,
        "m_rod" => m_rod = v,
        "r1" => r1 = v,
        "r2" => r2 = v,
        "rod_radius" | "rod_r" => rod_r = v,
        "length" | "len" => len = v,
        _ => return Err(format!("`{field}` is not a dumbbell member")),
    }
    let (mass, b) = ::physical_object::boundary::dumbbell(m1, m2, m_rod, r1, r2, rod_r, len)?;
    o.set_mass(mass);
    o.set_boundary(b);
    o.recompute_inertia_from_boundary();
    Ok(())
}

/// `BOX <size>` / `BOX OFF` / `BOX` — the rigid, **infinitely massive**
/// bounding box. Infinite mass enters the dynamics only as inverse
/// mass zero (the equations of motion and the contact impulse use
/// `m⁻¹`, never `m`): each of the six wall slabs is a static
/// `Boundary::Cuboid` object with `inverse_mass = 0` and zero inverse
/// inertia, so it contributes nothing to the impulse denominator
/// `n·K n = m_i⁻¹ + m_j⁻¹ + (angular terms)` and receives no state
/// writes — bodies bounce off the inside faces elastically while the
/// walls stay bit-identically at rest.
fn exec_box(mode: BoxMode, state: &mut SimState, stack: &mut Vec<Value>) -> Result<String, String> {
    let remove_walls = |state: &mut SimState| {
        let mut idx = state.wall_indices.clone();
        idx.sort_unstable();
        for &i in idx.iter().rev() {
            state.system.remove_object(i);
            unregister_index(&mut state.names, i);
            shift_new_targets(state, i);
        }
        state.wall_indices.clear();
        state.box_size = None;
        idx.len()
    };
    match mode {
        BoxMode::Status => Ok(match state.box_size {
            Some(s) => format!(
                "box: inner size {s} x {s} x {s}, six static walls {} (inverse_mass = 0)",
                state
                    .wall_indices
                    .iter()
                    .map(|i| format!("obj{i}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None if !state.wall_indices.is_empty() => format!(
                "box: dissolved (a wall was deleted; {} tracked slab(s) remain — \
                 BOX <size> replaces them, BOX OFF removes them)",
                state.wall_indices.len()
            ),
            None => "box: none (BOX <size> creates one)".to_string(),
        }),
        BoxMode::Off => {
            if state.box_size.is_none() && state.wall_indices.is_empty() {
                return Ok("box: none".to_string());
            }
            let n = remove_walls(state);
            Ok(format!("box removed ({n} wall(s) deleted, indices renumbered)"))
        }
        BoxMode::Create => {
            let size = match stack.pop() {
                Some(Value::Num(n)) => n,
                Some(v) => return Err(format!("BOX expects a number, got {}", type_name(&v))),
                None => return Err("BOX needs a size (the inner side length)".to_string()),
            };
            if !(size.is_finite() && size > 0.0) {
                return Err(format!("BOX size must be a finite number > 0, got {size}"));
            }
            /* remove tracked slabs even when the box was dissolved by a
             * wall deletion — recreating must never leak orphans */
            let replaced = if state.box_size.is_some() || !state.wall_indices.is_empty() {
                remove_walls(state)
            } else {
                0
            };
            let half = 0.5 * size; // inner half-extent
            let t = 0.5 * half; // slab half-thickness
            let big = half + 2.0 * t; // slab cross-section: covers the corners
            let c = half + t; // slab center distance from the origin
            let slabs: [([f64; 3], Vec3); 6] = [
                ([t, big, big], Vec3::new(c, 0.0, 0.0)),
                ([t, big, big], Vec3::new(-c, 0.0, 0.0)),
                ([big, t, big], Vec3::new(0.0, c, 0.0)),
                ([big, t, big], Vec3::new(0.0, -c, 0.0)),
                ([big, big, t], Vec3::new(0.0, 0.0, c)),
                ([big, big, t], Vec3::new(0.0, 0.0, -c)),
            ];
            let mut names = Vec::new();
            for (h, pos) in slabs {
                let id = state.system.objects.len();
                let mut wall = physical_object::new_from_shape(
                    id,
                    1.0,
                    0.0,
                    pos,
                    Vec3::zeros(),
                    Vec3::zeros(),
                    Boundary::Cuboid { half_extents: h },
                );
                /* infinite mass, the only way the math ever sees it */
                wall.set_inverse_mass(0.0);
                wall.set_inverse_inertia_tensor(Mat3::zeros());
                let idx = state.system.add_object(wall);
                state.wall_indices.push(idx);
                names.push(format!("obj{idx}"));
            }
            state.box_size = Some(size);
            let replaced_note =
                if replaced > 0 { " (replacing the previous box)" } else { "" };
            let scene_note = if state.scene.is_some() {
                "\n(scene window open: SCENE REFRESH shows the box)"
            } else {
                ""
            };
            Ok(format!(
                "box: inner size {size} x {size} x {size}{replaced_note} — six static walls \
                 {} with inverse_mass = 0 (infinitely massive); objects collide elastically \
                 off the inside faces{scene_note}",
                names.join(", ")
            ))
        }
    }
}

/// Executes one `SCENE` sub-command. Every command except `CREATE`
/// needs an open scene window.
fn exec_scene(cmd: &SceneCmd, state: &mut SimState, stack: &mut Vec<Value>) -> Result<String, String> {
    use crate::scene::{RunMode, SceneHandle};

    if let SceneCmd::Create { port } = cmd {
        if let Some(s) = &state.scene {
            return Ok(format!(
                "scene already open at {} — SCENE REFRESH re-syncs it, SCENE CLOSE closes it",
                s.url
            ));
        }
        let handle = SceneHandle::start(state.system.clone(), *port, state.machine_mode, true)?;
        handle.set_box(state.box_size, &state.wall_indices, &state.names)?;
        let url = handle.url.clone();
        // Say which of the two things actually happened. Announcing "opened in
        // your browser" unconditionally is false on macOS and Windows (no
        // xdg-open) and whenever POSIM_NO_BROWSER is set, and it sends the
        // reader off to wait for a window that is not coming.
        let opened = if handle.browser_launched {
            "(asked your desktop to open it; if no window appeared, open that address yourself)"
        } else {
            "(no browser was launched — open that address yourself)"
        };
        state.scene = Some(handle);
        let n = state.system.objects.len();
        let tail = if n == 0 {
            "showing 0 entities — the window will be an empty grid. The scene\n\
             draws rigid bodies only (NEW adds them, then SCENE REFRESH);\n\
             quantum problems are viewed with QM ANIMATE / QM2 ANIMATE instead"
                .to_string()
        } else {
            format!(
                "showing {n} entit{}; SCENE START begins the evolution — HELP lists all scene commands",
                if n == 1 { "y" } else { "ies" },
            )
        };
        return Ok(format!("scene window created: {url}\n{opened}\n{tail}"));
    }
    if matches!(cmd, SceneCmd::Close) {
        return match state.scene.take() {
            Some(s) => {
                let url = s.url.clone();
                drop(s);
                Ok(format!("scene closed ({url})"))
            }
            None => Err("no scene window is open".to_string()),
        };
    }

    let scene = state
        .scene
        .as_ref()
        .ok_or_else(|| "no scene window — run SCENE CREATE first".to_string())?;
    match cmd {
        SceneCmd::Create { .. } | SceneCmd::Close => unreachable!("handled above"),
        SceneCmd::Translate => {
            let dz = pop_num(stack)?;
            let dy = pop_num(stack)?;
            let dx = pop_num(stack)?;
            scene.translate(dx, dy, dz)
        }
        SceneCmd::Rotate => {
            let dpitch = pop_num(stack)?;
            let dyaw = pop_num(stack)?;
            scene.rotate(dyaw, dpitch)
        }
        SceneCmd::Zoom => {
            let f = pop_num(stack)?;
            scene.zoom(f)
        }
        SceneCmd::ZoomIn => scene.zoom(1.25),
        SceneCmd::ZoomOut => scene.zoom(1.0 / 1.25),
        SceneCmd::Hide(which) => scene.set_visibility(*which, true),
        SceneCmd::Show(which) => scene.set_visibility(*which, false),
        SceneCmd::Refresh => {
            scene.sync(&state.system)?;
            scene.set_box(state.box_size, &state.wall_indices, &state.names)?;
            Ok(format!(
                "scene refreshed from the notebook state ({} entities, t = {})",
                state.system.objects.len(),
                state.system.time
            ))
        }
        SceneCmd::Redraw => {
            scene.redraw()?;
            Ok("scene redraw queued for every window".to_string())
        }
        SceneCmd::Start => scene.set_mode(RunMode::Running),
        SceneCmd::Stop => scene.set_mode(RunMode::Stopped),
        SceneCmd::Pause => scene.set_mode(RunMode::Paused),
        SceneCmd::ResetPlayback => scene.reset_playback(),
        SceneCmd::Reverse => scene.set_mode(RunMode::Reversing),
        SceneCmd::SetTimeStep => {
            let dt = pop_num(stack)?;
            scene.set_dt(dt)
        }
        SceneCmd::Status => scene.status(),
        SceneCmd::Events => {
            let events = scene.drain_events()?;
            if events.is_empty() {
                Ok("(no scene events)".to_string())
            } else {
                Ok(events.join("\n"))
            }
        }
    }
}

/// Promote a real to a complex so the mixed cases below need only one
/// arm each. A complex result whose imaginary part has vanished stays
/// complex: `(2+3i) - 3i` displays as `2 + 0i`, which is honest about
/// the type rather than silently collapsing it.
fn as_c(v: &Value) -> Option<Complex64> {
    match v {
        Value::Num(x) => Some(Complex64::real(*x)),
        Value::Complex(z) => Some(*z),
        _ => None,
    }
}

/// Numeric comparison, yielding 1.0 or 0.0.
///
/// Only numbers are ordered here. `==`/`!=` additionally accept two
/// vectors or two quaternions, because asking whether two positions
/// coincide is a reasonable thing to want; ordering them is not.
///
/// NaN compares false to everything, INCLUDING itself, so `x != x` is
/// the idiomatic NaN test and is deliberately not special-cased.
fn compare(op: CmpOp, a: Value, b: Value) -> Result<f64, String> {
    let t = |b: bool| if b { 1.0 } else { 0.0 };
    match (&a, &b) {
        (Value::Num(x), Value::Num(y)) => Ok(match op {
            CmpOp::Lt => t(x < y),
            CmpOp::Le => t(x <= y),
            CmpOp::Gt => t(x > y),
            CmpOp::Ge => t(x >= y),
            CmpOp::Eq => t(x == y),
            CmpOp::Ne => t(x != y),
        }),
        (Value::Vec3(x), Value::Vec3(y)) if matches!(op, CmpOp::Eq | CmpOp::Ne) => {
            let same = x == y;
            Ok(t(if matches!(op, CmpOp::Eq) { same } else { !same }))
        }
        (Value::Quat(x), Value::Quat(y)) if matches!(op, CmpOp::Eq | CmpOp::Ne) => {
            let same = x == y;
            Ok(t(if matches!(op, CmpOp::Eq) { same } else { !same }))
        }
        _ => Err(format!(
            "cannot compare {} {} {} (ordering is defined for numbers; `==` and `!=` also \
             accept two vectors or two quaternions)",
            type_name(&a),
            op.symbol(),
            type_name(&b)
        )),
    }
}

fn binary_add(a: Value, b: Value, op: &str) -> Result<Value, String> {
    if matches!(a, Value::Complex(_)) || matches!(b, Value::Complex(_)) {
        if let (Some(x), Some(y)) = (as_c(&a), as_c(&b)) {
            return Ok(Value::Complex(x + y));
        }
    }
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x + y)),
        (Value::Vec3(x), Value::Vec3(y)) => Ok(Value::Vec3(x + y)),
        (Value::Quat(x), Value::Quat(y)) => Ok(Value::Quat(x + y)),
        (a, b) => Err(format!(
            "cannot apply `{op}` to {} and {} (for vectors use dot()/cross())",
            type_name(&a),
            type_name(&b)
        )),
    }
}

fn binary_mul(a: Value, b: Value) -> Result<Value, String> {
    if matches!(a, Value::Complex(_)) || matches!(b, Value::Complex(_)) {
        if let (Some(x), Some(y)) = (as_c(&a), as_c(&b)) {
            return Ok(Value::Complex(x * y));
        }
    }
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => Ok(Value::Num(x * y)),
        (Value::Num(x), Value::Vec3(v)) | (Value::Vec3(v), Value::Num(x)) => Ok(Value::Vec3(v * x)),
        (Value::Num(x), Value::Quat(q)) | (Value::Quat(q), Value::Num(x)) => Ok(Value::Quat(q * x)),
        (Value::Num(x), Value::Mat3(m)) | (Value::Mat3(m), Value::Num(x)) => Ok(Value::Mat3(m * x)),
        (Value::Mat3(m), Value::Vec3(v)) => Ok(Value::Vec3(m * v)),
        (Value::Mat3(m), Value::Mat3(n)) => Ok(Value::Mat3(m * n)),
        (Value::Quat(p), Value::Quat(q)) => Ok(Value::Quat(p * q)),
        (a, b) => Err(format!(
            "cannot multiply {} by {} (for vec3*vec3 use dot()/cross())",
            type_name(&a),
            type_name(&b)
        )),
    }
}

/// Names taken by the expression builtins — a user function may not
/// shadow them.
const CORE_BUILTIN_NAMES: &[&str] =
    &["dot", "cross", "norm", "normalize", "sqrt", "abs", "sin", "cos", "exp", "log"];

/// A user function may shadow neither a core builtin nor a special
/// function. Kept as one predicate so the two lists cannot drift.
fn is_builtin_name(n: &str) -> bool {
    CORE_BUILTIN_NAMES.contains(&n) || crate::special::SPECIAL_NAMES.contains(&n)
}

/// Maximum user-function call depth (a plain safety net — the language
/// has no conditionals, so recursion can never terminate anyway).
const MAX_CALL_DEPTH: usize = 32;

/// Executes a user-defined function: binds arguments (missing trailing
/// ones take their defaults), pushes a call frame, runs the body lines
/// through the ordinary compile/execute pipeline, and returns the last
/// line's value.
/// `QM POTENTIAL` needs to evaluate a user function at each grid point,
/// and it lives in another module.
pub fn call_user_function_public(
    name: &str,
    args: Vec<Value>,
    state: &mut SimState,
) -> Result<Value, String> {
    call_user_function(name, args, state)
}

fn call_user_function(name: &str, args: Vec<Value>, state: &mut SimState) -> Result<Value, String> {
    let f = state.functions.get(name).cloned().expect("caller checked");
    if args.len() > f.params.len() {
        return Err(format!(
            "{name}() takes at most {} argument(s), got {} — signature: {name}({})",
            f.params.len(),
            args.len(),
            f.params
                .iter()
                .map(|(p, d)| match d {
                    Some(v) => format!("{p} = {v}"),
                    None => p.clone(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if state.env_stack.len() >= MAX_CALL_DEPTH {
        return Err(format!("function call depth limit ({MAX_CALL_DEPTH}) exceeded"));
    }
    let mut env = BTreeMap::new();
    for (k, (pname, default)) in f.params.iter().enumerate() {
        let v = if k < args.len() {
            args[k].clone()
        } else if let Some(d) = default {
            d.clone()
        } else {
            return Err(format!(
                "{name}(): missing argument `{pname}` (it has no default)"
            ));
        };
        env.insert(pname.clone(), v);
    }
    state.env_stack.push(env);
    let mut last = Value::Unit;
    for line in &f.body {
        match crate::parser::compile_line(line).and_then(|prog| execute(&prog, state)) {
            Ok(v) => last = v,
            Err(e) => {
                state.env_stack.pop();
                return Err(format!("{name}(): {e}\n  in body line: {line}"));
            }
        }
    }
    state.env_stack.pop();
    Ok(last)
}

/// True when a source line is the DEF line form (handled before the
/// ordinary grammar).
pub fn is_def_line(line: &str) -> bool {
    let t = line.trim_start();
    t.len() >= 4
        && t[..3].eq_ignore_ascii_case("def")
        && t.as_bytes()[3].is_ascii_whitespace()
}

/// Splits `text` on `sep` at brace/bracket/paren depth 0, outside
/// string literals.
fn split_top_level(text: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev_escape = false;
    for c in text.chars() {
        if in_str {
            cur.push(c);
            if prev_escape {
                prev_escape = false;
            } else if c == '\\' {
                prev_escape = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                cur.push(c);
            }
            '{' | '[' | '(' => {
                depth += 1;
                cur.push(c);
            }
            '}' | ']' | ')' => {
                depth -= 1;
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                parts.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    parts.push(cur);
    parts
}

/// Parses and installs a `DEF name(param [= default], ...) { body }`
/// definition. The body is captured as source lines; every line is
/// compile-checked NOW (correct syntax at definition time), defaults
/// are evaluated now, and redefinition replaces the previous version
/// (that is how a function is edited: SHOW it, adjust, re-DEF).
pub fn define_function(source: &str, state: &mut SimState) -> Result<Value, String> {
    /* split off the header at the first top-level '{' */
    let mut brace = None;
    {
        let mut in_str = false;
        let mut prev_escape = false;
        for (i, c) in source.char_indices() {
            if in_str {
                if prev_escape {
                    prev_escape = false;
                } else if c == '\\' {
                    prev_escape = true;
                } else if c == '"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' => {
                    brace = Some(i);
                    break;
                }
                _ => {}
            }
        }
    }
    let Some(open) = brace else {
        return Err("DEF needs a `{ body }` block: DEF name(params) { commands }".to_string());
    };
    /* find the brace that CLOSES the block (depth-matched, blind to
     * braces inside string literals and `#` comments — a `}` in a
     * trailing comment must not end the body) */
    let close = {
        let mut depth = 0i32;
        let mut in_str = false;
        let mut prev_escape = false;
        let mut in_comment = false;
        let mut found = None;
        for (i, c) in source.char_indices() {
            if i < open {
                continue;
            }
            if in_comment {
                if c == '\n' {
                    in_comment = false;
                }
                continue;
            }
            if in_str {
                if prev_escape {
                    prev_escape = false;
                } else if c == '\\' {
                    prev_escape = true;
                } else if c == '"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '#' => in_comment = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        found = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        found.ok_or_else(|| "DEF: missing the closing `}` (unbalanced braces)".to_string())?
    };
    let header = &source[..open];
    let body_text = &source[open + 1..close];
    let trailer = source[close + 1..].trim();
    if !trailer.is_empty() && !trailer.starts_with('#') {
        return Err(format!("DEF: unexpected text after the closing brace: `{trailer}`"));
    }

    /* header: def NAME ( params ) */
    let ht = header.trim();
    let after_def = ht[3..].trim_start();
    let lp = after_def
        .find('(')
        .ok_or_else(|| "DEF needs a parameter list: DEF name(params) { ... }".to_string())?;
    let fname = after_def[..lp].trim().to_ascii_lowercase();
    let valid_name = !fname.is_empty()
        && fname.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && fname.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid_name {
        return Err(format!("DEF: `{fname}` is not a valid function name"));
    }
    if is_builtin_name(&fname) {
        return Err(format!("DEF: `{fname}` is a builtin function and cannot be redefined"));
    }
    /* the name must lex as a plain identifier — a keyword (NEW, BOX,
     * SHOW, ...) could never be called back */
    match crate::lexer::tokenize(&fname) {
        Ok(toks)
            if toks.len() == 1 && matches!(toks[0].kind, crate::lexer::TokKind::Ident(_)) => {}
        _ => {
            return Err(format!(
                "DEF: `{fname}` is a reserved word and cannot name a function"
            ))
        }
    }
    let params_text = after_def[lp + 1..]
        .trim_end()
        .strip_suffix(')')
        .ok_or_else(|| "DEF: the parameter list is missing its `)`".to_string())?;
    let mut params: Vec<(String, Option<Value>)> = Vec::new();
    if !params_text.trim().is_empty() {
        for part in split_top_level(params_text, ',') {
            let part = part.trim();
            let (pname, default) = match part.split_once('=') {
                Some((n, d)) => (n.trim(), Some(d.trim())),
                None => (part, None),
            };
            let ok = !pname.is_empty()
                && pname.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && pname.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !ok {
                return Err(format!("DEF {fname}: `{pname}` is not a valid parameter name"));
            }
            /* a parameter must lex as a plain identifier: a keyword
             * (NEW, RESET, ...) would silently EXECUTE instead of
             * reading the argument; the builtins and pi/tau could
             * never be referenced */
            let lname = pname.to_ascii_lowercase();
            match crate::lexer::tokenize(&lname) {
                Ok(toks)
                    if toks.len() == 1
                        && matches!(toks[0].kind, crate::lexer::TokKind::Ident(_)) => {}
                _ => {
                    return Err(format!(
                        "DEF {fname}: `{pname}` is a reserved word and cannot name a \
                         parameter"
                    ))
                }
            }
            if is_builtin_name(&lname) || lname == "pi" || lname == "tau" {
                return Err(format!(
                    "DEF {fname}: `{pname}` is a builtin and cannot name a parameter"
                ));
            }
            if params.iter().any(|(existing, _)| *existing == lname) {
                return Err(format!("DEF {fname}: duplicate parameter `{pname}`"));
            }
            let dval = match default {
                None => None,
                Some(d) => {
                    /* defaults are EXPRESSIONS ONLY (a command like
                     * `reset` here is a definition-time error), and are
                     * evaluated once NOW (LET variables are visible) */
                    let prog = crate::parser::compile_expression(d)
                        .map_err(|e| format!("DEF {fname}: default for `{pname}`: {e}"))?;
                    Some(execute(&prog, state).map_err(|e| {
                        format!("DEF {fname}: default for `{pname}`: {e}")
                    })?)
                }
            };
            params.push((pname.to_ascii_lowercase(), dval));
        }
    }

    /* body: newline- and ;-separated commands, each compile-checked */
    let mut body = Vec::new();
    for raw_line in body_text.lines() {
        for cmd in split_top_level(raw_line, ';') {
            let cmd = cmd.trim();
            if cmd.is_empty() || cmd.starts_with('#') {
                continue;
            }
            if is_def_line(cmd) {
                return Err(format!("DEF {fname}: nested DEF is not allowed"));
            }
            crate::parser::compile_line(cmd)
                .map_err(|e| format!("DEF {fname}: body line `{cmd}`: {e}"))?;
            body.push(cmd.to_string());
        }
    }
    if body.is_empty() {
        return Err(format!("DEF {fname}: the body is empty"));
    }

    let redefined = state.functions.contains_key(&fname);
    let nparams = params.len();
    let nlines = body.len();
    state.functions.insert(
        fname.clone(),
        FuncDef { params, body, source: source.trim().to_string() },
    );
    Ok(Value::Str(format!(
        "function {fname}({nparams} parameter(s)) defined — {nlines} body line(s){}",
        if redefined { " (redefined)" } else { "" }
    )))
}

fn call_builtin(name: &str, mut args: Vec<Value>) -> Result<Value, String> {
    let arity_err = |want: usize, got: usize| {
        Err::<Value, String>(format!("{name}() takes {want} argument(s), got {got}"))
    };
    let num1 = |args: &mut Vec<Value>| -> Result<f64, String> {
        match args.remove(0) {
            Value::Num(n) => Ok(n),
            v => Err(format!("{name}() expects a number, got {}", type_name(&v))),
        }
    };
    match name {
        "dot" => {
            if args.len() != 2 {
                return arity_err(2, args.len());
            }
            let a = as_vec3(args.remove(0))?;
            let b = as_vec3(args.remove(0))?;
            Ok(Value::Num(a.dot(b)))
        }
        "cross" => {
            if args.len() != 2 {
                return arity_err(2, args.len());
            }
            let a = as_vec3(args.remove(0))?;
            let b = as_vec3(args.remove(0))?;
            Ok(Value::Vec3(a.cross(b)))
        }
        "norm" => {
            if args.len() != 1 {
                return arity_err(1, args.len());
            }
            match args.remove(0) {
                Value::Vec3(v) => Ok(Value::Num(v.norm())),
                Value::Quat(q) => Ok(Value::Num(q.norm())),
                Value::Num(n) => Ok(Value::Num(n.abs())),
                v => Err(format!("norm() expects a vector, got {}", type_name(&v))),
            }
        }
        "normalize" => {
            if args.len() != 1 {
                return arity_err(1, args.len());
            }
            Ok(Value::Vec3(as_vec3(args.remove(0))?.normalize()))
        }
        "sqrt" | "abs" | "sin" | "cos" | "exp" | "log" => {
            if args.len() != 1 {
                return arity_err(1, args.len());
            }
            // abs() of a complex number is its modulus. Without this,
            // the single most natural question to ask about a complex
            // result — how big is it — could not be asked at all, which
            // is why the outgoing-wave check |x h1(x)| = 1 was
            // unwritable in the language before the Hankel functions
            // arrived. The other five stay real-only: complex sqrt and
            // log carry branch-cut choices this language has not
            // committed to.
            if name == "abs" {
                if let Value::Complex(c) = &args[0] {
                    return Ok(Value::Num(c.abs()));
                }
            }
            let x = num1(&mut args)?;
            let y = match name {
                "sqrt" => x.sqrt(),
                "abs" => x.abs(),
                "sin" => x.sin(),
                "cos" => x.cos(),
                "exp" => x.exp(),
                _ => x.ln(),
            };
            Ok(Value::Num(y))
        }
        other => match crate::special::call(other, &args) {
            Some(r) => r,
            None => Err(format!("unknown function `{other}()` — see HELP")),
        },
    }
}

fn get_object(state: &SimState, i: usize) -> Result<&physical_object, String> {
    state.system.objects.get(i).ok_or_else(|| format!("no object obj{i}"))
}

fn component(v: Vec3, c: usize) -> Result<Value, String> {
    match c {
        0 => Ok(Value::Num(v.x)),
        1 => Ok(Value::Num(v.y)),
        2 => Ok(Value::Num(v.z)),
        _ => Err("component `.w` only exists on quaternions".to_string()),
    }
}

fn apply_comp(base: Value, comp: Option<usize>, path: &Path) -> Result<Value, String> {
    let Some(c) = comp else { return Ok(base) };
    match base {
        Value::Vec3(v) => component(v, c),
        Value::Quat(q) => Ok(Value::Num([q.w, q.x, q.y, q.z][match c {
            3 => 0, /* .w */
            0 => 1,
            1 => 2,
            _ => 3,
        }])),
        other => Err(format!("`{path}`: {} has no components", type_name(&other))),
    }
}

fn load_path(state: &SimState, path: &Path) -> Result<Value, String> {
    if let PathRoot::Named(n) = &path.root {
        let i = resolve_name(state, n)?;
        return load_path(
            state,
            &Path { root: PathRoot::Object(i), field: path.field.clone(), comp: path.comp },
        );
    }
    let base = match &path.root {
        PathRoot::System => match path.field.as_str() {
            "g_constant" | "g" => Value::Num(state.system.g_constant),
            "softening" => Value::Num(state.system.softening),
            "uniform_gravity" | "gravity" => Value::Vec3(state.system.uniform_gravity),
            "e_field" => Value::Vec3(state.system.e_field),
            "b_field" => Value::Vec3(state.system.b_field),
            "rtol" => Value::Num(state.system.rtol),
            "atol" => Value::Num(state.system.atol),
            "time" | "t" => Value::Num(state.system.time),
            "method" => Value::Str(format!("{:?}", state.system.method)),
            "count" | "n" => Value::Num(state.system.objects.len() as f64),
            "collide" => Value::Num(if state.system.collide_enabled { 1.0 } else { 0.0 }),
            "contacts" => Value::Num(state.system.contacts.len() as f64),
            "collisions" => Value::Num(state.system.collision_count as f64),
            "restitution_threshold" => Value::Num(state.system.restitution_threshold),
            "contact_slop" => Value::Num(state.system.contact_slop),
            "box" => Value::Num(state.box_size.unwrap_or(0.0)),
            other => {
                return Err(format!(
                    "unknown system field `{other}` (g_constant, softening, uniform_gravity, \
                     e_field, b_field, rtol, atol, time, method, count, collide, contacts, \
                     collisions, restitution_threshold, contact_slop, box)"
                ))
            }
        },
        PathRoot::Object(i) => {
            let o = get_object(state, *i)?;
            match path.field.as_str() {
                "id" => Value::Num(o.get_id() as f64),
                /* component shorthands: name.x, name.vx, ... */
                "x" => Value::Num(o.get_position().x),
                "y" => Value::Num(o.get_position().y),
                "z" => Value::Num(o.get_position().z),
                "vx" => Value::Num(o.get_velocity().x),
                "vy" => Value::Num(o.get_velocity().y),
                "vz" => Value::Num(o.get_velocity().z),
                /* dumbbell members (part masses recoverable from the
                 * stored mass fractions) */
                "m1" | "m2" | "m_rod" if matches!(o.get_boundary(), Boundary::Dumbbell { .. }) => {
                    let (m1, m2, m_rod, ..) = dumbbell_members(o).expect("checked");
                    Value::Num(match path.field.as_str() {
                        "m1" => m1,
                        "m2" => m2,
                        _ => m_rod,
                    })
                }
                "r1" | "r2" if matches!(o.get_boundary(), Boundary::Dumbbell { .. }) => {
                    let (_, _, _, r1, r2, ..) = dumbbell_members(o).expect("checked");
                    Value::Num(if path.field == "r1" { r1 } else { r2 })
                }
                "rod_radius" | "rod_r"
                    if matches!(o.get_boundary(), Boundary::Dumbbell { .. }) =>
                {
                    let (.., rod_r, _) = dumbbell_members(o).expect("checked");
                    Value::Num(rod_r)
                }
                "length" | "len" if matches!(o.get_boundary(), Boundary::Dumbbell { .. }) => {
                    let (.., len) = dumbbell_members(o).expect("checked");
                    Value::Num(len)
                }
                "mass" => Value::Num(o.get_mass()),
                "inverse_mass" => Value::Num(o.get_inverse_mass()),
                "charge" => Value::Num(o.get_charge()),
                "position" | "pos" => Value::Vec3(o.get_position()),
                "velocity" | "vel" => Value::Vec3(o.get_velocity()),
                "momentum" => Value::Vec3(o.get_momentum()),
                "orientation" => Value::Quat(o.get_orientation()),
                "angular_velocity" => Value::Vec3(o.get_angular_velocity()),
                "angular_momentum" => Value::Vec3(o.get_angular_momentum()),
                "inertia_tensor" => Value::Mat3(o.get_inertia_tensor()),
                "inverse_inertia_tensor" => Value::Mat3(o.get_inverse_inertia_tensor()),
                "magnetic_moment_tensor" => Value::Mat3(o.get_magnetic_moment_tensor()),
                "kinetic_energy" | "energy" => Value::Num(o.kinetic_energy()),
                "boundary" | "shape" => Value::Str(format!("{:?}", o.get_boundary())),
                "radius" => match o.get_boundary() {
                    Boundary::Sphere { radius }
                    | Boundary::Disk { radius }
                    | Boundary::Cylinder { radius, .. } => Value::Num(radius),
                    b => {
                        return Err(format!(
                            "obj{i} has no radius (boundary = {b:?}; radius reads a sphere, \
                             disk or cylinder)"
                        ))
                    }
                },
                "half_extents" => match o.get_boundary() {
                    Boundary::Cuboid { half_extents } => Value::Vec3(Vec3::from_array(half_extents)),
                    b => return Err(format!("obj{i} is not a cuboid (boundary = {b:?})")),
                },
                "ring_radius" => match o.get_boundary() {
                    Boundary::Torus { ring_radius, .. } => Value::Num(ring_radius),
                    b => return Err(format!("obj{i} is not a torus (boundary = {b:?})")),
                },
                "tube_radius" => match o.get_boundary() {
                    Boundary::Torus { tube_radius, .. } => Value::Num(tube_radius),
                    b => return Err(format!("obj{i} is not a torus (boundary = {b:?})")),
                },
                "inner_radius" => match o.get_boundary() {
                    Boundary::Torus { ring_radius, tube_radius } => {
                        Value::Num(ring_radius - tube_radius)
                    }
                    b => return Err(format!("obj{i} is not a torus (boundary = {b:?})")),
                },
                "outer_radius" => match o.get_boundary() {
                    Boundary::Torus { ring_radius, tube_radius } => {
                        Value::Num(ring_radius + tube_radius)
                    }
                    b => return Err(format!("obj{i} is not a torus (boundary = {b:?})")),
                },
                "height" | "half_height" => match o.get_boundary() {
                    Boundary::Cylinder { half_height, .. } => {
                        if path.field == "height" {
                            Value::Num(2.0 * half_height)
                        } else {
                            Value::Num(half_height)
                        }
                    }
                    b => return Err(format!("obj{i} is not a cylinder (boundary = {b:?})")),
                },
                "force" => Value::Vec3(state.system.external_forces[*i]),
                "torque" => Value::Vec3(state.system.external_torques[*i]),
                "restitution" => Value::Num(o.get_restitution()),
                other => {
                    return Err(format!(
                        "unknown object field `{other}` — see HELP for the field list"
                    ))
                }
            }
        }
        PathRoot::Contact(k) => {
            let c = state.system.contacts.get(*k).ok_or_else(|| {
                format!(
                    "no contact{k} — the last STEP/RUN recorded {} contact(s); \
                     CONTACTS lists them",
                    state.system.contacts.len()
                )
            })?;
            match path.field.as_str() {
                "i" => Value::Num(c.i as f64),
                "j" => Value::Num(c.j as f64),
                "t" | "time" => Value::Num(c.t),
                "point" => Value::Vec3(c.point),
                "normal" => Value::Vec3(c.normal),
                "depth" => Value::Num(c.depth),
                "rel_vel_n" | "approach" => Value::Num(c.rel_vel_n),
                "impulse" | "impulse_n" => Value::Num(c.impulse_n),
                other => {
                    return Err(format!(
                        "unknown contact field `{other}` (i, j, t, point, normal, depth, \
                         rel_vel_n, impulse)"
                    ))
                }
            }
        }
        PathRoot::Named(_) => unreachable!("named roots resolve to Object above"),
    };
    apply_comp(base, path.comp, path)
}

fn store_path(state: &mut SimState, path: &Path, value: Value) -> Result<(), String> {
    if let PathRoot::Named(n) = &path.root {
        let i = resolve_name(state, n)?;
        return store_path(
            state,
            &Path { root: PathRoot::Object(i), field: path.field.clone(), comp: path.comp },
            value,
        );
    }
    /* component store: read-modify-write through the full-field API */
    if let Some(c) = path.comp {
        let full = Path { comp: None, ..path.clone() };
        let base = load_path(state, &full)?;
        let n = match value {
            Value::Num(n) => n,
            v => return Err(format!("component store expects a number, got {}", type_name(&v))),
        };
        let updated = match base {
            Value::Vec3(mut v) => {
                match c {
                    0 => v.x = n,
                    1 => v.y = n,
                    2 => v.z = n,
                    _ => return Err("component `.w` only exists on quaternions".to_string()),
                }
                Value::Vec3(v)
            }
            Value::Quat(mut q) => {
                match c {
                    3 => q.w = n,
                    0 => q.x = n,
                    1 => q.y = n,
                    _ => q.z = n,
                }
                Value::Quat(q)
            }
            other => return Err(format!("`{path}`: {} has no components", type_name(&other))),
        };
        return store_path(state, &full, updated);
    }

    match &path.root {
        PathRoot::System => {
            let num = |v: Value| match v {
                Value::Num(n) => Ok(n),
                v => Err(format!("system.{} expects a number, got {}", path.field, type_name(&v))),
            };
            match path.field.as_str() {
                "g_constant" | "g" => state.system.g_constant = num(value)?,
                "softening" => state.system.softening = num(value)?,
                "uniform_gravity" | "gravity" => state.system.uniform_gravity = as_vec3(value)?,
                "e_field" => state.system.e_field = as_vec3(value)?,
                "b_field" => state.system.b_field = as_vec3(value)?,
                "rtol" => state.system.rtol = num(value)?,
                "atol" => state.system.atol = num(value)?,
                "time" | "t" => state.system.time = num(value)?,
                "restitution_threshold" => {
                    let v = num(value)?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err("restitution_threshold must be a finite number >= 0".into());
                    }
                    state.system.restitution_threshold = v;
                }
                "contact_slop" => {
                    let v = num(value)?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err("contact_slop must be a finite number >= 0".into());
                    }
                    state.system.contact_slop = v;
                }
                "collide" => {
                    return Err(
                        "use COLLIDE ON / COLLIDE OFF to switch collision detection".into()
                    )
                }
                other => {
                    return Err(format!(
                        "system field `{other}` is not writable (or unknown) — see HELP"
                    ))
                }
            }
        }
        PathRoot::Object(i) => {
            let i = *i;
            if i >= state.system.objects.len() {
                return Err(format!("no object obj{i}"));
            }
            let num = |v: Value| match v {
                Value::Num(n) => Ok(n),
                v => Err(format!("obj.{} expects a number, got {}", path.field, type_name(&v))),
            };
            match path.field.as_str() {
                "id" => {
                    let n = num(value)?;
                    state.system.objects[i].set_id(n as usize);
                }
                "x" | "y" | "z" => {
                    let n = num(value)?;
                    let o = &mut state.system.objects[i];
                    let mut pos = o.get_position();
                    match path.field.as_str() {
                        "x" => pos.x = n,
                        "y" => pos.y = n,
                        _ => pos.z = n,
                    }
                    o.set_position(pos);
                }
                "vx" | "vy" | "vz" => {
                    let n = num(value)?;
                    let o = &mut state.system.objects[i];
                    let mut vel = o.get_velocity();
                    match path.field.as_str() {
                        "vx" => vel.x = n,
                        "vy" => vel.y = n,
                        _ => vel.z = n,
                    }
                    o.set_velocity(vel);
                }
                "m1" | "m2" | "m_rod"
                    if matches!(
                        state.system.objects[i].get_boundary(),
                        Boundary::Dumbbell { .. }
                    ) =>
                {
                    let n = num(value)?;
                    dumbbell_member_write(&mut state.system.objects[i], path.field.as_str(), n)?;
                }
                "r1" | "r2" | "rod_r"
                    if matches!(
                        state.system.objects[i].get_boundary(),
                        Boundary::Dumbbell { .. }
                    ) =>
                {
                    let n = num(value)?;
                    dumbbell_member_write(&mut state.system.objects[i], path.field.as_str(), n)?;
                }
                "rod_radius" | "length" | "len"
                    if matches!(
                        state.system.objects[i].get_boundary(),
                        Boundary::Dumbbell { .. }
                    ) =>
                {
                    let n = num(value)?;
                    dumbbell_member_write(&mut state.system.objects[i], path.field.as_str(), n)?;
                }
                "mass" => {
                    let n = num(value)?;
                    state.system.objects[i].set_mass(n);
                }
                "inverse_mass" => {
                    let n = num(value)?;
                    state.system.objects[i].set_inverse_mass(n);
                }
                "charge" => {
                    let n = num(value)?;
                    state.system.objects[i].set_charge(n);
                }
                "position" | "pos" => {
                    let v = as_vec3(value)?;
                    state.system.objects[i].set_position(v);
                }
                "velocity" | "vel" => {
                    let v = as_vec3(value)?;
                    state.system.objects[i].set_velocity(v);
                }
                "momentum" => {
                    let v = as_vec3(value)?;
                    state.system.objects[i].set_momentum(v);
                }
                "orientation" => {
                    let q = as_quat(value)?;
                    state.system.objects[i].set_orientation(q);
                }
                "angular_velocity" => {
                    let v = as_vec3(value)?;
                    state.system.objects[i].set_angular_velocity(v);
                }
                "angular_momentum" => {
                    let v = as_vec3(value)?;
                    state.system.objects[i].set_angular_momentum(v);
                }
                "inertia_tensor" => {
                    let m = as_mat3(value)?;
                    state.system.objects[i].set_inertia_tensor(m);
                }
                "inverse_inertia_tensor" => {
                    let m = as_mat3(value)?;
                    state.system.objects[i].set_inverse_inertia_tensor(m);
                }
                "magnetic_moment_tensor" => {
                    let m = as_mat3(value)?;
                    state.system.objects[i].set_magnetic_moment_tensor(m);
                }
                "radius" => {
                    let r = num(value)?;
                    if !(r.is_finite() && r > 0.0) {
                        return Err(format!("radius must be a finite number > 0, got {r}"));
                    }
                    /* radius keeps the shape family: a disk stays a
                     * disk, a cylinder keeps its height; a torus is
                     * refused (it has two radii — mirroring the read
                     * side); anything else becomes a sphere (the
                     * historical behavior) */
                    let b = match state.system.objects[i].get_boundary() {
                        Boundary::Disk { .. } => Boundary::Disk { radius: r },
                        Boundary::Cylinder { half_height, .. } => {
                            Boundary::Cylinder { radius: r, half_height }
                        }
                        Boundary::Torus { .. } => {
                            return Err(format!(
                                "obj{i} is a torus — set ring_radius/tube_radius or \
                                 inner_radius/outer_radius instead of radius"
                            ))
                        }
                        _ => Boundary::Sphere { radius: r },
                    };
                    state.system.objects[i].set_boundary(b);
                    state.system.objects[i].recompute_inertia_from_boundary();
                }
                "half_extents" => {
                    let v = as_vec3(value)?;
                    state.system.objects[i].set_boundary(Boundary::Cuboid {
                        half_extents: v.to_array(),
                    });
                    state.system.objects[i].recompute_inertia_from_boundary();
                }
                "ring_radius" | "tube_radius" | "inner_radius" | "outer_radius" => {
                    let n = num(value)?;
                    /* inner_radius admits 0 — the horn torus is a valid
                     * body (readable via GET, so writable too) */
                    let lo_ok = if path.field == "inner_radius" { n >= 0.0 } else { n > 0.0 };
                    if !(n.is_finite() && lo_ok) {
                        return Err(format!(
                            "{} must be a finite number {} 0, got {n}",
                            path.field,
                            if path.field == "inner_radius" { ">=" } else { ">" }
                        ));
                    }
                    let (ring, tube) = match state.system.objects[i].get_boundary() {
                        Boundary::Torus { ring_radius, tube_radius } => (ring_radius, tube_radius),
                        b => {
                            return Err(format!(
                                "obj{i} is not a torus (boundary = {b:?}); NEW TORUS creates one"
                            ))
                        }
                    };
                    /* inner/outer update the derived pair holding the
                     * other one fixed: ring = (inner+outer)/2,
                     * tube = (outer-inner)/2 — order-independent when
                     * both are given in a NEW initializer list */
                    let (inner, outer) = (ring - tube, ring + tube);
                    let (new_inner, new_outer) = match path.field.as_str() {
                        "ring_radius" => (n - tube, n + tube),
                        "tube_radius" => (ring - n, ring + n),
                        "inner_radius" => (n, outer),
                        _ => (inner, n),
                    };
                    if new_outer <= new_inner || new_inner < 0.0 {
                        return Err(format!(
                            "torus needs 0 <= inner < outer (got inner = {new_inner}, \
                             outer = {new_outer})"
                        ));
                    }
                    state.system.objects[i].set_boundary(Boundary::Torus {
                        ring_radius: 0.5 * (new_inner + new_outer),
                        tube_radius: 0.5 * (new_outer - new_inner),
                    });
                    state.system.objects[i].recompute_inertia_from_boundary();
                }
                "height" | "half_height" => {
                    let n = num(value)?;
                    if !(n.is_finite() && n > 0.0) {
                        return Err(format!("{} must be a finite number > 0, got {n}", path.field));
                    }
                    let radius = match state.system.objects[i].get_boundary() {
                        Boundary::Cylinder { radius, .. } => radius,
                        b => {
                            return Err(format!(
                                "obj{i} is not a cylinder (boundary = {b:?}); NEW CYLINDER \
                                 creates one"
                            ))
                        }
                    };
                    let half_height = if path.field == "height" { 0.5 * n } else { n };
                    state.system.objects[i]
                        .set_boundary(Boundary::Cylinder { radius, half_height });
                    state.system.objects[i].recompute_inertia_from_boundary();
                }
                "force" => {
                    let v = as_vec3(value)?;
                    state.system.external_forces[i] = v;
                }
                "torque" => {
                    let v = as_vec3(value)?;
                    state.system.external_torques[i] = v;
                }
                "restitution" => {
                    let n = num(value)?;
                    if !(n.is_finite() && (0.0..=1.0).contains(&n)) {
                        return Err(format!(
                            "restitution must be between 0 (plastic) and 1 (elastic), got {n}"
                        ));
                    }
                    state.system.objects[i].set_restitution(n);
                }
                other => {
                    return Err(format!(
                        "object field `{other}` is not writable (or unknown) — see HELP"
                    ))
                }
            }
        }
        PathRoot::Contact(k) => {
            return Err(format!(
                "contact{k} fields are read-only — contacts are the record of what the \
                 solver already resolved"
            ))
        }
        PathRoot::Named(_) => unreachable!("named roots resolve to Object above"),
    }
    Ok(())
}

/// Convenience: lex + parse + execute one command line. `DEF` is a
/// line form handled here, before the expression grammar (its body may
/// span multiple lines — the notebook joins them).
pub fn execute_line(line: &str, state: &mut SimState) -> Result<Value, String> {
    if is_def_line(line) {
        return define_function(line, state);
    }
    let prog = crate::parser::compile_line(line)?;
    execute(&prog, state)
}

#[cfg(test)]
mod tests {
    /// `abs()` accepts a complex value and returns its modulus. This
    /// arrived with the Hankel functions, because until then there was
    /// no way to ask how big a complex result was — which made the
    /// outgoing-wave property `|x h1_n(x)| -> 1` unwritable in the
    /// language even though every ingredient was present.
    #[test]
    fn abs_of_a_complex_value_is_its_modulus() {
        use special_functions::complex::Complex64 as Cx;
        let call = |v: super::Value| super::call_builtin("abs", vec![v]);
        // 3 - 4i has modulus exactly 5.
        match call(super::Value::Complex(Cx::new(3.0, -4.0))) {
            Ok(super::Value::Num(x)) => assert_eq!(x, 5.0),
            other => panic!("abs(3-4i) gave {other:?}"),
        }
        // The real case is unchanged.
        match call(super::Value::Num(-2.5)) {
            Ok(super::Value::Num(x)) => assert_eq!(x, 2.5),
            other => panic!("abs(-2.5) gave {other:?}"),
        }
        // ... and the outgoing-wave check the language can now express.
        let h = special_functions::hankel::sph_hankel_h1(3, 800.0).unwrap() * 800.0;
        match call(super::Value::Complex(h)) {
            Ok(super::Value::Num(x)) => {
                // n(n+1)/(4x^2) = 12/(4 * 640000)
                assert!((x - 1.0 - 12.0 / (4.0 * 640_000.0)).abs() < 1e-10, "|x h1_3| = {x}");
            }
            other => panic!("abs of a Hankel value gave {other:?}"),
        }
        // sqrt/log stay real-only: a complex argument is an error, not
        // a silent branch-cut choice.
        assert!(call_is_err("sqrt", Cx::new(-1.0, 0.0)));
        assert!(call_is_err("log", Cx::new(-1.0, 0.0)));
    }

    fn call_is_err(name: &str, z: special_functions::complex::Complex64) -> bool {
        super::call_builtin(name, vec![super::Value::Complex(z)]).is_err()
    }


    /// Comparison operators yield 1/0, which is what lets
    /// `(x > a) * (x < b)` act as an indicator function.
    #[test]
    fn comparison_operators() {
        let mut st = SimState::default();
        let n = |l: &str, st: &mut SimState| match execute_line(l, st).unwrap() {
            Value::Num(v) => v,
            other => panic!("`{l}` gave {other}"),
        };
        assert_eq!(n("3 < 5", &mut st), 1.0);
        assert_eq!(n("3 > 5", &mut st), 0.0);
        assert_eq!(n("3 <= 3", &mut st), 1.0);
        assert_eq!(n("3 >= 4", &mut st), 0.0);
        assert_eq!(n("2 == 2", &mut st), 1.0);
        assert_eq!(n("2 != 2", &mut st), 0.0);
        // lower precedence than + and -, so this is (1+1) > 1
        assert_eq!(n("1 + 1 > 1", &mut st), 1.0);
        // indicator function
        assert_eq!(n("(0.5 > 0) * (0.5 < 1)", &mut st), 1.0);
        assert_eq!(n("(2 > 0) * (2 < 1)", &mut st), 0.0);
        // NaN compares false to everything, including itself — so
        // `x != x` is the NaN test, and is deliberately not special-cased
        assert_eq!(n("0/0 != 0/0", &mut st), 1.0);
        assert_eq!(n("0/0 == 0/0", &mut st), 0.0);
        // `=` is still assignment, not comparison
        assert!(execute_line("let q = 3", &mut st).is_ok());
        assert_eq!(n("q == 3", &mut st), 1.0);
        // vectors: equality yes, ordering no
        assert_eq!(n("[1,2,3] == [1,2,3]", &mut st), 1.0);
        assert_eq!(n("[1,2,3] != [1,2,4]", &mut st), 1.0);
        assert!(execute_line("[1,2,3] < [1,2,4]", &mut st).is_err(), "ordering vectors");
        assert!(execute_line("1 < \"a\"", &mut st).is_err(), "comparing a string");
    }
    use super::*;

    #[test]
    fn new_set_get_roundtrip() {
        let mut st = SimState::default();
        let v = execute_line(
            "new sphere { mass = 2, radius = 0.5, charge = -1.5, position = [0, 10, 0], velocity = [1, 0, -0.5] }",
            &mut st,
        )
        .unwrap();
        assert_eq!(v, Value::Str("obj0".to_string()));
        assert_eq!(execute_line("get obj0.mass", &mut st).unwrap(), Value::Num(2.0));
        /* velocity init deferred after mass: momentum = m v */
        assert_eq!(
            execute_line("get obj0.momentum", &mut st).unwrap(),
            Value::Vec3(Vec3::new(2.0, 0.0, -1.0))
        );
        /* inertia recomputed from shape: 0.4 * 2 * 0.25 = 0.2 */
        match execute_line("get obj0.inertia_tensor", &mut st).unwrap() {
            Value::Mat3(m) => assert!((m.0[0][0] - 0.2).abs() < 1e-15),
            v => panic!("expected mat3, got {v}"),
        }
        execute_line("set obj0.position.y = 42", &mut st).unwrap();
        assert_eq!(execute_line("get obj0.position.y", &mut st).unwrap(), Value::Num(42.0));
    }

    #[test]
    fn expressions_and_builtins() {
        let mut st = SimState::default();
        assert_eq!(execute_line("1 + 2 * 3", &mut st).unwrap(), Value::Num(7.0));
        assert_eq!(
            execute_line("cross([1,0,0], [0,1,0])", &mut st).unwrap(),
            Value::Vec3(Vec3::new(0.0, 0.0, 1.0))
        );
        assert_eq!(execute_line("dot([1,2,3], [4,5,6])", &mut st).unwrap(), Value::Num(32.0));
        assert_eq!(execute_line("norm([3,4,0])", &mut st).unwrap(), Value::Num(5.0));
        let e = execute_line("[1,0,0] * [0,1,0]", &mut st).unwrap_err();
        assert!(e.contains("dot()/cross()"), "{e}");
    }

    #[test]
    fn step_runs_sundials() {
        let mut st = SimState::default();
        execute_line("new point { mass = 2, position = [0, 10, 0], velocity = [1, 0, 0] }", &mut st)
            .unwrap();
        execute_line("set system.gravity = [0, -9.81, 0]", &mut st).unwrap();
        execute_line("step 1", &mut st).unwrap();
        /* y(1) = 10 - 9.81/2 = 5.095 */
        match execute_line("get obj0.position.y", &mut st).unwrap() {
            Value::Num(y) => assert!((y - 5.095).abs() < 1e-8, "y = {y}"),
            v => panic!("expected number, got {v}"),
        }
        match execute_line("get system.time", &mut st).unwrap() {
            Value::Num(t) => assert!((t - 1.0).abs() < 1e-12),
            v => panic!("expected number, got {v}"),
        }
    }

    #[test]
    fn observables_and_errors() {
        let mut st = SimState::default();
        execute_line("new point { mass = 1, position = [1, 0, 0] }", &mut st).unwrap();
        execute_line("new point { mass = 3, position = [-1, 0, 0] }", &mut st).unwrap();
        assert_eq!(
            execute_line("com", &mut st).unwrap(),
            Value::Vec3(Vec3::new(-0.5, 0.0, 0.0))
        );
        assert!(execute_line("get obj7.mass", &mut st).unwrap_err().contains("no object"));
        assert!(execute_line("get obj0.bogus", &mut st).unwrap_err().contains("unknown object field"));
    }

    #[test]
    fn collide_command_and_contact_paths() {
        let mut st = SimState::default();
        // Two spheres on a collision course (G = 1 default is irrelevant
        // at these masses/speeds over the short run).
        execute_line(
            "new sphere { mass = 1, radius = 0.5, position = [-2, 0, 0], velocity = [1, 0, 0], restitution = 0.5 }",
            &mut st,
        )
        .unwrap();
        execute_line(
            "new sphere { mass = 1, radius = 0.5, position = [2, 0, 0], velocity = [-1, 0, 0] }",
            &mut st,
        )
        .unwrap();
        // NEW-block restitution round-trips.
        assert_eq!(execute_line("get obj0.restitution", &mut st).unwrap(), Value::Num(0.5));
        // COLLIDE status / OFF / ON.
        match execute_line("collide", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("ON") && s.contains("1 collidable pair"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        execute_line("collide off", &mut st).unwrap();
        assert_eq!(execute_line("get system.collide", &mut st).unwrap(), Value::Num(0.0));
        execute_line("collide on", &mut st).unwrap();

        // Step across the impact; contacts become readable.
        match execute_line("step 3", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("collision(s)"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        match execute_line("get system.contacts", &mut st).unwrap() {
            Value::Num(n) => assert!(n >= 1.0, "contacts = {n}"),
            v => panic!("expected number, got {v}"),
        }
        match execute_line("get contact0.normal", &mut st).unwrap() {
            Value::Vec3(n) => assert!((n.x - 1.0).abs() < 1e-9, "normal i→j = +x, got {n:?}"),
            v => panic!("expected vec3, got {v}"),
        }
        match execute_line("get contact0.normal.x", &mut st).unwrap() {
            Value::Num(x) => assert!((x - 1.0).abs() < 1e-9),
            v => panic!("expected number, got {v}"),
        }
        match execute_line("contacts", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("normal") && s.contains("impulse"), "{s}");
            }
            v => panic!("expected string, got {v}"),
        }
        // `system.collisions` (the running impulse count) must stay a
        // distinct path from `system.collide` (the on/off flag) — the
        // word `collisions` is deliberately not a keyword alias.
        assert_eq!(execute_line("get system.collisions", &mut st).unwrap(), Value::Num(1.0));
        execute_line("collide off", &mut st).unwrap();
        assert_eq!(execute_line("get system.collisions", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get system.collide", &mut st).unwrap(), Value::Num(0.0));
        execute_line("collide on", &mut st).unwrap();
        // Read-only + range errors.
        let e = execute_line("set contact0.depth = 1", &mut st).unwrap_err();
        assert!(e.contains("read-only"), "{e}");
        let e = execute_line("get contact9.normal", &mut st).unwrap_err();
        assert!(e.contains("no contact9"), "{e}");
        // Restitution validation.
        let e = execute_line("set obj0.restitution = 2", &mut st).unwrap_err();
        assert!(e.contains("between 0"), "{e}");
        // Bad path root suggestion mentions contactK.
        let e = execute_line("get contactx.normal", &mut st).unwrap_err();
        assert!(e.contains("contactK"), "{e}");
        // HELP documents the family (lockstep guard).
        match execute_line("help", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("COLLIDE") && s.contains("CONTACTS") && s.contains("restitution"));
            }
            v => panic!("expected string, got {v}"),
        }
    }

    #[test]
    fn scene_commands_require_a_window() {
        let mut st = SimState::default();
        for line in [
            "scene status",
            "scene start",
            "scene translate 1 0",
            "scene zoom in",
            "scene events",
        ] {
            let e = execute_line(line, &mut st).unwrap_err();
            assert!(e.contains("SCENE CREATE"), "{line}: {e}");
        }
        let e = execute_line("scene close", &mut st).unwrap_err();
        assert!(e.contains("no scene window"), "{e}");
    }

    #[test]
    fn method_switch_and_sprk_gate() {
        let mut st = SimState::default();
        execute_line("new sphere { mass = 1, angular_velocity = [0, 2, 0] }", &mut st).unwrap();
        execute_line("method sprk leapfrog_2_2 0.01", &mut st).unwrap();
        let e = execute_line("run 1", &mut st).unwrap_err();
        assert!(e.contains("SPRK"), "{e}");
        execute_line("method adams", &mut st).unwrap();
        execute_line("run 1 steps 2", &mut st).unwrap();
    }

    #[test]
    fn new_shapes_and_parameter_paths() {
        let mut st = SimState::default();
        // Torus by inner/outer radius (order-independent pair): ring
        // 1.5, tube 0.5 — the manager-demo torus.
        let v = execute_line("new torus { mass = 1, inner_radius = 1, outer_radius = 2 }", &mut st)
            .unwrap();
        assert_eq!(v, Value::Str("obj0".to_string()));
        assert_eq!(execute_line("get obj0.ring_radius", &mut st).unwrap(), Value::Num(1.5));
        assert_eq!(execute_line("get obj0.tube_radius", &mut st).unwrap(), Value::Num(0.5));
        assert_eq!(execute_line("get obj0.inner_radius", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get obj0.outer_radius", &mut st).unwrap(), Value::Num(2.0));
        // Solid-torus inertia about the axis: Iz = m(c² + ¾a²) = 2.4375.
        match execute_line("get obj0.inertia_tensor", &mut st).unwrap() {
            Value::Mat3(m) => assert!((m.0[2][2] - 2.4375).abs() < 1e-12),
            v => panic!("expected mat3, got {v}"),
        }
        // Disk m = 2/3, r = 1: Iz = ½ m a² = 1/3.
        execute_line("new disk { mass = 2/3, radius = 1 }", &mut st).unwrap();
        match execute_line("get obj1.inertia_tensor", &mut st).unwrap() {
            Value::Mat3(m) => assert!((m.0[2][2] - 1.0 / 3.0).abs() < 1e-12),
            v => panic!("expected mat3, got {v}"),
        }
        // Cylinder m = 2, r = 1/4, height 1.5 (HEIGHT is the full
        // height; half_height reads 0.75): Iz = ½ m r² = 0.0625.
        execute_line("new cylinder { mass = 2, radius = 1/4, height = 1.5 }", &mut st).unwrap();
        assert_eq!(execute_line("get obj2.half_height", &mut st).unwrap(), Value::Num(0.75));
        assert_eq!(execute_line("get obj2.height", &mut st).unwrap(), Value::Num(1.5));
        assert_eq!(execute_line("get obj2.radius", &mut st).unwrap(), Value::Num(0.25));
        match execute_line("get obj2.inertia_tensor", &mut st).unwrap() {
            Value::Mat3(m) => assert!((m.0[2][2] - 0.0625).abs() < 1e-12),
            v => panic!("expected mat3, got {v}"),
        }
        // LIST names the shapes; shape params error on the wrong shape.
        match execute_line("list", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("torus ring=1.5 tube=0.5"), "{s}");
                assert!(s.contains("disk r=1"), "{s}");
                assert!(s.contains("cylinder r=0.25 h=1.5"), "{s}");
            }
            v => panic!("expected string, got {v}"),
        }
        let e = execute_line("get obj1.ring_radius", &mut st).unwrap_err();
        assert!(e.contains("not a torus"), "{e}");
        // HELP documents the whole family.
        match execute_line("help", &mut st).unwrap() {
            Value::Str(h) => {
                assert!(h.contains("TORUS"), "help lists TORUS");
                assert!(h.contains("BOX <size>"), "help lists BOX");
                assert!(h.contains("inverse_mass = 0"), "help explains inverse mass");
            }
            v => panic!("expected string, got {v}"),
        }
    }

    /// The four solver families reached through the language: a rod
    /// (IDA), a rest state (KINSOL), a derivative (CVODES), and the
    /// refusals that keep each honest.
    #[test]
    fn constrain_equilibrium_and_sensitivity_round_trip() {
        let mut st = SimState::default();
        for line in [
            "set system.g_constant = 0",
            "set system.uniform_gravity = [0, -9.81, 0]",
            "collide off",
            "new sphere as pivot { mass = 1, radius = 0.05, inverse_mass = 0 }",
            "new sphere as bob { mass = 1, radius = 0.1, position = [0.6, -0.8, 0] }",
        ] {
            execute_line(line, &mut st).unwrap();
        }

        // Bare CONSTRAIN freezes the separation they already have (3-4-5).
        match execute_line("constrain pivot bob", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("rod obj0 <-> obj1"), "{s}");
                assert!(s.contains("1 row(s)"), "{s}");
                assert!(s.contains("METHOD IDA"), "the reply must say what to do next: {s}");
            }
            v => panic!("expected string, got {v}"),
        }
        match execute_line("constraints", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("worst |g| = 0e0"), "{s}"),
            v => panic!("expected string, got {v}"),
        }

        // A constrained system refuses every method but IDA, by name.
        let e = execute_line("run 1 steps 5", &mut st).unwrap_err();
        assert!(e.contains("METHOD IDA"), "{e}");

        match execute_line("method ida", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("IDA"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        execute_line("run 1 steps 10", &mut st).unwrap();
        // the rod is still exactly a rod
        let (g, _) = st.system.constraints.drift(&st.system);
        assert!(g < 1e-10, "rod stretched by {g:e}");

        // EQUILIBRIUM hangs it straight down and stops there.
        match execute_line("equilibrium", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("equilibrium found"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        let bob = st.system.objects[1].get_position();
        assert!(bob.x.abs() < 1e-9 && (bob.y + 1.0).abs() < 1e-9, "bob at {bob:?}");
        assert_eq!(st.system.objects[0].get_position(), Vec3::zeros(), "anchor fixed");

        // CONSTRAIN OFF drops it, and then SENSITIVITY runs through CVODES.
        match execute_line("constrain off", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("removed 1"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        execute_line("method adams", &mut st).unwrap();
        match execute_line("sensitivity 3 \"gravity.y\"", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("CVODES"), "{s}");
                /* Free fall: d(y)/d(g) = T²/2 = 4.5. Parse the number out
                 * rather than substring-matching its decimal expansion —
                 * the last digits are solver noise and a literal match
                 * would be a brittle golden-output test. */
                let line = s
                    .lines()
                    .find(|l| l.trim_start().starts_with("obj1 position"))
                    .unwrap_or_else(|| panic!("no obj1 row in:\n{s}"));
                let inner = line.split('[').nth(1).unwrap().trim_end_matches(']');
                let dy: f64 = inner.split(',').nth(1).unwrap().trim().parse().unwrap();
                assert!((dy - 4.5).abs() < 1e-5, "d(y)/d(g) = {dy}, expected 4.5");
            }
            v => panic!("expected string, got {v}"),
        }

        // Bad parameter names name every accepted spelling.
        let e = execute_line("sensitivity 1 \"wobble\"", &mut st).unwrap_err();
        assert!(e.contains("expected g_constant"), "{e}");
        // and a duration with no parameter is refused at parse time
        assert!(execute_line("sensitivity 1", &mut st).unwrap_err().contains("quoted parameter"));
    }

    /// The three orientation joints, through the language — including
    /// the compatibility point that `ball` is still a legal object name.
    #[test]
    fn ball_hinge_and_universal_round_trip() {
        let mut st = SimState::default();
        for line in [
            "set system.g_constant = 0",
            "set system.uniform_gravity = [0, -9.81, 0]",
            "collide off",
            "new sphere as jamb { mass = 1, radius = 0.02, inverse_mass = 0 }",
            "new cuboid as door { mass = 1, half_extents = [0.2, 0.4, 0.2], \
             position = [0.02, -0.9998, 0] }",
        ] {
            execute_line(line, &mut st).unwrap();
        }
        match execute_line("hinge jamb door [0, 0, 1]", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("hinge obj0 <-> obj1"), "{s}");
                assert!(s.contains("5 row(s)"), "a hinge removes five freedoms: {s}");
                assert!(s.contains("METHOD IDA"), "{s}");
            }
            v => panic!("expected string, got {v}"),
        }
        assert_eq!(st.system.constraints.len(), 5);
        execute_line("method ida", &mut st).unwrap();
        execute_line("run 1 steps 10", &mut st).unwrap();
        let (g, _) = st.system.constraints.drift(&st.system);
        assert!(g < 1e-8, "the hinge should hold: |g| = {g:e}");
        // gravity about the jamb swings the door
        assert!(st.system.objects[1].get_angular_momentum().z.abs() > 1e-6);

        // one joint per pair
        assert!(execute_line("ball jamb door", &mut st).unwrap_err().contains("already joined"));
        execute_line("constrain off", &mut st).unwrap();
        match execute_line("ball jamb door", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("3 row(s)"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        execute_line("constrain off", &mut st).unwrap();
        match execute_line("universal jamb door [1, 0, 0] [0, 1, 0]", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("4 row(s)"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        // shafts that are not square name the angle
        execute_line("constrain off", &mut st).unwrap();
        let e = execute_line("universal jamb door [1, 0, 0] [1, 1, 0]", &mut st).unwrap_err();
        assert!(e.contains("perpendicular"), "{e}");

        /* BALL/HINGE/UNIVERSAL are CONTEXTUAL keywords — commands only.
         * `ball` has to stay usable as an object name, because it is
         * exactly what a physics user calls a sphere. */
        let mut st2 = SimState::default();
        execute_line("new sphere as ball { mass = 2, radius = 0.5 }", &mut st2).unwrap();
        assert_eq!(execute_line("get ball.radius", &mut st2).unwrap(), Value::Num(0.5));
        execute_line("new sphere as hinge { mass = 1, radius = 0.1 }", &mut st2).unwrap();
        assert_eq!(execute_line("get hinge.mass", &mut st2).unwrap(), Value::Num(1.0));
    }

    /// Every new command appears in the quick-reference card — the
    /// lexer/parser/VM/HELP lockstep rule.
    #[test]
    fn help_lists_the_dae_family() {
        let mut st = SimState::default();
        match execute_line("help", &mut st).unwrap() {
            Value::Str(h) => {
                assert!(h.contains("CONSTRAIN <a> <b> [len]"), "help lists CONSTRAIN");
                assert!(h.contains("CONSTRAINTS"), "help lists CONSTRAINTS");
                assert!(h.contains("EQUILIBRIUM"), "help lists EQUILIBRIUM");
                assert!(h.contains("SENSITIVITY"), "help lists SENSITIVITY");
                assert!(h.contains("| IDA"), "help lists METHOD IDA");
                assert!(h.contains("ANCHOR"), "help explains what an anchor is");
                assert!(h.contains("BALL <a> <b>"), "help lists BALL");
                assert!(h.contains("HINGE <a> <b> <axis>"), "help lists HINGE");
                assert!(h.contains("UNIVERSAL"), "help lists UNIVERSAL");
            }
            v => panic!("expected string, got {v}"),
        }
    }

    #[test]
    fn box_family_and_infinite_mass_walls() {
        let mut st = SimState::default();
        match execute_line("box", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("none"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        assert_eq!(execute_line("get system.box", &mut st).unwrap(), Value::Num(0.0));

        // BOX 4: six static walls obj0..obj5, inverse mass 0.
        match execute_line("box 4", &mut st).unwrap() {
            Value::Str(s) => {
                assert!(s.contains("inverse_mass = 0"), "{s}");
                assert!(s.contains("obj0") && s.contains("obj5"), "{s}");
            }
            v => panic!("expected string, got {v}"),
        }
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(6.0));
        assert_eq!(execute_line("get system.box", &mut st).unwrap(), Value::Num(4.0));
        assert_eq!(execute_line("get obj0.inverse_mass", &mut st).unwrap(), Value::Num(0.0));

        // A ball rattling inside: elastic bounces conserve energy and
        // the infinitely massive walls never move.
        execute_line("set system.g_constant = 0", &mut st).unwrap();
        execute_line("new sphere { mass = 1, radius = 0.5, velocity = [3, 1.7, 0.9] }", &mut st)
            .unwrap();
        let e0 = match execute_line("energy", &mut st).unwrap() {
            Value::Num(e) => e,
            v => panic!("expected number, got {v}"),
        };
        execute_line("run 2 steps 20", &mut st).unwrap();
        let e1 = match execute_line("energy", &mut st).unwrap() {
            Value::Num(e) => e,
            v => panic!("expected number, got {v}"),
        };
        assert!((e1 - e0).abs() < 1e-9 * e0.abs(), "elastic box: E {e0} -> {e1}");
        assert!(st.system.collision_count > 0, "bounces happened");
        assert_eq!(
            execute_line("get obj0.momentum", &mut st).unwrap(),
            Value::Vec3(Vec3::zeros()),
            "wall momentum stays zero"
        );

        // BOX OFF removes the walls; the ball renumbers to obj0.
        match execute_line("box off", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("removed"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get system.box", &mut st).unwrap(), Value::Num(0.0));
        assert_eq!(execute_line("get obj0.mass", &mut st).unwrap(), Value::Num(1.0));
    }

    #[test]
    fn torus_pair_is_order_independent_and_new_is_transactional() {
        // The inner/outer pair resolves at FinishNew: both orders work,
        // including pairs where sequential validation used to fail.
        for line in [
            "new torus { inner_radius = 1, outer_radius = 2 }",
            "new torus { outer_radius = 2, inner_radius = 1 }",
            "new torus { outer_radius = 0.5, inner_radius = 0.2 }",
        ] {
            let mut st = SimState::default();
            execute_line(line, &mut st).unwrap_or_else(|e| panic!("{line}: {e}"));
        }
        let mut st = SimState::default();
        execute_line("new torus { outer_radius = 0.5, inner_radius = 0.2 }", &mut st).unwrap();
        assert_eq!(execute_line("get obj0.ring_radius", &mut st).unwrap(), Value::Num(0.35));
        assert_eq!(execute_line("get obj0.tube_radius", &mut st).unwrap(), Value::Num(0.15));

        // A genuinely invalid pair fails once, against the FINAL values —
        // and the failed NEW leaves NO ghost object behind.
        let e = execute_line("new torus { inner_radius = 2, outer_radius = 1 }", &mut st)
            .unwrap_err();
        assert!(e.contains("inner < outer"), "{e}");
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(1.0));
        let e2 = execute_line("new sphere { radius = -1 }", &mut st).unwrap_err();
        assert!(e2.contains("finite number > 0"), "{e2}");
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(1.0));

        // Horn torus: inner_radius = 0 is a valid body, on NEW and SET.
        execute_line("new torus { inner_radius = 0, outer_radius = 1 }", &mut st).unwrap();
        assert_eq!(execute_line("get obj1.ring_radius", &mut st).unwrap(), Value::Num(0.5));
        execute_line("set obj1.inner_radius = 0", &mut st).unwrap();

        // SET radius on a torus is refused (it has two radii), instead
        // of silently turning it into a sphere.
        let e3 = execute_line("set obj1.radius = 1", &mut st).unwrap_err();
        assert!(e3.contains("ring_radius"), "{e3}");
        match execute_line("get obj1.shape", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("Torus"), "still a torus: {s}"),
            v => panic!("expected string, got {v}"),
        }
    }

    #[test]
    fn def_call_named_objects_and_dumbbell_members() {
        let mut st = SimState::default();
        // Multi-line DEF with defaults; every body line syntax-checked.
        let def_src = "def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, \
                       r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], \
                       velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) {\n  \
                       new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, \
                       r2 = r2, rod_radius = rod_radius, length = length, \
                       position = position, velocity = velocity, \
                       angular_velocity = angular_velocity }\n}";
        match execute_line(def_src, &mut st).unwrap() {
            Value::Str(m) => assert!(m.contains("11 parameter(s)"), "{m}"),
            v => panic!("expected string, got {v}"),
        }
        // Call with a string name + partial arguments (defaults fill in).
        assert_eq!(
            execute_line("create_dumbell(\"dumbell0\", 1, 2, 0.5)", &mut st).unwrap(),
            Value::Str("obj0 as dumbell0".to_string())
        );
        assert_eq!(execute_line("get dumbell0.m1", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get dumbell0.m2", &mut st).unwrap(), Value::Num(2.0));
        match execute_line("get dumbell0.m_rod", &mut st).unwrap() {
            Value::Num(m) => assert!((m - 0.5).abs() < 1e-12),
            v => panic!("expected number, got {v}"),
        }
        assert_eq!(execute_line("get dumbell0.mass", &mut st).unwrap(), Value::Num(3.5));

        // Component shorthands on named paths.
        execute_line("set dumbell0.vx = 1.5", &mut st).unwrap();
        assert_eq!(execute_line("get dumbell0.vx", &mut st).unwrap(), Value::Num(1.5));
        assert_eq!(execute_line("get dumbell0.x", &mut st).unwrap(), Value::Num(0.0));

        // Member writes rebuild the body: mass, offsets, inertia.
        execute_line("set dumbell0.m1 = 3", &mut st).unwrap();
        assert_eq!(execute_line("get dumbell0.mass", &mut st).unwrap(), Value::Num(5.5));
        assert_eq!(execute_line("get dumbell0.length", &mut st).unwrap(), Value::Num(1.0));

        // Names renumber on DEL: a second named object survives.
        execute_line("new sphere as ball { radius = 0.5, position = [3, 0, 0] }", &mut st)
            .unwrap();
        execute_line("del 0", &mut st).unwrap();
        assert_eq!(execute_line("get ball.radius", &mut st).unwrap(), Value::Num(0.5));
        let e = execute_line("get dumbell0.m1", &mut st).unwrap_err();
        assert!(e.contains("no object named"), "{e}");

        // Duplicate names are refused; reserved names are refused.
        let e2 = execute_line("new sphere as ball", &mut st).unwrap_err();
        assert!(e2.contains("already refers"), "{e2}");
        let e3 = execute_line("new sphere as obj7", &mut st).unwrap_err();
        assert!(e3.contains("reserved"), "{e3}");

        // Editing = SHOW + re-DEF (replacement is reported).
        match execute_line("show create_dumbell", &mut st).unwrap() {
            Value::Str(src) => assert!(src.starts_with("def create_dumbell"), "{src}"),
            v => panic!("expected string, got {v}"),
        }
        match execute_line(def_src, &mut st).unwrap() {
            Value::Str(m) => assert!(m.contains("(redefined)"), "{m}"),
            v => panic!("expected string, got {v}"),
        }

        // Argument arity errors are actionable.
        let e4 = execute_line("create_dumbell()", &mut st).unwrap_err();
        assert!(e4.contains("missing argument `name`"), "{e4}");

        // LET variables feed calls and defaults.
        execute_line("let heavy = 7", &mut st).unwrap();
        execute_line("def probe(m = heavy) { new sphere { mass = m } }", &mut st).unwrap();
        execute_line("probe()", &mut st).unwrap();
        let n = st.system.objects.len();
        assert_eq!(
            execute_line(&format!("get obj{}.mass", n - 1), &mut st).unwrap(),
            Value::Num(7.0)
        );

        // A failing body line rolls back and names the function.
        execute_line("def bad() { new sphere { radius = -1 } }", &mut st).unwrap();
        let before = st.system.objects.len();
        let e5 = execute_line("bad()", &mut st).unwrap_err();
        assert!(e5.contains("bad()"), "{e5}");
        assert_eq!(st.system.objects.len(), before, "no ghost from a failing function");
    }

    #[test]
    fn def_hardening_reserved_words_duplicates_defaults_and_nesting() {
        let mut st = SimState::default();

        // A function that CREATES an object may be called from inside a
        // NEW initializer: the outer NEW context is stashed/restored.
        execute_line("def spawn() { new point { mass = 3 } ; 7 }", &mut st).unwrap();
        execute_line("new sphere { mass = spawn() }", &mut st).unwrap();
        assert_eq!(st.system.objects.len(), 2, "outer sphere + inner point");
        // The outer NEW registers first (obj0), the point spawned
        // mid-initializer lands after it (obj1).
        assert_eq!(
            execute_line("get obj0.mass", &mut st).unwrap(),
            Value::Num(7.0),
            "the outer sphere got the function's return value"
        );
        assert_eq!(execute_line("get obj1.mass", &mut st).unwrap(), Value::Num(3.0));

        // A FAILING function inside a NEW rolls the outer object back.
        execute_line("def bad() { new sphere { radius = -1 } }", &mut st).unwrap();
        let before = st.system.objects.len();
        assert!(execute_line("new sphere { mass = bad() }", &mut st).is_err());
        assert_eq!(st.system.objects.len(), before, "no ghosts on either level");

        // Defaults are EXPRESSIONS only: a command is refused at
        // definition time (and the system is untouched).
        let e = execute_line("def evil(a = reset) { a }", &mut st).unwrap_err();
        assert!(e.contains("default for `a`"), "{e}");
        assert_eq!(st.system.objects.len(), before, "RESET did not run");

        // Reserved words / builtins / pi/tau cannot name parameters.
        for bad_def in [
            "def f(new) { new }",
            "def f(box) { box }",
            "def f(sqrt) { sqrt }",
            "def f(pi) { pi }",
        ] {
            let e = execute_line(bad_def, &mut st).unwrap_err();
            assert!(
                e.contains("reserved word") || e.contains("builtin"),
                "{bad_def}: {e}"
            );
        }
        // Duplicate parameters are refused (case-insensitively).
        let e = execute_line("def f(a, A = 1) { a }", &mut st).unwrap_err();
        assert!(e.contains("duplicate parameter"), "{e}");

        // LET cannot shadow the built-in constants.
        let e = execute_line("let pi = 3", &mut st).unwrap_err();
        assert!(e.contains("built-in constant"), "{e}");

        // A `}` inside a trailing # comment does not end the DEF body.
        execute_line(
            "def g() {\n  new sphere { mass = 2 } # note: } brace in comment\n}",
            &mut st,
        )
        .unwrap();
        execute_line("g()", &mut st).unwrap();
        let n = st.system.objects.len();
        assert_eq!(
            execute_line(&format!("get obj{}.mass", n - 1), &mut st).unwrap(),
            Value::Num(2.0)
        );

        // AS with a NON-string binding is an error, not a silent
        // literal name.
        execute_line("let n = 2", &mut st).unwrap();
        let e = execute_line("new sphere as n", &mut st).unwrap_err();
        assert!(e.contains("not a string name"), "{e}");
    }

    /// A failing NEW whose initializer already created objects through
    /// a nested call must renumber the name registry exactly as DEL
    /// does. Before the fix, the survivor's name kept its stale index:
    /// `get p.mass` here read whatever object happened to land on the
    /// old slot (the cuboid), not the registered point.
    #[test]
    fn failing_new_with_nested_creation_renumbers_names() {
        let mut st = SimState::default();
        execute_line("def spawn(n) { new point as n { mass = 3 }\n7 }", &mut st).unwrap();
        // The outer sphere is appended first, the named point second;
        // the sphere's failing validation rolls back index 0, so the
        // point shifts down and `p` must follow it.
        let e = execute_line("new sphere { mass = spawn(\"p\"), radius = -1 }", &mut st)
            .unwrap_err();
        assert!(e.contains("radius"), "{e}");
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get p.mass", &mut st).unwrap(), Value::Num(3.0));
        // A later object must NOT be captured by the stale name.
        execute_line("new cuboid { mass = 8 }", &mut st).unwrap();
        assert_eq!(execute_line("get p.mass", &mut st).unwrap(), Value::Num(3.0));
        assert_eq!(execute_line("get obj0.mass", &mut st).unwrap(), Value::Num(3.0));
        assert_eq!(execute_line("get obj1.mass", &mut st).unwrap(), Value::Num(8.0));
        // And the registry stays consistent under a further DEL.
        execute_line("del 0", &mut st).unwrap();
        assert!(execute_line("get p.mass", &mut st).is_err(), "p should be gone");
        assert_eq!(execute_line("get obj0.mass", &mut st).unwrap(), Value::Num(8.0));
    }

    /// A user function called from a NEW initializer can DEL an object
    /// BELOW the one under construction. The objects renumber, so the
    /// NEW context — including the stashed copy the Call instruction
    /// holds during the call — must follow: later initializer fields
    /// must write to the shifted index, and a failing NEW must roll
    /// back the half-built object, not a neighbour (and never leave a
    /// ghost).
    #[test]
    fn del_inside_a_new_initializer_keeps_the_rollback_on_target() {
        let mut st = SimState::default();
        execute_line("new sphere as anchor { mass = 1 }", &mut st).unwrap(); // obj0
        execute_line("new sphere as target { mass = 5 }", &mut st).unwrap(); // obj1
        execute_line("def evil() { del 0\n7 }", &mut st).unwrap();
        // Failing NEW: appended at index 2, shifted to 1 by the DEL.
        let e = execute_line("new sphere { mass = evil(), radius = -1 }", &mut st).unwrap_err();
        assert!(e.contains("radius"), "{e}");
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get target.mass", &mut st).unwrap(), Value::Num(5.0));
        assert_eq!(execute_line("get obj0.mass", &mut st).unwrap(), Value::Num(5.0));
        assert!(execute_line("get anchor.mass", &mut st).is_err(), "anchor was deleted");

        // Succeeding NEW: the field AFTER the call must reach the
        // half-built object at its shifted index, and the AS name must
        // register there too.
        execute_line("def evil2() { del 0\n7 }", &mut st).unwrap();
        execute_line("new sphere as s { mass = evil2(), radius = 0.25 }", &mut st).unwrap();
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(1.0));
        assert_eq!(execute_line("get s.mass", &mut st).unwrap(), Value::Num(7.0));
        assert_eq!(execute_line("get obj0.radius", &mut st).unwrap(), Value::Num(0.25));
    }

    /// The same staleness through the OTHER deletion path: BOX OFF
    /// removes six wall slabs below the object under construction.
    #[test]
    fn box_off_inside_a_new_initializer_keeps_the_rollback_on_target() {
        let mut st = SimState::default();
        execute_line("box 4", &mut st).unwrap(); // walls obj0..obj5
        execute_line("def evil() { box off\n7 }", &mut st).unwrap();
        // Half-built sphere appended at index 6, shifted to 0.
        let e = execute_line("new sphere { mass = evil(), radius = -1 }", &mut st).unwrap_err();
        assert!(e.contains("radius"), "{e}");
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(0.0));
    }

    #[test]
    fn box_recreate_after_wall_deletion_leaks_nothing() {
        let mut st = SimState::default();
        execute_line("box 4", &mut st).unwrap();
        execute_line("del 0", &mut st).unwrap(); // dissolves the box
        match execute_line("box", &mut st).unwrap() {
            Value::Str(s) => assert!(s.contains("dissolved"), "{s}"),
            v => panic!("expected string, got {v}"),
        }
        assert_eq!(execute_line("get system.box", &mut st).unwrap(), Value::Num(0.0));
        // Recreating removes the five surviving tracked slabs first.
        execute_line("box 10", &mut st).unwrap();
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(6.0));
        assert_eq!(execute_line("get system.box", &mut st).unwrap(), Value::Num(10.0));
        execute_line("box off", &mut st).unwrap();
        assert_eq!(execute_line("get system.count", &mut st).unwrap(), Value::Num(0.0));
    }
}
