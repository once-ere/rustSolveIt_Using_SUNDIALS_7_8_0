//! Parser ("yacc/bison" analog): a recursive-descent grammar compiler
//! that turns the token stream into a postfix instruction program for
//! the stack machine in [`crate::vm`].
//!
//! Grammar (EBNF):
//!
//! ```text
//! command  := "NEW" shape [ "AS" (IDENT | STRING) ]
//!                          [ "{" init { "," init } "}" ]
//!                                               (* AS registers a user
//!                                                  name for the object *)
//!           | "SET" path "=" expr
//!           | "GET" path
//!           | "DEL" NUMBER
//!           | "LIST"
//!           | "STEP" expr                       (* advance by dt      *)
//!           | "RUN" expr [ "STEPS" NUMBER ]     (* advance by t, n outs *)
//!           | "METHOD" ( "ADAMS" | "BDF" | "SPRK" IDENT [ NUMBER ]
//!                       | "IDA" )              (* IDA: constrained DAE *)
//!           | "ENERGY" | "COM" | "MOMENTUM" | "ANGMOM"
//!           | "LAPLACE" NUMBER
//!           | "RESET" | "HELP"
//!           | "SCENE" scenecmd                  (* graphical scene     *)
//!           | "COLLIDE" [ "ON" | "OFF" ]        (* bare: report status *)
//!           | "CONTACTS"                        (* list last contacts  *)
//!           | "CONSTRAIN" ( "OFF" | IDENT IDENT [ expr ] )
//!                                               (* rigid rod between two
//!                                                  objects; no length =
//!                                                  freeze the current
//!                                                  separation. Needs
//!                                                  METHOD IDA to run.   *)
//!           | "BALL" IDENT IDENT                (* shared point, 3 rows *)
//!           | "HINGE" IDENT IDENT expr          (* + shared axis, 5     *)
//!           | "UNIVERSAL" IDENT IDENT expr expr (* Cardan joint, 4      *)
//!           | "CONSTRAINTS"                     (* list joints + drift  *)
//!           | "EQUILIBRIUM"                     (* KINSOL: rest state   *)
//!           | "SENSITIVITY" expr STRING { STRING }
//!                                               (* CVODES, or IDAS when
//!                                                  constrained: run for
//!                                                  expr and report
//!                                                  d(state)/d(param).
//!                                                  Parameter names are
//!                                                  STRINGS because
//!                                                  `mass 0` is two
//!                                                  tokens.              *)
//!           | "LET" IDENT "=" expr              (* session variable    *)
//!           | "FUNCS"                           (* list user functions *)
//!           | "SHOW" IDENT                      (* print a function    *)
//!           | "BOX" [ "OFF" | expr ]            (* rigid bounding box:
//!                                                  expr = inner side
//!                                                  length; bare = status;
//!                                                  OFF removes it      *)
//!           | expr ;                            (* bare expression     *)
//! scenecmd := "CREATE" [ NUMBER ]               (* open window [port]  *)
//!           | "CLOSE"                           (* aka DESTROY         *)
//!           | "TRANSLATE" term term [ term ]    (* camera dx dy [dz]   *)
//!           | "ROTATE" term term                (* camera dyaw dpitch  *)
//!           | "ZOOM" ( "IN" | "OUT" | term )    (* factor > 1 zooms in *)
//!           | "HIDE" [ NUMBER | "ALL" ]         (* default: ALL        *)
//!           | "SHOW" [ NUMBER | "ALL" ]
//!           | "REFRESH"                         (* re-sync from state  *)
//!           | "REDRAW"                          (* re-send full scene  *)
//!           | "START" | "STOP" | "PAUSE" | "REVERSE"
//!           | "RESET"                           (* re-initialize: all
//!                                                  values and the time
//!                                                  return to initial;
//!                                                  START re-starts    *)
//!           | "SET_TIME_STEP" term              (* args are term-level:
//!                                                  -5 is negative five;
//!                                                  parenthesize sums  *)
//!           | "STATUS" | "EVENTS" ;
//! shape    := "POINT" | "SPHERE" | "CUBOID" | "TORUS" | "DISK" | "CYLINDER"
//!           | "DUMBBELL" ;                      (* two solid spheres +
//!                                                  a rigid rod, one
//!                                                  rigid body          *)
//!
//! (* User-defined functions are a LINE FORM handled before this
//!    grammar: DEF name(param [= default], ...) { body } — the body is
//!    newline/;-separated commands using the parameters as variables;
//!    each body line must itself satisfy this grammar. Invocation uses
//!    the ordinary call syntax name(arg, ...); trailing parameters
//!    take their defaults. *)
//! init     := IDENT "=" expr ;
//! path     := IDENT { "." IDENT } ;             (* objN.field[.x|y|z|w],
//!                                                  system.field,
//!                                                  contactK.field,
//!                                                  name.field for
//!                                                  AS-registered names *)
//! expr     := sum { ("<" | "<=" | ">" | ">=" | "==" | "!=") sum } ;
//!                                     (* comparisons yield 1 or 0;
//!                                        LOWEST precedence, so
//!                                        `x + 1 > 2` is `(x+1) > 2`.
//!                                        There is no boolean type, and
//!                                        that is deliberate: 1/0 makes
//!                                        `(x > a) * (x < b)` an
//!                                        indicator function, which is
//!                                        how a piecewise potential is
//!                                        written. `=` remains
//!                                        assignment; `==` is equality. *)
//! sum      := term { ("+" | "-") term } ;
//! term     := unary { ("*" | "/") unary } ;
//! qmcmd    := [ "STATUS" ]
//!           | "GRID" expr expr expr
//!           | "POTENTIAL" ( "ZERO"
//!                         | "BARRIER" expr expr expr
//!                         | "WELL" expr expr expr
//!                         | IDENT )              (* a DEF'd V(x)        *)
//!           | "MASS" expr | "HBAR" expr
//!           | "METHOD" ( "CAYLEY" | "NASH" [ "LIE" | "STRANG" ] )
//!           | "STATES" expr | "STATE" expr
//!           | "PACKET" expr expr expr
//!           | "STEP" expr | "RUN" expr [ "STEPS" expr ]
//!           | "TRANSMISSION" expr
//!           | "SCAN" expr expr expr
//!           | "NORM" | "ENERGY" | "POSITION" | "MOMENTUM"
//!           | "PROB" expr expr
//!           | "DRIVE" ( "OFF" | IDENT [ "," ] IDENT )
//!           | "ABSORB" ( "OFF" | expr expr [ expr ] )
//!           | "DENSITY" | "RESET"
//!           | "ANIMATE" STRING expr [ "FRAMES" expr ] ;
//!
//! (* The QM subcommand word is read as an IDENT-or-keyword rather than
//!    being lexed as a keyword of its own, because `run`, `step`,
//!    `state`, `energy`, `momentum` and `reset` are ALREADY keywords
//!    here. Matching on the lowercased text keeps the whole quantum
//!    vocabulary out of the global keyword namespace: `QM` is the only
//!    word this family reserves.
//!
//!    Argument lists accept an optional comma between arguments. That
//!    is not decoration: `QM POTENTIAL WELL 5 -2 2` parses `-2` as
//!    SUBTRACTION, yielding two arguments where three were wanted.
//!    `5, -2, 2` is unambiguous.
//!
//!    QM TRANSMISSION and QM SCAN are time-INDEPENDENT: they solve the
//!    fixed-energy scattering problem by transfer matrix rather than
//!    propagating anything, which is the only way to resolve a
//!    resonance narrower than a wavepacket's own momentum spread.
//!
//!    QM METHOD selects the propagator, and with it the BOUNDARY
//!    CONDITION: CAYLEY is Crank-Nicolson with Dirichlet walls that
//!    reflect, NASH is the Bessel-stencil split-operator scheme and is
//!    PERIODIC. The trailing word chooses the splitting — LIE is the
//!    default and is what the original C++ does; STRANG is second order
//!    in dt at essentially the same cost. *)
//!
//! qm2cmd   := [ "STATUS" ]
//!           | "GRID" expr expr expr expr expr expr
//!           | "POTENTIAL" ( "ZERO" | IDENT )    (* a DEF'd V(x, y)     *)
//!           | "PACKET" expr expr expr expr expr expr
//!           | "DRIVE" ( "OFF" | IDENT [ "," ] IDENT )
//!           | "STATES" expr | "STATE" expr
//!           | "STEP" expr | "RUN" expr [ "STEPS" expr ]
//!           | "NORM" | "ENERGY" | "CENTROID"
//!           | "PROB" expr expr expr expr
//!           | "ABSORB" ( "OFF" | expr expr [ expr ] )
//!           | "ANIMATE" STRING expr [ "FRAMES" expr ]
//!           | "RESET" ;
//!
//! (* QM2 has NO `ISO`. An earlier version of this comment listed one,
//!    which `qm2_command` never implemented — so reading the grammar
//!    was enough to believe in a command that errors when you type it.
//!    A 2-D density is already drawable flat, as a heat map, so the
//!    isosurface exists only where it buys something: QM3. The
//!    `every_qm2_subcommand_is_documented_in_lockstep` test now checks
//!    this production in BOTH directions, so a phantom cannot come
//!    back. *)
//!
//! qm3cmd   := [ "STATUS" ]
//!           | "GRID" expr{9} | "POTENTIAL" ( "ZERO" | IDENT )
//!           | "PACKET" expr{9}
//!           | "STATES" expr | "STATE" expr
//!           | "STEP" expr | "RUN" expr [ "STEPS" expr ]
//!           | "NORM" | "ENERGY" | "CENTROID" | "PROB" expr{6}
//!           | "DRIVE" ( "OFF" | IDENT [ "," ] IDENT )
//!           | "ABSORB" ( "OFF" | expr expr [ expr ] )
//!           | "ANIMATE" STRING expr [ "FRAMES" expr ]
//!           | "ISO" STRING expr [ "FRAMES" expr ] [ "LEVEL" expr ]
//!           | "RESET" ;
//!
//! (* QM2 is a SEPARATE family rather than a mode on QM: a 2-D problem
//!    differs in almost every argument list, and a hidden mode that
//!    silently reinterprets your commands is worse than a second word.
//!    QM3 follows for the same reason. *)
//!
//! unary    := "-" unary | atom ;
//! atom     := NUMBER | IMAGINARY | STRING
//!           | "[" expr { "," expr } "]" | "(" expr ")"
//!                                     (* IMAGINARY is a number with an
//!                                        `i` suffix: 3i. So `2 + 3i`
//!                                        is ordinary addition and
//!                                        needs no complex literal
//!                                        syntax of its own. The suffix
//!                                        only applies when `i` is not
//!                                        followed by another identifier
//!                                        character, so `2intercept`
//!                                        still lexes as it always did. *)
//!           | IDENT "(" [ expr { "," expr } ] ")"   (* builtin or user
//!                                                      function call   *)
//!
//! (* The call production above is uniform: it already admits every
//!    builtin, so adding a function is a REGISTRATION, not a grammar
//!    change. Two classes of name are reachable through it:
//!
//!      core      dot cross norm normalize sqrt abs sin cos exp log
//!      special   sph_j sph_y sph_j_prime sph_y_prime legendre_p
//!                legendre_p_prime assoc_legendre_p
//!                norm_assoc_legendre_p sph_harm sph_harm_real
//!                hermite_h hermite_he laguerre_l laguerre_l_assoc
//!                chebyshev_t chebyshev_u gegenbauer_c jacobi_p
//!                bessel_j bessel_j_array bessel_j_z bessel_i_z
//!                bessel_y_z bessel_k_z bessel_j_nu bessel_i_nu
//!                bessel_y_nu bessel_k_nu
//!                hankel_h1_z hankel_h2_z hankel_h1_nu hankel_h2_nu
//!                hankel_h1_prime_z hankel_h2_prime_z
//!                hankel_h1_prime_nu hankel_h2_prime_nu
//!                sph_hankel_h1 sph_hankel_h2
//!                sph_hankel_h1_prime sph_hankel_h2_prime
//!                bessel_j_scaled bessel_y_scaled bessel_i_scaled
//!                bessel_k_scaled hankel_h1_scaled hankel_h2_scaled
//!                gamma_z ln_gamma_z rgamma_z airy_z
//!                gauss_legendre eigenvalues
//!                jacobi_eigen solve_tridiag solve_tridiag_c
//!                solve_cyclic_tridiag_c wigner_3j wigner_6j
//!                clebsch_gordan wigner_9j rel_err
//!
//!    The genuine parse-time obligation the special functions add is
//!    ARGUMENT DOMAIN checking, not syntax: an integer order must be a
//!    whole number. `hermite_h(2.5, 1)` is rejected rather than
//!    truncated to `hermite_h(2, 1)`, which would return a confident
//!    wrong answer. See `crate::special`. *)
//!           | path
//!           | IDENT ;                           (* parameter / LET var *)
//! ```

use crate::lexer::{tokenize, Keyword, TokKind, Token};
use crate::vm::{CmpOp, Instr, MethodSpec, NameArg, Path, PathRoot, ShapeKind, Value};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

/// Compiles exactly ONE expression (never a command form) — the entry
/// point for DEF parameter defaults, so a default like `reset` or
/// `del 0` is a definition-time error instead of a command that runs.
pub fn compile_expression(src: &str) -> Result<Vec<Instr>, String> {
    let toks = tokenize(src)?;
    if toks.is_empty() {
        return Err("expected an expression".to_string());
    }
    let mut p = Parser { toks, pos: 0 };
    let mut prog = Vec::new();
    p.expr(&mut prog)?;
    if let Some(t) = p.peek() {
        return Err(format!(
            "parse error at column {}: unexpected {} after the expression",
            t.col, t.kind
        ));
    }
    Ok(prog)
}

/// Compiles one command line into a stack-machine program.
pub fn compile_line(line: &str) -> Result<Vec<Instr>, String> {
    let toks = tokenize(line)?;
    if toks.is_empty() {
        return Ok(Vec::new());
    }
    let mut p = Parser { toks, pos: 0 };
    let prog = p.command()?;
    if let Some(t) = p.peek() {
        return Err(format!("parse error at column {}: unexpected {}", t.col, t.kind));
    }
    Ok(prog)
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, want: &TokKind) -> Result<(), String> {
        match self.next() {
            Some(t) if t.kind == *want => Ok(()),
            Some(t) => Err(format!(
                "parse error at column {}: expected {}, found {}",
                t.col, want, t.kind
            )),
            None => Err(format!("parse error: expected {} at end of line", want)),
        }
    }

    fn eat_keyword(&mut self, kw: Keyword) -> bool {
        if matches!(self.peek(), Some(t) if t.kind == TokKind::Keyword(kw)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_number(&mut self, what: &str) -> Result<f64, String> {
        match self.next() {
            Some(Token { kind: TokKind::Number(n), .. }) => Ok(n),
            Some(t) => Err(format!(
                "parse error at column {}: expected {what}, found {}",
                t.col, t.kind
            )),
            None => Err(format!("parse error: expected {what} at end of line")),
        }
    }

    /// As [`Parser::expect_number`], but for whole-number arguments
    /// (object indices, step counts): a fractional value is rejected
    /// rather than truncated, which would act on a confidently wrong
    /// target — the same policy the special functions apply to integer
    /// orders and `SCENE CREATE` applies to its port.
    fn expect_index(&mut self, what: &str) -> Result<usize, String> {
        let n = self.expect_number(what)?;
        if n.fract() != 0.0 {
            return Err(format!("{what} must be a whole number, got {n}"));
        }
        Ok(n as usize)
    }

    fn expect_ident(&mut self, what: &str) -> Result<String, String> {
        match self.next() {
            Some(Token { kind: TokKind::Ident(s), .. }) => Ok(s),
            Some(t) => Err(format!(
                "parse error at column {}: expected {what}, found {}",
                t.col, t.kind
            )),
            None => Err(format!("parse error: expected {what} at end of line")),
        }
    }

    /// A field name may collide with a keyword (`momentum`, `energy`,
    /// `method`, ...): after a `.` or inside `NEW { ... }` both are
    /// accepted.
    fn expect_field(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Token { kind: TokKind::Ident(s), .. }) => Ok(s.to_ascii_lowercase()),
            Some(Token { kind: TokKind::Keyword(k), .. }) => Ok(format!("{k:?}").to_ascii_lowercase()),
            Some(t) => Err(format!(
                "parse error at column {}: expected a field name, found {}",
                t.col, t.kind
            )),
            None => Err("parse error: expected a field name at end of line".to_string()),
        }
    }

    fn command(&mut self) -> Result<Vec<Instr>, String> {
        let mut prog = Vec::new();
        let t = self.peek().cloned().expect("nonempty");
        match t.kind {
            TokKind::Keyword(Keyword::New) => {
                self.pos += 1;
                let shape = match self.next() {
                    Some(Token { kind: TokKind::Keyword(Keyword::Point), .. }) => ShapeKind::Point,
                    Some(Token { kind: TokKind::Keyword(Keyword::Sphere), .. }) => ShapeKind::Sphere,
                    Some(Token { kind: TokKind::Keyword(Keyword::Cuboid), .. }) => ShapeKind::Cuboid,
                    Some(Token { kind: TokKind::Keyword(Keyword::Torus), .. }) => ShapeKind::Torus,
                    Some(Token { kind: TokKind::Keyword(Keyword::Disk), .. }) => ShapeKind::Disk,
                    Some(Token { kind: TokKind::Keyword(Keyword::Cylinder), .. }) => {
                        ShapeKind::Cylinder
                    }
                    Some(Token { kind: TokKind::Keyword(Keyword::Dumbbell), .. }) => {
                        ShapeKind::Dumbbell
                    }
                    Some(t) => {
                        return Err(format!(
                            "parse error at column {}: expected POINT, SPHERE, CUBOID, TORUS, \
                             DISK, CYLINDER or DUMBBELL, found {}",
                            t.col, t.kind
                        ))
                    }
                    None => {
                        return Err("parse error: NEW needs a shape (POINT, SPHERE, CUBOID, \
                                    TORUS, DISK, CYLINDER, DUMBBELL)"
                            .into())
                    }
                };
                prog.push(Instr::NewObject(shape));
                /* optional `AS <name>`: a bare identifier (resolved
                 * against function parameters / LET variables holding a
                 * string, else taken literally) or a string literal */
                let name = if self.eat_keyword(Keyword::As) {
                    match self.next() {
                        Some(Token { kind: TokKind::Ident(n), .. }) => {
                            Some(NameArg::Ident(n.to_ascii_lowercase()))
                        }
                        Some(Token { kind: TokKind::Str(st), .. }) => Some(NameArg::Str(st)),
                        Some(t) => {
                            return Err(format!(
                                "parse error at column {}: AS needs a name (identifier or \
                                 string), found {}",
                                t.col, t.kind
                            ))
                        }
                        None => return Err("parse error: AS needs a name".into()),
                    }
                } else {
                    None
                };
                let mut explicit_inertia = false;
                if matches!(self.peek(), Some(t) if t.kind == TokKind::LBrace) {
                    self.pos += 1;
                    loop {
                        let field = self.expect_field()?;
                        if field == "inertia_tensor" || field == "inverse_inertia_tensor" {
                            explicit_inertia = true;
                        }
                        self.eat(&TokKind::Equals)?;
                        self.expr(&mut prog)?;
                        prog.push(Instr::InitField(field));
                        if matches!(self.peek(), Some(t) if t.kind == TokKind::Comma) {
                            self.pos += 1;
                            continue;
                        }
                        break;
                    }
                    self.eat(&TokKind::RBrace)?;
                }
                prog.push(Instr::FinishNew { recompute_inertia: !explicit_inertia, name });
            }
            TokKind::Keyword(Keyword::Set) => {
                self.pos += 1;
                let path = self.path()?;
                self.eat(&TokKind::Equals)?;
                self.expr(&mut prog)?;
                prog.push(Instr::Store(path));
            }
            TokKind::Keyword(Keyword::Get) => {
                self.pos += 1;
                let path = self.path()?;
                prog.push(Instr::Load(path));
            }
            TokKind::Keyword(Keyword::Del) => {
                self.pos += 1;
                let n = self.expect_index("an object index")?;
                prog.push(Instr::Delete(n));
            }
            TokKind::Keyword(Keyword::List) => {
                self.pos += 1;
                prog.push(Instr::ListObjects);
            }
            TokKind::Keyword(Keyword::Step) => {
                self.pos += 1;
                self.expr(&mut prog)?;
                prog.push(Instr::Step);
            }
            TokKind::Keyword(Keyword::Run) => {
                self.pos += 1;
                self.expr(&mut prog)?;
                let steps = if self.eat_keyword(Keyword::Steps) {
                    self.expect_index("a step count")?
                } else {
                    10
                };
                prog.push(Instr::Run { outputs: steps.max(1) });
            }
            TokKind::Keyword(Keyword::Method) => {
                self.pos += 1;
                let spec = match self.next() {
                    Some(Token { kind: TokKind::Keyword(Keyword::Adams), .. }) => MethodSpec::Adams,
                    Some(Token { kind: TokKind::Keyword(Keyword::Bdf), .. }) => MethodSpec::Bdf,
                    Some(Token { kind: TokKind::Keyword(Keyword::Ida), .. }) => MethodSpec::Ida,
                    Some(Token { kind: TokKind::Keyword(Keyword::Sprk), .. }) => {
                        let raw = self.expect_ident("an SPRK table name")?;
                        let upper = raw.to_ascii_uppercase();
                        let table = if upper.starts_with("ARKODE_") {
                            upper
                        } else {
                            format!("ARKODE_SPRK_{upper}")
                        };
                        let dt = match self.peek() {
                            Some(Token { kind: TokKind::Number(_), .. }) => {
                                self.expect_number("a fixed step dt")?
                            }
                            _ => 0.01,
                        };
                        MethodSpec::Sprk { table, dt }
                    }
                    Some(t) => {
                        return Err(format!(
                            "parse error at column {}: expected ADAMS, BDF, SPRK or IDA, \
                             found {}",
                            t.col, t.kind
                        ))
                    }
                    None => {
                        return Err("parse error: METHOD needs ADAMS, BDF, SPRK or IDA".into())
                    }
                };
                prog.push(Instr::SetMethod(spec));
            }
            TokKind::Keyword(Keyword::Energy) => {
                self.pos += 1;
                prog.push(Instr::Energy);
            }
            TokKind::Keyword(Keyword::Com) => {
                self.pos += 1;
                prog.push(Instr::CenterOfMass);
            }
            TokKind::Keyword(Keyword::Momentum) => {
                self.pos += 1;
                prog.push(Instr::TotalMomentum);
            }
            TokKind::Keyword(Keyword::Angmom) => {
                self.pos += 1;
                prog.push(Instr::TotalAngularMomentum);
            }
            TokKind::Keyword(Keyword::Laplace) => {
                self.pos += 1;
                let n = self.expect_index("an object index")?;
                prog.push(Instr::Laplace(n));
            }
            TokKind::Keyword(Keyword::Qm) => {
                self.pos += 1;
                prog.extend(self.qm_command()?);
            }
            TokKind::Keyword(Keyword::Qm2) => {
                self.pos += 1;
                prog.extend(self.qm2_command()?);
            }
            TokKind::Keyword(Keyword::Qm3) => {
                self.pos += 1;
                prog.extend(self.qm3_command()?);
            }
            TokKind::Keyword(Keyword::Reset) => {
                self.pos += 1;
                prog.push(Instr::Reset);
            }
            TokKind::Keyword(Keyword::Help) => {
                self.pos += 1;
                prog.push(Instr::Help);
            }
            TokKind::Keyword(Keyword::Scene) => {
                self.pos += 1;
                self.scene_command(&mut prog)?;
            }
            TokKind::Keyword(Keyword::Collide) => {
                self.pos += 1;
                let mode = match self.peek() {
                    Some(Token { kind: TokKind::Keyword(Keyword::On), .. }) => {
                        self.pos += 1;
                        Some(true)
                    }
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. }) => {
                        self.pos += 1;
                        Some(false)
                    }
                    _ => None,
                };
                prog.push(Instr::Collide(mode));
            }
            TokKind::Keyword(Keyword::Contacts) => {
                self.pos += 1;
                prog.push(Instr::Contacts);
            }
            TokKind::Keyword(Keyword::Constrain) => {
                self.pos += 1;
                if let Some(Token { kind: TokKind::Keyword(Keyword::Off), .. }) = self.peek() {
                    self.pos += 1;
                    prog.push(Instr::ConstrainOff);
                } else {
                    let a = self.expect_ident("the first object (objN or a registered name)")?;
                    let b = self.expect_ident("the second object (objN or a registered name)")?;
                    /* A bare CONSTRAIN freezes the separation the bodies
                     * already have, which is always consistent; an
                     * explicit length is a full expression. */
                    let has_len = self.peek().is_some();
                    if has_len {
                        self.expr(&mut prog)?;
                    }
                    prog.push(Instr::Constrain { a, b, has_len });
                }
            }
            TokKind::Keyword(Keyword::Ball) => {
                self.pos += 1;
                let a = self.expect_ident("the first object")?;
                let b = self.expect_ident("the second object")?;
                prog.push(Instr::Ball { a, b });
            }
            TokKind::Keyword(Keyword::Hinge) => {
                self.pos += 1;
                let a = self.expect_ident("the first object")?;
                let b = self.expect_ident("the second object")?;
                /* the hinge axis is a full expression, so `[0, 0, 1]`
                 * and `normalize([1, 1, 0])` both work */
                self.expr(&mut prog)?;
                prog.push(Instr::Hinge { a, b });
            }
            TokKind::Keyword(Keyword::Gear) => {
                self.pos += 1;
                let a = self.expect_ident("the first object")?;
                let b = self.expect_ident("the second object")?;
                /* axis first, then the ratio — both full expressions */
                self.expr(&mut prog)?;
                self.expr(&mut prog)?;
                prog.push(Instr::Gear { a, b });
            }
            TokKind::Keyword(Keyword::Prismatic) => {
                self.pos += 1;
                let a = self.expect_ident("the first object")?;
                let b = self.expect_ident("the sliding object")?;
                self.expr(&mut prog)?;
                prog.push(Instr::Prismatic { a, b });
            }
            TokKind::Keyword(Keyword::Rack) => {
                self.pos += 1;
                let a = self.expect_ident("the pinion")?;
                let b = self.expect_ident("the rack")?;
                /* axis, then direction, then pitch radius */
                self.expr(&mut prog)?;
                self.expr(&mut prog)?;
                self.expr(&mut prog)?;
                prog.push(Instr::Rack { a, b });
            }
            TokKind::Keyword(Keyword::Universal) => {
                self.pos += 1;
                let a = self.expect_ident("the first object")?;
                let b = self.expect_ident("the second object")?;
                self.expr(&mut prog)?;
                self.expr(&mut prog)?;
                prog.push(Instr::Universal { a, b });
            }
            TokKind::Keyword(Keyword::Constraints) => {
                self.pos += 1;
                prog.push(Instr::Constraints);
            }
            TokKind::Keyword(Keyword::Equilibrium) => {
                self.pos += 1;
                prog.push(Instr::Equilibrium);
            }
            TokKind::Keyword(Keyword::Sensitivity) => {
                self.pos += 1;
                /* SENSITIVITY <duration> "<param>" ["<param>" ...]
                 * Parameter names are STRINGS on purpose: `mass 0` is two
                 * tokens and would be indistinguishable from a duration
                 * followed by a number. */
                self.expr(&mut prog)?;
                let mut params = Vec::new();
                while let Some(Token { kind: TokKind::Str(_), .. }) = self.peek() {
                    if let Some(Token { kind: TokKind::Str(st), .. }) = self.next() {
                        params.push(st.clone());
                    }
                }
                if params.is_empty() {
                    return Err(
                        "SENSITIVITY needs a duration then one or more quoted parameter \
                         names, e.g. `sensitivity 3 \"gravity.y\" \"mass 0\"`"
                            .into(),
                    );
                }
                prog.push(Instr::Sensitivity(params));
            }
            TokKind::Keyword(Keyword::Let) => {
                self.pos += 1;
                let name = self.expect_ident("a variable name")?.to_ascii_lowercase();
                if name == "pi" || name == "tau" {
                    return Err(format!(
                        "`{name}` is a built-in constant and cannot be a LET variable"
                    ));
                }
                self.eat(&TokKind::Equals)?;
                self.expr(&mut prog)?;
                prog.push(Instr::StoreGlobal(name));
            }
            TokKind::Keyword(Keyword::Funcs) => {
                self.pos += 1;
                prog.push(Instr::ListFns);
            }
            TokKind::Keyword(Keyword::Show) => {
                self.pos += 1;
                let name = self.expect_ident("a function name")?.to_ascii_lowercase();
                prog.push(Instr::ShowFn(name));
            }
            TokKind::Keyword(Keyword::Box) => {
                self.pos += 1;
                use crate::vm::BoxMode;
                match self.peek() {
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. }) => {
                        self.pos += 1;
                        prog.push(Instr::Box(BoxMode::Off));
                    }
                    None => prog.push(Instr::Box(BoxMode::Status)),
                    _ => {
                        self.expr(&mut prog)?;
                        prog.push(Instr::Box(BoxMode::Create));
                    }
                }
            }
            _ => {
                self.expr(&mut prog)?;
            }
        }
        Ok(prog)
    }

    /// `scenecmd` — the sub-command after the `SCENE` keyword.
    fn scene_command(&mut self, prog: &mut Vec<Instr>) -> Result<(), String> {
        use crate::vm::SceneCmd;
        let t = match self.next() {
            Some(t) => t,
            None => {
                return Err(
                    "parse error: SCENE needs a sub-command (CREATE, CLOSE, TRANSLATE, ROTATE, \
                     ZOOM, HIDE, SHOW, REFRESH, REDRAW, START, STOP, PAUSE, REVERSE, RESET, \
                     SET_TIME_STEP, STATUS, EVENTS)"
                        .into(),
                )
            }
        };
        let cmd = match t.kind {
            TokKind::Keyword(Keyword::Create) => {
                let port = match self.peek() {
                    Some(Token { kind: TokKind::Number(_), .. }) => {
                        let n = self.expect_number("a TCP port")?;
                        if !(0.0..=65_535.0).contains(&n) || n.fract() != 0.0 {
                            return Err("SCENE CREATE port must be an integer in 0..=65535".into());
                        }
                        n as u16
                    }
                    _ => 0,
                };
                SceneCmd::Create { port }
            }
            TokKind::Keyword(Keyword::Close) => SceneCmd::Close,
            TokKind::Keyword(Keyword::Translate) => {
                self.term(prog)?;
                self.term(prog)?;
                if self.peek().is_some() {
                    self.term(prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(0.0)));
                }
                SceneCmd::Translate
            }
            TokKind::Keyword(Keyword::Rotate) => {
                self.term(prog)?;
                self.term(prog)?;
                SceneCmd::Rotate
            }
            TokKind::Keyword(Keyword::Zoom) => match self.peek().map(|t| t.kind.clone()) {
                Some(TokKind::Keyword(Keyword::In)) => {
                    self.pos += 1;
                    SceneCmd::ZoomIn
                }
                Some(TokKind::Keyword(Keyword::Out)) => {
                    self.pos += 1;
                    SceneCmd::ZoomOut
                }
                _ => {
                    self.term(prog)?;
                    SceneCmd::Zoom
                }
            },
            TokKind::Keyword(Keyword::Hide) => SceneCmd::Hide(self.scene_which()?),
            TokKind::Keyword(Keyword::Show) => SceneCmd::Show(self.scene_which()?),
            TokKind::Keyword(Keyword::Refresh) => SceneCmd::Refresh,
            TokKind::Keyword(Keyword::Redraw) => SceneCmd::Redraw,
            TokKind::Keyword(Keyword::Start) => SceneCmd::Start,
            TokKind::Keyword(Keyword::Stop) => SceneCmd::Stop,
            TokKind::Keyword(Keyword::Pause) => SceneCmd::Pause,
            TokKind::Keyword(Keyword::Reverse) => SceneCmd::Reverse,
            TokKind::Keyword(Keyword::Reset) => SceneCmd::ResetPlayback,
            TokKind::Keyword(Keyword::SetTimeStep) => {
                self.term(prog)?;
                SceneCmd::SetTimeStep
            }
            TokKind::Keyword(Keyword::Status) => SceneCmd::Status,
            TokKind::Keyword(Keyword::Events) => SceneCmd::Events,
            other => {
                return Err(format!(
                    "parse error at column {}: unknown SCENE sub-command {other} \
                     (expected CREATE, CLOSE, TRANSLATE, ROTATE, ZOOM, HIDE, SHOW, REFRESH, \
                     REDRAW, START, STOP, PAUSE, REVERSE, RESET, SET_TIME_STEP, STATUS or EVENTS)",
                    t.col
                ))
            }
        };
        prog.push(Instr::Scene(cmd));
        Ok(())
    }

    /// `[ NUMBER | "ALL" ]` after HIDE/SHOW — `None` means every object.
    fn scene_which(&mut self) -> Result<Option<usize>, String> {
        match self.peek().map(|t| t.kind.clone()) {
            Some(TokKind::Keyword(Keyword::All)) | None => {
                if self.peek().is_some() {
                    self.pos += 1;
                }
                Ok(None)
            }
            Some(TokKind::Number(_)) => {
                let n = self.expect_index("an object index")?;
                Ok(Some(n))
            }
            Some(other) => Err(format!(
                "parse error: HIDE/SHOW takes an object index or ALL, found {other}"
            )),
        }
    }

    /// `path := IDENT { "." IDENT }` — root `objN` or `system`.
    fn path(&mut self) -> Result<Path, String> {
        let root_name = self.expect_ident("a path root (`objN` or `system`)")?;
        let root = parse_root(&root_name)?;
        self.eat(&TokKind::Dot)?;
        let field = self.expect_field()?;
        let comp = self.component_suffix()?;
        Ok(Path { root, field, comp })
    }

    /// Optional `.x|.y|.z|.w` component suffix on a dotted path
    /// (`Ok(None)` when the next token is not a dot). One home for the
    /// suffix grammar and its error text — `path()` and `atom()` both
    /// parse it.
    fn component_suffix(&mut self) -> Result<Option<usize>, String> {
        if !matches!(self.peek(), Some(t) if t.kind == TokKind::Dot) {
            return Ok(None);
        }
        self.pos += 1;
        let c = self.expect_ident("a component (x, y, z or w)")?;
        Ok(Some(match c.to_ascii_lowercase().as_str() {
            "x" => 0usize,
            "y" => 1,
            "z" => 2,
            "w" => 3,
            other => return Err(format!("unknown component `.{other}` (use x, y, z or w)")),
        }))
    }

    /// `QM3 <word> [args]` — the three-dimensional family. Same
    /// conventions as [`Self::qm2_command`]; argument groups come in
    /// threes, so commas between axes are worth using.
    /// Compiles `n` expression arguments for a QM-family subcommand
    /// (shared by `qm_command`, `qm2_command` and `qm3_command`).
    /// Arguments may be separated by spaces or by commas. Commas are
    /// not decoration: `qm potential well 5 -2 2` parses the `-2` as
    /// SUBTRACTION, giving `(5-2)` and `2` — two arguments where three
    /// were wanted. Writing `5, -2, 2` is unambiguous. Space separation
    /// is kept because it reads better when every argument is positive,
    /// which is the common case.
    fn qm_args(&mut self, n: usize, prog: &mut Vec<Instr>) -> Result<(), String> {
        for i in 0..n {
            if i > 0 {
                if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                    self.pos += 1;
                }
            }
            self.expr(prog)?;
        }
        Ok(())
    }

    fn qm3_command(&mut self) -> Result<Vec<Instr>, String> {
        use crate::qm3::Qm3Cmd;
        let mut prog = Vec::new();
        if self.peek().is_none() {
            return Ok(vec![Instr::Qm3(Qm3Cmd::Status)]);
        }
        let word = self.expect_field()?;
        let cmd = match word.as_str() {
            "status" => Qm3Cmd::Status,
            "grid" => {
                self.qm_args(9, &mut prog)?;
                Qm3Cmd::Grid
            }
            "potential" => Qm3Cmd::Potential(self.expect_field()?),
            "packet" => {
                self.qm_args(9, &mut prog)?;
                Qm3Cmd::Packet
            }
            "states" => {
                self.qm_args(1, &mut prog)?;
                Qm3Cmd::States
            }
            "state" => {
                self.qm_args(1, &mut prog)?;
                Qm3Cmd::LoadState
            }
            "step" => {
                self.qm_args(1, &mut prog)?;
                Qm3Cmd::Step
            }
            "run" => {
                self.expr(&mut prog)?;
                let has_steps = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Steps), .. })
                );
                if has_steps {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(1.0)));
                }
                Qm3Cmd::Run
            }
            "norm" => Qm3Cmd::Norm,
            "energy" => Qm3Cmd::Energy,
            "centroid" | "position" => Qm3Cmd::Centroid,
            "prob" | "probability" => {
                self.qm_args(6, &mut prog)?;
                Qm3Cmd::Prob
            }
            "absorb" => {
                let off = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. })
                );
                if off {
                    self.pos += 1;
                    Qm3Cmd::AbsorbOff
                } else {
                    self.qm_args(2, &mut prog)?;
                    if self.peek().is_some() {
                        if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                            self.pos += 1;
                        }
                        self.expr(&mut prog)?;
                    } else {
                        prog.push(Instr::Push(Value::Num(2.0)));
                    }
                    Qm3Cmd::Absorb
                }
            }
            "drive" => {
                let off = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. })
                );
                if off {
                    self.pos += 1;
                    Qm3Cmd::DriveOff
                } else {
                    let shape = self.expect_field()?;
                    if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                        self.pos += 1;
                    }
                    Qm3Cmd::Drive(shape, self.expect_field()?)
                }
            }
            "animate" => {
                let path = match self.next() {
                    Some(Token { kind: TokKind::Str(p), .. }) => p,
                    Some(t) => {
                        return Err(format!(
                            "parse error at column {}: QM3 ANIMATE needs a quoted file path, \
                             found {}",
                            t.col, t.kind
                        ))
                    }
                    None => return Err("QM3 ANIMATE: expected a quoted file path".to_string()),
                };
                self.expr(&mut prog)?;
                let has_frames = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Ident(w), .. }) if w.eq_ignore_ascii_case("frames")
                );
                if has_frames {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(60.0)));
                }
                Qm3Cmd::Animate(path)
            }
            "iso" | "isosurface" => {
                let path = match self.next() {
                    Some(Token { kind: TokKind::Str(p), .. }) => p,
                    Some(t) => {
                        return Err(format!(
                            "parse error at column {}: QM3 ISO needs a quoted file path, found {}",
                            t.col, t.kind
                        ))
                    }
                    None => return Err("QM3 ISO: expected a quoted file path".to_string()),
                };
                self.expr(&mut prog)?;
                let word = |me: &Self, w: &str| {
                    matches!(me.peek(), Some(Token { kind: TokKind::Ident(x), .. })
                        if x.eq_ignore_ascii_case(w))
                };
                if word(self, "frames") {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(20.0)));
                }
                if word(self, "level") {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(0.25)));
                }
                Qm3Cmd::Iso(path)
            }
            "reset" => Qm3Cmd::Reset,
            other => {
                // Generated from the one authoritative list rather than
                // hand-maintained beside it: a duplicated list is a
                // list that drifts.
                return Err(format!(
                    "QM3: unknown subcommand `{other}` ({})",
                    crate::qm3::QM3_SUBCOMMANDS.join(", ")
                ))
            }
        };
        prog.push(Instr::Qm3(cmd));
        Ok(prog)
    }

    /// `QM2 <word> [args]` — the two-dimensional family.
    ///
    /// Same conventions as [`Self::qm_command`]: the subcommand word is
    /// an ident-or-keyword, and argument lists take optional commas.
    /// Commas are worth using here even for positive values, because
    /// the argument groups are naturally pairs and triples —
    /// `qm2 grid -8 8 80, -8 8 80` reads as two axes rather than six
    /// loose numbers.
    fn qm2_command(&mut self) -> Result<Vec<Instr>, String> {
        use crate::qm2::Qm2Cmd;
        let mut prog = Vec::new();
        if self.peek().is_none() {
            return Ok(vec![Instr::Qm2(Qm2Cmd::Status)]);
        }
        let word = self.expect_field()?;
        let cmd = match word.as_str() {
            "status" => Qm2Cmd::Status,
            "grid" => {
                self.qm_args(6, &mut prog)?;
                Qm2Cmd::Grid
            }
            "potential" => Qm2Cmd::Potential(self.expect_field()?),
            "packet" => {
                self.qm_args(6, &mut prog)?;
                Qm2Cmd::Packet
            }
            "step" => {
                self.qm_args(1, &mut prog)?;
                Qm2Cmd::Step
            }
            "run" => {
                self.expr(&mut prog)?;
                let has_steps = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Steps), .. })
                );
                if has_steps {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(1.0)));
                }
                Qm2Cmd::Run
            }
            "norm" => Qm2Cmd::Norm,
            "energy" => Qm2Cmd::Energy,
            "centroid" | "position" => Qm2Cmd::Centroid,
            "prob" | "probability" => {
                self.qm_args(4, &mut prog)?;
                Qm2Cmd::Prob
            }
            "absorb" => {
                let off = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. })
                );
                if off {
                    self.pos += 1;
                    Qm2Cmd::AbsorbOff
                } else {
                    self.qm_args(2, &mut prog)?;
                    if self.peek().is_some() {
                        if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                            self.pos += 1;
                        }
                        self.expr(&mut prog)?;
                    } else {
                        prog.push(Instr::Push(Value::Num(2.0)));
                    }
                    Qm2Cmd::Absorb
                }
            }
            "drive" => {
                let off = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. })
                );
                if off {
                    self.pos += 1;
                    Qm2Cmd::DriveOff
                } else {
                    let shape = self.expect_field()?;
                    if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                        self.pos += 1;
                    }
                    Qm2Cmd::Drive(shape, self.expect_field()?)
                }
            }
            "states" => {
                self.qm_args(1, &mut prog)?;
                Qm2Cmd::States
            }
            "state" => {
                self.qm_args(1, &mut prog)?;
                Qm2Cmd::LoadState
            }
            "reset" => Qm2Cmd::Reset,
            "animate" => {
                let path = match self.next() {
                    Some(Token { kind: TokKind::Str(p), .. }) => p,
                    Some(t) => {
                        return Err(format!(
                            "parse error at column {}: QM2 ANIMATE needs a quoted file path, \
                             found {}",
                            t.col, t.kind
                        ))
                    }
                    None => return Err("QM2 ANIMATE: expected a quoted file path".to_string()),
                };
                self.expr(&mut prog)?;
                let has_frames = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Ident(w), .. }) if w.eq_ignore_ascii_case("frames")
                );
                if has_frames {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(80.0)));
                }
                Qm2Cmd::Animate(path)
            }
            other => {
                // Generated from the one authoritative list rather than
                // hand-maintained beside it: a duplicated list is a
                // list that drifts.
                return Err(format!(
                    "QM2: unknown subcommand `{other}` ({})",
                    crate::qm2::QM2_SUBCOMMANDS.join(", ")
                ))
            }
        };
        prog.push(Instr::Qm2(cmd));
        Ok(prog)
    }

    /// `QM <word> [args]`.
    ///
    /// The subcommand word is read with [`Self::expect_field`], which
    /// accepts an identifier OR a keyword — necessary because `run`,
    /// `step`, `energy`, `momentum`, `state` and `reset` are already
    /// keywords in this language. Matching on the lowercased text keeps
    /// the QM vocabulary out of the global keyword namespace entirely,
    /// so `QM` is the only word this family reserves.
    ///
    /// Numeric arguments are ordinary expressions, compiled before the
    /// instruction so it can pop them.
    fn qm_command(&mut self) -> Result<Vec<Instr>, String> {
        use crate::qm::QmCmd;
        let mut prog = Vec::new();
        // bare `QM` reports status
        if self.peek().is_none() {
            return Ok(vec![Instr::Qm(QmCmd::Status)]);
        }
        let word = self.expect_field()?;
        let cmd = match word.as_str() {
            "status" => QmCmd::Status,
            "grid" => {
                self.qm_args(3, &mut prog)?;
                QmCmd::Grid
            }
            "potential" => {
                use crate::qm::PotentialSpec;
                let name = self.expect_field()?;
                // `barrier` and `well` are built-in shapes, but they are
                // also perfectly reasonable names for a user's own
                // potential — and now that comparison operators exist,
                // writing one as a DEF is the natural thing to do. So
                // the two are told apart by whether ARGUMENTS FOLLOW:
                //
                //   qm potential barrier            -> the DEF'd barrier(x)
                //   qm potential barrier 2.5 0 1    -> the built-in shape
                //
                // A bare name is always a user function, which is the
                // reading that respects what the user actually wrote.
                let bare = self.peek().is_none();
                match name.as_str() {
                    "zero" | "free" => QmCmd::Potential(PotentialSpec::Zero),
                    "barrier" if !bare => {
                        self.qm_args(3, &mut prog)?;
                        QmCmd::Potential(PotentialSpec::Barrier)
                    }
                    "well" if !bare => {
                        self.qm_args(3, &mut prog)?;
                        QmCmd::Potential(PotentialSpec::Well)
                    }
                    _ => QmCmd::Potential(PotentialSpec::Named(name)),
                }
            }
            "mass" => {
                self.qm_args(1, &mut prog)?;
                QmCmd::Mass
            }
            "hbar" => {
                self.qm_args(1, &mut prog)?;
                QmCmd::Hbar
            }
            "states" => {
                self.qm_args(1, &mut prog)?;
                QmCmd::States
            }
            "state" => {
                self.qm_args(1, &mut prog)?;
                QmCmd::LoadState
            }
            "packet" => {
                self.qm_args(3, &mut prog)?;
                QmCmd::Packet
            }
            "step" => {
                self.qm_args(1, &mut prog)?;
                QmCmd::Step
            }
            "run" => {
                self.expr(&mut prog)?;
                // `STEPS <n>` is optional; default to 1 step
                let has_steps = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Steps), .. })
                );
                if has_steps {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(1.0)));
                }
                QmCmd::Run
            }
            "norm" => QmCmd::Norm,
            "energy" => QmCmd::Energy,
            "position" | "x" => QmCmd::Position,
            "momentum" | "p" => QmCmd::Momentum,
            "prob" | "probability" => {
                self.qm_args(2, &mut prog)?;
                QmCmd::Prob
            }
            "density" => QmCmd::Density,
            "animate" => {
                let path = match self.next() {
                    Some(Token { kind: TokKind::Str(p), .. }) => p,
                    Some(t) => {
                        return Err(format!(
                            "parse error at column {}: QM ANIMATE needs a quoted file path, \
                             found {}",
                            t.col, t.kind
                        ))
                    }
                    None => {
                        return Err(
                            "QM ANIMATE: expected a quoted file path, e.g. \"scatter.html\""
                                .to_string(),
                        )
                    }
                };
                self.expr(&mut prog)?;
                let has_frames = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Ident(w), .. }) if w.eq_ignore_ascii_case("frames")
                );
                if has_frames {
                    self.pos += 1;
                    self.expr(&mut prog)?;
                } else {
                    prog.push(Instr::Push(Value::Num(120.0)));
                }
                QmCmd::Animate(path)
            }
            "transmission" => {
                self.qm_args(1, &mut prog)?;
                QmCmd::Transmission
            }
            "scan" => {
                self.qm_args(3, &mut prog)?;
                QmCmd::Scan
            }
            "method" => {
                use crate::qm::{EvolveMethod, Splitting};
                let which = self.expect_field()?;
                match which.as_str() {
                    "cayley" => QmCmd::Method(EvolveMethod::Cayley),
                    "nash" => {
                        // An optional trailing LIE or STRANG. LIE is the
                        // default because this is a port and the default
                        // has to be what the original does.
                        let word = match self.peek() {
                            Some(Token { kind: TokKind::Ident(w), .. }) => {
                                Some(w.to_ascii_lowercase())
                            }
                            _ => None,
                        };
                        let sp = match word.as_deref() {
                            Some("strang") => {
                                self.pos += 1;
                                Splitting::Strang
                            }
                            Some("lie") => {
                                self.pos += 1;
                                Splitting::Lie
                            }
                            _ => Splitting::Lie,
                        };
                        QmCmd::Method(EvolveMethod::Nash(sp))
                    }
                    other => {
                        return Err(format!(
                            "QM METHOD: unknown method `{other}` — use `cayley` \
                             (Crank-Nicolson, Dirichlet walls) or `nash` \
                             (Bessel stencil, periodic), optionally `nash strang`"
                        ))
                    }
                }
            }
            "drive" => {
                let off = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. })
                );
                if off {
                    self.pos += 1;
                    QmCmd::DriveOff
                } else {
                    let shape = self.expect_field()?;
                    if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                        self.pos += 1;
                    }
                    let modulation = self.expect_field()?;
                    QmCmd::Drive(shape, modulation)
                }
            }
            "absorb" => {
                // `QM ABSORB OFF` removes it; otherwise width, strength
                // and an optional ramp exponent (2 is the measured
                // optimum — see quantum/examples/absorber_tuning.rs).
                let off = matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Keyword(Keyword::Off), .. })
                ) || matches!(
                    self.peek(),
                    Some(Token { kind: TokKind::Ident(w), .. }) if w.eq_ignore_ascii_case("off")
                );
                if off {
                    self.pos += 1;
                    QmCmd::AbsorbOff
                } else {
                    self.qm_args(2, &mut prog)?;
                    if self.peek().is_some() {
                        if let Some(Token { kind: TokKind::Comma, .. }) = self.peek() {
                            self.pos += 1;
                        }
                        self.expr(&mut prog)?;
                    } else {
                        prog.push(Instr::Push(Value::Num(2.0)));
                    }
                    QmCmd::Absorb
                }
            }
            "reset" => QmCmd::Reset,
            other => {
                // Generated from the one authoritative list rather than
                // hand-maintained beside it: a duplicated list is a
                // list that drifts.
                return Err(format!(
                    "QM: unknown subcommand `{other}` ({})",
                    crate::qm::QM_SUBCOMMANDS.join(", ")
                ))
            }
        };
        prog.push(Instr::Qm(cmd));
        Ok(prog)
    }

    /// The lowest precedence level: comparisons.
    ///
    /// Below `+`/`-`, so `x + 1 > 2` groups as `(x + 1) > 2`, which is
    /// what anyone writing a piecewise potential expects. Left
    /// associative, so `a < b < c` means `(a < b) < c` — legal, and
    /// almost certainly not what you meant; the documentation says so.
    fn expr(&mut self, prog: &mut Vec<Instr>) -> Result<(), String> {
        self.sum(prog)?;
        loop {
            let op = match self.peek().map(|t| t.kind.clone()) {
                Some(TokKind::Lt) => CmpOp::Lt,
                Some(TokKind::Le) => CmpOp::Le,
                Some(TokKind::Gt) => CmpOp::Gt,
                Some(TokKind::Ge) => CmpOp::Ge,
                Some(TokKind::EqEq) => CmpOp::Eq,
                Some(TokKind::Ne) => CmpOp::Ne,
                _ => break,
            };
            self.pos += 1;
            self.sum(prog)?;
            prog.push(Instr::Cmp(op));
        }
        Ok(())
    }

    fn sum(&mut self, prog: &mut Vec<Instr>) -> Result<(), String> {
        self.term(prog)?;
        loop {
            match self.peek().map(|t| t.kind.clone()) {
                Some(TokKind::Plus) => {
                    self.pos += 1;
                    self.term(prog)?;
                    prog.push(Instr::Add);
                }
                Some(TokKind::Minus) => {
                    self.pos += 1;
                    self.term(prog)?;
                    prog.push(Instr::Sub);
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn term(&mut self, prog: &mut Vec<Instr>) -> Result<(), String> {
        self.unary(prog)?;
        loop {
            match self.peek().map(|t| t.kind.clone()) {
                Some(TokKind::Star) => {
                    self.pos += 1;
                    self.unary(prog)?;
                    prog.push(Instr::Mul);
                }
                Some(TokKind::Slash) => {
                    self.pos += 1;
                    self.unary(prog)?;
                    prog.push(Instr::Div);
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn unary(&mut self, prog: &mut Vec<Instr>) -> Result<(), String> {
        if matches!(self.peek(), Some(t) if t.kind == TokKind::Minus) {
            self.pos += 1;
            self.unary(prog)?;
            prog.push(Instr::Neg);
            return Ok(());
        }
        self.atom(prog)
    }

    fn atom(&mut self, prog: &mut Vec<Instr>) -> Result<(), String> {
        let t = match self.next() {
            Some(t) => t,
            None => return Err("parse error: expected an expression at end of line".into()),
        };
        match t.kind {
            TokKind::Number(n) => {
                prog.push(Instr::Push(Value::Num(n)));
            }
            TokKind::Imaginary(n) => {
                prog.push(Instr::Push(Value::Complex(
                    ::special_functions::complex::Complex64::new(0.0, n),
                )));
            }
            TokKind::Str(st) => {
                prog.push(Instr::Push(Value::Str(st)));
            }
            TokKind::LBracket => {
                let mut count = 0usize;
                if matches!(self.peek(), Some(t) if t.kind == TokKind::RBracket) {
                    return Err(format!("parse error at column {}: empty vector `[]`", t.col));
                }
                loop {
                    self.expr(prog)?;
                    count += 1;
                    match self.peek().map(|t| t.kind.clone()) {
                        Some(TokKind::Comma) => {
                            self.pos += 1;
                        }
                        _ => break,
                    }
                }
                self.eat(&TokKind::RBracket)?;
                prog.push(Instr::PackList(count));
            }
            TokKind::LParen => {
                self.expr(prog)?;
                self.eat(&TokKind::RParen)?;
            }
            TokKind::Ident(name) => {
                /* builtin call, constant, or path load */
                if matches!(self.peek(), Some(t) if t.kind == TokKind::LParen) {
                    self.pos += 1;
                    let mut argc = 0usize;
                    if !matches!(self.peek(), Some(t) if t.kind == TokKind::RParen) {
                        loop {
                            self.expr(prog)?;
                            argc += 1;
                            match self.peek().map(|t| t.kind.clone()) {
                                Some(TokKind::Comma) => {
                                    self.pos += 1;
                                }
                                _ => break,
                            }
                        }
                    }
                    self.eat(&TokKind::RParen)?;
                    prog.push(Instr::Call(name.to_ascii_lowercase(), argc));
                } else if matches!(self.peek(), Some(t) if t.kind == TokKind::Dot) {
                    /* a dotted path used inside an expression */
                    let root = parse_root(&name)?;
                    self.pos += 1;
                    let field = self.expect_field()?;
                    let comp = self.component_suffix()?;
                    prog.push(Instr::Load(Path { root, field, comp }));
                } else {
                    match name.to_ascii_lowercase().as_str() {
                        "pi" => prog.push(Instr::Push(Value::Num(std::f64::consts::PI))),
                        "tau" => prog.push(Instr::Push(Value::Num(std::f64::consts::TAU))),
                        /* a bare identifier: a function parameter or a
                         * LET variable, resolved at execution time */
                        lower => prog.push(Instr::LoadIdent(lower.to_string())),
                    }
                }
            }
            other => {
                return Err(format!(
                    "parse error at column {}: unexpected {other} in expression",
                    t.col
                ));
            }
        }
        Ok(())
    }
}

fn parse_root(name: &str) -> Result<PathRoot, String> {
    let lower = name.to_ascii_lowercase();
    if lower == "system" || lower == "sys" {
        return Ok(PathRoot::System);
    }
    if let Some(idx) = lower.strip_prefix("obj") {
        if let Ok(i) = idx.parse::<usize>() {
            return Ok(PathRoot::Object(i));
        }
    }
    if let Some(idx) = lower.strip_prefix("contact") {
        if let Ok(i) = idx.parse::<usize>() {
            return Ok(PathRoot::Contact(i));
        }
    }
    /* anything else is a USER NAME (an object registered with
     * `NEW ... AS name`, e.g. `dumbell0.m1`), resolved at execution */
    Ok(PathRoot::Named(lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_command_compiles_to_postfix() {
        let p = compile_line("set obj0.mass = 2 + 3 * 4").unwrap();
        assert_eq!(
            p,
            vec![
                Instr::Push(Value::Num(2.0)),
                Instr::Push(Value::Num(3.0)),
                Instr::Push(Value::Num(4.0)),
                Instr::Mul,
                Instr::Add,
                Instr::Store(Path {
                    root: PathRoot::Object(0),
                    field: "mass".to_string(),
                    comp: None
                }),
            ]
        );
    }

    #[test]
    fn new_with_inits() {
        let p = compile_line("new sphere { mass = 2, radius = 0.5 }").unwrap();
        assert_eq!(p[0], Instr::NewObject(ShapeKind::Sphere));
        assert_eq!(p[2], Instr::InitField("mass".to_string()));
        assert_eq!(p[4], Instr::InitField("radius".to_string()));
        assert_eq!(p[5], Instr::FinishNew { recompute_inertia: true, name: None });
    }

    #[test]
    fn vector_literal_and_component_path() {
        let p = compile_line("get obj2.position.y").unwrap();
        assert_eq!(
            p,
            vec![Instr::Load(Path {
                root: PathRoot::Object(2),
                field: "position".to_string(),
                comp: Some(1)
            })]
        );
        let p = compile_line("[1, 2, 3]").unwrap();
        assert_eq!(p[3], Instr::PackList(3));
    }

    #[test]
    fn run_and_method() {
        assert_eq!(
            compile_line("run 10 steps 100").unwrap().last().unwrap(),
            &Instr::Run { outputs: 100 }
        );
        assert_eq!(
            compile_line("method sprk leapfrog_2_2 0.001").unwrap()[0],
            Instr::SetMethod(MethodSpec::Sprk {
                table: "ARKODE_SPRK_LEAPFROG_2_2".to_string(),
                dt: 0.001
            })
        );
    }

    #[test]
    fn scene_commands_compile() {
        use crate::vm::SceneCmd;
        assert_eq!(
            compile_line("scene create").unwrap(),
            vec![Instr::Scene(SceneCmd::Create { port: 0 })]
        );
        assert_eq!(
            compile_line("SCENE CREATE 8080").unwrap(),
            vec![Instr::Scene(SceneCmd::Create { port: 8080 })]
        );
        /* translate with an omitted dz gets an implicit 0 */
        let p = compile_line("scene translate 1 2").unwrap();
        assert_eq!(
            p,
            vec![
                Instr::Push(Value::Num(1.0)),
                Instr::Push(Value::Num(2.0)),
                Instr::Push(Value::Num(0.0)),
                Instr::Scene(SceneCmd::Translate),
            ]
        );
        assert_eq!(
            compile_line("scene rotate 15 -5").unwrap().last().unwrap(),
            &Instr::Scene(SceneCmd::Rotate)
        );
        assert_eq!(compile_line("scene zoom in").unwrap(), vec![Instr::Scene(SceneCmd::ZoomIn)]);
        assert_eq!(compile_line("scene zoom out").unwrap(), vec![Instr::Scene(SceneCmd::ZoomOut)]);
        assert_eq!(
            compile_line("scene zoom 2.5").unwrap(),
            vec![Instr::Push(Value::Num(2.5)), Instr::Scene(SceneCmd::Zoom)]
        );
        assert_eq!(compile_line("scene hide all").unwrap(), vec![Instr::Scene(SceneCmd::Hide(None))]);
        assert_eq!(compile_line("scene hide").unwrap(), vec![Instr::Scene(SceneCmd::Hide(None))]);
        assert_eq!(
            compile_line("scene show 2").unwrap(),
            vec![Instr::Scene(SceneCmd::Show(Some(2)))]
        );
        assert_eq!(
            compile_line("scene set_time_step 0.01").unwrap().last().unwrap(),
            &Instr::Scene(SceneCmd::SetTimeStep)
        );
        for (line, cmd) in [
            ("scene start", SceneCmd::Start),
            ("scene stop", SceneCmd::Stop),
            ("scene pause", SceneCmd::Pause),
            ("scene reverse", SceneCmd::Reverse),
            ("scene reset", SceneCmd::ResetPlayback),
            ("scene refresh", SceneCmd::Refresh),
            ("scene redraw", SceneCmd::Redraw),
            ("scene status", SceneCmd::Status),
            ("scene events", SceneCmd::Events),
            ("scene close", SceneCmd::Close),
            ("scene destroy", SceneCmd::Close),
        ] {
            assert_eq!(compile_line(line).unwrap(), vec![Instr::Scene(cmd)], "{line}");
        }
        assert!(compile_line("scene").is_err());
        assert!(compile_line("scene create 99999").is_err());
        assert!(compile_line("scene rotate 15").is_err());
    }

    #[test]
    fn errors_have_positions() {
        let e = compile_line("set = 3").unwrap_err();
        assert!(e.contains("column 5"), "{e}");
        let e = compile_line("get obj0.position.q").unwrap_err();
        assert!(e.contains("component"), "{e}");
        /* a bare identifier now COMPILES (function parameters and LET
         * variables) and resolves at execution instead */
        let p = compile_line("bogusname").unwrap();
        assert_eq!(p, vec![Instr::LoadIdent("bogusname".to_string())]);
    }

    /// Whole-number arguments are rejected when fractional, never
    /// truncated: `DEL 1.9` acting on obj1 (and renumbering everything
    /// above it) is exactly the confident-wrong-answer the language's
    /// documented policy forbids. Same policy at every index site.
    #[test]
    fn fractional_indices_are_refused_not_truncated() {
        for line in ["del 1.9", "run 10 steps 2.7", "laplace 0.5", "scene hide 1.5"] {
            let e = compile_line(line).unwrap_err();
            assert!(e.contains("whole number"), "`{line}` gave: {e}");
        }
        // Whole values still compile to the same instructions as before.
        assert_eq!(compile_line("del 1").unwrap(), vec![Instr::Delete(1)]);
        assert_eq!(compile_line("laplace 0").unwrap(), vec![Instr::Laplace(0)]);
        assert!(compile_line("run 10 steps 3").is_ok());
        assert!(compile_line("scene show 2").is_ok());
    }
}
