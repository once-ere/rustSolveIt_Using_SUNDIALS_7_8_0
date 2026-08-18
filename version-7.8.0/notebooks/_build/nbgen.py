#!/usr/bin/env python3
"""Turn one posim example script into one notebook spec.

Everything the spec says about the physics is DERIVED from the script:
the object table comes from the `new` statements, the inertia formulas
from the shapes, the constraint equations and the degree-of-freedom
arithmetic from the joints, the state-vector size from the body count,
and the solver from the `method` line. The example's own header comment
is carried through as the author's note, because that is where the
insight peculiar to that example lives.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lang import SHAPES, FIELDS, SYSTEM_FIELDS, JOINTS, METHODS

# ======================================================================
# 1. Parsing
# ======================================================================

def split_comment(line):
    """Return (code, comment). A '#' inside quotes is not a comment."""
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            if ch == q: q = None
            out.append(ch)
        elif ch in "\"'":
            q = ch; out.append(ch)
        elif ch == "#":
            return "".join(out), line[i+1:].strip()
        else:
            out.append(ch)
    return "".join(out), None

def parse(text):
    """[{comments: [str], stmt: str}] plus the leading header comment."""
    units, pending, header = [], [], []
    buf, depth = "", 0
    seen_stmt = False
    for raw in text.splitlines():
        code, comment = split_comment(raw)
        if comment is not None:
            (pending if seen_stmt else header).append(comment)
        if not code.strip():
            continue
        seen_stmt = True
        buf = (buf + " " + code.strip()).strip() if buf else code.strip()
        depth += sum(code.count(c) for c in "([{") - sum(code.count(c) for c in ")]}")
        if depth <= 0:
            units.append({"comments": pending, "stmt": buf})
            pending, buf, depth = [], "", 0
    if buf:
        units.append({"comments": pending, "stmt": buf})
    return header, units

def split_top(s, sep=","):
    """Split on `sep` at bracket depth zero."""
    parts, cur, depth, q = [], "", 0, None
    for ch in s:
        if q:
            cur += ch
            if ch == q: q = None
            continue
        if ch in "\"'": q = ch; cur += ch; continue
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts

NEW_RE = re.compile(r"^new\s+(\w+)(?:\s+as\s+(\w+))?\s*\{(.*)\}\s*$", re.I | re.S)

def parse_new(stmt):
    m = NEW_RE.match(stmt.strip())
    if not m: return None
    shape, name, body = m.group(1).lower(), m.group(2), m.group(3)
    fields = {}
    for part in split_top(body):
        if "=" in part:
            k, v = part.split("=", 1)
            fields[k.strip().lower()] = v.strip()
    return {"shape": shape, "name": name, "fields": fields}

def kind(stmt):
    """Which phase of the script a statement belongs to."""
    head = stmt.strip().split()[0].lower() if stmt.strip() else ""
    if head in JOINTS: return "joints"
    if head == "new": return "bodies"
    if head in ("def", "call", "let", "del"): return "bodies"
    if head == "method": return "solver"
    if head in ("run", "step"): return "run"
    if head == "set":
        return "world" if re.match(r"set\s+system\.", stmt, re.I) else "bodies"
    if head in ("collide", "box"): return "world"
    if head == "scene": return "scene"
    if head in ("get", "energy", "momentum", "angmom", "com", "laplace",
                "constraints", "contacts", "list", "funcs", "reset",
                "equilibrium", "sensitivity"):
        return "measure"
    if head in ("qm", "qm2", "qm3", "special"): return "special"
    return "measure"

# ======================================================================
# 2. Explaining one statement
# ======================================================================

def fmt_fields(fields, shape):
    lines = []
    for k, v in fields.items():
        doc = FIELDS.get(k, "a property of this shape.")
        lines.append(f"- `{k} = {v}` — {doc}")
    return "\n".join(lines)

def explain(stmt, ctx):
    """Precisely what this one input line expects, and what it will do."""
    s = stmt.strip()
    head = s.split()[0].lower()

    # ---- new ----------------------------------------------------------
    if head == "new":
        spec = parse_new(s)
        if spec:
            sh = SHAPES.get(spec["shape"], {})
            nm = spec["name"] or "the next automatic name `objN`"
            named = (f"named `{spec['name']}`" if spec["name"]
                     else "left unnamed, so it takes the next automatic name `objN`")
            out = [
              f"`new {spec['shape']}` creates one rigid body whose shape is "
              f"**{sh.get('blurb', spec['shape'])}**, {named}. The general form is",
              "",
              "```",
              "new <shape> [as <name>] { <field> = <value>, ... }",
              "```",
              "",
              f"Its inertia tensor is not typed in — the simulator computes it "
              f"from the shape: {sh.get('inertia', '')} {sh.get('why','')}",
              "",
              "The fields given here are:", "",
              fmt_fields(spec["fields"], spec["shape"]),
            ]
            if spec["fields"].get("inverse_mass", "").strip() in ("0", "0.0"):
                out += ["", "Because `inverse_mass` is zero this body is **static**: "
                        "no force can move it. It is part of the world, not part of "
                        "the motion."]
            out += ["", "The command prints the name the simulator assigned, which is "
                    "how you confirm the body exists."]
            return "\n".join(out)
        return f"Creates a body. The general form is `new <shape> [as <name>] {{ field = value, ... }}`."

    # ---- set ----------------------------------------------------------
    if head == "set":
        m = re.match(r"set\s+([\w.]+)\s*=\s*(.+)$", s, re.I)
        if m:
            path, val = m.group(1), m.group(2).strip()
            if path.lower().startswith("system."):
                f = path.split(".", 1)[1].lower()
                return (f"`set {path} = {val}` sets a property of the whole "
                        f"simulation rather than of one body. `{f}` is "
                        f"{SYSTEM_FIELDS.get(f, 'a system-wide setting.')}\n\n"
                        f"The value expected here is `{val}`. A successful `set` "
                        f"prints nothing.")
            obj, f = (path.split(".", 1) + [""])[:2]
            return (f"`set {path} = {val}` changes one property of the body "
                    f"`{obj}` after creation. `{f}` is "
                    f"{FIELDS.get(f.lower(), 'a property of that body.')}\n\n"
                    f"Setters keep the coupled quantities consistent: setting a "
                    f"velocity updates the momentum, setting a mass updates the "
                    f"inverse mass, and setting an orientation renormalises the "
                    f"quaternion. A successful `set` prints nothing.")

    # ---- joints -------------------------------------------------------
    if head in JOINTS:
        j = JOINTS[head]
        args = s.split()[1:]
        return (f"`{head.upper()}` declares a joint. Its form is\n\n"
                f"```\n{j['form']}\n```\n\n"
                f"and it contributes **{j['rows']} scalar constraint "
                f"equation{'s' if j['rows'] != 1 else ''}** to the system, "
                f"leaving {j['frees']} of the 6 relative freedoms between the "
                f"two bodies.\n\nWhat it holds: {j['holds']}.\n\n{j['detail']}\n\n"
                f"Here it is applied to `{' '.join(args)}`. The pivot goes at the "
                f"**midpoint of the two bodies' centres as they stand right now**, "
                f"and is then remembered in each body's own frame — which is why "
                f"bodies are positioned before they are joined. The command prints "
                f"a confirmation naming the joint and its row count.")

    # ---- method -------------------------------------------------------
    if head == "method":
        which = s.split()[1].lower() if len(s.split()) > 1 else "adams"
        return (f"`method {which}` chooses the time integrator. The form is "
                f"`METHOD <adams|bdf|sprk|ida>`.\n\nHere it selects "
                f"{METHODS.get(which, 'an integrator.')}\n\nAll stepping in this "
                f"simulator goes through the vendored pure-Rust SUNDIALS "
                f"translation. There is no hand-written Euler or Verlet anywhere.")

    # ---- run / step ---------------------------------------------------
    if head in ("run", "step"):
        parts = s.split()
        arg = parts[1] if len(parts) > 1 else "?"
        extra = ""
        if "steps" in s.lower():
            n = s.lower().split("steps")[1].strip()
            extra = (f"\n\n`steps {n}` asks for the result to be reported at {n} "
                     f"equally spaced points across that interval. It does not "
                     f"set the integrator's internal step size — SUNDIALS chooses "
                     f"that itself from the error tolerances, and takes as many "
                     f"internal steps between reports as accuracy demands.")
        if head == "run":
            return (f"`run {arg}` advances the simulation by **{arg} seconds of "
                    f"duration** — not to absolute time {arg}. This trips people "
                    f"up: `run 1.7` followed by `run 2.89` leaves you at t = 4.59, "
                    f"not at 2.89. The form is `RUN <duration> [STEPS <n>]`.{extra}\n\n"
                    f"This is the cell that actually integrates, so it is the slow "
                    f"one. It prints the time reached and how many internal solver "
                    f"steps were needed.")
        return (f"`step {arg}` advances the simulation by {arg} seconds. Each "
                f"`STEP` is a **cold restart**: a fresh solver instance, a fresh "
                f"multiplier seed and no carried-over history, which makes every "
                f"step independent and exactly reproducible at some cost in "
                f"efficiency. It prints the time reached and the number of "
                f"internal solver steps taken.")

    # ---- collide ------------------------------------------------------
    if head == "collide":
        rest = s.split()[1:] 
        if rest and rest[0].lower() == "off":
            return ("`collide off` disarms contact detection. Nothing in this "
                    "model is meant to touch anything else, so leaving it on "
                    "would have the solver arm a rootfinder hunting for impacts "
                    "that never happen. The form is `COLLIDE [ON|OFF]`; with no "
                    "argument it reports the current setting.\n\nIt prints the "
                    "resulting state and the number of collidable pairs.")
        return ("`collide` arms contact detection. Impacts are then found as "
                "**roots** of the separation function by SUNDIALS' rootfinder, "
                "not by checking for overlap after the fact — so the time of "
                "impact is located to solver tolerance rather than to within a "
                "step, and fast bodies cannot tunnel through thin ones. The form "
                "is `COLLIDE [ON|OFF]`.\n\nIt prints the resulting state and how "
                "many pairs can collide.")

    # ---- box ----------------------------------------------------------
    if head == "box":
        rest = s.split()[1:]
        if rest and rest[0].lower() == "off":
            return ("`box off` removes the enclosing box and its six walls."
                    "\n\nThe form is `BOX <size> | OFF`; with no argument it "
                    "reports the current box.")
        size = rest[0] if rest else "?"
        return (f"`box {size}` builds a rigid cube of side {size} centred on the "
                f"origin, out of six static wall slabs with `inverse_mass = 0`. "
                f"The bodies inside can then bounce off the walls, and the walls "
                f"never move. The form is `BOX <size> | OFF`.\n\nThe walls are "
                f"real bodies in the model but are excluded from the drawing and "
                f"from the camera fit, so the box shows as a wireframe rather "
                f"than as six huge blocks.")

    # ---- measurement --------------------------------------------------
    if head == "energy":
        return ("`energy` prints the total kinetic plus potential energy of the "
                "whole system. Take it before and after a run: for a closed "
                "system with no dissipation the two should agree to near machine "
                "precision, and how well they agree is the honest measure of the "
                "integration. It takes no arguments.")
    if head == "momentum":
        return ("`momentum` prints the total linear momentum, summed over every "
                "body. With no external force it is exactly conserved, and it is "
                "conserved through collisions too, because every impulse is "
                "applied equal and opposite to the pair. It takes no arguments.")
    if head == "angmom":
        return ("`angmom` prints the total angular momentum about the origin, "
                "including both the spin of each body and the orbital "
                "contribution of its motion about the origin. With no external "
                "torque it is exactly conserved. It takes no arguments.")
    if head == "com":
        return ("`com` prints the centre of mass of the whole system. With no "
                "external force it moves in a straight line at constant speed, "
                "whatever the bodies do among themselves. It takes no arguments.")
    if head == "laplace":
        return ("`laplace` prints the Laplace-Runge-Lenz vector, the extra "
                "conserved quantity peculiar to the inverse-square law. It points "
                "along the orbit's major axis toward perihelion and its length is "
                "the eccentricity, so watching it is how you tell whether an "
                "ellipse is precessing. It takes no arguments.")
    if head == "constraints":
        return ("`constraints` lists every joint and prints the worst absolute "
                "violation of the constraint equations: `|g|` at the position "
                "level and `|g_dot|` at the velocity level.\n\n**These two numbers "
                "are how you tell whether a mechanism is really being held "
                "together.** They should sit near the solver's tolerance, and "
                "above all they should not GROW. A formulation that enforced only "
                "the acceleration level would let `|g|` creep up quadratically in "
                "time while nothing failed loudly. It takes no arguments.")
    if head == "contacts":
        return ("`contacts` lists every contact recorded so far, each with the "
                "exact time of impact found by the rootfinder, the contact "
                "normal, the point, and the impulse delivered. It takes no "
                "arguments.")
    if head == "list":
        return ("`list` prints every body with its name, shape, mass and current "
                "state, tagging static bodies and walls. It takes no arguments.")
    if head == "reset":
        return ("`reset` returns the simulation to t = 0 and to the state the "
                "bodies were created in. It takes no arguments.")
    if head == "get":
        expr = s[3:].strip()
        return (f"`get` evaluates an expression and prints it. The form is "
                f"`GET <expression>`, where an expression may name a body's "
                f"field (`obj0.position`, `bob.velocity.x`), index into it, or "
                f"combine several with arithmetic and the built-in functions "
                f"`norm`, `normalize`, `dot`, `cross`, `sqrt` and the rest.\n\n"
                f"Here it evaluates:\n\n```\n{expr}\n```\n\nThe corresponding "
                f"Python call `sim.get(...)` returns the value as a Python "
                f"object instead of printing it, which is what you want when the "
                f"number is about to be used in a calculation.")
    if head == "equilibrium":
        return ("`equilibrium` solves for a rest state with KINSOL, the "
                "pure-Rust translation of SUNDIALS' nonlinear algebraic solver: "
                "it finds a configuration at which all the forces balance.\n\n"
                "A completely free system has no isolated equilibrium at all — "
                "translate the whole thing and nothing changes — so at least one "
                "body must be pinned, and the command says so if none is.")
    if head == "sensitivity":
        return ("`sensitivity` computes derivatives of the trajectory with "
                "respect to a parameter, using CVODES and IDAS, the "
                "sensitivity-capable members of the SUNDIALS family. It answers "
                "'how much would the answer move if this input moved', without "
                "re-running the whole simulation for each perturbed input.")
    if head == "funcs":
        return ("`funcs` lists every built-in function available to expressions. "
                "It takes no arguments.")
    if head == "scene":
        rest = " ".join(s.split()[1:]).lower()
        if rest.startswith("create"):
            return ("`scene create` opens the graphical scene window. It starts a "
                    "small HTTP and WebSocket server — hand-rolled on `std::net`, "
                    "with no web framework — and prints the address to open in a "
                    "browser. The window shows the bodies as wireframes, with a "
                    "readout of energy, momentum and angular momentum that updates "
                    "live.\n\nThe window evolves its **own copy** of the system. "
                    "Stepping in the notebook does not move the window, and "
                    "playing the window does not move the notebook. That "
                    "isolation is deliberate.\n\nIf the environment variable "
                    "`POSIM_NO_BROWSER` is set, the address is printed but no "
                    "browser is launched — which is what happens when this "
                    "notebook is run non-interactively.")
        if rest.startswith(("start", "play")):
            return ("`scene start` begins playback in the scene window. Its "
                    "thread calls the same integrator the notebook does; it never "
                    "integrates by hand. Playback runs until paused.")
        if rest.startswith("pause"):
            return "`scene pause` suspends playback in the scene window, leaving it where it is."
        if rest.startswith("reset"):
            return ("`scene reset` restores the window's playback to its initial "
                    "state bit-identically, clears the history and the step "
                    "counter, and leaves it stopped so that Start re-runs from "
                    "the beginning.")
        if rest.startswith("reverse"):
            return ("`scene reverse` plays the window backwards. It does this by "
                    "replaying snapshots from the history ring, never by "
                    "integrating with a negative step, so reversing all the way "
                    "returns to exactly t = 0.")
        return (f"`{s}` controls the graphical scene window. The scene command "
                f"family covers `CREATE`, `START`, `PAUSE`, `REVERSE`, `RESET`, "
                f"`REFRESH`, and the camera controls.")
    if head in ("qm", "qm2", "qm3"):
        return (f"`{s}` reaches the quantum-mechanics command family, which "
                f"solves the Schrodinger equation on a grid rather than moving "
                f"rigid bodies. Discretisation error here CONVERGES as the grid "
                f"is refined, which is how you tell it apart from a defect: "
                f"refine and check the ratio before calling a disagreement a bug.")
    if head == "special":
        return (f"`{s}` reaches the special-function command family — Bessel, "
                f"Legendre, gamma, Clebsch-Gordan and the rest — evaluated by the "
                f"project's own pure-Rust implementations.")
    if head == "def":
        return ("`def` defines a reusable function. The form is "
                "`DEF <name>(<params>) { <body> }`, where parameters may carry "
                "default values given as expressions. Parameters may not be "
                "reserved words, built-in function names, `pi`/`tau`, or "
                "duplicates, and a definition that fails leaves nothing behind.")
    if head == "let":
        return (f"`{s}` binds a name to the value of an expression, so it can be "
                f"used in later commands. The form is `LET <name> = <expression>`.")
    if head == "call" or "(" in s.split()[0]:
        return (f"`{s}` calls a previously defined function with the arguments "
                f"given. A call that fails part way through leaves no half-built "
                f"body behind.")
    if head == "del":
        return f"`{s}` deletes a body from the simulation."
    return (f"`{s}` — a command of the simulator's language. Sent as typed; the "
            f"simulator prints whatever it has to say about it.")

# ======================================================================
# 3. Deriving the physics
# ======================================================================

def plural(n, one, many=None):
    """`n` with a correctly agreeing noun."""
    return f"{n} {one if n == 1 else (many or one + 's')}"

def verb(n, singular, plural_form):
    return singular if n == 1 else plural_form

def objects_table(bodies):
    rows = ["| name | shape | given properties |", "|---|---|---|"]
    for i, b in enumerate(bodies):
        nm = b["name"] or f"obj{i}"
        props = ", ".join(f"`{k} = {v}`" for k, v in b["fields"].items())
        static = " **(static)**" if b["fields"].get("inverse_mass","").strip() in ("0","0.0") else ""
        rows.append(f"| `{nm}`{static} | `{b['shape']}` | {props} |")
    return "\n".join(rows)

def shape_notes(bodies):
    seen, out = set(), []
    for b in bodies:
        if b["shape"] in seen: continue
        seen.add(b["shape"])
        sh = SHAPES.get(b["shape"])
        if not sh: continue
        out.append(f"**`{b['shape']}`** — {sh['blurb']}. {sh['inertia']} {sh['why']}")
    return "\n\n".join(out)

def field_notes(bodies):
    seen, out = set(), []
    for b in bodies:
        for k in b["fields"]:
            if k in seen: continue
            seen.add(k)
            if k in FIELDS:
                out.append(f"- **`{k}`** — {FIELDS[k]}")
    return "\n".join(out)

def derive(units, header):
    bodies  = [parse_new(u["stmt"]) for u in units if u["stmt"].strip().lower().startswith("new ")]
    bodies  = [b for b in bodies if b]
    joints  = [u["stmt"] for u in units if kind(u["stmt"]) == "joints"]
    sets    = {}
    for u in units:
        m = re.match(r"set\s+system\.(\w+)\s*=\s*(.+)$", u["stmt"].strip(), re.I)
        if m: sets[m.group(1).lower()] = m.group(2).strip()
    methods = [u["stmt"].split()[1].lower() for u in units
               if u["stmt"].strip().lower().startswith("method ")]
    method  = methods[-1] if methods else ("ida" if joints else "adams")
    collide_on = any(u["stmt"].strip().lower() == "collide" or
                     u["stmt"].strip().lower().startswith("collide on") for u in units)
    collide_off = any(u["stmt"].strip().lower().startswith("collide off") for u in units)
    has_box = any(u["stmt"].strip().lower().startswith("box ") and
                  not u["stmt"].strip().lower().startswith("box off") for u in units)
    return dict(bodies=bodies, joints=joints, sets=sets, method=method,
                collide=(collide_on and not collide_off), has_box=has_box,
                header=header)

def situation_md(d, header_note):
    b = d["bodies"]
    parts = []
    if b:
        parts.append("### The objects\n\n" + objects_table(b) +
          "\n\nEach row is one rigid body. The simulator computes each body's "
          "inertia tensor from its shape; you never type an inertia tensor in.")
        sn = shape_notes(b)
        if sn: parts.append("### The shapes, and the inertia each implies\n\n" + sn)
        fn = field_notes(b)
        if fn: parts.append("### What each property means\n\n" + fn)
    else:
        parts.append("### The objects\n\nThis example creates no rigid bodies. It "
                     "exercises a part of the system that works on fields or on "
                     "numbers rather than on moving objects.")
    # interactions
    inter = ["### The interactions\n"]
    g = d["sets"].get("g_constant")
    if g is not None and g.strip() not in ("0", "0.0", "0e0"):
        inter.append(f"- **Mutual gravity is ON**, with `G = {g}`. Every pair of "
                     f"bodies attracts every other by `F = G m1 m2 / r^2`, "
                     f"softened at short range if `softening` is set.")
    elif g is not None:
        inter.append("- **Mutual gravity is OFF** (`g_constant = 0`). The "
                     "simulator's default is 1, so this had to be said "
                     "explicitly; without it, bodies that are supposed only to "
                     "bounce or to be held by joints would also attract each other.")
    else:
        inter.append("- **Mutual gravity is at its default `G = 1`**, so every "
                     "pair of bodies attracts every other by `F = G m1 m2 / r^2`.")
    ug = d["sets"].get("uniform_gravity")
    if ug:
        zero = re.sub(r"[\s\[\]]", "", ug) in ("0,0,0", "0.0,0.0,0.0")
        inter.append(f"- **Uniform gravity** is `{ug}`" +
                     (" — that is, none." if zero else
                      " metres per second squared, applied to every body alike."))
    if d["sets"].get("b_field"):
        inter.append(f"- **A magnetic field** `{d['sets']['b_field']}` acts on any "
                     f"charged body through the Lorentz force `F = q v x B`.")
    if d["sets"].get("e_field"):
        inter.append(f"- **An electric field** `{d['sets']['e_field']}` acts on any "
                     f"charged body through `F = q E`.")
    inter.append("- **Contact is " + ("ON" if d["collide"] else "OFF") + "**. " +
        ("Impacts are located as roots of the separation function by SUNDIALS' "
         "rootfinder, so the time of impact is exact to solver tolerance and "
         "nothing tunnels through a thin body."
         if d["collide"] else
         "Nothing here is meant to touch anything else, so the rootfinder is not armed."))
    if d["has_box"]:
        inter.append("- **A rigid box** encloses the scene: six static wall slabs "
                     "that the bodies bounce off and that never move.")
    if d["joints"]:
        nj = len(d["joints"])
        inter.append(f"- **{plural(nj, 'joint')}** {verb(nj, 'holds', 'hold')} the "
                     f"bodies together. A joint is an exact geometric relation "
                     f"held for all time by constraint forces the solver "
                     f"computes — not a stiff spring.")
    parts.append("\n".join(inter))
    if header_note:
        parts.append("### Note carried over from the example this notebook pairs with\n\n"
                     + header_note)
    return "\n\n".join(parts)

def eom_md(d):
    n = len(d["bodies"])
    terms, names = [], []
    g = d["sets"].get("g_constant")
    if g is None or g.strip() not in ("0", "0.0", "0e0"):
        terms.append("F_grav_i"); names.append(
          "`F_grav_i = sum over j != i of G m_i m_j (x_j - x_i) / |x_j - x_i|^3` — "
          "the mutual attraction, softened at short range if `softening` is set")
    ug = d["sets"].get("uniform_gravity")
    if ug and re.sub(r"[\s\[\]]", "", ug) not in ("0,0,0", "0.0,0.0,0.0"):
        terms.append("m_i g"); names.append(f"`m_i g` — the uniform field, here `g = {ug}`")
    if d["sets"].get("b_field") or d["sets"].get("e_field"):
        terms.append("q_i (E + v_i x B)"); names.append(
          "`q_i (E + v_i x B)` — the Lorentz force on a charged body")
    if d["collide"]:
        terms.append("F_contact_i"); names.append(
          "`F_contact_i` — impulsive, delivered at the instant of contact along "
          "the contact normal, scaled by the coefficient of restitution")
    if d["joints"]:
        terms.append("J^T lambda"); names.append(
          "`J^T lambda` — the constraint forces the joints generate. These are "
          "NOT written down by you: they are the Lagrange multipliers, solved "
          "for at every step so the joint equations hold exactly")
    rhs = " + ".join(terms) if terms else "0"
    out = [
      f"There are {n} bod{'ies' if n != 1 else 'y'}. Each obeys Newton's and "
      f"Euler's equations, both of which are **second order**:", "",
      "```", f"m_i * d2(x_i)/dt2 = {rhs}", "",
      "I_i * d(omega_i)/dt + omega_i x (I_i omega_i) = T_i", "```", "",
      "The `omega x (I omega)` term in Euler's equation is the gyroscopic term. "
      "It is what makes a tumbling body's spin axis wander even with no torque "
      "at all, and it is why rigid-body motion cannot be reduced to three "
      "independent rotations.", "",
      "The forces on the right are:", "",
    ]
    out += [f"- {t}" for t in names] or ["- nothing: every body coasts freely."]
    if d["joints"]:
        out += ["", "and the torques `T_i` likewise carry a constraint part "
                "`J_omega^T lambda`, which is how a bearing resists being "
                "twisted out of line."]
    return "\n".join(out)

def constraints_md(d):
    if not d["joints"]:
        return ("**This model has no joints.** Nothing algebraic relates the "
                "coordinates, so the system is an ordinary differential "
                "equation rather than a differential-algebraic one, and it "
                "needs no Lagrange multipliers and no constraint Jacobian.\n\n"
                "That does not mean nothing is conserved. Energy, linear "
                "momentum and angular momentum are still exact consequences of "
                "the equations of motion, and comparing them before and after a "
                "run is the honest way to judge the integration.")
    lines, total = [], 0
    for n, stmt in enumerate(d["joints"], start=1):
        head = stmt.split()[0].lower()
        j = JOINTS[head]
        total += j["rows"]
        args = " ".join(stmt.split()[1:])
        lines.append(f"**{n}. `{stmt}` — {j['rows']} row"
                     f"{'s' if j['rows'] != 1 else ''}.** {j['holds'].capitalize()}. "
                     f"{j['detail']}")
    nfree = sum(1 for b in d["bodies"]
                if b["fields"].get("inverse_mass","").strip() not in ("0","0.0"))
    dof = 6 * nfree
    body = "\n\n".join(lines)
    arith = (f"\n\n**The arithmetic that decides whether this can be solved at "
             f"all.** {plural(nfree, 'non-static body', 'non-static bodies')} "
             f"{verb(nfree, 'gives', 'give')} `6 x {nfree} = {dof}` degrees of "
             f"freedom, against **{plural(total, 'constraint row')}**, leaving "
             f"`{dof} - {total} = {dof - total}` "
             f"{'freedom' if dof - total == 1 else 'freedoms'}.\n\n")
    if dof - total < 0:
        arith += ("More rows than freedoms means the model is over-determined. "
                  "In a planar linkage this usually means the rows are "
                  "REDUNDANT rather than contradictory — several joints each "
                  "insisting on the same plane — which makes the solver's matrix "
                  "singular. The remedy is to replace some revolute joints with "
                  "spherical ones, which is what real multibody codes do.")
    elif dof - total == 0:
        arith += ("Zero freedoms left means the mechanism is fully locked: it "
                  "has a shape but no motion.")
    else:
        arith += ("Those remaining freedoms are the motion the mechanism "
                  "actually has. Some of them may be passive — a freedom "
                  "nothing applies a force along simply stays where it started.")
    arith += ("\n\nEvery row is held at the position level (`g = 0`) **and** at "
              "the velocity level (`d(g)/dt = J u = 0`) at the same time. That "
              "is what the GGL formulation is for, and it is why these joints do "
              "not slowly come apart.")
    return body + arith

ROT_FLOOR = (
  "**Tolerances.** This model has a joint that grips ORIENTATION, and such a "
  "joint imposes a floor of `rtol = 1e-6` and `atol = 1e-8` however tight a "
  "tolerance you ask for. That is not timidity: the index-2 accuracy ceiling "
  "is real, and asking for more produces convergence failures rather than "
  "better answers. Every joint except `CONSTRAIN` grips orientation.")

NO_ROT_FLOOR = (
  "**Tolerances.** Every joint here is a plain `CONSTRAIN` distance rod, which "
  "is the one joint that does NOT grip orientation. Its Jacobian has no "
  "angular part at all, so the orientation-joint tolerance floor of "
  "`rtol = 1e-6` does not apply and you may ask for as tight a tolerance as "
  "you like. A bare `CONSTRAIN` also guarantees consistent initial conditions, "
  "so nothing has to be projected before the first step.")

def reduction_md(d):
    n = len(d["bodies"])
    out = [f"**Sizing this particular problem.** There are {n} "
           f"bod{'ies' if n != 1 else 'y'}, so the state vector `y` has", "",
           f"```\n{n} bodies x 13 numbers = {13*n} components\n```", "",
           "laid out as consecutive 13-number blocks in creation order, each "
           "holding position (3), linear momentum (3), orientation quaternion "
           "w-first (4), and angular momentum (3)."]
    statics = [b for b in d["bodies"]
               if b["fields"].get("inverse_mass","").strip() in ("0","0.0")]
    if statics:
        ns = len(statics)
        out += ["", f"The {plural(ns, 'static body', 'static bodies')} still "
                f"{verb(ns, 'occupies its', 'occupy their')} 13 slots. "
                f"{verb(ns, 'Its', 'Their')} inverse mass is zero, so "
                f"{verb(ns, 'its', 'their')} momentum-to-velocity map returns "
                f"zero and {verb(ns, 'it', 'they')} never "
                f"{verb(ns, 'moves', 'move')} — carried as constants rather "
                f"than special-cased out of the layout."]
    if d["joints"]:
        rows = sum(JOINTS[s.split()[0].lower()]["rows"] for s in d["joints"])
        out += ["", f"On top of those {13*n} differential unknowns are the "
                f"algebraic ones: one Lagrange multiplier `lambda` and one "
                f"position-level multiplier `mu` for each of the "
                f"{plural(rows, 'constraint row')}, so", "",
                f"```\n{rows} lambda + {rows} mu = {2*rows} algebraic unknowns\n```",
                "", f"giving **{13*n + 2*rows} unknowns in total**, solved as one "
                f"implicit system."]
    out += ["", "**What is handed to the Rust SUNDIALS translation.**"]
    m = d["method"]
    rot = any(st.split()[0].lower() != "constrain" for st in d["joints"])
    TOL_NOTE = ROT_FLOOR if rot else NO_ROT_FLOOR
    if m == "ida":
        out += ["", "Because there are joints this is a DAE, and `METHOD IDA` "
                "selects the pure-Rust translation of **IDA**. IDA is handed a "
                "residual of the implicit form", "", "```\nF(t, y, y') = 0\n```",
                "", "built from four blocks: the position rows carrying the "
                "`-M^-1 J^T mu` correction, the momentum rows carrying "
                "`+J^T lambda`, the rows `g(q) = 0`, and the rows `J u = 0`. IDA "
                "runs its variable-order variable-step BDF method and calls back "
                "into that residual; the Newton solve at each step is what "
                "determines the constraint forces.", "",
                TOL_NOTE, "",
                "**Consistent start.** A DAE cannot start from an arbitrary "
                "state: the start must satisfy `g = 0` and `J u = 0` together. "
                "If the initial velocities are not compatible with the joints — "
                "a body turning about an offset pivot must have its centre "
                "moving — the simulator projects them onto the nearest "
                "compatible set and reports how much it changed them."]
    else:
        out += ["", f"There are no algebraic constraints, so the whole of the "
                f"mechanics above is one first-order system", "",
                f"```\ndy/dt = f(t, y),   y in R^{13*n}\n```", "",
                f"handed to {METHODS.get(m, 'the chosen integrator')}", "",
                "The right-hand side `f` unpacks `y` into bodies, computes every "
                "force and torque, and packs the derivatives back in the same "
                "13-per-body order."]
    if d["collide"]:
        out += ["", "**Contact is handled as rootfinding, not as a force.** The "
                "separation between each collidable pair is registered with the "
                "solver as a root function. The integrator advances normally "
                "until a separation crosses zero, stops exactly there, applies "
                "the impulse, and restarts. This is why the time of impact comes "
                "out to solver tolerance and why a fast body cannot pass through "
                "a thin one between steps."]
    return "\n".join(out)

# ======================================================================
# 4. Grouping the script into explained steps
# ======================================================================

TITLES = {"world":"Set up the world","bodies":"Create the bodies",
          "joints":"Declare the joints","solver":"Choose the integrator",
          "run":"Integrate","measure":"Measure","scene":"The graphical scene window",
          "special":"Evaluate"}

def py_for(stmt):
    esc = stmt.replace("\\", "\\\\").replace('"', '\\"')
    return f'sim.do("{esc}")'

def group(units, max_per=4):
    groups, cur = [], None
    for u in units:
        k = kind(u["stmt"])
        brk = (cur is None or cur["kind"] != k or u["comments"]
               or len(cur["units"]) >= max_per)
        if brk:
            cur = {"kind": k, "units": []}
            groups.append(cur)
        cur["units"].append(u)
    return groups

def steps_from(groups):
    steps, n = [], {}
    for g in groups:
        k = g["kind"]
        n[k] = n.get(k, 0) + 1
        title = TITLES.get(k, "Run") + (f" ({n[k]})" if n[k] > 1 else "")
        pre = []
        for u in g["units"]:
            if u["comments"]:
                pre.append("> " + "\n> ".join(u["comments"]))
        body = []
        if pre:
            body.append("**A note carried over from the example this notebook "
                        "pairs with:**\n\n" + "\n>\n".join(pre))
        count = len(g["units"])
        what  = "this command" if count == 1 else f"these {count} commands"
        each  = "it" if count == 1 else "each one"
        body.append(f"The next cell sends {what} to the simulator. Here is "
                    f"exactly what {each} expects:")
        for u in g["units"]:
            body.append("---\n\n" + explain(u["stmt"], None))
        code = "\n".join(py_for(u["stmt"]) for u in g["units"])
        steps.append({"title": title, "explain": "\n\n".join(body), "code": code})
    return steps

# ======================================================================
# 5. Assembling one whole spec
# ======================================================================

def clean_header(header):
    """The example's own header comment, as markdown paragraphs."""
    lines = [l.rstrip() for l in header]
    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    return "\n".join(lines)

def title_from(header, key):
    for l in header:
        t = l.strip()
        if not t: continue
        t = re.sub(r"^\d+\.\s*", "", t)
        t = re.sub(r"\s*[-—]+\s*dynamic notebook\s*$", "", t, flags=re.I)
        t = re.sub(r"^Example\s+\d+\s*[-—:]\s*", "", t, flags=re.I)
        t = t.rstrip(".:")
        if len(t) > 4:
            return t[0].upper() + t[1:]
    return key.replace("_", " ").capitalize()

def abstract_from(header):
    """The first paragraph of the example's own header comment."""
    para, started = [], False
    for l in header:
        if l.strip():
            para.append(l.strip()); started = True
        elif started:
            break
    return " ".join(para) if para else ""

def discussion_md(d, header_note, units):
    out = []
    measures = {u["stmt"].split()[0].lower() for u in units
                if kind(u["stmt"]) == "measure"}
    out.append("**What the cells above actually established.**\n")
    bullets = []
    if "energy" in measures:
        bullets.append("- `energy` was printed. Compare the value before the run "
                       "with the value after it. For a closed system with no "
                       "dissipation they should agree to near machine precision, "
                       "and the size of the disagreement is the honest measure of "
                       "the integration — not a number to be talked around.")
    if "momentum" in measures:
        bullets.append("- `momentum` was printed. With no external force the total "
                       "linear momentum is exactly conserved, collisions included, "
                       "because every impulse is applied equal and opposite.")
    if "angmom" in measures:
        bullets.append("- `angmom` was printed. With no external torque the total "
                       "angular momentum about the origin is exactly conserved. "
                       "Note that `L` is conserved, not `omega`: for a body whose "
                       "three moments of inertia differ, the spin axis moves even "
                       "though the angular momentum does not.")
    if "constraints" in measures:
        bullets.append("- `constraints` was printed. `|g|` and `|g_dot|` should sit "
                       "near the solver's tolerance and, above all, should not be "
                       "growing. Growth would mean the joints were slowly letting "
                       "go, which is exactly what carrying both `g` and `g_dot` as "
                       "algebraic equations prevents.")
    if "contacts" in measures:
        bullets.append("- `contacts` was printed. Each entry carries the exact time "
                       "of impact found by the rootfinder, the normal, the point "
                       "and the impulse. The time of impact is accurate to solver "
                       "tolerance, not to within one step.")
    if "laplace" in measures:
        bullets.append("- `laplace` was printed. The Laplace-Runge-Lenz vector is "
                       "conserved only for an exact inverse-square law; its drift "
                       "measures how far the integration has moved the orbit's "
                       "orientation.")
    if "com" in measures:
        bullets.append("- `com` was printed. With no external force the centre of "
                       "mass travels in a straight line at constant speed no matter "
                       "what the bodies do among themselves.")
    if not bullets:
        bullets.append("- The values printed above are the result. Read them "
                       "against the physical expectations set out earlier in this "
                       "notebook.")
    out.append("\n".join(bullets))
    out.append("\n**How to judge a number here.** A disagreement with theory is "
               "only a defect if it fails to shrink when you ask for more accuracy. "
               "Tighten `system.rtol` and `system.atol` and re-run: a "
               "discretisation error converges, and a bug does not. The one place "
               "this does not apply is a model with orientation-gripping joints, "
               "where `rtol` is floored at `1e-6` and tightening past it buys "
               "nothing.")
    if header_note:
        out.append("\n**The full note from the example this notebook pairs with**, "
                   "which sets out why it was built this way:\n\n" + header_note)
    return "\n".join(out)

CAT_RUN = {
 "video":    "cargo run --release -p posim -- --script videos/scenes/{stem}.posim",
 "collision":"cargo run --release -p posim -- --script scripts/collisions/{stem}.posim",
 "solveit":  "cargo run --release -p posim -- --script scripts/solveit/{stem}.posim",
 "dynamic":  "cargo run -p posim --release -- --notebook dynamic_notebooks/{stem}.posim",
}

def make_spec(path, key, category, repo_rel):
    text = Path(path).read_text()
    header, units = parse(text)
    d = derive(units, header)
    note = clean_header(header)
    stem = Path(path).stem
    abstract = abstract_from(header) or (
        f"This notebook runs the `{stem}` example and explains every line of it.")
    return {
      "key": key,
      "title": title_from(header, stem),
      "source": repo_rel,
      "category": category,
      "howtorun": CAT_RUN[category].format(stem=stem),
      "abstract": abstract,
      "situation": situation_md(d, None),
      "eom": eom_md(d),
      "constraints": constraints_md(d),
      "reduction": reduction_md(d),
      "steps": steps_from(group(units)),
      "discussion": discussion_md(d, note, units),
    }

if __name__ == "__main__":
    src, key, category, repo_rel, out = sys.argv[1:6]
    spec = make_spec(src, key, category, repo_rel)
    Path(out).write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{out}  ({len(spec['steps'])} steps)")
