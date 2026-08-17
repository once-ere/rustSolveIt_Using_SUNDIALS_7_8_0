#!/usr/bin/env python3
"""Phase-3: the 107 command entries for the Index of Entities.

Writes index_data/entries_commands.json, which tools/build_catalog.py folds
into catalog.json. Ladders are hand-authored for the core and SCENE families
and templated per sub-command for QM / QM2 / QM3, where argument shapes are
regular. Every fragment is self-contained and starts from a clean session.
Examples carry expected: null / verified: null — Phase 4 runs them.

Stdlib only.
"""

import json
import re

LEVELS = ["trivial", "intermediate", "advanced", "expert"]
P = "posim/src/parser.rs"
V = "posim/src/vm.rs"
G = "grammar.md"


def ex(level, code, medium="posim", runner="posim --script"):
    return {"level": level, "medium": medium, "code": code.strip(),
            "expected": None, "verified": None, "runner": runner}


def E(id, name, summary, definition, syntax, rungs, **kw):
    return {"id": id, "name": name, "kind": "command", "tier": "A",
            "aliases": kw.get("aliases", []),
            "indexKeys": kw.get("indexKeys") or [name.lstrip("<{").upper()[:1]],
            "summary": summary, "definition": definition, "syntax": syntax,
            "parameters": kw.get("parameters", []), "returns": kw.get("returns"),
            "errors": kw.get("errors", []),
            "locations": kw.get("locations", [{"file": P, "line": 8, "role": "grammar"}]),
            "examples": [ex(l, c, medium=kw.get("medium", "posim"),
                            runner=kw.get("runner", "posim --script"))
                         for l, c in zip(LEVELS, rungs)],
            "seeAlso": kw.get("seeAlso", []), "invariants": kw.get("invariants", []),
            "status": "complete" if rungs else "stub"}


TWO_BALLS = ("set system.g_constant = 0\n"
             "new sphere { mass = 1, radius = 0.5, position = [-2, 0, 0], velocity = [1, 0, 0] }\n"
             "new sphere { mass = 1, radius = 0.5, position = [2, 0, 0], velocity = [-1, 0, 0] }\n"
             "run 2 steps 2")

KEPLER = ("new point { mass = 1e9, position = [0, 0, 0] }\n"
          "new point { mass = 1, position = [0.4, 0, 0], velocity = [0, 2, 0] }\n"
          "set system.g_constant = 1e-9\nset system.softening = 0")


def gsec(sec):
    """Line of `### <sec> …` in grammar.md, found rather than hardcoded.

    The catalog cites grammar.md by line, and grammar.md is edited — the
    hardcoded numbers this replaced had already drifted by up to 30 lines
    after one round of documentation fixes, which turns a precise citation
    into a confidently wrong one. Resolving the heading at build time means
    the citation cannot rot.
    """
    if not hasattr(gsec, "_idx"):
        gsec._idx = {}
        for i, line in enumerate(open(G, encoding="utf-8"), 1):
            m = re.match(r"^#{2,3} (\d+(?:\.\d+)?)[ .]", line)
            if m:
                gsec._idx[m.group(1)] = i
    return gsec._idx.get(sec, 1)


out = []

# ==========================================================================
# core commands
# ==========================================================================

out.append(E(
    "cmd.new", "NEW",
    "Create one physical object and print its handle objN.",
    "Creates one physical object and prints its handle (obj0, obj1, ..., numbered by position "
    "in the system). The { ... } block is an optional list of initializers. Four guarantees "
    "make initializers forgiving: (1) ORDER DOES NOT MATTER FOR VELOCITIES -- velocity and "
    "angular_velocity are applied after mass and the inertia tensor are final; (2) INERTIA IS "
    "COMPUTED FOR YOU from the final mass and shape, unless you supplied inertia_tensor "
    "yourself; (3) COUPLED FIELDS STAY CONSISTENT; (4) NEW IS TRANSACTIONAL -- if any "
    "initializer or the final geometry validation fails, the half-built object is removed "
    "before the error is reported, so a failing NEW never leaves a ghost behind and "
    "system.count is exactly what it was.",
    ['NEW <shape> [ "AS" (IDENT | STRING) ] [ "{" init { "," init } "}" ]',
     'init := IDENT "=" expr'],
    ["new sphere { mass = 2, radius = 0.5 }",
     "new point { mass = 1, position = [0, 10, 0], velocity = [1, 0, 0] }\nlist",
     'new sphere as "ball" { mass = 2, radius = 0.5, velocity = [1, 0, 0] }\nget ball.momentum',
     "new torus { mass = 1, inner_radius = 1, outer_radius = 2 }\nget obj0.inertia_tensor\n"
     "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\nget obj1.mass"],
    returns="string -- the handle, e.g. obj0, or `obj0 as ball` when AS was used",
    errors=["torus needs 0 <= inner < outer (got inner = 2, outer = 1)",
            "the name `d` already refers to obj0 - DEL it or pick another name",
            "`system` is reserved"],
    locations=[{"file": P, "line": 8, "role": "production"},
               {"file": V, "line": 1, "role": "NewObject/InitField/FinishNew"},
               {"file": G, "line": gsec("5.1"), "role": "spec (5.1)"}],
    seeAlso=["cmd.del", "cmd.list", "type.shape.sphere", "kw.as"],
    invariants=["NEW is transactional: a failing initializer or final validation leaves no "
                "ghost object.",
                "The torus radius pair and the dumbbell part fields are DEFERRED and resolved "
                "once at the end of the list, so they are genuinely order-independent."]))

out.append(E(
    "cmd.set", "SET",
    "Write any field through the object's get/set API.",
    "Writes a field. A path is objN.field, system.field, contactK.field or name.field for a "
    "name registered with NEW ... AS, optionally followed by a component .x .y .z (vectors) "
    "or .w .x .y .z (quaternions). Component writes are safe read-modify-write operations "
    "through the full field's get/set pair. Every write goes through the physical_object "
    "setters, so every coupled invariant (mass<->inverse, inertia<->inverse, unit "
    "quaternions, momentum-canonical velocity) is enforced. SET prints no Out[n].",
    ["SET <path> = <expr>"],
    ["new point { mass = 1 }\nset obj0.mass = 5\nget obj0.mass",
     "new point { mass = 1 }\nset obj0.velocity = [1, 2, 3]\nget obj0.momentum",
     "new point { mass = 1 }\nset obj0.position.y = 10\nget obj0.position",
     "new sphere { mass = 4, radius = 0.5 }\nset obj0.inverse_mass = 0\n"
     "get obj0.mass\nget obj0.inverse_mass"],
    returns="nothing -- SET prints no Out[n]",
    errors=["parse error at column 5: expected a path root (`objN` or `system`), found `=`",
            "unknown object field `bogus` - see HELP for the field list",
            "system field `<name>` is not writable (or unknown) - see HELP"],
    locations=[{"file": P, "line": 12, "role": "production"},
               {"file": V, "line": 2411, "role": "store_path"},
               {"file": G, "line": gsec("5.2"), "role": "spec (5.2)"}],
    seeAlso=["cmd.get", "prop.obj.mass", "cmd.let"]))

out.append(E(
    "cmd.get", "GET",
    "Read any field. Takes a PATH ONLY -- no arithmetic.",
    "Reads a field and prints it. GET takes only a path. To do arithmetic with a field, use a "
    "bare expression instead: `get obj0.position.x - 5` is an error, while "
    "`obj0.position.x - 5` is fine. That is the single most common early surprise, and it is "
    "deliberate -- it keeps GET unambiguous.",
    ["GET <path>", 'path := IDENT { "." IDENT }'],
    ["new point { mass = 1 }\nget obj0.mass",
     "new point { mass = 1, position = [1, 2, 3] }\nget obj0.position.y",
     'new sphere as "ball" { mass = 2 }\nget ball.mass\nget system.count',
     "new point { mass = 0.145, position = [0, 1, 0], velocity = [30, 30, 0] }\n"
     "set system.gravity = [0, -9.81, 0]\nrun 6.12 steps 3\nget obj0.position.x\n"
     "obj0.position.x - 30 * 6.12"],
    returns="the field's value",
    errors=["no object obj7", "unknown object field `bogus` - see HELP for the field list"],
    locations=[{"file": P, "line": 13, "role": "production"},
               {"file": V, "line": 2243, "role": "load_path"},
               {"file": G, "line": gsec("5.2"), "role": "spec (5.2)"}],
    seeAlso=["cmd.set", "cmd.expr"],
    invariants=["GET takes a path, never an expression - use a bare expression for arithmetic."]))

out.append(E(
    "cmd.del", "DEL",
    "Delete object n. LATER OBJECTS RENUMBER.",
    "Removes object n. Later objects renumber, and the AS-name registry follows them down: the "
    "deleted object's name is removed and higher-numbered names shift with their objects. Wall "
    "indices of a BOX are tracked through every renumbering, so deleting a non-wall object "
    "never confuses the box -- but deleting a WALL dissolves the box (system.box drops to 0) "
    "while the surviving slabs stay tracked.",
    ["DEL <NUMBER>", "DELETE <NUMBER>"],
    ["new point { mass = 1 }\nnew point { mass = 2 }\ndel 0\nlist",
     "new point { mass = 1 }\ndelete 0\nlist",
     'new sphere as "ball" { mass = 1 }\nnew sphere as "pebble" { mass = 2 }\ndel 0\n'
     "get pebble.mass",
     "box 4\ndel 0\nbox"],
    aliases=["DELETE"],
    returns="string -- `deleted obj2; 2 object(s) remain (indices renumbered)`",
    locations=[{"file": P, "line": 14, "role": "production"},
               {"file": G, "line": gsec("5.5"), "role": "spec (5.5)"}],
    seeAlso=["cmd.new", "cmd.list", "cmd.box"],
    invariants=["DEL renumbers later objects and carries the AS-name registry with them.",
                "Deleting a wall dissolves the box but leaks no slab."]))

out.append(E(
    "cmd.list", "LIST",
    "One line per object.",
    "Prints one line per object: handle, shape with its defining dimensions, mass, charge and "
    "position. Wall slabs of a BOX carry the tag [wall: static, inverse_mass=0] and show "
    "mass=0 -- the canonical stored quantity for a static body is the inverse.",
    ["LIST"],
    ["new sphere { mass = 2, radius = 0.5 }\nlist",
     "new point { mass = 1 }\nnew cuboid { mass = 3, half_extents = [0.5, 1, 2] }\nlist",
     "box 4\nlist",
     "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\n"
     "new torus { mass = 1, inner_radius = 1, outer_radius = 2 }\nlist"],
    returns="string -- one line per object",
    locations=[{"file": P, "line": 15, "role": "production"},
               {"file": G, "line": gsec("5.5"), "role": "spec (5.5)"}],
    seeAlso=["cmd.new", "cmd.del", "cmd.box"]))

out.append(E(
    "cmd.step", "STEP",
    "Advance time by dt -- always through SUNDIALS.",
    "Advances time by dt. Accepts a full expression. All integration is performed by the "
    "pure-Rust SUNDIALS solvers (see METHOD); there is no hand-rolled stepper anywhere in the "
    "project. When collisions are on and the system holds at least one collidable pair, "
    "impacts are found DURING the step by SUNDIALS event rootfinding.",
    ["STEP <expr>"],
    ["new point { mass = 1, velocity = [1, 0, 0] }\nstep 1\nget obj0.position",
     "new sphere { mass = 2, radius = 0.5, position = [0, 10, 0] }\n"
     "set system.gravity = [0, -9.81, 0]\nstep 0.5\nget obj0.velocity",
     "new point { mass = 1, velocity = [0, 1, 0] }\nstep pi / 4\nget system.time",
     TWO_BALLS.replace("run 2 steps 2", "step 2") + "\nget system.collisions"],
    returns="string -- `t = 1 (advanced by 1, 12 solver steps)`",
    locations=[{"file": P, "line": 16, "role": "production"},
               {"file": "physical_object/src/integrate.rs", "line": 1, "role": "the driver"},
               {"file": G, "line": gsec("5.3"), "role": "spec (5.3)"}],
    seeAlso=["cmd.run", "cmd.method", "cmd.collide"],
    invariants=["Every step goes through cvode_rs/arkode_rs. Never a hand-rolled stepper."]))

out.append(E(
    "cmd.run", "RUN",
    "Advance by t with n evenly spaced output snapshots.",
    "Advances by t, stopping at n evenly spaced output points (default 10). Both arguments "
    "accept full expressions, so `run 2 * pi steps 8` is legal. The reply summarises the run, "
    "and |dE/E| -- the relative change in total energy across it -- is your built-in sanity "
    "check. When contacts occurred the banner names them and CONTACTS lists them.",
    ['RUN <expr> [ "STEPS" NUMBER ]'],
    ["new point { mass = 1, velocity = [1, 0, 0] }\nrun 2 steps 2",
     "new point { mass = 0.145, position = [0, 1, 0], velocity = [30, 30, 0] }\n"
     "set system.gravity = [0, -9.81, 0]\nrun 6.12 steps 3\nget obj0.position.x",
     "new sphere { mass = 2, radius = 0.1, charge = -1.5, velocity = [3, 0, 0] }\n"
     "set system.b_field = [0, 0, 4]\nmethod bdf\nrun 2 * pi * 2 / (1.5 * 4) steps 8\n"
     "get obj0.position",
     KEPLER + "\nlaplace 1\nmethod sprk mclachlan_4_4 0.001\nrun 12.6 steps 2\nlaplace 1"],
    returns="string -- `t = 12.6 (12600 solver steps, 2 snapshots, |dE/E| = 9.237e-14)`",
    errors=["SPRK method requires a separable Hamiltonian: magnetic field B must be zero "
            "(the Lorentz force q v x B is velocity-dependent); use METHOD ADAMS or BDF"],
    locations=[{"file": P, "line": 17, "role": "production"},
               {"file": "physical_object/src/integrate.rs", "line": 1, "role": "the driver"},
               {"file": G, "line": gsec("5.3"), "role": "spec (5.3)"}],
    seeAlso=["cmd.step", "cmd.steps", "cmd.method", "cmd.energy"]))

out.append(E(
    "cmd.steps", "STEPS",
    "The optional output-snapshot count of RUN.",
    "The optional trailing clause of RUN: the number of evenly spaced output points at which "
    "the run reports. Default 10. It controls REPORTING, not accuracy -- the solver chooses "
    "its own internal step sizes, and the run banner tells you how many it actually took.",
    ['RUN <expr> "STEPS" NUMBER'],
    ["new point { mass = 1, velocity = [1, 0, 0] }\nrun 2 steps 4",
     "new point { mass = 1, velocity = [1, 0, 0] }\nrun 2 steps 1",
     "new point { mass = 1, velocity = [1, 0, 0] }\nrun 2\nget system.time",
     "new point { mass = 0.145, position = [0, 1, 0], velocity = [30, 30, 0] }\n"
     "set system.gravity = [0, -9.81, 0]\nrun 6.12 steps 100\nget obj0.position.x"],
    locations=[{"file": P, "line": 17, "role": "production"}],
    seeAlso=["cmd.run"]))

out.append(E(
    "cmd.method", "METHOD",
    "Choose the SUNDIALS integrator: ADAMS, BDF or SPRK.",
    "Chooses the integrator. ADAMS (the default) is CVODE Adams-Moulton, adaptive, for "
    "non-stiff problems. BDF is CVODE BDF, for stiff ones -- fast magnetic gyration is the "
    "canonical case. SPRK <table> [dt] is ARKODE symplectic partitioned Runge-Kutta at a FIXED "
    "step (default dt 0.01); table names may be abbreviated, so leapfrog_2_2 becomes "
    "ARKODE_SPRK_LEAPFROG_2_2. SPRK requires a SEPARABLE system: if anything "
    "velocity-dependent or rotational is active, RUN refuses with an error naming the "
    "offending feature.",
    ['METHOD ( "ADAMS" | "BDF" | "SPRK" IDENT [ NUMBER ] )'],
    ["method adams", "method bdf",
     "new point { mass = 1, velocity = [0, 1, 0] }\nmethod sprk leapfrog_2_2 0.01\n"
     "run 1 steps 2",
     KEPLER + "\nmethod sprk mclachlan_4_4 0.001\nrun 12.6 steps 2\nget system.method"],
    returns="string -- `method = ARKODE SPRK ARKODE_SPRK_MCLACHLAN_4_4, fixed dt = 0.001`",
    errors=["SPRK method requires a separable Hamiltonian: magnetic field B must be zero "
            "(the Lorentz force q v x B is velocity-dependent); use METHOD ADAMS or BDF"],
    locations=[{"file": P, "line": 18, "role": "production"},
               {"file": G, "line": gsec("5.4"), "role": "spec (5.4)"}],
    seeAlso=["kw.adams", "kw.bdf", "kw.sprk", "prop.system.method"],
    invariants=["Useful SPRK tables: EULER_1_1, LEAPFROG_2_2 (velocity-Verlet), "
                "MCLACHLAN_2_2/3_3/4_4/5_6, RUTH_3_3, YOSHIDA_6_8."]))

for cid, nm, blurb, rungs in [
    ("energy", "ENERGY",
     "Total energy: kinetic + softened pairwise gravitational potential + uniform-field potentials.",
     ["new point { mass = 2, velocity = [3, 0, 0] }\nenergy",
      "new point { mass = 1, velocity = [100, 200, 100] }\nenergy",
      "new sphere { mass = 1, radius = 0.5, magnetic_moment_tensor = "
      "[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]] }\nset system.b_field = [0, 0.5, 0]\nenergy\n"
      "run 4 steps 4\nenergy",
      TWO_BALLS + "\nenergy"]),
    ("com", "COM", "The system's centre of mass, as a vec3.",
     ["new point { mass = 1, position = [2, 0, 0] }\n"
      "new point { mass = 1, position = [-2, 0, 0] }\ncom",
      "new point { mass = 3, position = [4, 0, 0] }\n"
      "new point { mass = 1, position = [0, 0, 0] }\ncom",
      "new point { mass = 1, position = [1, 0, 0], velocity = [0, 1, 0] }\n"
      "new point { mass = 4, position = [-1, 0, 0], velocity = [0, -0.25, 0] }\ncom",
      "new point { mass = 1, position = [1, 0, 0], velocity = [0, 1, 0] }\n"
      "new point { mass = 4, position = [-1, 0, 0], velocity = [0, -0.25, 0] }\n"
      "set system.g_constant = 0.001\nrun 3 steps 3\ncom"]),
    ("momentum", "MOMENTUM", "Total linear momentum of the system, as a vec3.",
     ["new point { mass = 2, velocity = [3, 0, 0] }\nmomentum",
      "new point { mass = 1, velocity = [0, 1, 0] }\n"
      "new point { mass = 4, velocity = [0, -0.25, 0] }\nmomentum",
      TWO_BALLS + "\nmomentum",
      "set system.g_constant = 0\nbox 4\n"
      "new point { mass = 1, position = [0.5, 0.3, 0.2], velocity = [100, 200, 100] }\n"
      "momentum\nrun 0.05 steps 10\nmomentum"]),
    ("angmom", "ANGMOM", "Total angular momentum about the ORIGIN -- orbital r x p plus spin.",
     ["new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\nangmom",
      "new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\n"
      "cross(obj0.position, obj0.momentum)\nangmom",
      "new cuboid { mass = 3, half_extents = [0.5, 1, 2], "
      "angular_velocity = [0.01, 3, 0.01] }\nangmom",
      "set system.g_constant = 0\n"
      "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5, position = [-2, 0.15, 0], "
      "velocity = [1.5, 0, 0], angular_velocity = [0, 0, 0.6] }\n"
      "new dumbbell { m1 = 2, m2 = 1, m_rod = 0.4, r1 = 0.3, r2 = 0.2, rod_radius = 0.08, "
      "length = 1.2, position = [2, -0.15, 0], velocity = [-1.5, 0, 0], "
      "angular_velocity = [0.4, 0, 0] }\nangmom\nrun 3 steps 60\nangmom"]),
]:
    out.append(E(f"cmd.{cid}", nm, blurb,
                 blurb + " An observable: it reads the live system and prints, changing nothing.",
                 [nm], rungs,
                 locations=[{"file": P, "line": 19, "role": "production"},
                            {"file": G, "line": gsec("5.5"), "role": "spec (5.5)"}],
                 seeAlso=["cmd.run", "cmd.list"]))

out.append(E(
    "cmd.laplace", "LAPLACE",
    "The Laplace-Runge-Lenz vector of object n about the system COM.",
    "Prints the Laplace-Runge-Lenz vector of object n about the system's centre of mass, with "
    "k = G * M_total. A points along a Kepler orbit's major axis and is conserved ONLY for a "
    "perfect 1/r^2 force, which makes it the most delicate conservation test available. "
    "Divided by m*k it is the eccentricity vector, so the orbit's e can be read straight off. "
    "Set system.softening = 0 first: any epsilon != 0 slightly breaks the 1/r^2 law and WOULD "
    "precess the axis.",
    ["LAPLACE <NUMBER>"],
    [KEPLER + "\nlaplace 1",
     KEPLER + "\nlaplace 1\nnorm([0.6, 0, 0])",
     KEPLER + "\nlaplace 1\nrun 6.28 steps 2\nlaplace 1",
     KEPLER + "\nlaplace 1\nmethod sprk mclachlan_4_4 0.001\nrun 12.6 steps 2\nlaplace 1"],
    returns="vec3",
    locations=[{"file": P, "line": 20, "role": "production"},
               {"file": G, "line": gsec("5.5"), "role": "spec (5.5)"}],
    seeAlso=["cmd.energy", "cmd.angmom", "nb.kepler_orbit", "prop.system.softening"]))

out.append(E(
    "cmd.reset", "RESET",
    "Wipe everything back to an empty system. NOT `SCENE RESET`.",
    "Wipes the simulator back to an empty system. An open scene window SURVIVES and re-syncs "
    "to the now-empty system -- its box wireframe and wall flags are cleared too, so no stale "
    "overlay outlives the wipe. Not to be confused with SCENE RESET, which re-initialises "
    "only the window's playback copy and leaves your notebook system alone.",
    ["RESET"],
    ["new point { mass = 1 }\nreset\nlist",
     "new point { mass = 1 }\nreset\nget system.count",
     "box 4\nreset\nget system.box",
     'new sphere as "ball" { mass = 2 }\nreset\nnew sphere as "ball" { mass = 3 }\n'
     "get ball.mass"],
    locations=[{"file": P, "line": 21, "role": "production"},
               {"file": G, "line": gsec("5.5"), "role": "spec (5.5)"}],
    seeAlso=["cmd.scene.reset", "magic.reset"],
    invariants=["RESET wipes the SYSTEM. SCENE RESET re-initialises the WINDOW's playback "
                "copy. They are different commands with different scopes."]))

out.append(E(
    "cmd.help", "HELP",
    "The quick-reference card for the whole command language.",
    "Prints HELP_TEXT -- the quick-reference card covering NEW, BOX, DEF, SET/GET and their "
    "field lists, the observables, the QM/QM2/QM3 families and every registered special "
    "function.",
    ["HELP"],
    # the discovery path, which is genuinely four steps: the card, then the
    # three commands that answer what the card cannot — which functions you
    # have defined, what one of them says, and what the session is set to
    ["help",
     "help\nfuncs",
     "def probe(m = 2) { new sphere { mass = m } }\nfuncs\nshow probe",
     "box 4\ncollide\nbox\nget system.method\nget system.count"],
    locations=[{"file": V, "line": 363, "role": "HELP_TEXT"},
               {"file": P, "line": 21, "role": "production"}],
    seeAlso=["cmd.funcs"]))

out.append(E(
    "cmd.collide", "COLLIDE",
    "Switch rigid-body collision detection; bare reports status.",
    "Collisions are ON by default and detected in EVERY scene. When on and the system holds at "
    "least one collidable pair, STEP and RUN find impacts DURING the time step by SUNDIALS "
    "event rootfinding: the integrator lands on the instant where a pair's signed separation "
    "crosses zero, interpolated to solver precision. Nothing tunnels. A system with zero "
    "collidable pairs is BIT-IDENTICAL with COLLIDE ON and OFF -- a structural invariance the "
    "test suite protects.",
    ['COLLIDE [ "ON" | "OFF" ]'],
    ["new sphere { radius = 0.5 }\nnew sphere { radius = 0.5, position = [3, 0, 0] }\ncollide",
     "collide off\ncollide", "collide off\ncollide on\ncollide",
     TWO_BALLS + "\ncollide\nget system.collisions"],
    returns="string -- `collisions ON (51 collidable pair(s); 0 impulse(s) so far)`",
    locations=[{"file": P, "line": 23, "role": "production"},
               {"file": "physical_object/src/collide.rs", "line": 1, "role": "implementation"},
               {"file": G, "line": gsec("5.7"), "role": "spec (5.7)"}],
    seeAlso=["cmd.contacts", "prop.system.collide", "prop.contact.normal"],
    invariants=["Two points can never collide.",
                "Zero-collidable-pair systems are bit-identical with COLLIDE ON vs OFF."]))

out.append(E(
    "cmd.contacts", "CONTACTS",
    "List every contact of the last STEP/RUN.",
    "Lists every contact recorded by the last STEP/RUN. Each is also exposed to the rest of "
    "the language through read-only contactK paths -- ordinary expression atoms, so you can "
    "compute with them without any GET. The normal points from obji toward objj: objj "
    "receives +J n, obji receives -J n.",
    ["CONTACTS"],
    [TWO_BALLS + "\ncontacts", TWO_BALLS + "\nget contact0.normal",
     TWO_BALLS + "\nget contact0.t\nget contact0.impulse",
     TWO_BALLS + "\ncontact0.impulse * contact0.normal.x"],
    returns="string -- one line per contact",
    errors=["no contact0 - the last STEP/RUN recorded 0 contact(s); CONTACTS lists them"],
    locations=[{"file": P, "line": 24, "role": "production"},
               {"file": V, "line": 2381, "role": "contact paths"},
               {"file": G, "line": gsec("5.7"), "role": "spec (5.7)"}],
    seeAlso=["cmd.collide", "prop.contact.normal", "prop.contact.impulse"]))

out.append(E(
    "cmd.let", "LET",
    "Bind a session variable, visible in expressions, bodies and defaults.",
    "Binds a session variable. It is visible in bare expressions, in function bodies, and in "
    "DEF defaults -- and, holding a string, it can NAME things: NEW ... AS and named paths "
    "resolve a bare identifier through the parameter/LET bindings first. Note that GET takes "
    "only a path, so a variable is read back as a BARE EXPRESSION: `g0`, not `GET g0`.",
    ["LET IDENT = <expr>"],
    ["let g0 = 9.81\ng0", "let h = 0.2\nlet k = 1 / (h * h)\nk",
     'let n = "pebble"\nnew sphere as n { mass = 0.1 }\nget pebble.mass',
     "let g0 = 9.81\ndef drop(name, h = 10) {\n"
     "  new sphere as name { position = [0, h, 0] }\n"
     "  set system.gravity = [0, -g0, 0]\n}\ndrop(\"ball\")\nget ball.position.y"],
    returns="string -- `g0 set`",
    errors=["unknown name `speed` (define it with LET, pass it as a function parameter, or "
            "use `speed.field` for a registered object)"],
    locations=[{"file": P, "line": 25, "role": "production"},
               {"file": V, "line": 1, "role": "StoreGlobal"},
               {"file": G, "line": gsec("5.9"), "role": "spec (5.9)"}],
    seeAlso=["cmd.def", "cmd.expr", "kw.as"]))

out.append(E(
    "cmd.funcs", "FUNCS",
    "List every user-defined function signature with its defaults.",
    "Lists every user function's signature, with its defaults, and reminds you that "
    "SHOW <name> prints the source.",
    ["FUNCS", "FUNCTIONS"],
    ["def probe(m = 2) { new sphere { mass = m } }\nfuncs",
     "def probe(m = 2) { new sphere { mass = m } }\nfunctions",
     "def a(x = 1) { x }\ndef b(y = 2) { y }\nfuncs",
     "def drop(name, m = 1, h = 10) {\n"
     "  new sphere as name { mass = m, position = [0, h, 0] }\n}\nfuncs\nshow drop"],
    aliases=["FUNCTIONS"],
    locations=[{"file": P, "line": 26, "role": "production"},
               {"file": G, "line": gsec("5.9"), "role": "spec (5.9)"}],
    seeAlso=["cmd.def", "cmd.show"]))

out.append(E(
    "cmd.show", "SHOW",
    "Print a user function's source verbatim -- half of the edit loop.",
    "Prints the definition verbatim, exactly as you typed it. Editing is SHOW + re-DEF: copy "
    "the printed source, change what you want, and redefine the same name -- the reply appends "
    "(redefined). SHOW is also a SCENE sub-command (SCENE SHOW n|ALL); the parser tells them "
    "apart by position.",
    ["SHOW IDENT"],
    ["def probe(m = 2) { new sphere { mass = m } }\nshow probe",
     "def drop(name, m = 1, h = 10) {\n"
     "  new sphere as name { mass = m, position = [0, h, 0] }\n}\nshow drop",
     "def drop(name, h = 10) { new sphere as name { position = [0, h, 0] } }\nshow drop\n"
     "def drop(name, h = 100) { new sphere as name { position = [0, h, 0] } }\nshow drop",
     "def drop(name, h = 100) { new sphere as name { position = [0, h, 0] } }\n"
     'drop("skydiver")\nskydiver.y'],
    errors=["no user function `nosuch` - FUNCS lists the defined ones"],
    locations=[{"file": P, "line": 27, "role": "production"},
               {"file": G, "line": gsec("5.9"), "role": "spec (5.9)"}],
    seeAlso=["cmd.def", "cmd.funcs", "cmd.scene.show"]))

out.append(E(
    "cmd.box", "BOX",
    "The rigid, infinitely massive bounding box.",
    "BOX <size> builds a closed rigid room out of SIX ORDINARY CUBOID OBJECTS -- wall slabs "
    "behind the planes x,y,z = +/- size/2 -- and prints their handles. Infinite mass is exact, "
    "not approximate: the equations of motion never use the mass itself, only the INVERSE "
    "mass, so each wall's inverse_mass = 0 contributes exactly 0 to every impulse denominator "
    "and the walls stay bit-identically at rest. One measurable consequence: system momentum "
    "is NOT conserved inside a box -- the walls absorb it without moving. A second BOX <size> "
    "replaces the first; BOX OFF removes it; bare BOX reports.",
    ["BOX <expr>", 'BOX "OFF"', "BOX"],
    ["box 4\nget system.box", "box 4\nbox", "box 4\nbox off\nbox",
     "set system.g_constant = 0\nbox 4\n"
     "new point { mass = 1, position = [0.5, 0.3, 0.2], velocity = [100, 200, 100] }\n"
     "energy\nmomentum\nrun 0.05 steps 10\nenergy\nmomentum\nget obj0.inverse_mass"],
    returns="string -- `box: inner size 4 x 4 x 4 - six static walls obj0, ... with "
            "inverse_mass = 0 (infinitely massive)`",
    errors=["box: none (BOX <size> creates one)",
            "box: dissolved (a wall was deleted; 5 tracked slab(s) remain - BOX <size> "
            "replaces them, BOX OFF removes them)"],
    locations=[{"file": P, "line": 28, "role": "production"},
               {"file": G, "line": gsec("5.8"), "role": "spec (5.8)"}],
    seeAlso=["prop.system.box", "cmd.collide", "prop.obj.inverse_mass", "nb.box_of_shapes"],
    invariants=["Momentum is NOT conserved inside a box; energy is.",
                "The walls end every run bit-identically at rest."]))

out.append(E(
    "cmd.expr", "<expr>",
    "A bare expression -- the only way to do arithmetic on a field.",
    "Any expression on its own line is evaluated and printed. This is the escape hatch from "
    "GET's path-only rule: `obj0.position.x - 5` works where `get obj0.position.x - 5` is an "
    "error. Contact paths, registered names, LET variables, builtins and user functions are "
    "all ordinary atoms here. Comparisons yield 1 or 0, so (x > a) * (x < b) is an INDICATOR "
    "FUNCTION -- that is how a piecewise potential is written, since the language has no "
    "boolean type and no conditionals.",
    ["<expr>", 'expr := sum { ("<"|"<="|">"|">="|"=="|"!=") sum }',
     'sum := term { ("+"|"-") term }', 'term := unary { ("*"|"/") unary }'],
    ["2 + 3 * 4", "3 < 5",
     "new point { mass = 0.145, position = [0, 1, 0], velocity = [30, 30, 0] }\n"
     "set system.gravity = [0, -9.81, 0]\nrun 6.12 steps 3\n"
     "obj0.position.x - 30 * 6.12",
     "def barrier(x) { 2.5 * (x > 0) * (x < 1) }\nbarrier(-1)\nbarrier(0.5)\nbarrier(2)"],
    indexKeys=["<"],
    errors=["cannot multiply vec3 by vec3 (for vec3*vec3 use dot()/cross())",
            "unknown name `bogusname` (define it with LET, pass it as a function parameter, "
            "or use `bogusname.field` for a registered object)"],
    locations=[{"file": P, "line": 32, "role": "production"},
               {"file": P, "line": 68, "role": "expr/sum/term/unary"},
               {"file": G, "line": gsec("4.2"), "role": "spec (4.2)"}],
    seeAlso=["cmd.get", "type.number", "fn.dot"],
    invariants=["Comparisons sit at the LOWEST precedence and are left-associative, so "
                "`a < b < c` means `(a < b) < c` - legal, and almost certainly not what you "
                "meant.",
                "NaN compares false to everything, so `x != x` is the idiomatic NaN test."]))

out.append(E(
    "cmd.def", "DEF",
    "Define a user function. A LINE FORM, recognised before the grammar.",
    "DEF is a line form, not a grammar production. A line starting `DEF ` is recognised BEFORE "
    "the ordinary grammar: the notebook captures everything up to the closing } -- "
    "interactively it keeps prompting `  ...:= ` until the brace closes -- and installs the "
    "function. The body is a sequence of ordinary commands separated by newlines or ;, in "
    "which the parameters act as variables. Every body line is syntax-checked at definition "
    "time, and defaults are ordinary expressions evaluated once, at definition (LET variables "
    "are visible in them). Each call pushes a call frame; the depth cap is 32, because the "
    "language has no conditionals and recursion could never terminate anyway. The call returns "
    "the LAST body line's value.",
    ["DEF name(param [= default], ...) { body }", "name(arg, ...)"],
    ["def double(x = 2) { x * 2 }\ndouble()",
     "def probe(m = 2) {\n  new sphere { mass = m }\n}\nprobe()\nget obj0.mass",
     'def drop(name, m = 1, h = 10) {\n'
     '  new sphere as name { mass = m, position = [0, h, 0] }\n}\n'
     'drop("ball")\ndrop("pebble", 0.1, 2)\nget ball.position.y\nget pebble.mass',
     "set system.g_constant = 0\n"
     "def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, "
     "rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], "
     "angular_velocity = [0, 0, 0]) {\n"
     "  new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, "
     "rod_radius = rod_radius, length = length, position = position, velocity = velocity, "
     "angular_velocity = angular_velocity }\n}\n"
     'create_dumbell("dumbell0", 1, 2, 0.5, 0.25, 0.25, 0.1, 1, [-2, 0.15, 0], '
     "[1.5, 0, 0], [0, 0, 0.6])\nget dumbell0.m1\nget dumbell0.vx"],
    returns="string -- `function drop(3 parameter(s)) defined - 2 body line(s)`, plus "
            "`(redefined)` when it replaces an existing one",
    errors=["DEF: `norm` is a builtin function and cannot be redefined",
            "name(): missing argument `p` (it has no default)",
            "function call depth limit (32) exceeded"],
    locations=[{"file": P, "line": 56, "role": "line-form note"},
               {"file": V, "line": 1, "role": "Call/ListFns/ShowFn"},
               {"file": G, "line": gsec("5.9"), "role": "spec (5.9)"}],
    seeAlso=["cmd.funcs", "cmd.show", "cmd.let"],
    invariants=["DEF is a LINE FORM, not a keyword - `def` has no arm in Keyword::from_ident.",
                "A failing body line aborts the call; lines that already ran keep their "
                "effects, but each command's own guarantees still hold, so a failing NEW "
                "inside a body is still transactional.",
                "Editing is SHOW + re-DEF."]))

out.append(E(
    "cmd.call", "<call>",
    "Call a user function; trailing arguments take their defaults.",
    "Calling uses the ordinary call syntax name(arg, ...) -- the same atom as sqrt(2). A name "
    "that is not a builtin is looked up among your functions. Missing TRAILING arguments take "
    "their defaults; an argument with no default must be supplied. The call returns the last "
    "body line's value -- if that line is a SET, the call prints no Out[n], exactly like SET "
    "itself.",
    ["name(arg, ...)"],
    ["def double(x = 2) { x * 2 }\ndouble()",
     "def double(x = 2) { x * 2 }\ndouble(5)",
     'def drop(name, h = 10) { new sphere as name { position = [0, h, 0] } }\n'
     'drop("ball")\nget ball.y',
     'def mk(m = 3) {\n  new sphere as pip { mass = m }\n}\nmk()\nget pip.mass'],
    indexKeys=["<"],
    locations=[{"file": P, "line": 168, "role": "call production"},
               {"file": G, "line": gsec("5.9"), "role": "spec (5.9)"}],
    seeAlso=["cmd.def", "cmd.expr"]))

# ==========================================================================
# SCENE family
# ==========================================================================

SP = "new sphere { mass = 1, radius = 0.5 }\nscene create"
SM = "new point { mass = 1, velocity = [1, 0, 0] }\nscene create"

SPIN = ("# playback advances on WALL-CLOCK time, on its own thread, so a batch\n"
        "# script must actually spend some — otherwise PAUSE lands before the\n"
        "# first frame is recorded and REVERSE has nothing to rewind. The work\n"
        "# below also shows the isolation: the notebook computes while the\n"
        "# window animates, and neither moves the other.\n"
        "def v2(x, y) { 0.5 * (x * x + y * y) }\n"
        "qm2 grid -5 5 40, -5 5 40\n"
        "qm2 potential v2\n"
        "qm2 states 3")

SCENE = [
    ("create", "SCENE CREATE", "SCENE CREATE [ NUMBER ]",
     "Open the scene window. Starts a tiny web server INSIDE posim (pure Rust standard "
     "library, no external dependencies) and opens a page in your browser. Default port: one "
     "chosen by the OS; give a number in 0..=65535 to pin it. A second CREATE is harmless -- "
     "it just reminds you of the URL. Set POSIM_NO_BROWSER=1 to suppress the browser launch; "
     "the URL is always printed.",
     [SP + "\nscene close", "new sphere { mass = 1 }\nscene create 7878\nscene close",
      SP + "\nscene status\nscene close",
      "box 4\nnew point { mass = 1, velocity = [10, 20, 10] }\nscene create\n"
      "scene set_time_step 0.001\nscene status\nscene close"],
     ["SCENE CREATE port must be an integer in 0..=65535"]),
    ("close", "SCENE CLOSE", "SCENE CLOSE",
     "Shut the server down; every window disconnects. DESTROY is an alias.",
     [SP + "\nscene close", SP + "\nscene destroy",
      SP + "\nscene close\nscene create\nscene close", SP + "\nscene status\nscene close"],
     ["no scene window - run SCENE CREATE first"]),
    ("translate", "SCENE TRANSLATE", "SCENE TRANSLATE term term [ term ]",
     "Move the camera's look-at point by (dx, dy, dz) world units; dz defaults to 0. This is "
     "what the arrow keys do in the window. Arguments are TERM-level, so -5 is negative five "
     "rather than a subtraction; a sum must be parenthesised.",
     [SP + "\nscene translate 2 0\nscene close", SP + "\nscene translate 2 0 1\nscene close",
      SP + "\nscene translate 2*2 0 1/2\nscene close",
      SP + "\nscene translate (1 + 0.5) -2\nscene status\nscene close"], []),
    ("rotate", "SCENE ROTATE", "SCENE ROTATE term term",
     "Orbit the camera: yaw (azimuth) and pitch (elevation) in DEGREES. Pitch is clamped to "
     "+/-89 degrees. This is what left-dragging does. Term-level arguments: "
     "`scene rotate 15 -5` is TWO arguments, not one subtraction.",
     [SP + "\nscene rotate 30 -10\nscene close", SP + "\nscene rotate 15 -5\nscene close",
      SP + "\nscene rotate 90 0\nscene status\nscene close",
      SP + "\nscene rotate (10 + 20) -10\nscene status\nscene close"], []),
    ("zoom", "SCENE ZOOM", 'SCENE ZOOM ( "IN" | "OUT" | term )',
     "Zoom by 1.25x, by 1/1.25, or by any factor f > 0 (f > 1 zooms in). This is the mouse "
     "wheel.",
     [SP + "\nscene zoom in\nscene close", SP + "\nscene zoom out\nscene close",
      SP + "\nscene zoom 2\nscene close",
      SP + "\nscene zoom (1 + 0.5)\nscene status\nscene close"], []),
    ("hide", "SCENE HIDE", 'SCENE HIDE [ NUMBER | "ALL" ]',
     "Hide object n, or everything (bare HIDE = HIDE ALL). Hiding blanks the body out of every "
     "connected window WITHOUT deleting anything.",
     [SP + "\nscene hide 0\nscene close", SP + "\nscene hide all\nscene close",
      SP + "\nscene hide\nscene status\nscene close",
      SP + "\nscene hide 0\nscene show all\nscene status\nscene close"], []),
    ("show", "SCENE SHOW", 'SCENE SHOW [ NUMBER | "ALL" ]',
     "Undo SCENE HIDE. Distinct from the top-level SHOW <function>; the parser tells them "
     "apart by position.",
     [SP + "\nscene hide 0\nscene show 0\nscene close",
      SP + "\nscene hide all\nscene show all\nscene close",
      SP + "\nscene show all\nscene status\nscene close",
      SP + "\nscene hide 0\nscene status\nscene show all\nscene status\nscene close"], []),
    ("refresh", "SCENE REFRESH", "SCENE REFRESH",
     "Copy the notebook's current system into the window and CLEAR playback history. This is "
     "how new objects reach an already-open window: the window animates its own synchronized "
     "copy, taken at CREATE and again at every REFRESH.",
     [SP + "\nscene refresh\nscene close",
      SP + "\nnew point { mass = 1 }\nscene refresh\nscene status\nscene close",
      SP + "\nbox 4\nscene refresh\nscene status\nscene close",
      SM + "\nscene start\nscene pause\nscene refresh\nscene status\nscene close"], []),
    ("redraw", "SCENE REDRAW", "SCENE REDRAW",
     "Re-send the complete scene description to every window, forcing a full redraw. Unlike "
     "REFRESH it does not re-sync state or clear history -- it is a display operation only.",
     [SP + "\nscene redraw\nscene close", SP + "\nscene redraw\nscene status\nscene close",
      SP + "\nscene zoom 2\nscene redraw\nscene close",
      SM + "\nscene start\nscene pause\nscene redraw\nscene status\nscene close"], []),
    ("start", "SCENE START", "SCENE START",
     "Begin time-stepped evolution, forward. Playback runs on a BACKGROUND THREAD at about 30 "
     "frames per second, so the notebook prompt never blocks -- and all forward stepping goes "
     "through the same SUNDIALS integrators as STEP/RUN. There is no separate physics engine "
     "in the window.",
     [SM + "\nscene start\nscene pause\nscene close",
      SM + "\nscene start\nscene status\nscene close",
      SM + "\nscene set_time_step 0.005\nscene start\nscene pause\nscene close",
      SM + "\nscene start\nscene pause\nscene status\nget system.time\nscene close"], []),
    ("stop", "SCENE STOP", "SCENE STOP",
     "Halt AND clear the recorded history. A later START begins fresh. Contrast PAUSE, which "
     "keeps history so both START and REVERSE can continue from the freeze point.",
     [SM + "\nscene start\nscene stop\nscene close",
      SM + "\nscene start\nscene stop\nscene status\nscene close",
      SM + "\nscene start\nscene stop\nscene start\nscene pause\nscene close",
      SM + "\nscene start\nscene pause\nscene status\nscene stop\nscene status\nscene close"],
     []),
    ("pause", "SCENE PAUSE", "SCENE PAUSE",
     "Freeze. START resumes and history is KEPT, so REVERSE can also continue from here.",
     [SM + "\nscene start\nscene pause\nscene close",
      SM + "\nscene start\nscene pause\nscene status\nscene close",
      SM + "\nscene start\nscene pause\nscene start\nscene pause\nscene close",
      SM + "\nscene start\n" + SPIN + "\nscene pause\nscene reverse\nscene close"], []),
    ("reverse", "SCENE REVERSE", "SCENE REVERSE",
     "Play BACKWARD in time through the recorded history. While running forward the playback "
     "thread records a snapshot of the whole system before every step (a ring buffer, at most "
     "20 000 frames); REVERSE replays those newest-first, which is an EXACT rewind -- "
     "bit-for-bit the states you already visited, not negative-dt integration. When the buffer "
     "runs out it pauses and sends an event.",
     [SM + "\nscene start\n" + SPIN + "\nscene pause\nscene reverse\nscene close",
      SM + "\nscene set_time_step 0.001\nscene start\n" + SPIN +
      "\nscene pause\nscene reverse\nscene close",
      SM + "\nscene start\n" + SPIN + "\nscene pause\nscene reverse\n"
      "scene pause\nscene start\nscene pause\nscene close",
      SM + "\nscene start\n" + SPIN + "\nscene pause\nscene reverse\n"
      "scene events\nscene close"],
     ["scene: nothing to reverse - no forward history recorded yet (SCENE START first)"]),
    ("reset", "SCENE RESET", "SCENE RESET",
     "Re-initialise the playback: every mutable value and the time return to their initial "
     "values -- the state last synced at CREATE/REFRESH, restored BIT-IDENTICALLY -- history "
     "and the step counter clear, and the mode returns to stopped. START then re-runs the "
     "simulation from the beginning. The window's permanent toolbar Reset button calls the "
     "same primitive. Stronger than STOP, and NOT the top-level RESET, which wipes your "
     "notebook system instead.",
     [SP + "\nscene reset\nscene close",
      SM + "\nscene start\nscene pause\nscene reset\nscene status\nscene close",
      SM + "\nscene start\nscene reset\nscene start\nscene pause\nscene close",
      SM + "\nscene set_time_step 0.005\nscene start\nscene pause\nscene status\n"
      "scene reset\nscene status\nscene close"], []),
    ("set_time_step", "SCENE SET_TIME_STEP", "SCENE SET_TIME_STEP term",
     "Set the playback time step. Must be positive and finite. SETTIMESTEP is an alias. "
     "Term-level argument.",
     [SP + "\nscene set_time_step 0.005\nscene close",
      SP + "\nscene settimestep 0.0002\nscene close",
      SP + "\nscene set_time_step 1/100\nscene close",
      SM + "\nscene set_time_step 0.001\nscene start\nscene pause\nscene status\nscene close"],
     ["scene: set_time_step needs a positive, finite dt"]),
    ("status", "SCENE STATUS", "SCENE STATUS",
     "A four-line report: URL and connected windows; mode/t/dt/steps/history; entities and the "
     "hidden list; camera.",
     [SP + "\nscene status\nscene close", SP + "\nscene zoom 2\nscene status\nscene close",
      SM + "\nscene start\nscene pause\nscene status\nscene close",
      SM + "\nscene start\n" + SPIN + "\nscene pause\nscene status\n"
      "scene reverse\nscene close"],
     ["no scene window - run SCENE CREATE first"]),
    ("events", "SCENE EVENTS", "SCENE EVENTS",
     "Print (and clear) the asynchronous messages the window has sent: JavaScript errors, "
     "connect/disconnect notices, toolbar actions, data requests. Up to 1000 are kept. In "
     "JupyterLab these also arrive BY THEMSELVES, streamed by the kernel's reader thread as "
     "[scene] ... lines, without you asking.",
     [SP + "\nscene events\nscene close", SP + "\nscene events\nscene events\nscene close",
      SM + "\nscene start\nscene pause\nscene events\nscene close",
      SM + "\nscene start\n" + SPIN + "\nscene pause\nscene reverse\n"
      "scene events\nscene close"], []),
]

for sid, nm, syn, defn, rungs, errs in SCENE:
    aliases = []
    if sid == "close":
        aliases = ["SCENE DESTROY"]
    if sid == "set_time_step":
        aliases = ["SCENE SETTIMESTEP"]
    out.append(E(
        "cmd.scene." + sid, nm, defn.split(".")[0] + ".", defn, [syn], rungs,
        aliases=aliases, indexKeys=["S"], errors=errs,
        locations=[{"file": P, "line": 33, "role": "scenecmd production"},
                   {"file": "posim/src/scene/mod.rs", "line": 1, "role": "the server"},
                   {"file": "scene_info.md", "line": 287, "role": "spec (section 4)"}],
        seeAlso=["cmd.scene.create", "cmd.scene.status"],
        invariants=["The window evolves a synchronized COPY: notebook STEP/RUN never move the "
                    "window, and window playback never moves the notebook."]))

# ==========================================================================
# QM / QM2 / QM3 families
# ==========================================================================

QM_PRE = "qm grid -8 8 60\nqm potential zero\n"
QM_OSC = "def v(x) { 0.5 * x * x }\nqm grid -8 8 60\nqm potential v\n"
QM_PKT = QM_PRE + "qm packet -2 1 2\n"
Q2 = "qm2 grid -4 4 20, -4 4 20\nqm2 potential zero\n"
Q2P = Q2 + "qm2 packet 0 0, 1 1, 1 0\n"
Q2OSC = "def v2(x, y) { 0.5 * (x * x + y * y) }\nqm2 grid -4 4 24, -4 4 24\nqm2 potential v2\n"
Q3 = "qm3 grid -3 3 10, -3 3 10, -3 3 10\nqm3 potential zero\n"
Q3P = Q3 + "qm3 packet 0 0 0, 1 1 1, 1 0 0\n"
Q3OSC = ("def v3(x, y, z) { 0.5 * (x * x + y * y + z * z) }\n"
         "qm3 grid -4 4 14, -4 4 14, -4 4 14\nqm3 potential v3\n")

QM_DOC = {
 "status": "Report what is currently set up.",
 "grid": "The domain and its n interior points. Under CAYLEY the walls are INFINITE and psi "
         "is pinned to zero just outside -- exactly right for bound states, and a trap for "
         "scattering.",
 "potential": "Set the potential. ZERO is a free particle; BARRIER v0, x1, x2 and WELL depth, "
              "x1, x2 are built in BECAUSE the language has no conditionals, so a piecewise "
              "potential cannot be a DEF at all; any smooth potential should be a DEF. The "
              "potential is sampled when the command runs -- editing the function afterwards "
              "changes nothing until you re-issue it.",
 "mass": "The particle mass. Defaults to 1.",
 "hbar": "Planck's constant over 2*pi. Defaults to 1.",
 "method": "Choose the propagator -- and with it the BOUNDARY CONDITION. CAYLEY is "
           "Crank-Nicolson on the Dirichlet grid: the walls reflect. NASH is the "
           "Bessel-stencil split-operator scheme ported from the original C++, and it is "
           "PERIODIC. The grid points and potential samples are identical either way, so "
           "switching moves nothing; only the two ends change meaning. NASH alone is first "
           "order in dt (what the original does, so it is the default); NASH STRANG is second "
           "order for essentially the same cost.",
 "states": "The k lowest bound-state energies. In 1-D by cyclic Jacobi on a dense n x n "
           "matrix -- O(n^3), comfortable to a few hundred points; in 2-D and 3-D by "
           "matrix-free Lanczos with full reorthogonalisation AND deflation, because a Krylov "
           "space built from one starting vector contains exactly ONE direction from each "
           "degenerate eigenspace. REFUSED while the method is NASH (bound states need "
           "Dirichlet walls) or while an absorber is on (H - iW is not Hermitian).",
 "state": "Load bound state n as psi. A stationary state is stationary, so loading one and "
          "propagating it must change nothing -- the sharpest available test of the "
          "eigensolver and the propagator together.",
 "packet": "A normalised Gaussian wavepacket. Watches for probability accumulating within 5% "
           "of a wall and warns.",
 "step": "Propagate one step with the current method. Unitary at any step size.",
 "run": "Propagate for t, reporting at n output points. Prints <E> and the norm drift -- which "
        "measures the linear solver rather than dt, because the Cayley operator is unitary for "
        "ANY step.",
 "norm": "The norm of psi. Holds at 1 to machine precision unless an absorber is on.",
 "energy": "<psi|H|psi>/<psi|psi>. Divided by the norm on purpose, so an absorber does not "
           "look like an energy leak.",
 "position": "<x>.",
 "momentum": "<p>.",
 "prob": "Probability of being found in the given interval, rectangle or box.",
 "density": "|psi|^2 as a list.",
 "drive": "Make the potential time-dependent: V(x,t) = V0(x) + f(t)*g(x), from two DEF'd "
          "functions. FACTORISED rather than a general V(x,t) because a general form would "
          "need re-sampling every grid point on every step. The modulation is sampled at the "
          "MIDPOINT of each step, which keeps the scheme second order. Energy is then NOT "
          "conserved -- a driven system trades energy with its drive -- but propagation stays "
          "unitary.",
 "absorb": "Absorbing edges -- a complex absorbing potential, H - iW with W >= 0 ramping up "
           "smoothly over `width` at each edge, so probability arriving there is DRAINED "
           "instead of bounced. Propagation is then NOT unitary (the norm decays, which is the "
           "absorber working) and STATES is refused. The tuning is real: too weak and the "
           "packet reaches the wall, too strong and it reflects off the absorber's own leading "
           "edge.",
 "animate": "Propagate and write a self-contained HTML animation -- nothing fetched from the "
            "network. In 3-D it shows the three MARGINAL densities P(x,y), P(x,z), P(y,z), "
            "each a genuine observable rather than a rendering convention.",
 "reset": "Forget the quantum problem.",
 "transmission": "T(E) and R(E) at one energy, by TRANSFER MATRIX: exact, time-independent, no "
                 "packet and no time stepping.",
 "scan": "Scan T(E) over an energy range and report the resonances, pushing the transmission "
         "list. Resolves peaks narrower than any affordable wavepacket's own momentum spread.",
 "centroid": "The centroid <r> of the packet.",
 "iso": "Extract the surface where |psi|^2 equals a chosen fraction of its peak (default 0.25) "
        "and ship a rotatable view. MARCHING TETRAHEDRA, not marching cubes: marching cubes "
        "needs a 256-entry table which IS the algorithm, so one wrong entry gives a surface "
        "with a hole that looks fine from most angles. A tetrahedron has only 16 sign patterns, "
        "reducing to three cases derivable in a sentence, and the shared-face ambiguity cannot "
        "arise. Rendered by a software rasteriser on a 2-D canvas rather than WebGL, which can "
        "fail silently where there is no GPU. QM2 has no ISO -- a 2-D density is already "
        "drawable as a heat map — stated in grammar.md \u00a75.12 and gated by "
        "`every_qm2_subcommand_is_documented_in_lockstep`.",
}

LADDER = {"qm": {}, "qm2": {}, "qm3": {}}

LADDER["qm"] = {
 "status": ["qm", QM_PRE + "qm", QM_OSC + "qm", QM_PKT + "qm"],
 "grid": ["qm grid -8 8 60", "qm grid -100 100 500", "qm grid -8 8 60\nqm grid -4 4 30",
          QM_OSC + "qm"],
 "potential": ["qm grid -8 8 60\nqm potential zero",
               "qm grid -8 8 60\nqm potential barrier 2.5, 0, 1",
               "qm grid -8 8 60\nqm potential well 5, -2, 2", QM_OSC + "qm states 2"],
 "mass": ["qm mass 1", "qm grid -8 8 60\nqm mass 2\nqm", QM_OSC + "qm mass 1\nqm states 2",
          QM_OSC + "qm mass 2\nqm states 2"],
 "hbar": ["qm hbar 1", "qm grid -8 8 60\nqm hbar 1\nqm", QM_OSC + "qm hbar 1\nqm states 2",
          QM_OSC + "qm hbar 1\nqm states 1\nqm state 0\nqm energy"],
 "method": ["qm method cayley", "qm method nash", "qm method nash strang",
            QM_PRE + "qm method nash strang\nqm packet -2 1 2\nqm run 0.2 steps 20"],
 "states": [QM_OSC + "qm states 1", QM_OSC + "qm states 4",
            QM_OSC + "qm states 2\nqm state 0\nqm energy",
            QM_OSC + "qm states 4\nqm state 1\nqm run 1 steps 100\nqm energy"],
 "state": [QM_OSC + "qm states 2\nqm state 0", QM_OSC + "qm states 2\nqm state 1\nqm norm",
           QM_OSC + "qm states 2\nqm state 1\nqm energy",
           QM_OSC + "qm states 2\nqm state 1\nqm run 1 steps 100\nqm energy"],
 "packet": [QM_PRE + "qm packet 0 1 1", QM_PRE + "qm packet -2 1 2\nqm norm",
            QM_PRE + "qm packet -2 1 2\nqm position",
            QM_PKT + "qm run 0.5 steps 50\nqm position\nqm momentum"],
 "step": [QM_PKT + "qm step 0.01", QM_PKT + "qm step 0.01\nqm norm",
          QM_PKT + "qm step 0.01\nqm step 0.01\nqm energy",
          QM_OSC + "qm states 2\nqm state 1\nqm step 0.01\nqm energy"],
 "run": [QM_PKT + "qm run 0.2 steps 20", QM_PKT + "qm run 0.5 steps 50\nqm norm",
         QM_OSC + "qm states 2\nqm state 1\nqm run 1 steps 100\nqm energy",
         QM_PKT + "qm run 1 steps 100\nqm prob -8 0\nqm prob 0 8"],
 "norm": [QM_PKT + "qm norm", QM_PKT + "qm run 0.2 steps 20\nqm norm",
          QM_OSC + "qm states 2\nqm state 0\nqm norm", QM_PKT + "qm run 1 steps 100\nqm norm"],
 "energy": [QM_PKT + "qm energy", QM_OSC + "qm states 2\nqm state 0\nqm energy",
            QM_OSC + "qm states 2\nqm state 1\nqm energy",
            QM_OSC + "qm states 2\nqm state 1\nqm run 1 steps 100\nqm energy"],
 "position": [QM_PKT + "qm position", QM_PKT + "qm run 0.2 steps 20\nqm position",
              QM_PRE + "qm packet 0 1 0\nqm position", QM_PKT + "qm run 1 steps 100\nqm position"],
 "momentum": [QM_PKT + "qm momentum", QM_PRE + "qm packet 0 1 2\nqm momentum",
              QM_PKT + "qm run 0.2 steps 20\nqm momentum",
              QM_OSC + "qm states 2\nqm state 0\nqm momentum"],
 "prob": [QM_PKT + "qm prob -8 8", QM_PKT + "qm prob -8 0", QM_PKT + "qm prob -8 0\nqm prob 0 8",
          QM_PKT + "qm run 1 steps 100\nqm prob -8 0\nqm prob 0 8"],
 "density": [QM_PKT + "qm density", QM_PRE + "qm packet 0 1 0\nqm density",
             QM_PKT + "qm run 0.2 steps 20\nqm density",
             QM_OSC + "qm states 2\nqm state 0\nqm density"],
 "drive": ["def g(x) { x }\ndef f(t) { 0.3 * cos(0.7 * t) }\n" + QM_OSC +
           "qm states 2\nqm state 0\nqm drive g f",
           "def g(x) { x }\ndef f(t) { 0.3 * cos(0.7 * t) }\n" + QM_OSC +
           "qm states 2\nqm state 0\nqm drive g f\nqm drive off",
           "def g(x) { x }\ndef f(t) { 0.3 * cos(0.7 * t) }\n" + QM_OSC +
           "qm states 2\nqm state 0\nqm drive g f\nqm run 1 steps 100\nqm position",
           "def g(x) { x }\ndef f(t) { 0.3 * cos(0.7 * t) }\n" + QM_OSC +
           "qm states 2\nqm state 0\nqm drive g f\nqm run 5 steps 500\nqm position\nqm norm"],
 "absorb": [QM_PKT + "qm absorb 3 1", QM_PKT + "qm absorb 3 1\nqm absorb off",
            QM_PKT + "qm absorb 3 1 2\nqm run 0.5 steps 50\nqm norm",
            QM_PKT + "qm absorb 3 3\nqm run 1 steps 100\nqm norm\nqm prob -8 8"],
 "animate": [QM_PKT + 'qm animate "scatter.html" 0.2 frames 5',
             QM_PKT + 'qm animate "scatter.html" 0.5 frames 10',
             QM_OSC + 'qm states 2\nqm state 0\nqm animate "scatter.html" 1 frames 10',
             QM_PKT + 'qm animate "scatter.html" 1 frames 20'],
 "reset": [QM_PKT + "qm reset", QM_PKT + "qm reset\nqm", QM_OSC + "qm states 2\nqm reset\nqm",
           QM_PKT + "qm reset\nqm grid -4 4 30\nqm"],
 "transmission": ["qm grid -60 60 600\nqm potential barrier 2.5, 0, 1\nqm transmission 2",
                  "qm grid -60 60 600\nqm potential barrier 2.5, 0, 1\nqm transmission 1",
                  "qm grid -60 60 600\nqm potential barrier 2.5, 0, 1\nqm transmission 2\n"
                  "qm transmission 3",
                  "def barrier(x) { 2.5 * (x > 0) * (x < 1) }\nqm grid -60 60 600\n"
                  "qm potential barrier\nqm transmission 2"],
 "scan": ["qm grid -60 60 600\nqm potential barrier 2.5, 0, 1\nqm scan 0.5 4 20",
          "qm grid -60 60 600\nqm potential barrier 2.5, 0, 1\nqm scan 1 3 10",
          "def double(x) { 2.5 * ((x > 0) * (x < 1) + (x > 3) * (x < 4)) }\n"
          "qm grid -60 60 600\nqm potential double\nqm scan 0.5 4 30",
          "qm grid -60 60 600\nqm potential barrier 2.5, 0, 1\nqm scan 0.5 4 40"],
}

LADDER["qm2"] = {
 "status": ["qm2", Q2 + "qm2", Q2P + "qm2", Q2P + "qm2 norm\nqm2"],
 "grid": ["qm2 grid -4 4 20, -4 4 20", "qm2 grid -6 6 30, -6 6 30",
          "qm2 grid -4 4 20, -4 4 20\nqm2 grid -2 2 16, -2 2 16", Q2 + "qm2"],
 "potential": ["qm2 grid -4 4 20, -4 4 20\nqm2 potential zero", Q2OSC + "qm2",
               Q2OSC + "qm2 states 2",
               "def openings(y) { (y > 1.5) * (y < 2.5) + (y > -2.5) * (y < -1.5) }\n"
               "def slit(x, y) { 60 * (x > 0) * (x < 0.4) * (1 - openings(y)) }\n"
               "qm2 grid -6 6 30, -6 6 30\nqm2 potential slit"],
 "packet": [Q2 + "qm2 packet 0 0, 1 1, 0 0", Q2P + "qm2 norm", Q2P + "qm2 centroid",
            Q2P + "qm2 run 0.2 steps 10\nqm2 centroid"],
 "step": [Q2P + "qm2 step 0.01", Q2P + "qm2 step 0.01\nqm2 norm",
          Q2P + "qm2 step 0.01\nqm2 step 0.01\nqm2 energy",
          Q2P + "qm2 step 0.05\nqm2 norm\nqm2 centroid"],
 "run": [Q2P + "qm2 run 0.2 steps 10", Q2P + "qm2 run 0.2 steps 10\nqm2 norm",
         Q2P + "qm2 run 0.5 steps 20\nqm2 centroid",
         Q2P + "qm2 run 0.5 steps 20\nqm2 energy\nqm2 norm"],
 "norm": [Q2P + "qm2 norm", Q2P + "qm2 run 0.2 steps 10\nqm2 norm",
          Q2P + "qm2 step 0.01\nqm2 norm", Q2P + "qm2 run 0.5 steps 20\nqm2 norm"],
 "energy": [Q2P + "qm2 energy", Q2P + "qm2 run 0.2 steps 10\nqm2 energy",
            Q2OSC + "qm2 states 1\nqm2 state 0\nqm2 energy",
            Q2P + "qm2 run 0.5 steps 20\nqm2 energy"],
 "centroid": [Q2P + "qm2 centroid", Q2P + "qm2 run 0.2 steps 10\nqm2 centroid",
              Q2 + "qm2 packet 1 0, 1 1, 0 0\nqm2 centroid",
              Q2P + "qm2 run 0.5 steps 20\nqm2 centroid"],
 "prob": [Q2P + "qm2 prob -4 4, -4 4", Q2P + "qm2 prob -1 1, -1 1", Q2P + "qm2 prob 0 4, -4 4",
          Q2P + "qm2 run 0.2 steps 10\nqm2 prob 0 4, -4 4"],
 "absorb": [Q2P + "qm2 absorb 1 1", Q2P + "qm2 absorb 1 1\nqm2 absorb off",
            Q2P + "qm2 absorb 1 1 2\nqm2 run 0.2 steps 10\nqm2 norm",
            Q2P + "qm2 absorb 1 3\nqm2 run 0.5 steps 20\nqm2 norm"],
 "drive": ["def g2(x, y) { x }\ndef f2(t) { 0.3 * cos(t) }\n" + Q2P + "qm2 drive g2 f2",
           "def g2(x, y) { x }\ndef f2(t) { 0.3 * cos(t) }\n" + Q2P +
           "qm2 drive g2 f2\nqm2 drive off",
           "def g2(x, y) { x }\ndef f2(t) { 0.3 * cos(t) }\n" + Q2P +
           "qm2 drive g2 f2\nqm2 run 0.2 steps 10\nqm2 centroid",
           "def g2(x, y) { x }\ndef f2(t) { 0.3 * cos(t) }\n" + Q2P +
           "qm2 drive g2 f2\nqm2 run 0.5 steps 20\nqm2 norm"],
 "states": [Q2OSC + "qm2 states 1", Q2OSC + "qm2 states 3",
            "def v2(x, y) { 0.5 * (x * x + y * y) }\nqm2 grid -5 5 30, -5 5 30\n"
            "qm2 potential v2\nqm2 states 3",
            Q2OSC + "qm2 states 3\nqm2 state 0\nqm2 energy"],
 "state": [Q2OSC + "qm2 states 1\nqm2 state 0", Q2OSC + "qm2 states 2\nqm2 state 1\nqm2 norm",
           Q2OSC + "qm2 states 1\nqm2 state 0\nqm2 energy",
           Q2OSC + "qm2 states 1\nqm2 state 0\nqm2 run 0.2 steps 10\nqm2 energy"],
 "reset": [Q2P + "qm2 reset", Q2P + "qm2 reset\nqm2", Q2 + "qm2 reset\nqm2",
           Q2P + "qm2 reset\nqm2 grid -2 2 12, -2 2 12"],
 "animate": [Q2P + 'qm2 animate "double_slit.html" 0.1 frames 4',
             Q2P + 'qm2 animate "double_slit.html" 0.2 frames 6',
             Q2P + 'qm2 animate "double_slit.html" 0.3 frames 8',
             Q2P + 'qm2 animate "double_slit.html" 0.4 frames 10'],
}

LADDER["qm3"] = {
 "status": ["qm3", Q3 + "qm3", Q3P + "qm3", Q3P + "qm3 norm\nqm3"],
 "grid": ["qm3 grid -3 3 10, -3 3 10, -3 3 10", "qm3 grid -4 4 12, -4 4 12, -4 4 12",
          "qm3 grid -3 3 10, -3 3 10, -3 3 10\nqm3 grid -2 2 8, -2 2 8, -2 2 8", Q3 + "qm3"],
 "potential": ["qm3 grid -3 3 10, -3 3 10, -3 3 10\nqm3 potential zero", Q3OSC + "qm3",
               Q3OSC + "qm3 states 1", Q3OSC + "qm3 states 2"],
 "packet": [Q3 + "qm3 packet 0 0 0, 1 1 1, 0 0 0", Q3P + "qm3 norm", Q3P + "qm3 centroid",
            Q3P + "qm3 run 0.1 steps 5\nqm3 centroid"],
 "states": [Q3OSC + "qm3 states 1", Q3OSC + "qm3 states 2",
            Q3OSC + "qm3 states 2\nqm3 state 0\nqm3 energy",
            Q3OSC + "qm3 states 1\nqm3 state 0\nqm3 norm"],
 "state": [Q3OSC + "qm3 states 1\nqm3 state 0", Q3OSC + "qm3 states 1\nqm3 state 0\nqm3 norm",
           Q3OSC + "qm3 states 1\nqm3 state 0\nqm3 energy",
           Q3OSC + "qm3 states 1\nqm3 state 0\nqm3 run 0.1 steps 5\nqm3 energy"],
 "step": [Q3P + "qm3 step 0.01", Q3P + "qm3 step 0.01\nqm3 norm",
          Q3P + "qm3 step 0.01\nqm3 step 0.01\nqm3 energy",
          Q3P + "qm3 step 0.02\nqm3 norm\nqm3 centroid"],
 "run": [Q3P + "qm3 run 0.1 steps 5", Q3P + "qm3 run 0.1 steps 5\nqm3 norm",
         Q3P + "qm3 run 0.2 steps 10\nqm3 centroid",
         Q3P + "qm3 run 0.2 steps 10\nqm3 energy\nqm3 norm"],
 "norm": [Q3P + "qm3 norm", Q3P + "qm3 run 0.1 steps 5\nqm3 norm",
          Q3P + "qm3 step 0.01\nqm3 norm", Q3P + "qm3 run 0.2 steps 10\nqm3 norm"],
 "energy": [Q3P + "qm3 energy", Q3P + "qm3 run 0.1 steps 5\nqm3 energy",
            Q3P + "qm3 step 0.01\nqm3 energy", Q3P + "qm3 run 0.2 steps 10\nqm3 energy"],
 "centroid": [Q3P + "qm3 centroid", Q3P + "qm3 run 0.1 steps 5\nqm3 centroid",
              Q3 + "qm3 packet 1 0 0, 1 1 1, 0 0 0\nqm3 centroid",
              Q3P + "qm3 run 0.2 steps 10\nqm3 centroid"],
 "prob": [Q3P + "qm3 prob -3 3, -3 3, -3 3", Q3P + "qm3 prob -1 1, -1 1, -1 1",
          Q3P + "qm3 prob 0 3, -3 3, -3 3", Q3P + "qm3 run 0.1 steps 5\nqm3 prob 0 3, -3 3, -3 3"],
 "absorb": [Q3P + "qm3 absorb 1 1", Q3P + "qm3 absorb 1 1\nqm3 absorb off",
            Q3P + "qm3 absorb 1 1 2\nqm3 run 0.1 steps 5\nqm3 norm",
            Q3P + "qm3 absorb 1 3\nqm3 run 0.2 steps 10\nqm3 norm"],
 "drive": ["def g3(x, y, z) { x }\ndef f3(t) { 0.3 * cos(t) }\n" + Q3P + "qm3 drive g3 f3",
           "def g3(x, y, z) { x }\ndef f3(t) { 0.3 * cos(t) }\n" + Q3P +
           "qm3 drive g3 f3\nqm3 drive off",
           "def g3(x, y, z) { x }\ndef f3(t) { 0.3 * cos(t) }\n" + Q3P +
           "qm3 drive g3 f3\nqm3 run 0.1 steps 5\nqm3 centroid",
           "def g3(x, y, z) { x }\ndef f3(t) { 0.3 * cos(t) }\n" + Q3P +
           "qm3 drive g3 f3\nqm3 run 0.2 steps 10\nqm3 norm"],
 "animate": [Q3P + 'qm3 animate "vol3d.html" 0.1 frames 3',
             Q3P + 'qm3 animate "vol3d.html" 0.15 frames 4',
             Q3P + 'qm3 animate "vol3d.html" 0.2 frames 5',
             Q3P + 'qm3 animate "vol3d.html" 0.25 frames 6'],
 "iso": [Q3P + 'qm3 iso "iso3d.html" 0.1 frames 3',
         Q3P + 'qm3 iso "iso3d.html" 0.1 frames 3 level 0.25',
         Q3P + 'qm3 iso "iso3d.html" 0.15 frames 4 level 0.4',
         Q3P + 'qm3 iso "iso3d.html" 0.2 frames 5 level 0.1'],
 "reset": [Q3P + "qm3 reset", Q3P + "qm3 reset\nqm3", Q3 + "qm3 reset\nqm3",
           Q3P + "qm3 reset\nqm3 grid -2 2 8, -2 2 8, -2 2 8"],
}

EBNF_LINE = {"qm": 81, "qm2": 125, "qm3": 139}

cmds = json.load(open("index_data/commands.json"))
for fam in ("qm", "qm2", "qm3"):
    for sub in cmds[fam + "_subcommands"]:
        canon = sub.split("|")[0]
        aliases = [fam.upper() + " " + a.upper() for a in sub.split("|")[1:]]
        rungs = LADDER[fam].get(canon, [])
        doc = QM_DOC.get(canon, "")
        note = ""
        if aliases:
            note = ("  `" + aliases[0] + "` is an accepted alias, documented in "
                    "grammar.md since the D3 fix.")
        out.append(E(
            "cmd." + fam + "." + canon, fam.upper() + " " + canon.upper(),
            (doc.split(".")[0] + "." if doc else fam.upper() + " " + canon.upper()),
            doc + note, [fam.upper() + " " + canon.upper() + " ..."], rungs,
            aliases=aliases, indexKeys=["Q"],
            errors=(["QM STATES is unavailable while the absorber is on"]
                    if canon == "states" else []),
            locations=[{"file": P, "line": EBNF_LINE[fam], "role": fam + "cmd EBNF"},
                       {"file": "posim/src/" + fam + ".rs", "line": 1, "role": "implementation"},
                       {"file": G, "line": gsec("5.10"), "role": "spec (5.10-5.12)"}],
            seeAlso=["kw." + fam, "cmd.def"],
            invariants=["Separate negative arguments with COMMAS: `well 5 -2 2` reads `5 - 2` "
                        "as subtraction and then finds two arguments where three were wanted."]))

# ==========================================================================
# machine-mode ops
# ==========================================================================

MACHINE_DOC = {
 "exec": "Execute one or more command lines. Multi-line cells run one line at a time, stopping "
         "at the first error. This is what the JupyterLab kernel sends for every shift-entered "
         "cell, and a multi-line DEF arrives intact as embedded newlines.",
 "get": "Read a field by path, returning a typed JSON result.",
 "set": "Write a field by path from a JSON value.",
 "state": "Dump the whole system as JSON -- objects, shapes, box, wall flags, inverse_mass.",
 "help": "Return HELP_TEXT.",
 "quit": "End the session.",
}

M_LADDER = {
 "exec": [
  "{\"op\":\"exec\",\"code\":\"new sphere { mass = 2, radius = 0.5 }\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1, velocity = [0, 1, 0] }\"}\n{\"op\":\"exec\",\"code\":\"energy\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"def mk(m = 3) {\\n  new sphere as pip { mass = m }\\n}\"}\n{\"op\":\"exec\",\"code\":\"mk()\"}\n{\"op\":\"get\",\"path\":\"pip.mass\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"set system.g_constant = 0\"}\n{\"op\":\"exec\",\"code\":\"new sphere { mass = 1, radius = 0.5, position = [-2, 0, 0], velocity = [1, 0, 0] }\"}\n{\"op\":\"exec\",\"code\":\"new sphere { mass = 1, radius = 0.5, position = [2, 0, 0], velocity = [-1, 0, 0] }\"}\n{\"op\":\"exec\",\"code\":\"run 2 steps 2\"}\n{\"op\":\"get\",\"path\":\"contact0.normal\"}\n{\"op\":\"quit\"}"
 ],
 "get": [
  "{\"op\":\"exec\",\"code\":\"new point { mass = 2, velocity = [3, 0, 0] }\"}\n{\"op\":\"get\",\"path\":\"obj0.momentum\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 2 }\"}\n{\"op\":\"get\",\"path\":\"obj0.mass\"}\n{\"op\":\"get\",\"path\":\"obj0.inverse_mass\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new sphere as ball { mass = 2, radius = 0.5 }\"}\n{\"op\":\"get\",\"path\":\"ball.inertia_tensor\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1, position = [1, 2, 3] }\"}\n{\"op\":\"get\",\"path\":\"obj0.position\"}\n{\"op\":\"get\",\"path\":\"obj0.position.y\"}\n{\"op\":\"get\",\"path\":\"system.count\"}\n{\"op\":\"quit\"}"
 ],
 "set": [
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1 }\"}\n{\"op\":\"set\",\"path\":\"obj0.mass\",\"value\":5}\n{\"op\":\"get\",\"path\":\"obj0.mass\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1 }\"}\n{\"op\":\"set\",\"path\":\"obj0.velocity\",\"value\":[1,2,3]}\n{\"op\":\"get\",\"path\":\"obj0.momentum\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new sphere { mass = 4, radius = 0.5 }\"}\n{\"op\":\"set\",\"path\":\"obj0.inverse_mass\",\"value\":0}\n{\"op\":\"get\",\"path\":\"obj0.mass\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1, velocity = [0, 1, 0] }\"}\n{\"op\":\"get\",\"path\":\"obj0.momentum\"}\n{\"op\":\"set\",\"path\":\"obj0.mass\",\"value\":5}\n{\"op\":\"get\",\"path\":\"obj0.momentum\"}\n{\"op\":\"get\",\"path\":\"obj0.velocity\"}\n{\"op\":\"quit\"}"
 ],
 "state": [
  "{\"op\":\"state\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new sphere { mass = 2, radius = 0.5 }\"}\n{\"op\":\"state\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"box 4\"}\n{\"op\":\"state\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"set system.g_constant = 0\"}\n{\"op\":\"exec\",\"code\":\"new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\"}\n{\"op\":\"exec\",\"code\":\"run 1 steps 2\"}\n{\"op\":\"state\"}\n{\"op\":\"quit\"}"
 ],
 "help": [
  "{\"op\":\"help\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"help\"}\n{\"op\":\"exec\",\"code\":\"funcs\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"def probe(m = 2) { new sphere { mass = m } }\"}\n{\"op\":\"help\"}\n{\"op\":\"exec\",\"code\":\"funcs\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"help\"}\n{\"op\":\"exec\",\"code\":\"collide\"}\n{\"op\":\"exec\",\"code\":\"box\"}\n{\"op\":\"quit\"}"
 ],
 "quit": [
  "{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1 }\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"new point { mass = 1 }\"}\n{\"op\":\"get\",\"path\":\"obj0.mass\"}\n{\"op\":\"quit\"}",
  "{\"op\":\"exec\",\"code\":\"energy\"}\n{\"op\":\"state\"}\n{\"op\":\"quit\"}"
 ]
}

for m in cmds["machine_ops"]:
    op = m["op"]
    out.append(E(
        "cmd.machine." + op, '{"op":"' + op + '"}',
        MACHINE_DOC[op].split(".")[0] + ".",
        MACHINE_DOC[op] + "  Machine mode (posim --machine) speaks one JSON document per line "
        "over stdin/stdout and replies with one per line: "
        '{"display": ..., "ok": true, "result": ...}. When a scene window is open, posim also '
        'pushes asynchronous {"event": ...} lines that are NOT replies to any request; the '
        "JupyterLab kernel has a reader thread that streams them into your notebook as "
        "[scene] ... lines the moment they happen.",
        [m["shape"],
         "printf '%s\\n' '" + m["shape"] + "' | posim --machine"],
        M_LADDER[op],
        medium="machine", runner="posim --machine",
        indexKeys=["{"],
        locations=[{"file": "posim/src/machine.rs", "line": 1, "role": "the protocol"},
                   {"file": "jupyter/README.md", "line": 53, "role": "spec"}],
        seeAlso=["cmd.machine.exec" if op != "exec" else "cmd.machine.get"]))

json.dump(out, open("index_data/entries_commands.json", "w"), indent=1)
print(str(len(out)) + " command entries -> index_data/entries_commands.json")
