#!/usr/bin/env python3
"""Phase-3 catalog builder: Tier-A entries for the Index of Entities.

Reads the Phase-2 inventories in index_data/ and emits index_data/catalog.json,
an array of entries conforming to prompt_01.md section 5.

Examples are DRAFTED here and VERIFIED in Phase 4: every example carries
`expected: null` and `verified: null` until tools/verify_index_examples.py
runs it and fills them in. Nothing in this file predicts output.

Stdlib only.
"""

import json
import os
import re
import sys

D = "index_data"

# --------------------------------------------------------------------------
# shared example scaffolding — every fragment is self-contained and starts
# from a clean session, because a reader pastes one example, not a sequence.
# --------------------------------------------------------------------------

MAKER = {
    "all":       "new sphere { mass = 2, radius = 0.5 }",
    "sphere":    "new sphere { mass = 2, radius = 0.5 }",
    "point":     "new point { mass = 1 }",
    "cuboid":    "new cuboid { mass = 3, half_extents = [0.5, 1, 2] }",
    "torus":     "new torus { mass = 1, ring_radius = 1.5, tube_radius = 0.5 }",
    "disk":      "new disk { mass = 1, radius = 1 }",
    "cylinder":  "new cylinder { mass = 2, radius = 0.5, height = 1.5 }",
    "dumbbell":  "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }",
    "sphere, disk, cylinder": "new sphere { mass = 2, radius = 0.5 }",
}

# Some fields have a range the generic sample would violate: writing
# tube_radius = 3 on a ring-1.5 torus asks for inner = -1.5, and
# restitution = 3 is not a restitution. Per-field overrides keep every
# generated SET example inside the field's own domain.
PROP_SAMPLE = {
    "tube_radius": "0.25",
    "inner_radius": "0.5",
    "outer_radius": "3",
    "restitution": "0.8",
    "rod_radius": "0.05",
    "r1": "0.2",
    "r2": "0.2",
    "length": "2",
}

SAMPLE = {           # a plausible literal to write into a field of each type
    "number": "3",
    "vec3": "[1, 2, 3]",
    "quaternion": "[1, 0, 0, 0]",
    "mat3": "[[2, 0, 0], [0, 3, 0], [0, 0, 4]]",
    "string": '"x"',
}

TWO_BALLS = (
    "set system.g_constant = 0\n"
    "new sphere { mass = 1, radius = 0.5, position = [-2, 0, 0], velocity = [1, 0, 0] }\n"
    "new sphere { mass = 1, radius = 0.5, position = [2, 0, 0], velocity = [-1, 0, 0] }\n"
    "run 2 steps 2"
)


# The media this project EXECUTES, as opposed to quotes or points at.
EXECUTED_MEDIA = ("posim", "rust", "machine")

LEVELS = ["trivial", "intermediate", "advanced", "expert"]


def ex(level, code, medium="posim", runner="posim --script"):
    return {"level": level, "medium": medium, "code": code.strip(),
            "expected": None, "verified": None, "runner": runner}


def bucket(name):
    """Home-screen bucket for a display name (prompt_01 section 6.2)."""
    c = name.lstrip("<").lstrip()[:1]
    if not c:
        return "Σ"
    if c in "%_.":
        return c
    if c.isdigit():
        return c
    if c.isalpha():
        return c.upper()
    return "Σ"


def entry(**kw):
    e = {"id": None, "name": None, "kind": None, "tier": "A", "aliases": [],
         "indexKeys": [], "summary": "", "definition": "", "syntax": [],
         "parameters": [], "returns": None, "errors": [], "locations": [],
         "examples": [], "seeAlso": [], "invariants": [], "status": "complete"}
    e.update(kw)
    if not e["indexKeys"]:
        e["indexKeys"] = [bucket(e["name"])]
    # levels must be contiguous and increasing: a skipped rung (a number-typed
    # field with no alias, say) would otherwise leave a gap like
    # trivial/intermediate/expert.
    for i, x in enumerate(e["examples"]):
        x["level"] = LEVELS[i] if i < len(LEVELS) else f"expert+{i - 3}"
    return e


# --------------------------------------------------------------------------
# 1. keywords
# --------------------------------------------------------------------------

KEYWORD_USE = {
    "NEW": "new sphere { mass = 2, radius = 0.5 }",
    "SET": "new point { mass = 1 }\nset obj0.mass = 5\nget obj0.mass",
    "GET": "new point { mass = 1 }\nget obj0.mass",
    "DEL": "new point { mass = 1 }\nnew point { mass = 2 }\ndel 0\nlist",
    "LIST": "new sphere { mass = 2, radius = 0.5 }\nlist",
    "STEP": "new point { mass = 1, velocity = [1, 0, 0] }\nstep 1\nget obj0.position",
    "RUN": "new point { mass = 1, velocity = [1, 0, 0] }\nrun 2 steps 2",
    "STEPS": "new point { mass = 1, velocity = [1, 0, 0] }\nrun 2 steps 4",
    "METHOD": "new point { mass = 1, velocity = [0, 1, 0] }\nmethod bdf",
    "ADAMS": "method adams",
    "BDF": "method bdf",
    "SPRK": "new point { mass = 1, velocity = [0, 1, 0] }\nmethod sprk leapfrog_2_2 0.01\nrun 1 steps 2",
    "ENERGY": "new point { mass = 2, velocity = [3, 0, 0] }\nenergy",
    "COM": "new point { mass = 1, position = [2, 0, 0] }\nnew point { mass = 1, position = [-2, 0, 0] }\ncom",
    "MOMENTUM": "new point { mass = 2, velocity = [3, 0, 0] }\nmomentum",
    "ANGMOM": "new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\nangmom",
    "LAPLACE": ("new point { mass = 1e9, position = [0, 0, 0] }\n"
                "new point { mass = 1, position = [0.4, 0, 0], velocity = [0, 2, 0] }\n"
                "set system.g_constant = 1e-9\nset system.softening = 0\nlaplace 1"),
    "HELP": "help",
    "RESET": "new point { mass = 1 }\nreset\nlist",
    "POINT": "new point { mass = 1, velocity = [1, 0, 0] }\nlist",
    "SPHERE": "new sphere { mass = 2, radius = 0.5 }\nget obj0.inertia_tensor",
    "CUBOID": "new cuboid { mass = 3, half_extents = [0.5, 1, 2] }\nget obj0.inertia_tensor",
    "TORUS": "new torus { mass = 1, inner_radius = 1, outer_radius = 2 }\nget obj0.inertia_tensor",
    "DISK": "new disk { mass = 1, radius = 1 }\nget obj0.inertia_tensor",
    "CYLINDER": "new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nget obj0.half_height",
    "DUMBBELL": "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\nget obj0.mass",
    "BOX": "box 4\nget system.box",
    "AS": 'new sphere as "ball" { mass = 2 }\nget ball.mass',
    "LET": "let g0 = 9.81\ng0",
    "FUNCS": "def probe(m = 2) { new sphere { mass = m } }\nfuncs",
    "SHOW": "def probe(m = 2) { new sphere { mass = m } }\nshow probe",
    "COLLIDE": "new sphere { radius = 0.5 }\nnew sphere { radius = 0.5, position = [3, 0, 0] }\ncollide",
    "CONTACTS": TWO_BALLS + "\ncontacts",
    "ON": "collide on",
    "OFF": "collide off",
    "SCENE": "new sphere { mass = 1 }\nscene create\nscene status\nscene close",
    "CREATE": "new sphere { mass = 1 }\nscene create\nscene close",
    "CLOSE": "new sphere { mass = 1 }\nscene create\nscene close",
    "TRANSLATE": "new sphere { mass = 1 }\nscene create\nscene translate 2 0\nscene close",
    "ROTATE": "new sphere { mass = 1 }\nscene create\nscene rotate 15 -5\nscene close",
    "ZOOM": "new sphere { mass = 1 }\nscene create\nscene zoom in\nscene close",
    "IN": "new sphere { mass = 1 }\nscene create\nscene zoom in\nscene close",
    "OUT": "new sphere { mass = 1 }\nscene create\nscene zoom out\nscene close",
    "HIDE": "new sphere { mass = 1 }\nscene create\nscene hide 0\nscene close",
    "REFRESH": "new sphere { mass = 1 }\nscene create\nscene refresh\nscene close",
    "REDRAW": "new sphere { mass = 1 }\nscene create\nscene redraw\nscene close",
    "START": "new point { mass = 1, velocity = [1, 0, 0] }\nscene create\nscene start\nscene pause\nscene close",
    "STOP": "new point { mass = 1, velocity = [1, 0, 0] }\nscene create\nscene start\nscene stop\nscene close",
    "PAUSE": "new point { mass = 1, velocity = [1, 0, 0] }\nscene create\nscene start\nscene pause\nscene close",
    "REVERSE": ("new point { mass = 1, velocity = [1, 0, 0] }\nscene create\n"
                "scene start\n"
                "# playback advances on WALL-CLOCK time, on its own thread, so a\n"
                "# batch script must actually spend some — otherwise PAUSE lands\n"
                "# before the first frame is recorded and REVERSE has nothing to\n"
                "# rewind.\n"
                "def v2(x, y) { 0.5 * (x * x + y * y) }\n"
                "qm2 grid -5 5 40, -5 5 40\nqm2 potential v2\nqm2 states 3\n"
                "scene pause\nscene reverse\nscene close"),
    "SET_TIME_STEP": "new sphere { mass = 1 }\nscene create\nscene set_time_step 0.005\nscene close",
    "STATUS": "new sphere { mass = 1 }\nscene create\nscene status\nscene close",
    "EVENTS": "new sphere { mass = 1 }\nscene create\nscene events\nscene close",
    "ALL": "new sphere { mass = 1 }\nscene create\nscene hide all\nscene show all\nscene close",
    "QM": "qm grid -8 8 60\nqm potential zero\nqm",
    "QM2": "qm2 grid -4 4 20, -4 4 20\nqm2 potential zero\nqm2",
    "QM3": "qm3 grid -3 3 12, -3 3 12, -3 3 12\nqm3 potential zero\nqm3",
}

KEYWORD_NOTE = {
    "DEL": "Later objects RENUMBER, and the AS-name registry follows them down.",
    "BOX": "The six wall slabs are ordinary cuboid objects with inverse_mass = 0.",
    "RESET": "Not SCENE RESET: this wipes the system, that re-initialises the window's "
             "playback copy.",
    "SHOW": "SHOW is also a SCENE sub-command (SCENE SHOW n|ALL); the parser tells them "
            "apart by position.",
    "OFF": "OFF serves three families: COLLIDE OFF, BOX OFF and QM/QM2/QM3 DRIVE|ABSORB OFF.",
    "ON": "Only COLLIDE takes ON; it is the default.",
    "SPRK": "SPRK requires a separable Hamiltonian: no b_field, magnetic tensor, external "
            "torque or spinning rigid body.",
    "QM": "One of only three words the quantum families reserve: every sub-command "
          "after it is matched on its lowercased text, so the whole quantum vocabulary "
          "stays out of the global keyword namespace.",
    "QM2": "See QM — the family head is the only reserved word.",
    "QM3": "See QM — the family head is the only reserved word.",
}


def build_keywords():
    kws = json.load(open(f"{D}/keywords.json"))
    out = []
    for k in kws:
        name = k["canonical"]
        use = KEYWORD_USE.get(name)
        exs = []
        if use:
            exs.append(ex("trivial", use))
            if k["aliases"]:
                exs.append(ex("intermediate",
                              use.replace(name.lower(), k["aliases"][0].lower(), 1)))
        out.append(entry(
            id=f"kw.{name.lower()}",
            name=name,
            kind="keyword",
            aliases=k["aliases"],
            summary=f"Reserved word of the posim command language (Keyword::{k['variant']}).",
            definition=(
                f"`{name}` is matched case-insensitively by the lexer and mapped to "
                f"`Keyword::{k['variant']}`."
                + (f" Accepted alternative spelling(s): {', '.join(k['aliases'])}."
                   if k["aliases"] else "")
                + " Keywords are reserved only at the START of paths: after a `.`, or inside "
                  "a `NEW { ... }` initializer list, the parser accepts a keyword spelling as "
                  "an ordinary field name — which is why `obj0.momentum` and `system.method` "
                  "both work."
                + (" " + KEYWORD_NOTE[name] if name in KEYWORD_NOTE else "")
            ),
            syntax=[name] + list(k["aliases"]),
            locations=[{"file": k["file"], "line": k["line"], "role": "lexer mapping"},
                       {"file": "posim/src/lexer.rs", "line": 10, "role": "Keyword enum"}],
            examples=exs,
            # A keyword links to its command entry when it heads one. Shape
            # words head no command of their own (they are an argument of
            # NEW), so they link to the shape's type entry instead — the
            # build prunes any reference whose target does not exist, and a
            # silently pruned link is a link the reader never gets.
            seeAlso=[f"cmd.{name.lower()}", f"type.shape.{name.lower()}",
                     f"cmd.scene.{name.lower()}", "cmd.method", "cmd.new"],
            status="complete" if exs else "stub",
        ))
    return out


# --------------------------------------------------------------------------
# 2. properties
# --------------------------------------------------------------------------

COUPLING = {
    "mass": ("new sphere { mass = 4, radius = 0.5 }\nget obj0.inverse_mass\n"
             "set obj0.mass = 2\nget obj0.inverse_mass",
             "Writing `mass` rewrites `inverse_mass`; the two are one quantity."),
    "inverse_mass": ("new sphere { mass = 4, radius = 0.5 }\nset obj0.inverse_mass = 0\n"
                     "get obj0.mass\nget obj0.inverse_mass",
                     "`inverse_mass = 0` is EXACT infinite mass: the equations of motion "
                     "only ever use the inverse, so a static body needs no approximation."),
    "velocity": ("new point { mass = 2, velocity = [3, 0, 0] }\nget obj0.momentum\n"
                 "set obj0.mass = 4\nget obj0.velocity\nget obj0.momentum",
                 "Momentum is the canonical stored state: changing the mass changes the "
                 "VELOCITY, not the momentum."),
    "momentum": ("new point { mass = 2, velocity = [3, 0, 0] }\nget obj0.momentum\n"
                 "set obj0.momentum = [8, 0, 0]\nget obj0.velocity",
                 "The canonical stored linear state; velocity is derived as p/m."),
    "orientation": ("new sphere { mass = 1, radius = 0.5, orientation = [2, 0, 0, 0] }\n"
                    "get obj0.orientation",
                    "Quaternions are w-first and renormalised to unit length on every write."),
    "angular_velocity": ("new cuboid { mass = 1, half_extents = [0.5, 1, 2], "
                         "angular_velocity = [0, 2, 0] }\nget obj0.angular_momentum",
                         "Derived through the full L = (R I R^T) omega transformation."),
    "inertia_tensor": ("new cuboid { mass = 1, inertia_tensor = [[2, 0, 0], [0, 3, 0], "
                       "[0, 0, 4]] }\nget obj0.inverse_inertia_tensor",
                       "Supplying `inertia_tensor` in a NEW initializer SUPPRESSES the "
                       "analytic shape inertia; the inverse is maintained for you."),
    "inner_radius": ("new torus { mass = 1, inner_radius = 1, outer_radius = 2 }\n"
                     "get obj0.ring_radius\nget obj0.tube_radius",
                     "The inner/outer pair is deferred and resolved ONCE at the end of the "
                     "initializer list, so it is genuinely order-independent. "
                     "`inner_radius = 0` is the horn torus and is legal."),
    "outer_radius": ("new torus { mass = 1, outer_radius = 0.5, inner_radius = 0.2 }\n"
                     "get obj0.ring_radius\nget obj0.tube_radius",
                     "Writing one of the derived pair holds the other fixed."),
    "height": ("new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nget obj0.half_height",
               "`height` is the FULL height = 2 * half_height."),
    "radius": ("new disk { mass = 1, radius = 1 }\nset obj0.radius = 2\nget obj0.boundary",
               "Writing `radius` KEEPS the shape family: a disk stays a disk, a cylinder "
               "keeps its height. A torus refuses rather than silently becoming a sphere."),
    "m1": ("new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\nget obj0.mass\n"
           "set obj0.m1 = 3\nget obj0.mass",
           "Writing a part field rebuilds the whole body: total mass, the COM offsets and "
           "the inertia tensor all follow."),
    "restitution": ("new sphere { mass = 1, radius = 0.5, restitution = 0.8 }\n"
                    "get obj0.restitution",
                    "A colliding pair uses min(e_i, e_j)."),
}


# The rung each of these fields actually needs: the refusal, the boundary
# case, or the physical consequence — not a fourth restatement.
FOURTH = {
 "mass": "new sphere { mass = 4, radius = 0.5 }\nset obj0.mass = 0\nget obj0.inverse_mass\nget obj0.mass",
 "inverse_mass": "set system.g_constant = 0\nnew sphere { mass = 1, radius = 0.5, position = [0, 2, 0], velocity = [0, -1, 0] }\nnew cuboid { mass = 1, half_extents = [2, 0.5, 2], inverse_mass = 0 }\nmomentum\nrun 3 steps 3\nmomentum\nget obj1.position",
 "inertia_tensor": "new cuboid { mass = 1, inertia_tensor = [[2, 0, 0], [0, 3, 0], [0, 0, 4]], orientation = [0.7071067811865476, 0, 0, 0.7071067811865476] }\nset obj0.angular_velocity = [0, 2, 0]\nget obj0.angular_momentum",
 "inner_radius": "new torus { mass = 1, inner_radius = 0, outer_radius = 2 }\nget obj0.ring_radius\nget obj0.tube_radius\nget obj0.inner_radius",
 "outer_radius": "new torus { mass = 1, outer_radius = 0.5, inner_radius = 0.2 }\nget obj0.ring_radius\nget obj0.tube_radius",
 "radius": "new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nset obj0.radius = 1\nget obj0.boundary\nget obj0.height",
 "height": "new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nget obj0.height\nset obj0.half_height = 2\nget obj0.height",
 "m1": "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5, velocity = [1, 0, 0] }\nget obj0.momentum\nset obj0.m1 = 3\nget obj0.momentum\nget obj0.velocity",
 "restitution": "set system.g_constant = 0\nset system.gravity = [0, -9.81, 0]\nnew sphere { mass = 1, radius = 0.5, position = [0, 5, 0], restitution = 0.8 }\nnew cuboid { mass = 1, half_extents = [4, 0.5, 4], position = [0, 0, 0], inverse_mass = 0 }\nrun 2 steps 20\nget system.collisions",
 "boundary": "new sphere { mass = 1, radius = 0.5 }\nnew torus { mass = 1, inner_radius = 1, outer_radius = 2 }\nnew dumbbell { m1 = 1, m2 = 2 }\nget obj0.boundary\nget obj1.boundary\nget obj2.boundary"
}


def prop_examples(p, root):
    """A ladder of self-contained fragments for one field path."""
    shapes = p.get("shapes", "all")
    maker = MAKER.get(shapes, MAKER["all"])
    n, t, rw = p["name"], p["type"], p["rw"]
    out = []

    if root == "system":
        out.append(ex("trivial", f"get system.{n}"))
        if rw == "RW":
            out.append(ex("intermediate", f"set system.{n} = {SAMPLE[t]}\nget system.{n}"))
        else:
            out.append(ex("intermediate", f"{MAKER['all']}\nget system.{n}"))
        if p["aliases"]:
            a = p["aliases"][0]
            out.append(ex("advanced",
                          f"set system.{a} = {SAMPLE[t]}\nget system.{n}" if rw == "RW"
                          else f"get system.{a}\nget system.{n}"))
        # watch it across a real integration: what a system field MEANS is what
        # it does to a trajectory, which a static read cannot show
        while len(out) < 4:
            out.append(ex("expert",
                          "new sphere { mass = 2, radius = 0.5, velocity = [1, 0, 0] }\n"
                          "set system.gravity = [0, -9.81, 0]\n"
                          f"get system.{n}\nrun 1 steps 2\nget system.{n}"))
        return out

    if root == "contact":
        out.append(ex("trivial", TWO_BALLS + f"\nget contact0.{n}"))
        out.append(ex("intermediate", TWO_BALLS + f"\ncontacts\ncontact0.{n}"))
        if p["aliases"]:
            out.append(ex("advanced",
                          TWO_BALLS + f"\nget contact0.{n}\nget contact0.{p['aliases'][0]}"))
        if t == "vec3":
            out.append(ex("expert", TWO_BALLS + f"\ncontact0.impulse * contact0.{n}.x"))
        # a contact field is only meaningful beside the pair it describes
        while len(out) < 4:
            out.append(ex("expert", TWO_BALLS
                          + f"\nget contact0.i\nget contact0.j\nget contact0.{n}"
                          + "\nget system.collisions"))
        return out

    out.append(ex("trivial", f"{maker}\nget obj0.{n}"))
    if rw == "RW":
        val = PROP_SAMPLE.get(n, SAMPLE[t])
        out.append(ex("intermediate", f"{maker}\nset obj0.{n} = {val}\nget obj0.{n}"))
    else:
        out.append(ex("intermediate", f"{maker}\nobj0.{n}"))
    if t in ("vec3", "quaternion"):
        comp = ".w" if t == "quaternion" else ".x"
        out.append(ex("advanced", f"{maker}\nget obj0.{n}{comp}"))
    elif p["aliases"]:
        out.append(ex("advanced", f"{maker}\nget obj0.{p['aliases'][0]}"))
    if n in COUPLING:
        out.append(ex("expert", COUPLING[n][0]))
    # FOURTH is applied independently of COUPLING: a read-only field like
    # `boundary` has no coupled invariant to demonstrate, but still deserves
    # the rung that shows what it is FOR.
    if n in FOURTH:
        out.append(ex("expert", FOURTH[n]))
    if not (n in COUPLING or n in FOURTH) \
            and t in ("number", "vec3", "quaternion", "mat3"):
        # watch the field across a real integration: a static read tells you
        # the type, a read either side of a RUN tells you what it MEANS
        while len(out) < 4:
            out.append(ex("expert",
                          f"{maker}\nset system.gravity = [0, -9.81, 0]\n"
                          f"get obj0.{n}\nrun 1 steps 2\nget obj0.{n}"))
    return out


def build_properties():
    props = json.load(open(f"{D}/properties.json"))
    out = []
    roots = {"object": "objN", "system": "system", "contact": "contactK"}
    err_for = {
        "object": "unknown object field `<name>` — see HELP for the field list",
        "system": ("unknown system field `<name>` (g_constant, softening, uniform_gravity, "
                   "e_field, b_field, rtol, atol, time, method, count, collide, contacts, "
                   "collisions, restitution_threshold, contact_slop, box)"),
        "contact": ("unknown contact field `<name>` (i, j, t, point, normal, depth, "
                    "rel_vel_n, impulse)"),
    }
    for root, prefix in roots.items():
        for p in props[root]:
            n = p["name"]
            defn = p["meaning"]
            if n in COUPLING:
                defn += "  " + COUPLING[n][1]
            errs = [err_for[root]]
            if p["rw"] == "R+refused":
                errs.insert(0, "use COLLIDE ON / COLLIDE OFF to switch collision detection")
            locs = [{"file": "posim/src/vm.rs", "line": p["read_line"], "role": "read arm"}]
            if p.get("write_line"):
                locs.append({"file": "posim/src/vm.rs", "line": p["write_line"],
                             "role": "write arm"})
            syn = [f"GET {prefix}.{n}"]
            if p["rw"] == "RW":
                syn.append(f"SET {prefix}.{n} = <expr>")
            syn.append(f"{prefix}.{n}")
            note = ""
            if p["aliases"] and root == "contact":
                note = ("  The spelling `%s` is an accepted alias, documented in "
                        "grammar.md §5.7 since the D3 fix." % p["aliases"][0])
            out.append(entry(
                id=f"prop.{'obj' if root == 'object' else root}.{n}",
                name=n, kind="property", aliases=p["aliases"],
                summary=f"{prefix}.{n} — {p['type']}, {p['rw']}.",
                definition=defn + note,
                syntax=syn,
                parameters=[{"name": n, "type": p["type"], "rw": p["rw"],
                             "shapes": p.get("shapes", "-"), "notes": p["meaning"]}],
                returns=p["type"], errors=errs, locations=locs,
                examples=prop_examples(p, root),
                seeAlso=["cmd.get", "cmd.set"],
                invariants=([COUPLING[n][1]] if n in COUPLING else []),
            ))
    return out


# --------------------------------------------------------------------------
# 3. builtins
# --------------------------------------------------------------------------

CORE_LADDER = {
    "dot": ["dot([1, 2, 3], [4, 5, 6])",
            "new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\n"
            "dot(obj0.position, obj0.velocity)",
            "dot([1, 0, 0], [0, 1, 0])",
            "new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\n"
            "dot(obj0.position, obj0.momentum) / (norm(obj0.position) * norm(obj0.momentum))"],
    "cross": ["cross([1, 0, 0], [0, 1, 0])",
              "new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\n"
              "cross(obj0.position, obj0.momentum)",
              "new point { mass = 2, position = [1, 0, 0], velocity = [0, 3, 0] }\n"
              "cross(obj0.position, obj0.momentum)\nangmom",
              "norm(cross([1, 0, 0], [0, 1, 0]))"],
    "norm": ["norm([3, 4, 0])", "norm([1, 0, 0, 0])",
             "new sphere { mass = 2, radius = 0.1, charge = -1.5, velocity = [3, 0, 0] }\n"
             "norm(obj0.velocity)",
             "new sphere { mass = 2, radius = 0.1, charge = -1.5, velocity = [3, 0, 0] }\n"
             "set system.b_field = [0, 0, 4]\nmethod bdf\n"
             "run 2 * pi * 2 / (1.5 * 4) steps 8\nnorm(obj0.velocity)"],
    "normalize": ["normalize([3, 4, 0])", "norm(normalize([3, 4, 0]))",
                  "new point { mass = 1, position = [2, 2, 1] }\nnormalize(obj0.position)",
                  "new point { mass = 1, position = [2, 2, 1] }\n"
                  "dot(normalize(obj0.position), normalize(obj0.position))"],
    "sqrt": ["sqrt(2)", "sqrt(2) * sqrt(2)", "sqrt(pi)", "sqrt(2/(pi*2))*sin(2)"],
    "abs": ["abs(-3)", "abs(0 - 3i)", "abs(sph_hankel_h1(3, 800) * 800)",
            "abs(bessel_j_z(0, 2.4048))"],
    "sin": ["sin(0)", "sin(pi/2)", "sph_j(0, 1.3) - sin(1.3) / 1.3",
            "let t = 0.7\nchebyshev_t(5, cos(t)) - cos(5 * t)"],
    "cos": ["cos(0)", "cos(pi)", "sqrt(2/(pi*2))*cos(2)",
            "let t = 0.7\nchebyshev_t(5, cos(t)) - cos(5 * t)"],
    "exp": ["exp(0)", "exp(1)", "log(exp(2))", "exp(log(5)) - 5"],
    "log": ["log(1)", "log(exp(3))", "exp(log(5)) - 5", "log(2) + log(3) - log(6)"],
}

CONST_LADDER = {
    "pi": ["pi", "2 * pi", "pi * pi / 2",
           "new point { mass = 1, velocity = [0, 1, 0] }\nrun 2 * pi steps 4"],
    "tau": ["tau", "tau - 2 * pi", "tau / 4",
            "new point { mass = 1, velocity = [0, 1, 0] }\nrun tau steps 4"],
}

SPECIAL_LADDER = {
    "sph_j": ["sph_j(0, 1.3)", "sph_j(0, 1.3) - sin(1.3) / 1.3", "sph_j(20, 1.0)"],
    "sph_y": ["sph_y(0, 1.3)", "sph_y(0, 1.3) + cos(1.3) / 1.3", "sph_y(2, 3.5)"],
    "sph_j_prime": ["sph_j_prime(0, 1.3)", "sph_j_prime(1, 2.0)",
                    "sph_j(1, 2.0) * sph_y_prime(1, 2.0) - sph_j_prime(1, 2.0) * sph_y(1, 2.0)"],
    "sph_y_prime": ["sph_y_prime(0, 1.3)", "sph_y_prime(1, 2.0)",
                    "sph_j(1, 2.0) * sph_y_prime(1, 2.0) - sph_j_prime(1, 2.0) * sph_y(1, 2.0)"],
    "legendre_p": ["legendre_p(3, 0.4)", "legendre_p(7, 1.0)",
                   "legendre_p(3, 0.4) - 0.5 * (5 * 0.4 * 0.4 * 0.4 - 3 * 0.4)"],
    "legendre_p_prime": ["legendre_p_prime(3, 0.4)", "legendre_p_prime(1, 0.5)",
                         "legendre_p_prime(2, 0.3) - 3 * 0.3"],
    "assoc_legendre_p": ["assoc_legendre_p(2, 2, 0.5)",
                         "assoc_legendre_p(2, 2, 0.5) - 3 * (1 - 0.5 * 0.5)",
                         "assoc_legendre_p(2, 0, 0.5) - legendre_p(2, 0.5)"],
    "norm_assoc_legendre_p": ["norm_assoc_legendre_p(2, 2, 0.5)",
                              "norm_assoc_legendre_p(100, 50, 0.3)",
                              "norm_assoc_legendre_p(170, 170, 0.3)"],
    "sph_harm": ["sph_harm(0, 0, 1.0, 2.0)", "sph_harm(1, 0, 0.5, 0.3)",
                 "sph_harm(2, 1, 1.0, 2.0)"],
    "sph_harm_real": ["sph_harm_real(0, 0, 1.0, 2.0)", "sph_harm_real(1, 1, 0.5, 0.3)",
                      "sph_harm_real(2, -1, 1.0, 2.0)"],
    "hermite_h": ["hermite_h(2, 0.37)", "hermite_h(2, 0.37) - (4 * 0.37 * 0.37 - 2)",
                  "hermite_h(5, 1.0)"],
    "hermite_he": ["hermite_he(2, 0.37)", "hermite_he(2, 0.37) - (0.37 * 0.37 - 1)",
                   "hermite_he(5, 1.0)"],
    "laguerre_l": ["laguerre_l(2, 0.5)", "laguerre_l(0, 3.0)",
                   "laguerre_l(1, 0.5) - (1 - 0.5)"],
    "laguerre_l_assoc": ["laguerre_l_assoc(2, 1, 0.5)",
                         "laguerre_l_assoc(2, 0, 0.5) - laguerre_l(2, 0.5)",
                         "laguerre_l_assoc(3, 0.5, 1.2)"],
    "chebyshev_t": ["chebyshev_t(5, 0.3)",
                    "let t = 0.7\nchebyshev_t(5, cos(t)) - cos(5 * t)",
                    "chebyshev_t(0, 0.9)"],
    "chebyshev_u": ["chebyshev_u(5, 0.3)", "chebyshev_u(1, 0.4) - 2 * 0.4",
                    "chebyshev_u(0, 0.9)"],
    "gegenbauer_c": ["gegenbauer_c(3, 1, 0.4)",
                     "gegenbauer_c(3, 1, 0.4) - chebyshev_u(3, 0.4)",
                     "gegenbauer_c(2, 0.5, 0.4) - legendre_p(2, 0.4)"],
    "jacobi_p": ["jacobi_p(3, 0, 0, 0.4)", "jacobi_p(3, 0, 0, 0.4) - legendre_p(3, 0.4)",
                 "jacobi_p(2, 1, 1, 0.3)"],
    "bessel_j": ["bessel_j(0, 2)", "bessel_j(0, 2.404825557695773)",
                 "bessel_j(1, -1.4) + bessel_j(1, 1.4)"],
    "bessel_j_array": ["bessel_j_array(4, 2)", "bessel_j_array(4, 0)",
                       "bessel_j_array(10, 5)"],
    "eigenvalues": ["eigenvalues([[2, 1], [1, 2]])",
                    "let h = 0.2\nlet k = 1 / (h * h)\n"
                    "eigenvalues([[k, -0.5*k, 0, 0], [-0.5*k, k, -0.5*k, 0], "
                    "[0, -0.5*k, k, -0.5*k], [0, 0, -0.5*k, k]])",
                    "eigenvalues([[1, 0, 0], [0, 2, 0], [0, 0, 3]])"],
    "jacobi_eigen": ["jacobi_eigen([[2, 1], [1, 2]])", "jacobi_eigen([[1, 0], [0, 3]])",
                     "jacobi_eigen([[2, 0, 0], [0, 3, 1], [0, 1, 3]])"],
    "gauss_legendre": ["gauss_legendre(3)", "gauss_legendre(2)", "gauss_legendre(8)"],
    "rel_err": ["rel_err(1, 1.0001)", "rel_err(2, 2)", "rel_err(sqrt(2) * sqrt(2), 2)"],
    "solve_tridiag": ["solve_tridiag([0, 1, 1], [2, 2, 2], [1, 1, 0], [1, 2, 3])",
                      "solve_tridiag([0, 0, 0], [1, 1, 1], [0, 0, 0], [4, 5, 6])",
                      "solve_tridiag([0, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 0], [1, 0, 0, 1])"],
    "solve_tridiag_c": ["solve_tridiag_c([0, 1, 1], [1i, 1i, 1i], [1, 1, 0], [1, 0, 0])",
                        "solve_tridiag_c([0, 0, 0], [1i, 1i, 1i], [0, 0, 0], [1i, 1i, 1i])",
                        "solve_tridiag_c([0, 1, 1, 1], [2i, 2i, 2i, 2i], [1, 1, 1, 0], "
                        "[1, 0, 0, 1])"],
    "solve_cyclic_tridiag_c": [
        "solve_cyclic_tridiag_c([0, 1, 1], [2, 2, 2], [1, 1, 0], [1, 0, 0], 0, 0)",
        "solve_cyclic_tridiag_c([0, 1, 1], [2i, 2i, 2i], [1, 1, 0], [1, 0, 0], 1, 1)",
        "solve_cyclic_tridiag_c([0, 1, 1, 1], [3, 3, 3, 3], [1, 1, 1, 0], [1, 0, 0, 1], 1, 1)"],
    "wigner_3j": ["wigner_3j(1, 1, 0, 0, 0, 0)", "wigner_3j(1, 1, 2, 0, 0, 0)",
                  "wigner_3j(0.5, 0.5, 1, 0.5, -0.5, 0)"],
    "wigner_6j": ["wigner_6j(1, 1, 0, 1, 1, 1)", "wigner_6j(1, 2, 3, 2, 1, 2)",
                  "wigner_6j(0.5, 0.5, 1, 0.5, 0.5, 1)"],
    "wigner_9j": ["wigner_9j(1, 1, 0, 1, 1, 0, 0, 0, 0)",
                  "wigner_9j(1, 1, 2, 1, 1, 2, 2, 2, 0)",
                  "wigner_9j(0.5, 0.5, 1, 0.5, 0.5, 1, 1, 1, 0)"],
    "clebsch_gordan": ["clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0)",
                       "clebsch_gordan(1, 0, 1, 0, 2, 0)",
                       "clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 1)"],
    "airy_z": ["airy_z(0)", "airy_z(2 - 3i)", "airy_z(-8)"],
    "gamma_z": ["gamma_z(0.5)", "gamma_z(0.5) - sqrt(pi)", "gamma_z(1 + 1i)"],
    "ln_gamma_z": ["ln_gamma_z(200)", "ln_gamma_z(1)", "ln_gamma_z(2 + 1i)"],
    "rgamma_z": ["rgamma_z(-3)", "rgamma_z(1)", "rgamma_z(0.5) * gamma_z(0.5)"],
}

# Builtins whose result is a LIST rather than a number or a complex value.
# `abs()` of a list is an error, so the generated ladder must know the
# difference — grammar.md section 4.1 marks each of these with `-> list`.
RETURNS_LIST = {"sph_harm", "bessel_j_array", "gauss_legendre", "eigenvalues",
                "jacobi_eigen", "airy_z", "solve_tridiag", "solve_tridiag_c",
                "solve_cyclic_tridiag_c"}

BESSEL_Z = ["bessel_j_z", "bessel_i_z", "bessel_y_z", "bessel_k_z"]
BESSEL_NU = ["bessel_j_nu", "bessel_i_nu", "bessel_y_nu", "bessel_k_nu"]
SCALED = ["bessel_j_scaled", "bessel_y_scaled", "bessel_i_scaled", "bessel_k_scaled",
          "hankel_h1_scaled", "hankel_h2_scaled"]
HANKEL_Z = ["hankel_h1_z", "hankel_h2_z", "hankel_h1_prime_z", "hankel_h2_prime_z"]
HANKEL_NU = ["hankel_h1_nu", "hankel_h2_nu", "hankel_h1_prime_nu", "hankel_h2_prime_nu"]
SPH_HANKEL = ["sph_hankel_h1", "sph_hankel_h2", "sph_hankel_h1_prime", "sph_hankel_h2_prime"]

FAMILY_DOC = {}
FAMILY_DOC.update({n: "cylindrical Bessel, complex argument, WHOLE order" for n in BESSEL_Z})
FAMILY_DOC.update({n: "cylindrical Bessel, any real or complex order" for n in BESSEL_NU})
FAMILY_DOC.update({n: "exponentially scaled form — the growing factor is never formed"
                   for n in SCALED})
FAMILY_DOC.update({n: "Hankel (travelling-wave) pair, whole order" for n in HANKEL_Z})
FAMILY_DOC.update({n: "Hankel (travelling-wave) pair, any real order" for n in HANKEL_NU})
FAMILY_DOC.update({n: "spherical Hankel — real argument, complex value" for n in SPH_HANKEL})


def special_ladder(name):
    if name in SPECIAL_LADDER:
        return SPECIAL_LADDER[name]
    if name in BESSEL_Z:
        return [f"{name}(0, 2)", f"{name}(1, 2 + 1i)", f"{name}(2, 3)"]
    if name in BESSEL_NU:
        return [f"{name}(0.5, 2)", f"{name}(1.3, 2 + 1i)", f"{name}(2, 1.6 + 0.9i)"]
    if name in SCALED:
        return [f"{name}(0, 40)", f"{name}(0.5, 3)", f"{name}(2, 25)"]
    if name in HANKEL_Z:
        return [f"{name}(0, 3)", f"{name}(1, 3)", f"{name}(2, 2 - 0.5i)"]
    if name in HANKEL_NU:
        return [f"{name}(0.5, 2 - 0.5i)", f"{name}(1.3, 3)", f"{name}(2, 1.6 + 0.9i)"]
    if name in SPH_HANKEL:
        return [f"{name}(0, 2.3)", f"{name}(1, 2.3)", f"{name}(3, 800)"]
    return [f"{name}(0, 1)"]


def build_builtins():
    bi = json.load(open(f"{D}/builtins.json"))
    out = []
    for b in bi["core"]:
        rungs = CORE_LADDER.get(b["name"], [f"{b['name']}(1)"])
        out.append(entry(
            id=f"fn.{b['name']}", name=b["name"], kind="builtin",
            summary=f"{b['syntax']} -> {b['returns']} — {b['meaning']}",
            definition=f"Core builtin of the expression language. {b['meaning'].capitalize()}. "
                       "Reached through the uniform call production "
                       "`IDENT \"(\" [expr {\",\" expr}] \")\"`, so it needs no grammar of "
                       "its own. A name that is not a builtin is looked up among your "
                       "DEF'd functions instead.",
            syntax=[b["syntax"]], returns=b["returns"],
            locations=[{"file": "posim/src/vm.rs", "line": 1, "role": "Call dispatch"},
                       {"file": "posim/src/parser.rs", "line": 168, "role": "call production"}],
            examples=[ex(l, c) for l, c in zip(LEVELS, rungs)],
            seeAlso=["type.number", "type.vec3"],
        ))
    for c in bi["constants"]:
        out.append(entry(
            id=f"const.{c['name']}", name=c["name"], kind="builtin",
            summary=f"{c['name']} — {c['meaning']}",
            definition=f"A constant of the expression language: {c['meaning']}. Resolved at "
                       "execution time like any bare identifier — a bare name that is bound "
                       "to nothing fails with an error listing the three ways to bind one.",
            syntax=[c["name"]], returns="number",
            locations=[{"file": "posim/src/vm.rs", "line": 1, "role": "LoadIdent constants"}],
            examples=[ex(l, cc) for l, cc in zip(LEVELS, CONST_LADDER[c["name"]])],
        ))
    for s in bi["special"]:
        n = s["name"]
        fam = FAMILY_DOC.get(n, "special function of the `special_functions` crate")
        out.append(entry(
            id=f"fn.{n}", name=n, kind="builtin",
            summary=f"{n}(...) — {fam}.",
            definition=(
                f"Registered special function ({fam}). Adding one of these is a "
                "REGISTRATION, not a grammar change: the call production already admits any "
                "builtin name. Where an argument is an integer order, a fractional value is "
                "an ERROR rather than being quietly truncated — a truncated order would "
                "return a confident, wrong number with nothing to notice."),
            syntax=[f"{n}(...)"],
            errors=[f"{n}(): argument 1 must be a whole number (an integer order), got 2.5"],
            locations=[{"file": s["file"], "line": s["line"], "role": "registration"},
                       {"file": "grammar.md", "line": 253, "role": "spec (section 4.1)"}],
            examples=[ex(l, c) for l, c in zip(LEVELS, special_ladder(n) + [
                # a 4th rung: the value bound to a LET and reused, which is how
                # these are actually written. `abs` only where the result is a
                # scalar or complex — the families below return LISTS, and the
                # scalar builtins are real-only by design.
                f"let v = {special_ladder(n)[0]}\nv"
                + ("" if n in RETURNS_LIST else "\nabs(v)")])],
            seeAlso=["fn.abs", "type.complex"],
        ))
    return out


# --------------------------------------------------------------------------
# 4. magics
# --------------------------------------------------------------------------

MAGIC_LADDER = {
    "%history": ["new point { mass = 2 }\n%history",
                 "new point { mass = 2 }\nget obj0.mass\n%history"],
    "%rerun": ["new point { mass = 2 }\n%rerun 1\nlist"],
    "%edit": ["new sphere { mass = 20, radius = 0.5 }\n"
              "%edit 1 new sphere { mass = 2, radius = 0.5 }\nlist"],
    "%save": ["new sphere { mass = 2, radius = 0.5 }\n%save session.posim"],
    "%load": ["new sphere { mass = 2, radius = 0.5 }\n%save session.posim\n"
              "%reset\n%load session.posim\nlist"],
    "%reset": ["new point { mass = 1 }\n%reset\nlist"],
    # These end the session — which a script fragment can perfectly well do.
    # An earlier version left them empty on the assumption that they could not
    # be demonstrated in batch; running them showed otherwise (exit 0, replay
    # stops at that line).
    "%quit": ["%quit",
              "new point { mass = 1 }\n%quit",
              "new point { mass = 1 }\nget obj0.mass\n%quit",
              "new point { mass = 1 }\n%quit\nlist"],
    "%exit": ["%exit",
              "new point { mass = 1 }\n%exit",
              "new point { mass = 1 }\nget obj0.mass\n%exit",
              "new sphere { mass = 2, radius = 0.5 }\nenergy\n%exit\nlist"],
}


def build_magics():
    mg = json.load(open(f"{D}/magics.json"))
    out = []
    for m in mg["entries"]:
        rungs = MAGIC_LADDER.get(m["name"], [])
        out.append(entry(
            id=f"magic.{m['name'].lstrip('%')}", name=m["name"], kind="magic",
            indexKeys=["%"],
            summary=f"{m['syntax']} — {m['meaning']}",
            definition=("A notebook magic. A line whose FIRST character is `%` is handled by "
                        "the notebook itself and never reaches the lexer. A plain terminal "
                        "has no cursor-addressable cells (posim uses only the Rust standard "
                        "library), so backward and forward movement is by these magics; in "
                        f"JupyterLab the cells themselves do the job. {m['meaning'].capitalize()}."),
            syntax=[m["syntax"]],
            errors=["unknown magic `<name>` — see HELP"],
            locations=[{"file": "posim/src/notebook.rs", "line": m["line"], "role": "magic arm"},
                       {"file": "posim/src/notebook.rs", "line": 56, "role": "Notebook::magic"}],
            examples=[ex(l, c) for l, c in zip(LEVELS, rungs)],
            seeAlso=["cmd.reset"],
            invariants=["Magics do not consume cell numbers — only executed commands become "
                        "numbered cells.",
                        "%quit and %exit end a SCRIPT replay too, at that line, with exit 0 — "
                        "anything after them is not executed.",
                        "%save writes only successful, non-magic inputs.",
                        "%load joins continuation lines by brace depth, so a saved "
                        "multi-line DEF replays as ONE cell."],
            status="complete" if rungs else "stub",
        ))
    return out


# --------------------------------------------------------------------------
# 5. types and shapes
# --------------------------------------------------------------------------

TYPES = [
    ("number", "64-bit float. `2`, `-0.5`, `.5`, `1e-3`, `2.5E+4`. A leading `-` is not part "
               "of the literal — it is the negation operator.",
     ["2", "1e-3", "2 + 3 * 4", "0/0 != 0/0"]),
    ("complex", "A number with an `i` suffix is imaginary, so `2 + 3i` needs no complex "
                "literal syntax of its own — it is ordinary addition. Reals promote. The "
                "result stays typed as complex and displays `13 + 0i` rather than collapsing: "
                "the type you get out should not depend on whether a cancellation happened to "
                "be exact. The suffix binds only when the `i` is not followed by another "
                "identifier character, so `2intercept` lexes as it always did.",
     ["3i", "2 + 3i", "(2 + 3i) * (2 - 3i)", "1 / (0 + 1i)"]),
    ("vec3", "Exactly 3 numeric entries: `[1, 2, 3]`. Components `.x .y .z`. `*` between two "
             "vec3s is FORBIDDEN — the error tells you to use `dot()` or `cross()`.",
     ["[1, 2, 3]", "[1, 2, 3] + [1, 1, 1]", "2 * [1, 2, 3]",
      "new point { mass = 1, position = [1, 2, 3] }\nget obj0.position.y"]),
    ("quaternion", "Exactly 4 numeric entries, W FIRST: `[w, x, y, z]`. Assigned to "
                   "`orientation` it is renormalised to unit length. Components `.w .x .y .z`. "
                   "`*` between two quaternions is the Hamilton product.",
     ["[1, 0, 0, 0]", "norm([1, 0, 0, 0])",
      "new sphere { orientation = [0.7071067811865476, 0, 0, 0.7071067811865476] }\n"
      "get obj0.orientation",
      "new sphere { orientation = [2, 0, 0, 0] }\nget obj0.orientation"]),
    ("mat3", "3 rows of 3 numbers — a vector of 3 vec3s: `[[a,b,c],[d,e,f],[g,h,i]]`. "
             "`mat3 * vec3` and `mat3 * mat3` are defined.",
     ["[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[2,0,0],[0,3,0],[0,0,4]] * [1, 1, 1]",
      "new cuboid { mass = 3, half_extents = [0.5, 1, 2] }\nget obj0.inertia_tensor",
      "new cuboid { mass = 1, inertia_tensor = [[2,0,0],[0,3,0],[0,0,4]] }\n"
      "get obj0.inverse_inertia_tensor"]),
    ("string", "A double-quoted literal with the escapes \\\" \\\\ \\n. Names objects "
               "(`NEW ... AS`) and feeds user functions.",
     ['"ball"', 'new sphere as "ball" { mass = 2 }\nget ball.mass',
      'let n = "pebble"\nnew sphere as n { mass = 0.1 }\nget pebble.mass',
      'def make(name) { new sphere as name { mass = 1 } }\nmake("ball")\nget ball.mass']),
    ("list", "Any bracket literal that is not 3, 4 or 3x3 numbers stays a generic list. Fields "
             "refuse it with a clear error, but every numeric-list argument of the special "
             "functions accepts all three bracket shapes — you should not have to think about "
             "quaternions to type a 4x4 matrix.",
     ["[1, 2, 3, 4, 5]", "bessel_j_array(4, 2)", "gauss_legendre(3)",
      "eigenvalues([[2, 1], [1, 2]])"]),
]

SHAPE_DOC = {
    "POINT": ("a massive point particle — no extent, no rotation, and two points can never "
              "collide", "new point { mass = 1, velocity = [1, 0, 0] }\nlist"),
    "SPHERE": ("a solid sphere; inertia 2/5 m r^2; default radius 1",
               "new sphere { mass = 2, radius = 0.5 }\nget obj0.inertia_tensor"),
    "CUBOID": ("a solid box; inertia m/3 (h_y^2 + h_z^2) and so on; default half_extents "
               "[1,1,1]",
               "new cuboid { mass = 3, half_extents = [0.5, 1, 2] }\nget obj0.inertia_tensor"),
    "TORUS": ("a solid torus; I_z = m(c^2 + 3/4 a^2), I_xy = m(1/2 c^2 + 5/8 a^2); sizeable "
              "by ring/tube OR by the order-independent inner/outer pair",
              "new torus { mass = 1, inner_radius = 1, outer_radius = 2 }\n"
              "get obj0.inertia_tensor"),
    "DISK": ("an ideal zero-thickness disk; I_z = 1/2 m a^2, I_xy = 1/4 m a^2; default "
             "radius 1", "new disk { mass = 1, radius = 1 }\nget obj0.inertia_tensor"),
    "CYLINDER": ("a solid cylinder; I_z = 1/2 m r^2, I_xy = m(3r^2 + 4h^2)/12 with h the "
                 "HALF-height; default radius 0.5, half_height 1",
                 "new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nget obj0.half_height"),
    "DUMBBELL": ("ONE rigid body: two solid spheres joined by a solid rod. The local origin "
                 "is the composite CENTRE OF MASS, so `position` is the COM exactly as for "
                 "every other shape. The simulator's first non-centrally-symmetric shape",
                 "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\nget obj0.mass\nlist"),
}


# Rungs 3 and 4 for each shape CHECK the documented inertia formula rather
# than restate it: build it, read the tensor, subtract the analytic value.
# An example that proves its own claim is worth more than one that repeats it.
SHAPE_LADDER = {
 "POINT": [
  "new point { mass = 1, position = [1, 0, 0], velocity = [0, 3, 0] }\nangmom\ncross(obj0.position, obj0.momentum)",
  "new point { mass = 1, position = [1, 0, 0], velocity = [0, 1, 0] }\nnew point { mass = 4, position = [-1, 0, 0], velocity = [0, -0.25, 0] }\nset system.g_constant = 0.001\nmomentum\nrun 3 steps 3\nmomentum"
 ],
 "SPHERE": [
  "new sphere { mass = 2, radius = 0.5 }\nget obj0.inertia_tensor\n0.4 * 2 * 0.5 * 0.5",
  "set system.g_constant = 0\nnew sphere { mass = 1, radius = 0.5, position = [-2, 0, 0], velocity = [1, 0, 0] }\nnew sphere { mass = 1, radius = 0.5, position = [2, 0, 0], velocity = [-1, 0, 0] }\nenergy\nrun 2 steps 2\nenergy\nget contact0.t"
 ],
 "CUBOID": [
  "new cuboid { mass = 3, half_extents = [0.5, 1, 2] }\nget obj0.inertia_tensor\n3 / 3 * (1 * 1 + 2 * 2)",
  "new cuboid { mass = 3, half_extents = [0.5, 1, 2], angular_velocity = [0.01, 3, 0.01] }\nangmom\nrun 40 steps 4\nget obj0.angular_velocity\nangmom"
 ],
 "TORUS": [
  "new torus { mass = 1, inner_radius = 1, outer_radius = 2 }\nget obj0.inertia_tensor\n1 * (1.5 * 1.5 + 0.75 * 0.5 * 0.5)\n1 * (0.5 * 1.5 * 1.5 + 0.625 * 0.5 * 0.5)",
  "new torus { mass = 1, inner_radius = 0, outer_radius = 2 }\nget obj0.ring_radius\nget obj0.tube_radius\nget obj0.inner_radius"
 ],
 "DISK": [
  "new disk { mass = 1, radius = 1 }\nget obj0.inertia_tensor\n0.5 * 1 * 1 * 1\n0.25 * 1 * 1 * 1",
  "new disk { mass = 1, radius = 1 }\nset obj0.radius = 2\nget obj0.boundary\nget obj0.inertia_tensor\n0.5 * 1 * 2 * 2"
 ],
 "CYLINDER": [
  "new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nget obj0.inertia_tensor\n0.5 * 2 * 0.5 * 0.5",
  "new cylinder { mass = 2, radius = 0.5, height = 1.5 }\nget obj0.half_height\nget obj0.inertia_tensor\n2 * (3 * 0.5 * 0.5 + 4 * 0.75 * 0.75) / 12"
 ],
 "DUMBBELL": [
  "new dumbbell { m1 = 1, m2 = 2, m_rod = 0.5 }\nget obj0.mass\nget obj0.m1\nget obj0.m2\nget obj0.position",
  "set system.g_constant = 0\nnew dumbbell { m1 = 1, m2 = 2, m_rod = 0.5, position = [-2, 0.15, 0], velocity = [1.5, 0, 0], angular_velocity = [0, 0, 0.6] }\nnew dumbbell { m1 = 2, m2 = 1, m_rod = 0.4, r1 = 0.3, r2 = 0.2, rod_radius = 0.08, length = 1.2, position = [2, -0.15, 0], velocity = [-1.5, 0, 0], angular_velocity = [0.4, 0, 0] }\nenergy\nangmom\nrun 3 steps 60\nenergy\nangmom"
 ]
}


def build_types():
    out = []
    for name, defn, rungs in TYPES:
        out.append(entry(
            id=f"type.{name}", name=name, kind="type",
            summary=f"Value type of the expression language: {name}.",
            definition=defn + "  Bracket literals are SHAPE-DIRECTED: 3 numbers make a vec3, "
                              "4 make a quaternion, 3 vec3s make a mat3, anything else stays "
                              "a generic list.",
            syntax=[rungs[0].splitlines()[-1]], returns=name,
            locations=[{"file": "posim/src/vm.rs", "line": 1, "role": "Value enum"},
                       {"file": "posim/src/lexer.rs", "line": 164, "role": "TokKind"},
                       {"file": "grammar.md", "line": 222, "role": "spec (section 4)"}],
            examples=[ex(l, c) for l, c in zip(LEVELS, rungs)],
        ))
    for shape, (defn, code) in SHAPE_DOC.items():
        out.append(entry(
            id=f"type.shape.{shape.lower()}", name=shape, kind="type",
            summary=f"NEW {shape} — {defn.split(';')[0]}.",
            definition=f"A boundary shape of `physical_object`: {defn}. Inertia is recomputed "
                       "from the final mass and shape unless you supplied `inertia_tensor` "
                       "yourself, in which case yours is kept.",
            syntax=[f"NEW {shape} [ AS <name> ] [ {{ field = expr, ... }} ]"],
            locations=[{"file": "physical_object/src/boundary.rs", "line": 1,
                        "role": "Boundary enum"},
                       {"file": "posim/src/lexer.rs", "line": 1, "role": "shape keyword"}],
            examples=[ex("trivial", f"new {shape.lower()}"), ex("intermediate", code)]
                     + [ex(l, c) for l, c in zip(["advanced", "expert"],
                                                 SHAPE_LADDER.get(shape, []))],
            seeAlso=["cmd.new", f"kw.{shape.lower()}"],
        ))
    return out


# --------------------------------------------------------------------------
# 6. notebooks and documented examples
# --------------------------------------------------------------------------

def nb_anchors():
    anchors = {}
    txt = open("dynamic_notebooks/README.md", encoding="utf-8").read()
    for m in re.finditer(r"^\| `([a-z_0-9]+)` \| (.+?) \| (.+?) \|\s*$", txt, re.M):
        anchors[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return anchors


def opens_window(path):
    """Does this notebook actually EXECUTE `SCENE CREATE`?

    Grepping the raw text is not good enough, and getting that wrong is
    how this function was first written: ten notebooks discuss the scene
    window in `#` comments without ever opening one, so a plain substring
    search says yes and the generated `scene close` then errors with `no
    scene window is open`. Strip comments the way the lexer does — `#` to
    end of line — before deciding.
    """
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line.startswith("scene ") and line.split()[1:2] == ["create"]:
            return True
    return False


def build_notebooks():
    co = json.load(open(f"{D}/corpora.json"))
    anchors = nb_anchors()
    out = []
    groups = [("dynamic_notebooks", "dynamic notebook", "notebook"),
              ("stage_notebooks", "stage notebook", "notebook"),
              ("collision_scripts", "collision script", "script")]
    for key, label, flag in groups:
        for nb in co[key]:
            name, path = nb["name"], nb["file"]
            watch, anchor = anchors.get(name, (None, None))
            defn = f"A {label}: `{path}`."
            if watch:
                defn += f" What you watch: {watch}. Numeric anchor: {anchor}."
            if key == "dynamic_notebooks":
                defn += (" It builds a system, prints its analytic baselines and ends with "
                         "`SCENE CREATE`, so launching it opens the graphical window with the "
                         "simulation loaded, Stopped and ready. The terminal STAYS "
                         "interactive: loaded cells keep their In[n] numbers and your next "
                         "command continues the numbering. No dynamic notebook runs the "
                         "simulation in batch — that is what the window's Start button is for.")
            elif key == "stage_notebooks":
                defn += (" Each stage notebook is complete in itself: it carries its own run "
                         "instructions, its own browser-GUI procedure, and its own statement "
                         "of whether the stage can be animated. `scripts/certify_clean.sh` "
                         "runs every one of them and fails the build if any emits an error "
                         "line.")
            syn = [f"cargo run -p posim --release -- --{flag} {path}"]
            if key == "dynamic_notebooks":
                syn.append(f"posim_notebook {name}")
            out.append(entry(
                id=f"nb.{name}", name=name, kind="notebook",
                summary=(watch or f"{label}: {path}"),
                definition=defn, syntax=syn,
                locations=[{"file": path, "line": 1, "role": "the notebook itself"}],
                examples=[
                    ex("trivial", syn[0], medium="shell", runner="cargo"),
                    # %load resolves the path relative to the CURRENT DIRECTORY, and a
                    # failing %load exits 0 with no Err[ line — so this fragment is only
                    # meaningful from the repository root, and the verifier must treat a
                    # `failed:` line as a failure. See _meta.failure_rules.
                    # Only close a window the file actually opened: the QM
                    # notebooks (tunneling, double_slit) write an HTML film
                    # instead and never call SCENE CREATE, so an
                    # unconditional `scene close` errors on exactly those.
                    ex("intermediate",
                       f"%load {path}" + ("\nscene close" if opens_window(path) else ""),
                       runner="posim --script (from the repository root)"),
                ],
                seeAlso=["cmd.scene.create"] if key == "dynamic_notebooks" else ["cmd.collide"],
            ))
    for r in co["rust_examples"]:
        out.append(entry(
            id=f"ex.rust.{r['name']}", name=r["name"], kind="example",
            summary=f"Self-checking Rust example in `{r['crate']}`.",
            definition=f"`{r['file']}` — a runnable, self-checking example. The six physics "
                       "examples print SUCCESS/FAILURE and exit nonzero on failure, so they "
                       "are regression anchors rather than demonstrations.",
            syntax=[f"cargo run -p {r['crate']} --release --example {r['name']}"],
            locations=[{"file": r["file"], "line": 1, "role": "the example itself"}],
            examples=[ex("trivial",
                         f"cargo run -p {r['crate']} --release --example {r['name']}",
                         medium="shell", runner="cargo")],
        ))
    return out


DEMO_SECTIONS = {
 "ex.specfn.4": [
  "assoc_legendre_p(2, 2, 0.5)\nnorm_assoc_legendre_p(2, 2, 0.5)",
  "assoc_legendre_p(100, 50, 0.3)\nnorm_assoc_legendre_p(100, 50, 0.3)",
  "norm_assoc_legendre_p(170, 170, 0.3)",
  "assoc_legendre_p(2, 0, 0.5)\nlegendre_p(2, 0.5)\nassoc_legendre_p(2, 0, 0.5) - legendre_p(2, 0.5)"
 ],
 "ex.specfn.8": [
  "eigenvalues([[2, 1], [1, 2]])",
  "eigenvalues([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])",
  "let n = 3\n4 * sin(pi / (2 * (n + 1))) * sin(pi / (2 * (n + 1)))\n4 * sin(2 * pi / (2 * (n + 1))) * sin(2 * pi / (2 * (n + 1)))\n4 * sin(3 * pi / (2 * (n + 1))) * sin(3 * pi / (2 * (n + 1)))",
  "jacobi_eigen([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])"
 ],
 "ex.specfn.10": [
  "gauss_legendre(3)",
  "gauss_legendre(2)\ngauss_legendre(8)",
  "let t = 0.7\nchebyshev_t(5, cos(t)) - cos(5 * t)",
  "sph_j(0, 1.3) - sin(1.3) / 1.3\nlegendre_p(3, 0.4) - 0.5 * (5 * 0.4 * 0.4 * 0.4 - 3 * 0.4)"
 ]
}


def build_doc_examples():
    dx = json.load(open(f"{D}/doc_examples.json"))
    try:
        TRANSCRIPTS = json.load(open(f"{D}/transcripts.json"))
    except OSError:
        TRANSCRIPTS = {}
    labels = {"grammar_md": ("ex.grammar", "grammar.md section 9"),
              "user_guide": ("ex.guide", "physical_object_simulator.md section 8"),
              "collision_detection": ("ex.collision", "collision_detection.md section 9"),
              "notebooks_md_captures": ("ex.notebook", "NOTEBOOKS.md"),
              "special_functions_md": ("ex.specfn", "special_functions.md")}
    out = []
    for key, (prefix, where) in labels.items():
        for i, e in enumerate(dx[key], 1):
            eid = f"{prefix}.{i}"
            t = TRANSCRIPTS.get(eid, {})
            exs, defn = [], (f"{e['title']} — a documented worked example with a genuine "
                             f"captured transcript, in {where}.")
            if t.get("kind") == "posim" and t.get("code"):
                # what you type, pasteable and executed by the verifier
                exs.append(ex("trivial", t["code"]))
                # and what the document published, quoted verbatim beside it —
                # medium `quoted`, so it is never badged as something this page ran
                exs.append(ex("intermediate", t["transcript"], medium="quoted",
                              runner="quoted from " + t["file"]))
                defn += ("  The first example below is the input half, stripped of its "
                         "prompts so it pastes cleanly, and it is executed by the "
                         "verifier like any other fragment. The second is the "
                         "document's own published transcript, quoted verbatim so you "
                         "can compare.")
            elif t.get("transcript"):
                exs.append(ex("trivial", t["transcript"], medium="quoted",
                              runner="quoted from " + t["file"]))
                defn += ("  Quoted verbatim from the source below. It is not a posim "
                         "fragment — it is " + ("Rust, Python or shell"
                         if t["kind"] == "rust" else "a captured session or table")
                         + " — so this page does not claim to have run it.")
            elif eid in DEMO_SECTIONS:
                # no code block in the source, but the section's CLAIM is
                # demonstrable — showing it beats quoting prose at the reader
                exs = [ex(l, c) for l, c in zip(LEVELS, DEMO_SECTIONS[eid])]
                defn += ("  The source section is prose and a table, with no code "
                         "block to quote. The examples below demonstrate the claim it "
                         "makes, and are executed like any other fragment.")
            else:
                defn += ("  This section carries no code block: it is the prose or "
                         "table half of its example, and the index says so rather "
                         "than inventing one.")
            out.append(entry(
                id=eid, name=e["title"], kind="example",
                summary=f"Worked example in {where}.",
                definition=defn,
                syntax=[f"see {e['file']}:{e['line']}"],
                locations=[{"file": e["file"], "line": e["line"], "role": "the example"}],
                examples=exs,
                status="complete" if exs else "stub",
            ))
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def main():
    catalog = []
    catalog += build_keywords()
    catalog += build_properties()
    catalog += build_builtins()
    catalog += build_magics()
    catalog += build_types()
    catalog += build_notebooks()
    catalog += build_doc_examples()

    # Tier C is NOT folded in here: it is emitted as a separate payload that
    # the app loads on demand (see build_app.py). Its bucket counts ARE
    # embedded below, so the home screen shows honest totals immediately
    # rather than numbers that jump when the second script arrives.
    tierc = []
    tierc_path = f"{D}/entries_tierc.json"
    if os.path.exists(tierc_path):
        tierc = [entry(**c) for c in json.load(open(tierc_path))]
        json.dump(tierc, open(f"{D}/catalog_c.json", "w"), indent=1)
        print(f"{len(tierc)} Tier-C entries -> {D}/catalog_c.json", file=sys.stderr)

    for path, what in ((f"{D}/entries_commands.json", "command"),
                       (f"{D}/entries_tierb.json", "Tier-B Rust")):
        if os.path.exists(path):
            catalog += [entry(**c) for c in json.load(open(path))]
        else:
            print(f"NOTE: {path} absent — no {what} entries", file=sys.stderr)

    # Cross-links, both ways. Tier B already points at its Tier-A twin
    # (special_functions::sph_j -> the builtin sph_j); without the reverse a
    # reader who arrives at the builtin never learns where it is implemented,
    # which is the direction they are more likely to travel.
    by_name = {}
    for e in catalog:
        by_name.setdefault(e["name"].split("::")[-1], []).append(e)
    added = 0
    for e in catalog:
        if e["kind"] != "builtin":
            continue
        for other in by_name.get(e["name"], []):
            if other["tier"] == "B" and other["id"] not in e["seeAlso"]:
                e["seeAlso"].append(other["id"])
                added += 1
    # A dynamic notebook and the documented example it animates are the same
    # subject reached two ways; dynamic_notebooks/README.md already states the
    # mapping, so the index should not make the reader re-derive it.
    nb_ids = {e["id"]: e for e in catalog if e["kind"] == "notebook"}
    for e in catalog:
        if e["kind"] != "example":
            continue
        for token in re.findall(r"[a-z_0-9]{4,}", e["name"].lower()):
            cand = "nb." + token
            if cand in nb_ids and cand not in e["seeAlso"]:
                e["seeAlso"].append(cand)
                nb_ids[cand]["seeAlso"].append(e["id"])
                added += 1
                break
    if added:
        print(f"added {added} cross-link(s)", file=sys.stderr)

    seen = set()
    for e in catalog:
        if e["id"] in seen:
            print(f"DUPLICATE id: {e['id']}", file=sys.stderr)
        seen.add(e["id"])

    # Carry verification forward. A rebuild regenerates every entry from the
    # inventories, which would silently discard the captured output of Phase 4
    # — and an example whose `verified` date vanished on an unrelated rebuild
    # is worse than one that never had it: the catalog would look unverified
    # while nothing had actually changed. Results are keyed on the example's
    # CODE, so an edited fragment correctly loses its date and must be re-run.
    prior = {}
    if os.path.exists(f"{D}/catalog.json"):
        try:
            for e in json.load(open(f"{D}/catalog.json")):
                for x in e.get("examples", []):
                    if x.get("verified"):
                        prior[(x["medium"], x["code"])] = (x["expected"], x["verified"])
        except (ValueError, OSError):
            pass
    carried = 0
    for e in catalog:
        for x in e["examples"]:
            # Only fill a GAP. Overwriting a date the source file already
            # supplies re-stamps a freshly verified example with the previous
            # run's date, which is how the Rust snippets ended up a day behind
            # the notebook ones.
            if x["verified"]:
                continue
            hit = prior.get((x["medium"], x["code"]))
            if hit:
                x["expected"], x["verified"] = hit
                carried += 1
    if prior:
        print(f"carried verification forward into {carried} example slot(s) "
              f"from {len(prior)} distinct fragment(s)", file=sys.stderr)

    # Prune see-also references whose target does not exist, and self-links.
    # Done here rather than in the renderer so the DATA is clean: the app
    # already filters dead refs before drawing, which would have hidden the
    # problem instead of fixing it.
    ids = {e["id"] for e in catalog}
    pruned = 0
    for e in catalog:
        keep = [r for r in e["seeAlso"] if r in ids and r != e["id"]]
        pruned += len(e["seeAlso"]) - len(keep)
        e["seeAlso"] = keep
    if pruned:
        print(f"pruned {pruned} dead see-also reference(s)", file=sys.stderr)

    tierc_counts = {}
    for e in tierc:
        for k in e["indexKeys"]:
            tierc_counts[k] = tierc_counts.get(k, 0) + 1

    meta = {
        "_meta": True,
        "tierC": {
            "file": "catalog-c.js",
            "entries": len(tierc),
            "buckets": tierc_counts,
            "kinds": {k: sum(1 for e in tierc if e["kind"] == k)
                      for k in sorted({e["kind"] for e in tierc})},
            "why_no_snippets":
                "sundials_rs is a faithful translation of a C library: its API is "
                "`&mut CVodeMem` plus a context, a matrix, a linear solver and a set "
                "of callbacks, so a one-line snippet would misrepresent how any of it "
                "is reached. The workspace ships 105 runnable example PROGRAMS whose "
                "stdout is diffed byte-for-byte against the upstream C references "
                "(sundials_rs/VERIFICATION.md); each entry links to the ones that "
                "actually call it. Status `reference` means exactly that — a full "
                "reference whose usage is demonstrated by a verified program "
                "elsewhere in the tree, as distinct from `stub`, which means nothing "
                "is there yet.",
        },
        "phase": 3,
        "schema": "prompt_01.md section 5",
        "examples_verified": True,
        # both media count: posim fragments are EXECUTED, Rust snippets are
        # COMPILED. Shell examples are neither (see shell_examples below), so
        # they are excluded from the total rather than counted as failures.
        "verified_pass": sum(1 for e in catalog for x in e["examples"]
                             if x["verified"]),
        # machine-mode fragments are EXECUTED too — omitting them from the
        # total while counting them in the pass gave the home screen the
        # nonsense headline "1435 of 1411 examples verified".
        "verified_total": sum(1 for e in catalog for x in e["examples"]
                              if x["medium"] in EXECUTED_MEDIA),
        # read off the examples rather than asserted: the meta cannot claim
        # a date no example actually carries
        "verified_date": max((x["verified"] for e in catalog
                              for x in e["examples"] if x["verified"]),
                             default=None),
        "verified_by": {
            "posim": "executed with `posim --script` from the repository root; "
                     "output captured into `expected`",
            "rust": "compiled by `cargo build -p posim --example …` via "
                    "tools/verify_tierb_examples.py; `expected` reads `compiles`",
            "machine": "executed through the JSONL protocol with `posim --machine`; "
                       "a reply carrying \"ok\":false is the machine-mode equivalent "
                       "of an Err[] line",
            "quoted": "NOT run by this page — a transcript quoted verbatim from the "
                      "document that published it",
            "shell": "NOT run by this page — a command you run yourself, pointing at a "
                     "program verified elsewhere",
        },
        "shell_examples": {
            "verified": False,
            "why": [
                "The `--notebook` launch lines load a file and then STAY "
                "INTERACTIVE by design, so they never exit and cannot be "
                "batch-verified. Their content is verified instead by the "
                "`%load <file>` rung on the same entry, which runs the same "
                "file through `--script`.",
                "The `cargo run --example` lines are runnable programs the "
                "project already exercises elsewhere — the six physical_object "
                "physics examples are self-checking, and the sundials_rs ones "
                "are diffed byte-for-byte against the upstream C references in "
                "sundials_rs/VERIFICATION.md. This pass does not re-run them, "
                "so it does not claim them.",
            ],
        },
        "failure_rules": [
            "A posim fragment FAILS if the process exits nonzero.",
            "A posim fragment FAILS if stdout contains any line starting `Err[`.",
            "A posim fragment FAILS if stdout contains a line matching `^%\\w+ .* failed:` "
            "— a failing magic (notably %load) exits 0 and emits NO Err[ line, so exit "
            "status alone would pass it vacuously.",
            "posim fragments must be run with cwd = the repository root: %load and "
            "%save resolve paths relative to the current directory.",
            "POSIM_NO_BROWSER=1 must be set so SCENE CREATE does not launch a browser.",
            "Captured output has run-to-run variation normalised: the "
            "OS-assigned scene port becomes <port>, and the playback step and "
            "history counters become <varies> because playback advances on "
            "wall-clock time. Solver step counts are NOT normalised — those "
            "are deterministic, and they are the anchors the documentation "
            "pins.",
        ],
    }
    json.dump([meta] + catalog, open(f"{D}/catalog.json", "w"), indent=1)
    print(f"{len(catalog)} entries (+1 _meta) -> {D}/catalog.json")


if __name__ == "__main__":
    main()
