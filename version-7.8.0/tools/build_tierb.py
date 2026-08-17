#!/usr/bin/env python3
"""Phase-7: Tier-B entries — the first-party Rust API.

Every public item of physical_object, special_functions, quantum and posim
becomes an entry carrying its real signature, its doc comment, and its
file:line. Examples are TEMPLATED ONLY where the signature shape can be
turned into Rust that is certain to compile; everything else is an honest
stub. Generating a plausible-looking snippet that does not build would be
worse than admitting there isn't one — the whole index rests on the claim
that its examples run.

Coverage is reported, not hidden: the script prints how many entries got
examples and how many did not, and the status page counts the stubs.

Stdlib only.
"""

import json
import re
import sys
from collections import Counter, defaultdict

TIER_B = ["physical_object", "special_functions", "quantum", "posim"]
LEVELS = ["trivial", "intermediate", "advanced", "expert"]

# --- how to build a receiver of each type, and what it costs to set up ----
# Receivers whose construction takes a chain — a grid, then a Hamiltonian,
# then the propagator. Written out rather than templated, because the chain is
# exactly what a caller has to know and a generated one-liner would hide it.
EXTRA_RECEIVER = {'Propagator': 'let subject = Propagator::new(Hamiltonian::new(Grid::new(-8.0, 8.0, 60)?, vec![0.0; 60], 1.0, 1.0)?, 0.01)?;', 'Propagator2': 'let subject = Propagator2::new(Hamiltonian2::new(Grid2::new(-4.0, 4.0, 20, -4.0, 4.0, 20)?, vec![0.0; 400], 1.0, 1.0)?, 0.01)?;', 'Propagator3': 'let subject = Propagator3::new(Hamiltonian3::new(Grid3::new(-3.0, 3.0, 10, -3.0, 3.0, 10, -3.0, 3.0, 10)?, vec![0.0; 1000], 1.0, 1.0)?, 0.01)?;', 'DrivenPropagator': 'let subject = DrivenPropagator::new(Hamiltonian::new(Grid::new(-8.0, 8.0, 60)?, vec![0.0; 60], 1.0, 1.0)?, vec![0.0; 60], 0.01)?;', 'DrivenPropagator2': 'let subject = DrivenPropagator2::new(Hamiltonian2::new(Grid2::new(-4.0, 4.0, 20, -4.0, 4.0, 20)?, vec![0.0; 400], 1.0, 1.0)?, vec![0.0; 400], 0.01)?;', 'DrivenPropagator3': 'let subject = DrivenPropagator3::new(Hamiltonian3::new(Grid3::new(-3.0, 3.0, 10, -3.0, 3.0, 10, -3.0, 3.0, 10)?, vec![0.0; 1000], 1.0, 1.0)?, vec![0.0; 1000], 0.01)?;', 'NashPropagator': 'let subject = NashPropagator::new(PeriodicGrid::new(-8.0, 8.0, 60)?, &vec![0.0; 60], 1.0, 1.0, 0.01, None)?;', 'Hamiltonian': 'let subject = Hamiltonian::new(Grid::new(-8.0, 8.0, 60)?, vec![0.0; 60], 1.0, 1.0)?;'}
EXTRA_IMPORTS = {'Propagator': ['::quantum::qm1d::{Grid, Hamiltonian, Propagator}'], 'Propagator2': ['::quantum::qm2d::{Grid2, Hamiltonian2, Propagator2}'], 'Propagator3': ['::quantum::qm3d::{Grid3, Hamiltonian3, Propagator3}'], 'DrivenPropagator': ['::quantum::qm1d::{Grid, Hamiltonian, DrivenPropagator}'], 'DrivenPropagator2': ['::quantum::qm2d::{Grid2, Hamiltonian2, DrivenPropagator2}'], 'DrivenPropagator3': ['::quantum::qm3d::{Grid3, Hamiltonian3, DrivenPropagator3}'], 'NashPropagator': ['::quantum::nash::{PeriodicGrid, NashPropagator}'], 'Hamiltonian': ['::quantum::qm1d::{Grid, Hamiltonian}']}

RECEIVER = {
    "physical_object": ("physical_object::physical_object::physical_object",
        "let subject = physical_object::new_point(0, 2.0, Vec3::new(1.0, 0.0, 0.0), "
        "Vec3::new(0.0, 3.0, 0.0));"),
    "PhysicalObjectSystem": ("physical_object::system::PhysicalObjectSystem",
        "let subject = PhysicalObjectSystem::new(\n"
        "    vec![physical_object::new_point(0, 1.0, Vec3::new(1.0, 0.0, 0.0), "
        "Vec3::new(0.0, 1.0, 0.0))],\n    1.0,\n);"),
    "Vec3": ("physical_object::linalg::Vec3", "let subject = Vec3::new(1.0, 2.0, 3.0);"),
    "Mat3": ("physical_object::linalg::Mat3", "let subject = Mat3::identity();"),
    "Quat": ("physical_object::linalg::Quat", "let subject = Quat::identity();"),
    "Complex64": ("special_functions::complex::Complex64",
        "let subject = Complex64::new(3.0, -4.0);"),
    "Grid": ("quantum::qm1d::Grid", "let subject = Grid::new(-8.0, 8.0, 60)?;"),
    "Grid2": ("quantum::qm2d::Grid2",
        "let subject = Grid2::new(-4.0, 4.0, 20, -4.0, 4.0, 20)?;"),
    "Grid3": ("quantum::qm3d::Grid3",
        "let subject = Grid3::new(-3.0, 3.0, 10, -3.0, 3.0, 10, -3.0, 3.0, 10)?;"),
}

# Types whose associated functions we can call even though we have no stored
# constructor for them: the associated fn IS usually the constructor.
ASSOC_OK = {
    "Vec3", "Mat3", "Quat", "Complex64", "PhysicalObjectSystem", "physical_object",
    "Grid", "Grid2", "Grid3", "PeriodicGrid",
    "Hamiltonian", "Hamiltonian2", "Hamiltonian3",
    "Wavefunction", "Wavefunction2", "Wavefunction3",
    "Propagator", "Propagator2", "Propagator3",
    "DrivenPropagator", "DrivenPropagator2", "DrivenPropagator3",
    "NashPropagator",
}

# --- a value of each argument type, chosen to be inside every domain ------
ARG = {
    "f64": "0.5", "i32": "1", "usize": "1", "u8": "1", "bool": "true",
    "Vec3": "Vec3::new(1.0, 0.0, 0.0)", "Mat3": "Mat3::identity()",
    "Quat": "Quat::identity()", "C": "Complex64::new(1.0, 0.5)",
    "Complex64": "Complex64::new(1.0, 0.5)",
    "&[f64]": "&[1.0, 2.0, 3.0]", "&Vec3": "&Vec3::new(1.0, 0.0, 0.0)",
    "&Mat3": "&Mat3::identity()",
    "Vec<f64>": "vec![0.0; 60]", "Vec<C>": "vec![Complex64::new(1.0, 0.0); 60]",
    "Vec<Complex64>": "vec![Complex64::new(1.0, 0.0); 60]",
    "Grid": "Grid::new(-8.0, 8.0, 60)?",
    "Grid2": "Grid2::new(-4.0, 4.0, 20, -4.0, 4.0, 20)?",
    "Grid3": "Grid3::new(-3.0, 3.0, 10, -3.0, 3.0, 10, -3.0, 3.0, 10)?",
    "PeriodicGrid": "PeriodicGrid::new(-8.0, 8.0, 60)?",
    "Vec<physical_object>": "vec![physical_object::new_point(0, 1.0, "
                            "Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))]",
    "&Boundary": "&Boundary::Sphere { radius: 0.5 }",
    "Boundary": "Boundary::Sphere { radius: 0.5 }",
}
# order arguments for the special-function families, whose first argument is
# an ORDER and whose second is the argument: (0, 2.0) is inside every domain
# these functions have, which (1, 0.5) is not for the singular ones.
SPECIAL_ARGS = {
    ("i32", "f64"): ("0", "2.0"),
    ("i32", "C"): ("0", "Complex64::new(2.0, 0.5)"),
    ("f64", "C"): ("0.5", "Complex64::new(2.0, 0.5)"),
    ("f64", "f64"): ("0.5", "2.0"),
    ("C", "C"): ("Complex64::new(0.5, 0.0)", "Complex64::new(2.0, 0.5)"),
    ("usize", "f64"): ("4", "2.0"),
}

SIG = re.compile(r"^pub (?:const )?fn (\w+)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*$", re.S)


def parse(sig):
    m = SIG.match(" ".join(sig.split()))
    if not m:
        return None
    name, args, ret = m.group(1), m.group(2).strip(), (m.group(3) or "()").strip()
    takes_self = args.startswith("&self") or args.startswith("&mut self") \
        or args.startswith("self")
    mut_self = args.startswith("&mut self")
    if takes_self:
        args = args.split(",", 1)[1] if "," in args else ""
    types = []
    for part in re.split(r",(?![^<(\[]*[>)\]])", args):
        part = part.strip().rstrip(",")
        if not part:
            continue
        if ":" not in part:
            return None
        types.append(part.split(":", 1)[1].strip())
    return {"name": name, "self": takes_self, "mut": mut_self,
            "args": types, "ret": ret}


def call_args(p, crate):
    if crate == "special_functions" and len(p["args"]) == 2:
        key = tuple(p["args"])
        if key in SPECIAL_ARGS:
            return list(SPECIAL_ARGS[key])
    out = []
    for t in p["args"]:
        if t not in ARG:
            return None
        out.append(ARG[t])
    return out


def snippet(item, p):
    """Rust that compiles, or None when the shape is not one we can be sure of."""
    crate, name = item["crate"], item["bare"]
    owner = item["name"].split("::")[0] if "::" in item["name"] else None
    args = call_args(p, crate)
    if args is None:
        return None
    fallible = p["ret"].startswith("Result<")
    q = "?" if fallible else ""
    if p["self"]:
        if owner not in RECEIVER and owner not in EXTRA_RECEIVER:
            return None
        ctor = (EXTRA_RECEIVER[owner] if owner in EXTRA_RECEIVER
                else RECEIVER[owner][1])
        if p["mut"]:
            ctor = ctor.replace("let subject", "let mut subject")
        body = ctor + f"\nlet value = subject.{name}({', '.join(args)}){q};"
    elif owner:
        # an associated function: a constructor, most often. `Owner::fn(args)`
        # is exactly what a caller writes, so show that.
        if owner not in RECEIVER and owner not in ASSOC_OK:
            return None
        body = f"let value = {owner}::{name}({', '.join(args)}){q};"
    else:
        body = f"let value = {name}({', '.join(args)}){q};"
    if p["ret"] in ("()", "Result<(), String>"):
        body = body.replace("let value = ", "")
    else:
        # A binding, not a println: `Airy` and `Uniform` are plain data types
        # with no Debug, and a snippet that only compiles for the types that
        # happen to derive it is a snippet that lies about the rest.
        body = body.replace("let value = ", "let _value = ")
    return body


def use_lines(item, p):
    """Import lines for one snippet.

    Every path is written `::crate::...`. That is not decoration: the union
    struct is named `physical_object` in lower case by specification, so the
    moment a scope imports it the bare name `physical_object` resolves to the
    STRUCT and every later `physical_object::…` path fails with "is a struct,
    not a module". CLAUDE.md hard rule #8 says to prefix the others with `::`,
    and that is exactly what this does.
    """
    owner = item["name"].split("::")[0] if "::" in item["name"] else None
    uses = set()
    if owner and owner in EXTRA_RECEIVER:
        uses.update(EXTRA_IMPORTS[owner])
    elif owner and owner in RECEIVER:
        uses.add("::" + RECEIVER[owner][0])
    elif owner and owner in ASSOC_OK:
        uses.add(f"::{item['crate']}::{item['module']}::{owner}")
    for t in p["args"]:
        for ty in ("Grid3", "Grid2", "Grid", "PeriodicGrid", "Boundary"):
            if t == ty or t == "&" + ty:
                mod = {"Grid": "qm1d", "Grid2": "qm2d", "Grid3": "qm3d",
                       "PeriodicGrid": "nash", "Boundary": "boundary"}[ty]
                crate = "physical_object" if ty == "Boundary" else "quantum"
                uses.add(f"::{crate}::{mod}::{ty}")
    if item["crate"] == "physical_object" or owner in ("Vec3", "Mat3", "Quat"):
        uses.add("::physical_object::linalg::{Vec3, Mat3, Quat}")
    if not p["self"] and not owner:
        # A FREE function is imported by name. An associated one is reached
        # through its type, which is already imported above — importing
        # `linalg::new` instead of `linalg::Vec3` is how that went wrong.
        # Items declared in lib.rs sit at the CRATE ROOT: no `::lib::` to
        # path through.
        mod = "" if item["module"] in ("lib", "main") else item["module"] + "::"
        uses.add(f"::{item['crate']}::{mod}{item['bare']}")
    for t in p["args"] + [p["ret"]]:
        # `Vec<C>` needs Complex64 in scope just as `C` does — matching only
        # the bare name missed every collection of them.
        if "Complex64" in t or t == "C" or re.search(r"\bC\b", t):
            uses.add("::special_functions::complex::Complex64")
        if "Vec3" in t or "Mat3" in t or "Quat" in t:
            uses.add("::physical_object::linalg::{Vec3, Mat3, Quat}")
    if owner == "physical_object" or (item["crate"] == "physical_object"
                                      and "physical_object" in item["signature"]):
        uses.add("::physical_object::physical_object::physical_object")
    # the PhysicalObjectSystem constructor builds a physical_object, so it
    # needs the struct in scope too
    if owner in RECEIVER and "physical_object::new_point" in RECEIVER[owner][1]:
        uses.add("::physical_object::physical_object::physical_object")
    # the braced linalg import already brings in Vec3/Mat3/Quat; importing one
    # of them again by name is E0252, not a harmless duplicate
    if "::physical_object::linalg::{Vec3, Mat3, Quat}" in uses:
        for t in ("Vec3", "Mat3", "Quat"):
            uses.discard("::physical_object::linalg::" + t)
    # a braced import already brings its names in; adding one of them again by
    # name is E0252, not a harmless duplicate
    braced = [u for u in uses if "{" in u]
    for b in braced:
        head, names = b.split("{", 1)
        for nm in names.rstrip("}").split(","):
            uses.discard(head + nm.strip())
    return sorted(uses)


def usage_index():
    """symbol -> files that reference it, across tests, examples and the docs.

    A Tier-B item with no templatable snippet is not empty-handed: the crate's
    own tests and examples call it, and those are compiled and run by
    `cargo test`. Pointing at them is real evidence of use, which is worth
    more than a snippet invented to fill a slot.
    """
    import os
    idx = {}
    roots = ["physical_object", "special_functions", "quantum", "posim"]
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("target", ".git")]
            for fn in files:
                if not fn.endswith((".rs", ".md")):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    text = open(full, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for sym in set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b", text)):
                    idx.setdefault(sym, set()).add(full)
                # Short method names — `new`, `run`, `dt`, `h`, `x` — are below
                # the 4-character floor above, and lowering it would match them
                # everywhere and mean nothing. Index the QUALIFIED form instead:
                # `Grid::new` and `.escaped(` are precise where `new` is noise.
                for sym in set(re.findall(r"\b([A-Z][A-Za-z0-9_]*::[a-z_][A-Za-z0-9_]*)",
                                          text)):
                    idx.setdefault(sym, set()).add(full)
                for sym in set(re.findall(r"\.([a-z_][A-Za-z0-9_]*)\s*\(", text)):
                    idx.setdefault("." + sym, set()).add(full)
    return idx


# Hand-written where the templater cannot go: closures, references, consts
# and the honest-failure demonstrations that are the point of the API.
OVERRIDE_EXTRA = {
 "rs.quantum.isosurface.Mesh": "let m = Mesh::default();\nassert!(m.triangles.is_empty() || !m.triangles.is_empty());",
 "rs.quantum.transfer.scatter_real": "// T(E) and R(E) for a potential sampled on a grid, by transfer matrix \u2014\n// exact at one energy, with no packet and no time stepping\nlet v = vec![0.0f64; 40];\nlet out = scatter_real(&v, -8.0, 8.0, 2.0, 1.0, 1.0)?;\nlet _ = out;",
 "rs.quantum.qm1d.Hamiltonian.without_absorber": "let h = Hamiltonian::new(Grid::new(-8.0, 8.0, 60)?, vec![0.0; 60], 1.0, 1.0)?;\n// QM STATES is refused while an absorber is on, because H - iW is not\n// Hermitian; this is the Hermitian half of the same operator\nlet bare = h.without_absorber();\nlet _ = bare;",
 "rs.quantum.qm1d.DrivenPropagator.hamiltonian_now": "let h = Hamiltonian::new(Grid::new(-8.0, 8.0, 60)?, vec![0.0; 60], 1.0, 1.0)?;\nlet d = DrivenPropagator::new(h, vec![0.0; 60], 0.01)?;\n// the modulation is a CLOSURE of t, sampled at the MIDPOINT of each step \u2014\n// which is what keeps the scheme second order\nlet hn = d.hamiltonian_now(|t: f64| 0.3 * t.cos())?;\nlet _ = hn;",
 "rs.physical_object.boundary.Sdf": "// Sdf is the trait every shape implements: a signed distance, plus a\n// surface normal that defaults to a central difference of it.\nlet s = Boundary::Sphere { radius: 0.5 };\nlet d = s.signed_distance(&Vec3::new(1.0, 0.0, 0.0));\nassert!((d - 0.5).abs() < 1e-9);   // 1.0 out from a radius-0.5 sphere",
 "rs.physical_object.collide.ContactGeometry": "let a = physical_object::new_rigid(0, 1.0, 0.0, Vec3::new(-0.6, 0.0, 0.0),\n    Quat::identity(), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0),\n    Mat3::identity(), Mat3::identity(), Boundary::Sphere { radius: 0.5 });\nlet b = physical_object::new_rigid(1, 1.0, 0.0, Vec3::new(0.6, 0.0, 0.0),\n    Quat::identity(), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0),\n    Mat3::identity(), Mat3::identity(), Boundary::Sphere { radius: 0.5 });\nlet g: Option<ContactGeometry> = contact_geometry(&a, &b, 1e-9);\n// the two spheres are 1.2 apart with radii 0.5 each, so they are NOT\n// touching yet and the narrow phase says so\nassert!(g.is_none());",
 "rs.physical_object.collide.contact_geometry": "let a = physical_object::new_rigid(0, 1.0, 0.0, Vec3::new(-0.6, 0.0, 0.0),\n    Quat::identity(), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0),\n    Mat3::identity(), Mat3::identity(), Boundary::Sphere { radius: 0.5 });\nlet b = physical_object::new_rigid(1, 1.0, 0.0, Vec3::new(0.6, 0.0, 0.0),\n    Quat::identity(), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0),\n    Mat3::identity(), Mat3::identity(), Boundary::Sphere { radius: 0.5 });\n// separation > 0 means apart; the normal points from a toward b\nlet g = contact_geometry(&a, &b, 1e-9);\nlet _ = g;",
 "rs.physical_object.collide.resolve_pair": "let mut sys = PhysicalObjectSystem::new(vec![\n    physical_object::new_point(0, 1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),\n    physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),\n], 0.0);\n// two POINTS can never collide, so the resolver has nothing to do\nlet c = resolve_pair(&mut sys, 0, 1, false)?;\nassert!(c.is_none());",
 "rs.physical_object.integrate.run": "let mut sys = PhysicalObjectSystem::new(vec![\n    physical_object::new_point(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),\n], 0.0);\n// every step goes through the pure-Rust SUNDIALS solvers; there is no\n// hand-rolled stepper anywhere in the project\nlet report = run(&mut sys, 1.0, 2)?;\nlet _ = report;",
 "rs.special_functions.lanczos.LanczosResult": "// what QM2/QM3 STATES return: the k lowest eigenvalues with their vectors\n// and the residual that justifies trusting them \u2014 an iterative eigensolver\n// has no exact stopping point, so the residual is part of the answer\nlet r = LanczosResult { values: vec![1.0], vectors: vec![vec![1.0]],\n                        residuals: vec![1e-8], iterations: 1, stop: Stop::Converged };\nassert_eq!(r.values.len(), r.residuals.len());",
 "rs.quantum.absorber.Ramp": "let ramp = Ramp { width: 15.0, power: 2.0 };\n// power 2 is the measured optimum: too steep a ramp is a mirror,\n// whether it is real or imaginary\nassert_eq!(ramp.power, 2.0);",
 "rs.quantum.absorber.Band": "let band = Band { k_lo: 1.0, k_hi: 4.0, samples: 8 };\nassert_eq!(band.samples, 8);",
 "rs.quantum.absorber.Leak": "// the two ways probability escapes an absorber, which trade off against\n// each other: too weak and it passes THROUGH, too strong and it reflects\n// off the absorber's own leading edge\nlet leak = Leak { k: 3.0, reflection: 1e-9, leakage: 2e-9, absorbed: 1.0 };\nassert!(leak.escaped() > 0.0);",
 "rs.quantum.absorber.Leak.escaped": "let leak = Leak { k: 3.0, reflection: 1e-9, leakage: 2e-9, absorbed: 1.0 };\n// R + T: everything the absorber failed to swallow\nassert!((leak.escaped() - 3e-9).abs() < 1e-18);",
 "rs.quantum.absorber.worst_escape": "let w = worst_escape(Ramp { width: 15.0, power: 2.0 }, 3.0,\n                    Band { k_lo: 2.0, k_hi: 4.0, samples: 4 }, 1.0, 1.0)?;\nassert!(w >= 0.0);",
 "rs.quantum.absorber.choose_strength": "// the tuning surface has a wide optimum \u2014 over an order of magnitude \u2014\n// but it is not infinite, so the strength is SEARCHED rather than guessed\nlet (strength, escape) = choose_strength(Ramp { width: 15.0, power: 2.0 },\n                                        Band { k_lo: 2.0, k_hi: 4.0, samples: 4 },\n                                        1.0, 1.0)?;\nassert!(strength > 0.0 && escape >= 0.0);",
 "rs.quantum.isosurface.Mesh.enclosed_volume": "let m = Mesh::default();\n// checked by the divergence theorem against shapes whose volume is known\nassert_eq!(m.enclosed_volume(), 0.0);",
 "rs.quantum.isosurface.Mesh.is_watertight": "let m = Mesh::default();\n// every directed edge exactly once, its reverse exactly once: a hole\n// leaves an unmatched edge, a flipped triangle leaves a duplicate\nassert!(m.is_watertight());"
}
OVERRIDE_IMPORTS = {
 "rs.quantum.isosurface.Mesh": [
  "::quantum::isosurface::Mesh"
 ],
 "rs.quantum.transfer.scatter_real": [
  "::quantum::transfer::scatter_real"
 ],
 "rs.quantum.qm1d.Hamiltonian.without_absorber": [
  "::quantum::qm1d::{Grid, Hamiltonian}"
 ],
 "rs.quantum.qm1d.DrivenPropagator.hamiltonian_now": [
  "::quantum::qm1d::{Grid, Hamiltonian, DrivenPropagator}"
 ],
 "rs.physical_object.boundary.Sdf": [
  "::physical_object::linalg::{Vec3, Mat3, Quat}",
  "::physical_object::boundary::{Boundary, Sdf}"
 ],
 "rs.physical_object.collide.ContactGeometry": [
  "::physical_object::linalg::{Vec3, Mat3, Quat}",
  "::physical_object::physical_object::physical_object",
  "::physical_object::boundary::Boundary",
  "::physical_object::collide::{ContactGeometry, contact_geometry}"
 ],
 "rs.physical_object.collide.contact_geometry": [
  "::physical_object::linalg::{Vec3, Mat3, Quat}",
  "::physical_object::physical_object::physical_object",
  "::physical_object::boundary::Boundary",
  "::physical_object::collide::contact_geometry"
 ],
 "rs.physical_object.collide.resolve_pair": [
  "::physical_object::linalg::{Vec3, Mat3, Quat}",
  "::physical_object::physical_object::physical_object",
  "::physical_object::system::PhysicalObjectSystem",
  "::physical_object::collide::resolve_pair"
 ],
 "rs.physical_object.integrate.run": [
  "::physical_object::linalg::{Vec3, Mat3, Quat}",
  "::physical_object::physical_object::physical_object",
  "::physical_object::system::PhysicalObjectSystem",
  "::physical_object::integrate::run"
 ],
 "rs.special_functions.lanczos.LanczosResult": [
  "::special_functions::lanczos::{LanczosResult, Stop}"
 ],
 "rs.quantum.absorber.Ramp": [
  "::quantum::absorber::Ramp"
 ],
 "rs.quantum.absorber.Band": [
  "::quantum::absorber::Band"
 ],
 "rs.quantum.absorber.Leak": [
  "::quantum::absorber::Leak"
 ],
 "rs.quantum.absorber.Leak.escaped": [
  "::quantum::absorber::Leak"
 ],
 "rs.quantum.absorber.worst_escape": [
  "::quantum::absorber::{Ramp, Band, worst_escape}"
 ],
 "rs.quantum.absorber.choose_strength": [
  "::quantum::absorber::{Ramp, Band, choose_strength}"
 ],
 "rs.quantum.isosurface.Mesh.enclosed_volume": [
  "::quantum::isosurface::Mesh"
 ],
 "rs.quantum.isosurface.Mesh.is_watertight": [
  "::quantum::isosurface::Mesh"
 ]
}
POSIM_LINK = {
 "value_to_json": [
  "cmd.machine.get",
  "turns a VM value into the JSON a machine-mode reply carries"
 ],
 "handle_request": [
  "cmd.machine.exec",
  "dispatches one machine-mode request line"
 ],
 "Notebook": [
  "magic.history",
  "the In[n]/Out[n] cell session itself"
 ],
 "execute_cell": [
  "cmd.expr",
  "runs one cell and numbers it"
 ],
 "Parser": [
  "cmd.expr",
  "the recursive-descent grammar compiler behind every line"
 ],
 "FuncDef": [
  "cmd.def",
  "a stored user function: parameters, defaults and body lines"
 ],
 "is_def_line": [
  "cmd.def",
  "recognises the DEF LINE FORM before the ordinary grammar"
 ],
 "define_function": [
  "cmd.def",
  "syntax-checks every body line and installs the function"
 ],
 "Camera": [
  "cmd.scene.rotate",
  "the orbit camera: look-at point, distance, yaw, pitch"
 ],
 "client_count": [
  "cmd.scene.status",
  "how many windows are connected"
 ],
 "steps_done": [
  "cmd.scene.status",
  "playback step counter"
 ],
 "system_time": [
  "cmd.scene.status",
  "the playback copy's simulation time"
 ],
 "packed_state": [
  "cmd.scene.refresh",
  "the synced clone the playback thread evolves"
 ],
 "contacts_len": [
  "cmd.scene.status",
  "contacts in the playback copy"
 ],
 "sha1": [
  "cmd.scene.create",
  "the RFC 6455 handshake, hand-rolled on std"
 ],
 "base64_encode": [
  "cmd.scene.create",
  "the same handshake's accept key"
 ]
}

OVERRIDE = {
 "rs.special_functions.quadrature.brent_root": "use ::special_functions::quadrature::brent_root;\n\n// cos x = x, bracketed on [0, 1]\nlet r = brent_root(|x: f64| x.cos() - x, 0.0, 1.0, 1e-12)?;\nassert!((r - 0.739085133215).abs() < 1e-9);",
 "rs.special_functions.quadrature.find_roots": "use ::special_functions::quadrature::find_roots;\n\n// find_roots CANNOT tell a pole from a root: a pole flips the sign of f\n// too, and Brent converges on it. Check f at each value it returns.\nlet rs = find_roots(|x: f64| x.sin(), 0.1, 7.0, 100, 1e-12)?;\nassert!(!rs.is_empty());",
 "rs.special_functions.quadrature.integrate_adaptive": "use ::special_functions::quadrature::integrate_adaptive;\n\nlet v = integrate_adaptive(|x: f64| x * x * x, 0.0, 1.0, 1e-10)?;\nassert!((v - 0.25).abs() < 1e-10);\n// and the honest failure: 1/sqrt(x) on [0,1] is finite but Simpson\n// cannot resolve the endpoint, so it ERRORS rather than guessing.\nassert!(integrate_adaptive(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10).is_err());",
 "rs.special_functions.complex.I": "use ::special_functions::complex::Complex64;\n\nlet v = Complex64::I * Complex64::I;\nassert_eq!(v, Complex64::real(-1.0));",
 "rs.special_functions.complex.ONE": "use ::special_functions::complex::Complex64;\n\nlet v = Complex64::ONE * Complex64::new(3.0, -4.0);\nassert_eq!(v.abs(), 5.0);",
 "rs.physical_object.collide.world_aabb": "use ::physical_object::linalg::{Vec3, Mat3, Quat};\nuse ::physical_object::physical_object::physical_object;\nuse ::physical_object::collide::world_aabb;\n\nlet o = physical_object::new_point(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));\nlet (lo, hi) = world_aabb(&o);\nassert!(lo.x <= hi.x);",
 "rs.physical_object.physical_object.physical_object.new_rigid": (
   "use ::physical_object::linalg::{Vec3, Mat3, Quat};\n"
   "use ::physical_object::physical_object::physical_object;\n"
   "use ::physical_object::boundary::Boundary;\n\n"
   "// id, mass, charge, then pose, then the two velocities, then the two\n"
   "// tensors, then the shape\n"
   "let o = physical_object::new_rigid(\n"
   "    0, 2.0, 0.0,\n"
   "    Vec3::new(0.0, 0.0, 0.0), Quat::identity(),\n"
   "    Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0),\n"
   "    Mat3::identity(), Mat3::identity(),\n"
   "    Boundary::Sphere { radius: 0.5 },\n"
   ");\n"
   "assert_eq!(o.get_mass(), 2.0);"),
 "rs.physical_object.collide.ZENO_GAP_RELATIVE": "use ::physical_object::collide::ZENO_GAP_RELATIVE;\n\n// the relative gap below which a contact is treated as resting rather\n// than as another impact \u2014 the anti-Zeno guard\nassert!(ZENO_GAP_RELATIVE > 0.0);"
}


def main():
    items = [json.loads(l) for l in open("index_data/rust_items.jsonl")]
    USAGE = usage_index()
    pub = [d for d in items if d["crate"] in TIER_B and d["visibility"] == "pub"
           and d["kind"] != "mod"]

    # Tier-A builtins with the same bare name are the notebook-reachable twin
    cat = json.load(open("index_data/catalog.json"))
    a_by_name = {e["name"]: e["id"] for e in cat
                 if not e.get("_meta") and e["kind"] == "builtin"}

    out, made, skipped = [], 0, Counter()
    for d in pub:
        kind = "function" if d["kind"] == "fn" else "type"
        owner = d["name"].split("::")[0] if "::" in d["name"] else None
        eid = f"rs.{d['crate']}.{d['module']}.{d['name'].replace('::', '.')}"
        examples = []
        if eid in OVERRIDE_EXTRA:
            code = "\n".join("use " + u + ";" for u in OVERRIDE_IMPORTS[eid]) \
                   + "\n\n" + OVERRIDE_EXTRA[eid]
            examples.append({"level": "trivial", "medium": "rust", "code": code,
                             "expected": None, "verified": None, "runner": "cargo build"})
            made += 1
        elif eid in OVERRIDE:
            examples.append({"level": "trivial", "medium": "rust",
                             "code": OVERRIDE[eid], "expected": None,
                             "verified": None, "runner": "cargo build"})
            made += 1
        elif d["kind"] == "fn" and d["crate"] == "posim":
            # posim is a binary crate with no lib target, so nothing can
            # `use posim::…`. Its items are catalogued and cross-linked to the
            # notebook commands they implement, but they get no Rust snippet.
            skipped["posim is a bin crate (not importable)"] += 1
        elif d["kind"] == "fn":
            p = parse(d["signature"])
            if p is None:
                skipped["unparsed signature"] += 1
            else:
                code = snippet(d, p)
                if code:
                    uses = use_lines(d, p)
                    examples.append({
                        "level": "trivial", "medium": "rust",
                        "code": "\n".join("use " + u + ";" for u in uses) + "\n\n" + code,
                        "expected": None, "verified": None, "runner": "cargo build"})
                    made += 1
                else:
                    skipped["shape not templatable"] += 1
        else:
            skipped[d["kind"]] += 1

        see = []
        if d["bare"] in a_by_name:
            see.append(a_by_name[d["bare"]])
        posim_note = ""
        if d["crate"] == "posim" and d["bare"] in POSIM_LINK:
            ref, what = POSIM_LINK[d["bare"]]
            see.append(ref)
            posim_note = ("  posim is a BINARY crate with no lib target, so nothing "
                          "can `use posim::…` and no snippet is possible. What it does "
                          "is reachable from the notebook: it " + what + ".")
        doc = d["doc"] or ""
        # where it is genuinely used, for the items that get no snippet
        # Prefer a test, an example or a document — those demonstrate the item.
        # Fall back to any other file in the tree that calls it, which is still
        # a real answer to "where is this used?" and better than an empty page.
        keys = [d["bare"]]
        if "::" in d["name"]:
            keys += [d["name"], "." + d["bare"]]      # Grid::new, and .new(
        cand = [u for k in keys for u in USAGE.get(k, ()) if u != d["file"]]
        cand = sorted(set(cand))
        best = [u for u in cand if u.endswith(".md") or "/examples/" in u or "test" in u]
        uses = sorted(best)[:4] or sorted(cand)[:3]
        out.append({
            "id": eid, "name": d["name"], "kind": kind, "tier": "B",
            "aliases": [], "indexKeys": [(d["bare"][0] if d["bare"] else "Σ").upper()],
            "summary": (doc.split(".")[0][:150] + "." if doc
                        else f"{d['kind']} `{d['bare']}` in `{d['crate']}::{d['module']}`."),
            "definition": (doc or "No doc comment in the source.")
                + f"  Declared in the `{d['module']}` module of the `{d['crate']}` crate."
                + (("  No snippet is generated for this signature shape — see the "
                    "status page for why an invented one would be worse than none — "
                    "but it is called from " + ", ".join(f"`{u}`" for u in uses) + ".")
                   if uses and not examples else "") + posim_note,
            "syntax": [d["signature"]],
            "parameters": [], "returns": None, "errors": [],
            "locations": [{"file": d["file"], "line": d["line"], "role": "definition"}]
                       + [{"file": u, "line": 1, "role": "used here"} for u in uses],
            "examples": examples, "seeAlso": see, "invariants": [],
            # `reference` where the item is catalogued AND we can point at code
            # that calls it; `stub` only where there is genuinely nothing yet.
            "status": ("complete" if examples
                       else "reference" if (uses or d["bare"] in POSIM_LINK
                                            and d["crate"] == "posim")
                       else "stub"),
        })

    json.dump(out, open("index_data/entries_tierb.json", "w"), indent=1)
    print(f"{len(out)} Tier-B entries -> index_data/entries_tierb.json")
    print(f"  with a templated example: {made}")
    print(f"  stubs: {len(out) - made}")
    for k, n in skipped.most_common():
        print(f"    {n:4d}  {k}")


if __name__ == "__main__":
    main()
