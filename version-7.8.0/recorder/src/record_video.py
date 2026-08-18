#!/usr/bin/env python3
"""Record a posim simulation into a self-contained browser 'video'.

The scene window (`SCENE CREATE`, grammar.md §5.6) is *live*: it needs a
running posim to talk to.  This tool produces the other thing a reader
usually wants — a file you can open next month, on a machine with no
Rust toolchain, and still watch the same motion, scrub it, and read the
conserved quantities off the frame you stopped on.

How it works, end to end:

1.  Start `posim --machine`, the documented JSONL protocol
    (grammar.md §7, ARCHITECTURE.md §3.6).
2.  Feed it the setup lines of a `.posim` script, one line per
    `{"op":"exec"}` request.
3.  Loop `--frames` times: ask for `{"op":"state"}`, keep the reply,
    then `{"op":"exec","code":"step <dt>"}`.
    **Every advance is a real SUNDIALS step** — this tool never
    integrates anything itself (CLAUDE.md hard rule 1).
4.  Write one HTML file with the recorded frames embedded as JSON and a
    vanilla-JS canvas player around them.  No CDN, no external asset,
    no network access at view time (CLAUDE.md hard rule 2).

Usage:

    record_video.py SETUP.posim -o out.html [--frames N] [--dt DT]
                    [--title "..."] [--caption "..."]

The setup script is ordinary posim source.  Anything it prints is
ignored; only the state dumps become frames.  A `STEP`/`RUN` inside the
setup script is fine — it just means the recording starts later.

This tool lives in its own directory, apart from the Rust workspace it
records, so it has to *find* that workspace rather than assume it sits
one level up.  See `find_workspace`; `--workspace` and `--posim` override
the search when the layout is unusual.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: Where a built posim can be, relative to a cargo workspace root.
BIN_CANDIDATES = ("target/release/posim", "target/debug/posim")


def _binary_in(root: pathlib.Path):
    """The built posim under `root`, release preferred, or None."""
    for rel in BIN_CANDIDATES:
        p = root / rel
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def find_workspace(explicit=None, near=None) -> pathlib.Path:
    """Locate the cargo workspace whose posim we should drive.

    Searched in order, first hit wins:

    1. `--workspace`, or `$POSIM_WORKSPACE`.
    2. The scene script's directory and each of its ancestors.
    3. The current directory and each of its ancestors.

    Only *ancestors* are searched, never siblings.  An earlier version
    also scanned each ancestor's immediate children, so that a recorder
    living in `recorder/` would find a release in `version-7.8.0/`
    without configuration.  That is exactly the search that goes wrong:
    a checkout holding more than one posim workspace — the port next to
    the upstream it was ported from — resolves to whichever name sorts
    first, which is silently the wrong engine.  Three of the five
    shipped scenes still recorded byte-identically against it, because
    the two engines agree to the bit; the fourth failed only because it
    uses a joint the older grammar has not got.  A recording must never
    depend on which sibling sorts first, so the scene decides: it lives
    inside the workspace it belongs to.
    """
    if explicit is None:
        explicit = os.environ.get("POSIM_WORKSPACE")
    if explicit:
        root = pathlib.Path(explicit).resolve()
        if _binary_in(root) is None:
            sys.exit(
                f"no built posim under {root}\n"
                "Build it first:  cargo build --release -p posim"
            )
        return root

    starts = []
    if near is not None:
        starts.append(pathlib.Path(near).resolve().parent)
    starts.append(pathlib.Path.cwd())
    for start in starts:
        for d in (start, *start.parents):
            if _binary_in(d) is not None:
                return d
    sys.exit(
        "posim binary not found.\n"
        "Build it first:  cargo build --release -p posim\n"
        "or point the recorder at the workspace:  --workspace DIR\n"
        "(searched upward from "
        + " and from ".join(str(s) for s in starts)
        + ")"
    )


def find_posim(root: pathlib.Path) -> str:
    """The built posim under an already-located workspace."""
    p = _binary_in(root)
    if p is None:
        sys.exit(f"no built posim under {root}")
    return str(p)


class Posim:
    """One `posim --machine` child process, spoken to in JSONL."""

    def __init__(self, binary: str, cwd=None):
        env = dict(os.environ, POSIM_NO_BROWSER="1")
        self.proc = subprocess.Popen(
            [binary, "--machine"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            # posim resolves relative paths against its own cwd, so it is
            # started in the workspace it belongs to, not in whatever
            # directory the recorder happened to be invoked from.
            cwd=str(cwd) if cwd else None,
            env=env,
        )

    def request(self, obj: dict) -> dict:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("posim closed the pipe")
            reply = json.loads(line)
            # Scene events are unsolicited pushes, not replies (§7).
            if "event" in reply:
                continue
            return reply

    def exec_line(self, code: str) -> dict:
        """One `{"op":"exec"}`. Replies are `{"ok":true,"result":...}`
        or `{"ok":false,"error":"..."}` (machine.rs `ok_reply`/
        `err_reply`); a refusal is reported, never swallowed."""
        reply = self.request({"op": "exec", "code": code})
        if not reply.get("ok"):
            raise RuntimeError(f"posim refused {code!r}: {reply.get('error')}")
        return reply

    def state(self) -> dict:
        reply = self.request({"op": "state"})
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error"))
        return reply["result"]

    def close(self):
        try:
            self.proc.stdin.write('{"op":"quit"}\n')
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait(timeout=30)
        # Closing the pipes is not optional once several recordings run
        # in one process: the child exits either way, but its pipe ends
        # stay open until the garbage collector gets to them.
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except (BrokenPipeError, ValueError, OSError):
                pass


def setup_lines(path: pathlib.Path):
    """Yield the executable lines of a .posim script.

    Blank lines, `#` comments and notebook magics (`%...`, which the
    notebook layer handles and machine mode has no use for) are skipped.
    `SCENE ...` lines are skipped too: a recording has no live window.
    """
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("%"):
            continue
        if line.split()[0].lower() == "scene":
            continue
        yield line


def totals(state: dict):
    """Total momentum and angular momentum about the origin, and the
    per-object kinetic energy sum, computed from the state dump."""
    px = py = pz = 0.0
    lx = ly = lz = 0.0
    for o in state["objects"]:
        m = o["momentum"]
        r = o["position"]
        px += m[0]
        py += m[1]
        pz += m[2]
        # orbital r x p, plus the body's own spin angular momentum
        lx += r[1] * m[2] - r[2] * m[1] + o["angular_momentum"][0]
        ly += r[2] * m[0] - r[0] * m[2] + o["angular_momentum"][1]
        lz += r[0] * m[1] - r[1] * m[0] + o["angular_momentum"][2]
    return [px, py, pz], [lx, ly, lz]


def frame_of(state: dict) -> dict:
    """Shrink a state dump to what the player draws."""
    p, l = totals(state)
    return {
        "t": state["time"],
        "E": state["total_energy"],
        "P": p,
        "L": l,
        "n": state["collision_count"],
        "o": [
            [o["position"], o["orientation"]] for o in state["objects"]
        ],
        "c": [
            [c["point"], c["normal"], c["impulse"]] for c in state["contacts"]
        ],
        # joints: world pivot, the axis a hinge/universal turns about, and
        # the far end. BALL/HINGE/UNIVERSAL hold one shared point, so both
        # ends coincide; a rod holds two points apart, and the player draws
        # the strut between them.
        "j": [
            [jt["point"], jt.get("axis"), jt["point_j"]]
            for jt in state.get("joints", [])
        ],
        # worst |g| over the joint set at this instant
        "gd": state.get("joint_drift", 0.0),
    }


def record(script: pathlib.Path, frames: int, dt: float, workspace=None):
    workspace = workspace or find_workspace(near=script)
    posim = Posim(find_posim(workspace), cwd=workspace)
    try:
        for line in setup_lines(script):
            posim.exec_line(line)
        first = posim.state()
        # Static per-object description: shape, size, mass, colour key.
        bodies = []
        for o in first["objects"]:
            bodies.append(
                {
                    "name": o.get("name") or f"obj{int(o['index'])}",
                    "shape": o["boundary"],
                    "mass": o["mass"],
                    "wall": o["wall"],
                }
            )
        out = [frame_of(first)]
        for _ in range(frames):
            posim.exec_line(f"step {dt!r}")
            out.append(frame_of(posim.state()))
        meta = {
            "joints": [
                {"kind": jt["kind"], "rows": int(jt["rows"])}
                for jt in first.get("joints", [])
            ],
            "method": first["method"],
            "box": first["box"],
            "gravity": first["uniform_gravity"],
            "g_constant": first["g_constant"],
            "dt": dt,
        }
        return bodies, out, meta
    finally:
        posim.close()


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3;
    --dim:#8b949e; --accent:#5d84a8; --gold:#d9a441;
  }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
              font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  #wrap { display:flex; flex-direction:column; height:100%; }
  header { padding:.6rem 1rem; border-bottom:1px solid var(--line);
           background:var(--panel); }
  header h1 { margin:0; font-size:1rem; font-weight:600; }
  header p  { margin:.25rem 0 0; color:var(--dim); font-size:.8rem; }
  #stage { flex:1; position:relative; min-height:0; }
  canvas { display:block; width:100%; height:100%; cursor:grab; }
  canvas:active { cursor:grabbing; }
  #hud { position:absolute; top:.6rem; left:.6rem; background:rgba(13,17,23,.82);
         border:1px solid var(--line); border-radius:6px; padding:.5rem .7rem;
         font-size:.76rem; color:var(--dim); pointer-events:none; }
  #hud b { color:var(--fg); font-weight:600; }
  footer { border-top:1px solid var(--line); background:var(--panel);
           padding:.5rem 1rem; display:flex; gap:.6rem; align-items:center;
           flex-wrap:wrap; }
  button, select { background:#21262d; color:var(--fg); border:1px solid var(--line);
                   border-radius:5px; padding:.3rem .7rem; font:inherit;
                   font-size:.8rem; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  input[type=range] { flex:1; min-width:12rem; accent-color:var(--accent); }
  label { color:var(--dim); font-size:.8rem; display:flex; gap:.3rem;
          align-items:center; }
</style>
<div id="wrap">
  <header>
    <h1>__TITLE__</h1>
    <p>__CAPTION__</p>
  </header>
  <div id="stage">
    <canvas id="c"></canvas>
    <div id="hud"></div>
  </div>
  <footer>
    <button id="play">&#9654; Play</button>
    <button id="back">&#9664;&#9664;</button>
    <button id="fwd">&#9654;&#9654;</button>
    <button id="rst">&#8635; Reset view</button>
    <input id="scrub" type="range" min="0" value="0" step="1">
    <label>speed <select id="speed">
      <option value="0.25">0.25&times;</option>
      <option value="0.5">0.5&times;</option>
      <option value="1" selected>1&times;</option>
      <option value="2">2&times;</option>
      <option value="4">4&times;</option>
    </select></label>
    <label><input id="trails" type="checkbox" checked> trails</label>
    <label><input id="labels" type="checkbox" checked> labels</label>
    <label><input id="arrows" type="checkbox" checked> contacts</label>
    <label><input id="joints" type="checkbox" checked> joints</label>
  </footer>
</div>
<script>
const BODIES = __BODIES__;
const FRAMES = __FRAMES__;
const META   = __META__;

/* ---- shape parsing -------------------------------------------------
   The state dump writes boundaries as text ("sphere r=0.5"), which is
   what a human reads in LIST.  Parse the numbers back out so the player
   can draw the right silhouette at the right size. */
function parseShape(s) {
  const num = k => { const m = s.match(new RegExp(k + "=(-?[0-9.eE+]+)"));
                     return m ? parseFloat(m[1]) : 0; };
  if (s.startsWith("sphere"))   return {k:"sphere",   r:num("r")};
  if (s.startsWith("cuboid")) {
    const m = s.match(/he=\[(-?[0-9.eE+]+),(-?[0-9.eE+]+),(-?[0-9.eE+]+)\]/);
    return {k:"cuboid", he: m ? [+m[1],+m[2],+m[3]] : [.5,.5,.5]};
  }
  if (s.startsWith("torus"))    return {k:"torus",    ring:num("ring"), tube:num("tube")};
  if (s.startsWith("disk"))     return {k:"disk",     r:num("r")};
  if (s.startsWith("cylinder")) return {k:"cylinder", r:num("r"), h:num("h")};
  if (s.startsWith("dumbbell")) return {k:"dumbbell", r1:num("r1"), r2:num("r2"),
                                        rod:num("rod_r"), len:num("len")};
  return {k:"point", r:0};
}
BODIES.forEach(b => b.geom = parseShape(b.shape));

/* stable, colour-blind-safe hues; walls stay grey */
const HUES = ["#6cb2eb","#f0a35e","#7ec98f","#c98fd6","#e2777a","#d9c46a",
              "#63c8c4","#b0a0e0"];
BODIES.forEach((b,i) => b.color = b.wall ? "#3b4450" : HUES[i % HUES.length]);

/* ---- camera --------------------------------------------------------
   Orbit camera: yaw/pitch about a target, perspective divide.  Same
   controls as the live scene window (drag to orbit, wheel to zoom,
   arrows to pan). */
/* A planar linkage — every hinge axis along z — reads as the mechanism
   it is only when you look straight down that axis. An isometric view is
   the better default for a 3-D scene, so the recorder picks. */
const HOME = "__VIEW__" === "front"
  ? {yaw:0, pitch:0, dist:0, tx:0, ty:0, target:[0,0,0]}
  : {yaw:0.6, pitch:0.35, dist:0, tx:0, ty:0, target:[0,0,0]};
let cam = Object.assign({}, HOME, {target: HOME.target.slice()});

function autoFit() {
  /* Frame the bodies the reader came to watch, CENTRED ON THEM rather
     than on the origin — a pendulum hangs below its pivot, and centring
     on (0,0,0) wastes half the picture on empty sky. Wall slabs are
     excluded: they are half-space-sized and would push the camera out to
     nothing. */
  const lo = [ Infinity,  Infinity,  Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  let pad = 0;
  for (const b of BODIES) {
    if (b.wall) continue;
    const g = b.geom;
    pad = Math.max(pad, g.r||0, g.ring||0, (g.len||0)/2,
                   g.he ? Math.max.apply(null, g.he) : 0);
  }
  for (const f of FRAMES) f.o.forEach((o, i) => {
    if (BODIES[i] && BODIES[i].wall) return;
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], o[0][k]);
      hi[k] = Math.max(hi[k], o[0][k]);
    }
  });
  if (!isFinite(lo[0])) { lo.fill(-1); hi.fill(1); }
  if (META.box) {
    const h = META.box / 2;
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], -h); hi[k] = Math.max(hi[k], h); }
  }
  let half = 1e-9;
  for (let k = 0; k < 3; k++) {
    HOME.target[k] = 0.5 * (lo[k] + hi[k]);
    half = Math.max(half, 0.5 * (hi[k] - lo[k]) + pad);
  }
  cam.target = HOME.target.slice();
  /* 2.4 rather than a timid 3.2: the content should fill the frame,
     and the wheel is right there if the viewer wants to pull back. */
  HOME.dist = cam.dist = half * 2.4;
}
autoFit();

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
let W = 0, H = 0;
function resize() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener("resize", resize);

function project(p) {
  /* yaw about the world Y axis, then pitch about the camera X axis,
     then a plain perspective divide against an eye at distance
     cam.dist.  `s` is the pixels-per-world-unit scale AT that depth,
     which is what the sphere radii are drawn with. */
  const cy = Math.cos(cam.yaw),   sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  /* about the camera TARGET, not the world origin */
  const px = p[0] - cam.target[0], py = p[1] - cam.target[1], pz = p[2] - cam.target[2];
  const x1 =  px * cy + pz * sy;
  const z1 = -px * sy + pz * cy;
  const y  =  py * cp - z1 * sp;
  const z  =  py * sp + z1 * cp;
  const d = cam.dist - z;
  if (d <= 1e-6) return null;          // behind the camera
  const s = (H * 0.9) / d;
  return {x: W/2 + x1 * s + cam.tx, y: H/2 - y * s + cam.ty, s: s, z: z};
}

/* rotate a body-frame vector by the w-first quaternion (grammar §4) */
function qrot(q, v) {
  const [w,x,y,z] = q, [a,b,c] = v;
  const tx = 2*(y*c - z*b), ty = 2*(z*a - x*c), tz = 2*(x*b - y*a);
  return [a + w*tx + (y*tz - z*ty),
          b + w*ty + (z*tx - x*tz),
          c + w*tz + (x*ty - y*tx)];
}

function ring(center, q, radius, segs) {
  /* A circle of `radius` in the body's xy plane — every round boundary
     in physical_object (torus, disk, cylinder, dumbbell) is symmetric
     about its LOCAL Z AXIS (boundary.rs), so the quaternion is what
     makes spin visible. */
  const e1 = [1,0,0], e2 = [0,1,0];
  const pts = [];
  for (let i = 0; i <= segs; i++) {
    const a = i / segs * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
    const v = [ (e1[0]*ca + e2[0]*sa) * radius,
                (e1[1]*ca + e2[1]*sa) * radius,
                (e1[2]*ca + e2[2]*sa) * radius ];
    const r = qrot(q, v);
    pts.push([center[0]+r[0], center[1]+r[1], center[2]+r[2]]);
  }
  return pts;
}

function stroke(pts, color, width) {
  ctx.beginPath(); let started = false;
  for (const p of pts) {
    const q = project(p); if (!q) { started = false; continue; }
    if (!started) { ctx.moveTo(q.x, q.y); started = true; } else ctx.lineTo(q.x, q.y);
  }
  ctx.strokeStyle = color; ctx.lineWidth = width || 1; ctx.stroke();
}

let idx = 0, playing = false, lastTick = 0;
const trails = BODIES.map(() => []);

function rebuildTrails(upTo) {
  for (const t of trails) t.length = 0;
  const from = Math.max(0, upTo - 220);
  for (let k = from; k <= upTo; k++)
    FRAMES[k].o.forEach((o, i) => trails[i] && trails[i].push(o[0]));
}

function drawBox(size) {
  const h = size / 2, e = [-h, h], seg = [];
  for (const a of e) for (const b of e) {
    seg.push([[a,b,-h],[a,b,h]], [[a,-h,b],[a,h,b]], [[-h,a,b],[h,a,b]]);
  }
  ctx.setLineDash([5,4]);
  for (const s of seg) stroke(s, "#5d84a8", 1);
  ctx.setLineDash([]);
}

function draw() {
  const f = FRAMES[idx];
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);

  /* ground grid, drawn first so bodies sit on top */
  const span = HOME.dist / 3.2, step = span / 4;
  ctx.globalAlpha = .35;
  for (let i = -4; i <= 4; i++) {
    stroke([[-span, 0, i*step], [span, 0, i*step]], "#232a33", 1);
    stroke([[i*step, 0, -span], [i*step, 0, span]], "#232a33", 1);
  }
  ctx.globalAlpha = 1;

  if (META.box) drawBox(META.box);

  if (document.getElementById("trails").checked) {
    trails.forEach((tr, i) => {
      if (BODIES[i].wall || tr.length < 2) return;
      ctx.globalAlpha = .5; stroke(tr, BODIES[i].color, 1); ctx.globalAlpha = 1;
    });
  }

  /* painter's algorithm: far bodies first */
  const order = f.o.map((o, i) => ({i, z: (project(o[0]) || {z:-1e9}).z}))
                   .filter(({i}) => BODIES[i] && !BODIES[i].wall)
                   .sort((a, b) => a.z - b.z);

  for (const {i} of order) {
    const b = BODIES[i];
    const [p, q] = f.o[i], g = b.geom, pr = project(p);
    if (!pr) continue;
    if (g.k === "sphere" || g.k === "point") {
      const rad = Math.max(2, (g.r || 0) * pr.s);
      const grd = ctx.createRadialGradient(pr.x - rad*.35, pr.y - rad*.35, rad*.1,
                                           pr.x, pr.y, rad);
      grd.addColorStop(0, "#ffffff"); grd.addColorStop(.25, b.color);
      grd.addColorStop(1, "#10141a");
      ctx.beginPath(); ctx.arc(pr.x, pr.y, rad, 0, Math.PI*2);
      ctx.fillStyle = g.r ? grd : b.color; ctx.fill();
    } else if (g.k === "cuboid") {
      const [hx, hy, hz] = g.he, V = [];
      for (const sx of [-1,1]) for (const sy of [-1,1]) for (const sz of [-1,1]) {
        const r = qrot(q, [sx*hx, sy*hy, sz*hz]);
        V.push([p[0]+r[0], p[1]+r[1], p[2]+r[2]]);
      }
      const E = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],
                 [4,5],[4,6],[5,7],[6,7]];
      for (const [a,c] of E) stroke([V[a], V[c]], b.color, 1.3);
    } else if (g.k === "torus") {
      stroke(ring(p, q, g.ring + g.tube, 48), b.color, 1.2);
      stroke(ring(p, q, g.ring - g.tube, 48), b.color, 1.2);
    } else if (g.k === "disk") {
      stroke(ring(p, q, g.r, 48), b.color, 1.4);
      for (const a of [0, Math.PI/2]) {           // two diameters
        const u = qrot(q, [Math.cos(a)*g.r, Math.sin(a)*g.r, 0]);
        stroke([[p[0]-u[0],p[1]-u[1],p[2]-u[2]],
                [p[0]+u[0],p[1]+u[1],p[2]+u[2]]], b.color, 1);
      }
    } else if (g.k === "cylinder") {
      for (const s of [-1, 1]) {
        const off = qrot(q, [0, 0, s*g.h/2]);
        stroke(ring([p[0]+off[0], p[1]+off[1], p[2]+off[2]], q, g.r, 40),
               b.color, 1.2);
      }
      for (const a of [0, Math.PI/2, Math.PI, 3*Math.PI/2]) {   // side lines
        const lo = qrot(q, [Math.cos(a)*g.r, Math.sin(a)*g.r, -g.h/2]);
        const hi = qrot(q, [Math.cos(a)*g.r, Math.sin(a)*g.r,  g.h/2]);
        stroke([[p[0]+lo[0],p[1]+lo[1],p[2]+lo[2]],
                [p[0]+hi[0],p[1]+hi[1],p[2]+hi[2]]], b.color, 1);
      }
    } else if (g.k === "dumbbell") {
      const half = g.len / 2;
      for (const [s, rr] of [[-1, g.r1], [1, g.r2]]) {
        const off = qrot(q, [0, 0, s*half]);
        const cpt = [p[0]+off[0], p[1]+off[1], p[2]+off[2]];
        const pj = project(cpt); if (!pj) continue;
        ctx.beginPath(); ctx.arc(pj.x, pj.y, Math.max(2, rr*pj.s), 0, Math.PI*2);
        ctx.fillStyle = b.color; ctx.fill();
      }
      const a = qrot(q, [0,0,-half]), c = qrot(q, [0,0,half]);
      stroke([[p[0]+a[0],p[1]+a[1],p[2]+a[2]],
              [p[0]+c[0],p[1]+c[1],p[2]+c[2]]], b.color, 2);
    }
    if (document.getElementById("labels").checked && !b.wall) {
      ctx.fillStyle = "#8b949e"; ctx.font = "11px ui-monospace,monospace";
      ctx.fillText(b.name, pr.x + 8, pr.y - 8);
    }
  }

  /* Joints: a ring at the shared point, and for a hinge the axis it
     turns about. Drawn AFTER the bodies so the pivot is never hidden
     inside the link it belongs to — the joint is the thing this video is
     about. */
  if (document.getElementById("joints").checked && f.j) {
    const len = HOME.dist / 14;
    for (const [pt, axis, far] of f.j) {
      /* A rod holds two points a fixed distance apart, so its two ends do
         not coincide: draw the strut itself, else the shaft it braces
         would look unsupported. */
      if (far && (far[0]!==pt[0] || far[1]!==pt[1] || far[2]!==pt[2])) {
        stroke([pt, far], "#e8c46a", 1.4);
      }
      const pj = project(pt);
      if (pj) {
        ctx.beginPath();
        ctx.arc(pj.x, pj.y, 5, 0, Math.PI*2);
        ctx.strokeStyle = "#e8c46a"; ctx.lineWidth = 2; ctx.stroke();
        ctx.beginPath();
        ctx.arc(pj.x, pj.y, 1.6, 0, Math.PI*2);
        ctx.fillStyle = "#e8c46a"; ctx.fill();
      }
      if (axis) {
        stroke([[pt[0]-axis[0]*len, pt[1]-axis[1]*len, pt[2]-axis[2]*len],
                [pt[0]+axis[0]*len, pt[1]+axis[1]*len, pt[2]+axis[2]*len]],
               "#e8c46a", 1.4);
      }
    }
  }

  if (document.getElementById("arrows").checked) {
    /* Contact normals of the step that produced THIS frame: the arrow
       runs along the exact analytic normal (i -> j) and its length is
       the applied impulse, so a hard hit draws a long arrow. */
    let maxImp = 1e-12;
    for (const c of f.c) maxImp = Math.max(maxImp, Math.abs(c[2]));
    for (const [pt, n, imp] of f.c) {
      const len = (HOME.dist / 10) * (0.35 + 0.65 * Math.abs(imp) / maxImp);
      stroke([pt, [pt[0]+n[0]*len, pt[1]+n[1]*len, pt[2]+n[2]*len]], "#d9a441", 2);
      const pj = project(pt);
      if (pj) { ctx.beginPath(); ctx.arc(pj.x, pj.y, 3, 0, Math.PI*2);
                ctx.fillStyle = "#d9a441"; ctx.fill(); }
    }
  }

  const fx = v => (v >= 0 ? " " : "") + v.toPrecision(8);
  document.getElementById("hud").innerHTML =
    `frame <b>${idx}</b> / ${FRAMES.length - 1}` +
    `<br>t &nbsp;= <b>${fx(f.t)}</b>` +
    `<br>E &nbsp;= <b>${fx(f.E)}</b>` +
    `<br>|P| = <b>${fx(Math.hypot(f.P[0],f.P[1],f.P[2]))}</b>` +
    `<br>|L| = <b>${fx(Math.hypot(f.L[0],f.L[1],f.L[2]))}</b>` +
    `<br>collisions <b>${f.n}</b>` +
    (META.joints && META.joints.length
      ? `<br>joints <b>${META.joints.map(j => j.kind).join(", ")}</b>` +
        `<br>worst |g| = <b>${(f.gd || 0).toExponential(2)}</b>`
      : "") +
    `<br><span style="color:#8b949e">${META.method}, dt = ${META.dt}</span>`;
}

function seek(k) {
  idx = Math.max(0, Math.min(FRAMES.length - 1, k));
  document.getElementById("scrub").value = idx;
  rebuildTrails(idx);
  draw();
}

/* ---- controls ---- */
const scrub = document.getElementById("scrub");
scrub.max = FRAMES.length - 1;
scrub.addEventListener("input", () => seek(+scrub.value));
document.getElementById("back").onclick = () => seek(idx - 1);
document.getElementById("fwd").onclick  = () => seek(idx + 1);
document.getElementById("rst").onclick  = () => {
  cam = Object.assign({}, HOME, {target: HOME.target.slice()});
  draw();
};
const playBtn = document.getElementById("play");
playBtn.onclick = () => {
  playing = !playing;
  playBtn.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  lastTick = performance.now();
  if (playing) requestAnimationFrame(tick);
};
for (const id of ["trails","labels","arrows","joints"])
  document.getElementById(id).addEventListener("change", draw);

function tick(now) {
  if (!playing) return;
  const speed = parseFloat(document.getElementById("speed").value);
  if (now - lastTick > 33 / speed) {
    lastTick = now;
    if (idx >= FRAMES.length - 1) seek(0); else seek(idx + 1);
  }
  requestAnimationFrame(tick);
}

let dragging = null;
canvas.addEventListener("pointerdown", e => { dragging = [e.clientX, e.clientY];
                                              canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointerup",   e => { dragging = null;
                                              canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  cam.yaw   += (e.clientX - dragging[0]) * 0.008;
  cam.pitch += (e.clientY - dragging[1]) * 0.008;
  cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch));
  dragging = [e.clientX, e.clientY];
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  cam.dist *= e.deltaY > 0 ? 1.1 : 1/1.1;
  draw();
}, {passive:false});
window.addEventListener("keydown", e => {
  const k = e.key;
  if (k === " ")           { e.preventDefault(); playBtn.click(); }
  else if (k === "ArrowRight") seek(idx + 1);
  else if (k === "ArrowLeft")  seek(idx - 1);
  else if (k === "ArrowUp")     { cam.ty += 20; draw(); }
  else if (k === "ArrowDown")   { cam.ty -= 20; draw(); }
  else if (k === "+" || k === "=") { cam.dist /= 1.15; draw(); }
  else if (k === "-")              { cam.dist *= 1.15; draw(); }
});

resize();
seek(0);
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", type=pathlib.Path, help="a .posim setup script")
    ap.add_argument("-o", "--out", type=pathlib.Path, required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--title", default=None)
    ap.add_argument("--caption", default="")
    ap.add_argument(
        "--view",
        choices=["iso", "front"],
        default="iso",
        help="opening camera: 'iso' looks down on the scene, 'front' looks "
        "straight along -z, which is what a planar linkage wants",
    )
    ap.add_argument(
        "--workspace",
        type=pathlib.Path,
        default=None,
        help="the cargo workspace holding the posim to drive; found "
        "automatically, or from $POSIM_WORKSPACE, when not given",
    )
    args = ap.parse_args()

    bodies, frames, meta = record(
        args.script, args.frames, args.dt, workspace=args.workspace
    )
    title = args.title or args.script.stem.replace("_", " ")
    html = (PAGE
            .replace("__TITLE__", title)
            .replace("__CAPTION__", args.caption)
            .replace("__BODIES__", json.dumps(bodies))
            .replace("__FRAMES__", json.dumps(frames))
            .replace("__META__", json.dumps(meta))
            .replace("__VIEW__", args.view))
    args.out.write_text(html)
    print(f"{args.out}: {len(frames)} frames, {len(bodies)} bodies, "
          f"{len(html)/1024:.0f} kB, dt = {args.dt}")


if __name__ == "__main__":
    main()
