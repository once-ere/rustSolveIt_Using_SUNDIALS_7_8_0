# StageNbks — one stand-alone notebook per stage

Eleven posim notebooks, one for each stage of the port. **Every one is
complete in itself, and so is this file.** No notebook refers you to another
notebook, to this README, or to any other document; this README does not
send you into a notebook to find out how to run it. Everything needed is
written out in each place it is needed. That is deliberate: a reader who
opens one file should never have to go hunting.

Every command, output line, button, and key binding recorded here was read
off the running binary or the served page, not from memory. The exact
sources and the verification transcript are in
[STAGENBKS_PROVENANCE.md](STAGENBKS_PROVENANCE.md).

## What is here

| notebook | stage | interactive display |
|---|---|---|
| `Stage_24.posim` | The sliver at 4 ≤ \|ν\| ≤ 8 | none — pure numerics; recipe for a live scene included inside |
| `Stage_2A.posim` | A faithful port of EVOLVE_NASH, with a Strang variant | **yes — two HTML films** |
| `Stage_2B.posim` | T(E) by transfer matrix, and a designed absorber | **yes — one HTML film** |
| `Stage_2C.posim` | The output-granularity defect, repaired | **yes — scene window, 12 entities** |
| `Stage_2D.posim` | DLMF 10.20 past the turning point | none — pure numerics; recipe for a live scene included inside |
| `Stage_2E.posim` | Mechanising the staleness rule | none — build tooling; recipe for a live scene included inside |
| `Stage_2F.posim` | Mutation-probing the numerical core | none — build tooling; recipe for a live scene included inside |
| `Stage_2G.posim` | Closing the mutation survivors | **yes — scene window, 3 entities** |
| `Stage_2H.posim` | Resolving the last survivor | **yes — scene window, 2 entities** |
| `Stage_2I.posim` | The ridge is a real defect | none — pure numerics; recipe for a live scene included inside |
| `Stage_2J.posim` | A measured guard for the reflection route | none — pure numerics; recipe for a live scene included inside |

"none" means the stage produces numbers, not moving bodies: a function
value has no position and no time coordinate, so there is nothing for a 3-D
window to draw. Those notebooks say so plainly and then give you a
complete, self-contained recipe for a live interactive scene anyway, so you
are never left without one.

## Build once

From the repository root (the directory holding `Cargo.toml`):

```bash
git clone https://github.com/once-ere/rustSimulate.git
```

```bash
cargo build -p posim --release
```

There are no submodules and no crates to download — every dependency is
vendored. The first build takes a few minutes; later runs are immediate. It
produces `./target/release/posim`. If `cargo` is missing, install Rust from
<https://rustup.rs> and reopen the terminal.

## Run any notebook

```bash
./target/release/posim --notebook StageNbks/Stage_2C.posim
```

Three forms exist, and the difference matters:

| form | what it does | interactive afterwards? |
|---|---|---|
| `--notebook <file>` | executes the file, prints an `In[n]:=` / `Out[n]=` transcript, **stays at the prompt with all state alive** | **yes** |
| `--script <file>` | executes the file and **exits** | no |
| `cargo run -p posim --release -- --notebook <file>` | same as `--notebook`, but re-checks the build first | yes |

**Use `--notebook` for anything interactive.** `--script` exits the moment
the file ends and takes any open scene window with it.

To type commands by hand instead, start the bare interpreter:

```bash
./target/release/posim
```

`HELP` prints the complete command language; `%quit` leaves. Other magics:
`%history`, `%rerun <n>`, `%edit <n> <text>`, `%save <file>`, `%load <file>`,
`%reset`.

## Running `Stage_2C.posim` as an interactive simulation

```bash
./target/release/posim --notebook StageNbks/Stage_2C.posim
```

The notebook's last commands are `scene create`, `scene set_time_step
0.0002`, `scene zoom 2`, `scene start`, `scene status`. They run
automatically, so by the time the prompt appears the window exists and the
bodies are moving. You will see, in this order:

```
scene window created: http://127.0.0.1:<port>/
(asked your desktop to open it; if no window appeared, open that address yourself)
showing 12 entities; SCENE START begins the evolution — HELP lists all scene commands
scene time step dt = 0.0002
camera distance = <d>
scene playback: running
scene: http://127.0.0.1:<port>/  (<k> window(s) connected)
mode = running, t = 1, dt = 0.0002, steps = 0, history = 0 frame(s)
entities = 12 (hidden: none)
```

Twelve entities is six bodies plus the box's six static walls. `<k>` counts
browsers actually attached — 0 until one loads the page, then 1; a running
scene with 0 connected windows is normal, because the physics does not wait
for an audience.

You are left at an interactive prompt **with the window live**: every scene
command below takes effect immediately. Gravity is off and every restitution
is 1, so `energy` must keep reading 960000 while it runs — that invariant
holding is the point of this stage. `scene close` frees the port; `%quit`
ends the session.

## Running `Stage_2H.posim` as an interactive simulation

```bash
./target/release/posim --notebook StageNbks/Stage_2H.posim
```

Identical in form, with `scene set_time_step 0.005`, and it reports
`showing 2 entities`. Everything in *The scene window* below applies
unchanged.

`Stage_2G.posim` behaves the same way, with `scene set_time_step 0.002` and
`showing 3 entities`.

## Running `Stage_2A.posim` as an interactive simulation

This stage is a **quantum** problem — a wavefunction on a grid, not rigid
bodies. The scene window draws rigid bodies only, so `scene create` here
would open a window correctly reporting `showing 0 entities`. That is not a
fault; there is genuinely no rigid body in this stage. The interactive
viewer for a wavefunction is the HTML film:

```bash
./target/release/posim --notebook StageNbks/Stage_2A.posim
```

It writes two films into this folder, so the two boundary conditions can be
compared:

```
StageNbks/out_2A_nash_periodic.html
StageNbks/out_2A_cayley_reflecting.html
```

Open one:

```bash
xdg-open StageNbks/out_2A_nash_periodic.html
```

On macOS use `open`, on Windows `start`; or drag the file onto a browser
window. Each file is self-contained HTML and JavaScript — no server, no
network, no install.

`Stage_2B.posim` works identically and writes
`StageNbks/out_2B_double_barrier.html`.

The films are regenerated on every run and are therefore **not tracked by
git** (`.gitignore` carries `StageNbks/out_*.html`).

### The film's controls

It starts playing by itself and loops, at about 22 frames per second.

| control | behaviour |
|---|---|
| **Pause** | toggles playback; the label always names what the *next* click does, flipping to **Play** when paused |
| **Restart** | jumps to frame 0. It does **not** pause — if it was playing, it keeps playing from the start |
| **slider** | drag to any frame. Dragging pauses playback automatically (the button flips to **Play**), so you can step through a reflection or a tunnelling event frame by frame |

The readout beside the slider shows, for the frame on screen:

```
t = <time>   frame <i>/<n>   norm = <total>   transmitted = <part>
```

`norm` is the conserved quantity — watch it while you scrub. `transmitted`
is the probability in the region marked green. The colour key under the
canvas: blue is |ψ|², orange is V(x) scaled to fit the axes, green marks the
transmitted region.

To drive it yourself, start the bare interpreter, enter the stage's setup
lines, and write your own film with any end time and frame count:

```
qm animate "my_run.html" 2.0 frames 300
```

## Any future notebook whose display is "yes"

The rule that makes a notebook interactive, stated once so it can be
followed without reading an example:

1. **Build the bodies first.** `scene create` opens a *view onto the world
   that already exists*; it never constructs entities. A window reporting
   `showing 0 entities` means the world is genuinely empty, which is an
   honest report rather than a bug.
2. **End the notebook with the display commands**, not with a description of
   them — `scene create`, `scene set_time_step <dt>`, optionally
   `scene zoom <f>`, then `scene start`, then `scene status`. A notebook
   that only *describes* the GUI leaves the reader with a dead prompt.
3. **Choose `dt` for the scenario.** Too large and fast bodies pass through
   each other between frames; too small and the motion crawls.
4. **Run it with `--notebook`,** never `--script`.
5. **For a quantum stage, write a film instead** — `qm animate "<file>.html"
   <t> frames <n>` — and say in the notebook that the scene window cannot
   draw a wavefunction.

## Opening the window in a browser

`scene create` prints three lines. The **middle** one reports what was
actually attempted, and it has two forms:

```
(asked your desktop to open it; if no window appeared, open that address yourself)
(no browser was launched — open that address yourself)
```

posim tries exactly one command, `xdg-open <url>`, which is a Linux/BSD
utility. So:

* **Linux/BSD** with `xdg-utils` — the browser opens by itself.
* **macOS** — nothing opens, and nothing is broken: `xdg-open` does not
  exist there, so you get the second line. Run `open http://127.0.0.1:<port>/`.
* **Windows** — same, via `start http://127.0.0.1:<port>/`.
* **Headless or SSH** — nothing opens. Forward the port and browse locally:
  `ssh -L 41234:127.0.0.1:41234 <user>@<host>`.

Setting `POSIM_NO_BROWSER` to any value suppresses the attempt entirely; the
URL is still printed and still works. That is what the tests use.

The port is chosen by the OS each run. Pin it with `scene create 8080`. The
page is served by posim itself over a websocket, bound to `127.0.0.1`, so it
is not reachable from another machine by design.

## The scene window: every control, and its typed equivalent

Toolbar, left to right:

| button | typed equivalent | effect |
|---|---|---|
| ▶ Start | `scene start` | begin / resume forward evolution |
| ❚❚ Pause | `scene pause` | freeze, keep the window live |
| ■ Stop | `scene stop` | stop evolution **and clear history** |
| ◀ Reverse | `scene reverse` | play backward through recorded history |
| ↺ Reset | `scene reset` | every value and the time return to their initial values; Start then runs it again |
| ⇤ | *(button only)* | one step backward |
| ⇥ | *(button only)* | one step forward |
| dt \_\_\_ **Set** | `scene set_time_step <dt>` | simulated seconds per frame |
| 🔍+ / 🔍− | `scene zoom in` / `scene zoom out` | zoom |
| ⌂ View | *(button only)* | reset the camera to its home pose |
| Grid / Trails / Labels / Contacts | *(buttons only)* | toggle the ground grid, motion trails, object labels, contact-normal arrows |
| ? | *(button only)* | toggle the controls cheat-sheet |

Mouse and keyboard, with the pointer over the window:

| input | effect |
|---|---|
| left-drag | rotate (orbit) |
| shift-drag or right-drag | translate (pan) |
| wheel, or `+`/`=` and `-`/`_` | zoom in / out |
| ← → ↑ ↓ | translate the view |
| Space | start / pause playback |
| `R` | reset the view |
| `G` / `T` / `L` / `C` | toggle grid / trails / labels / contacts |
| `H` or `?` | toggle the cheat-sheet |

**The arrow keys move the camera, not the simulation.** To advance the
physics one frame at a time, use the ⇤ and ⇥ buttons.

The status bar reports live: body count, contacts, hidden count, simulated
time, dt, energy, fps, camera yaw/pitch/distance, playback mode, history
depth, and connection state.

Typed commands available while the window is open — the window and the
prompt drive the *same* state, so a button and the command it mirrors cannot
disagree:

```
scene status | refresh | redraw | close
scene hide <n> | all        scene show <n> | all
scene translate dx dy [dz]  scene rotate dyaw dpitch
scene zoom in | out | <f>   scene set_time_step <dt>
scene start | stop | pause | reverse | reset
scene events                energy
```

### One behaviour that looks like a fault and is not

`scene reverse` immediately after opening answers:

```
nothing to reverse - no forward history recorded yet
```

Reverse replays *recorded* history; until the scene has run forward there is
nothing to play back. Let it run a few seconds and it works. `scene stop`
clears that history deliberately, so Reverse is empty again after a Stop.

### Troubleshooting

| symptom | cause and fix |
|---|---|
| `no scene window - run SCENE CREATE first` | the window was closed or never opened — type `scene create` |
| window says `showing 0 entities` | the world holds no bodies; build them, then `scene refresh` |
| port already in use | `scene close`, then `scene create <another port>` |
| page loads but never moves | playback is not running — press Start or type `scene start` |
| bodies jump through each other | `dt` too large — `scene set_time_step <smaller>` |
| no window appeared | read the middle line of `scene create`; on macOS/Windows open the URL yourself |

## Running all eleven

```bash
for f in StageNbks/*.posim; do echo "=== $f ==="; ./target/release/posim --notebook "$f" < /dev/null; done
```

All eleven run with **zero errors**, and that is enforced by the build:
`scripts/certify_clean.sh` executes every notebook in this folder and fails
if any emits an error line. The check costs about 14 seconds.

That gate exists because the need was demonstrated rather than imagined.
When these notebooks were first written, *running* them found two errors
that *reading* them had not: two used a `version` command that does not
exist, and two more asserted that `bessel_y_nu` would refuse at points where
the language actually routes to a different, working implementation. Both
would have shipped as confident, wrong instructions.

The gate has a limit worth knowing: it tests that nothing errors. It cannot
tell a notebook that *displays* a simulation from one that merely
*describes* one, because both exit cleanly. That distinction is what the
`interactive display` column above records, and it was checked by running
each notebook and reading the entity count, not by reading the source.
