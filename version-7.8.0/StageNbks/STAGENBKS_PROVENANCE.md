# StageNbks provenance — where every instruction came from, and how it was checked

This file exists so that no claim in `README.md` or in any `.posim` notebook
has to be taken on trust. Every command, output line, button label and key
binding documented in this folder was read out of a source file or observed
coming out of the running binary. Nothing was written from memory.

The rule this folder is built on: **a notebook may never refer the reader to
another notebook, to this file, or to any other document.** Each notebook,
and `README.md`, is complete on its own. The cost is repetition; the benefit
is that opening any single file is sufficient.

## 1. Sources of truth

Each documented fact traces to exactly one of these:

| fact documented | source read |
|---|---|
| the three invocation modes (`--notebook`, `--script`, bare) | `posim/src/main.rs:30-61` |
| what `--notebook` leaves behind, and the sign-off text | `posim/src/notebook.rs`, `loaded_hint()` |
| `xdg-open`, `$POSIM_NO_BROWSER`, the URL form | `posim/src/scene/mod.rs`, `SceneHandle::start` |
| the `scene create` reply, all three lines | `posim/src/vm.rs`, `SceneCmd::Create` arm |
| toolbar button ids, labels, tooltips | `posim/src/scene/scene.html`, `#toolbar` |
| key bindings | `posim/src/scene/scene.html`, the `keydown` listener |
| mouse bindings (orbit / pan / zoom) | `posim/src/scene/scene.html`, `mousedown` + `drag.pan` |
| the in-window cheat-sheet | `posim/src/scene/scene.html`, `#help` |
| status-bar fields | `posim/src/scene/scene.html`, the `st-*` element ids |
| the `SCENE` command list | `HELP` from the built binary |
| the HTML film's controls and readout | `posim/src/qm.rs`, the `#play` / `#rew` / `#scrub` handlers |
| `qm animate` syntax | `HELP` from the built binary |

## 2. Corrections this pass made to previously-wrong documentation

These were not editorial improvements. Each was a statement that would have
misled someone following it.

**Eight notebooks pointed at other notebooks.** `Stage_24`, `2A`, `2B`,
`2D`, `2E`, `2F`, `2I` and `2J` each ended their GUI section by naming
`Stage_2C.posim` and `Stage_2H.posim` as the place to find a working window.
That is precisely the cross-reference the folder forbids. Every one is now
replaced by a complete, self-contained recipe: the reader builds and runs a
live three-body scene from within the notebook they already have open. The
recipe was executed before it was written down — it reports `showing 9
entities` (three bodies plus the box's six static walls) and a conserved
energy of `2495`.

**The arrow keys were documented backwards.** Every notebook claimed
"step -> the arrow keys advance or retreat one frame". The `keydown`
listener shows the arrow keys call `moveBy(...)` on the camera target: they
**translate the view** and do not touch the simulation. Stepping is the
`⇤` / `⇥` toolbar buttons (`bt-back`, `bt-step`). The notebooks now say so,
with an explicit warning, because the wrong assumption is the natural one.

**The button list was less than half the window.** The notebooks documented
Start, Pause, Stop, Reverse and three vague mouse gestures. The page has
sixteen controls: the five playback buttons plus Reset, one-step-back,
one-step-forward, a `dt` entry box with its Set button, zoom in, zoom out,
camera home, and the Grid / Trails / Labels / Contacts toggles, plus the
help overlay. All sixteen are now documented, each with its typed equivalent
or an explicit "(button only)".

**`scene create` claimed a browser had opened when none had.** The reply
line read `(opened in your browser; if no window appeared, open that address
yourself)` unconditionally. posim tries exactly one command, `xdg-open`,
which is a Linux/BSD utility: on macOS and Windows the spawn fails because
the binary does not exist, and when `$POSIM_NO_BROWSER` is set no attempt is
made at all. In all three cases the old line still announced success, and a
reader on macOS would wait for a window that was never coming. Fixed at the
source rather than papered over in prose: `SceneHandle` now carries
`browser_launched`, set from whether the spawn actually succeeded, and the
reply reads one of

```
(asked your desktop to open it; if no window appeared, open that address yourself)
(no browser was launched — open that address yourself)
```

Both paths were exercised. `scene_info.md`, `scene_info.tex` and
`scene_info.pdf` were updated in the same pass, per the project's
documentation-lockstep rule.

## 3. Verification actually performed

Not "it compiles". Each item below was observed.

### 3.1 Every notebook executes

All eleven run through the release binary with **zero** error lines:

```
Stage_24  errors=0      Stage_2E  errors=0      Stage_2I  errors=0
Stage_2A  errors=0      Stage_2F  errors=0      Stage_2J  errors=0
Stage_2B  errors=0      Stage_2G  errors=0
Stage_2C  errors=0      Stage_2H  errors=0
Stage_2D  errors=0
```

### 3.2 The animating notebooks actually display

Entity counts read from the transcript, not asserted:

| notebook | reported | playback |
|---|---|---|
| `Stage_2C` | `showing 12 entities` | `scene playback: running` |
| `Stage_2G` | `showing 3 entities` | `scene playback: running` |
| `Stage_2H` | `showing 2 entities` | `scene playback: running` |

Twelve for 2C is six bodies plus six static box walls, which is why the
number is larger than the body count in the scenario listing.

### 3.3 The films are written

```
StageNbks/out_2A_nash_periodic.html
StageNbks/out_2A_cayley_reflecting.html
StageNbks/out_2B_double_barrier.html
```

Each contains exactly one `#play`, one `#rew` and one `#scrub` control,
confirming the documented Pause / Restart / scrubber set is present in the
generated artifact and not merely in the generator's source.

### 3.4 The scene window really serves, and really has the controls

A live `Stage_2C` session was driven over its own HTTP port:

```
GET / -> http=200 bytes=32568
all 16 toolbar controls present
all 13 status-bar fields present
```

The sixteen ids checked: `bt-start`, `bt-pause`, `bt-stop`, `bt-reverse`,
`bt-reset`, `bt-step`, `bt-back`, `bt-setdt`, `bt-zin`, `bt-zout`,
`bt-home`, `bt-grid`, `bt-trails`, `bt-labels`, `bt-contacts`, `bt-help`.

### 3.5 The typed commands behave as documented

Issued against the same live session, in order:

```
scene status   -> entities = 12 (hidden: none)
                  camera: yaw = -60°, pitch = 55°, dist = 6
scene pause    -> scene playback: paused
scene reverse  -> Err: nothing to reverse — no forward history recorded yet
scene start    -> scene playback: running
scene zoom in  -> camera distance = 4.8      (from 6)
scene close    -> scene closed (http://127.0.0.1:42057/)
```

The `scene reverse` error is documented as expected behaviour rather than
hidden. Reverse replays *recorded* history; immediately after loading there
is none, so the refusal is correct. `scene stop` clears that history
deliberately, so Reverse is empty again after a Stop. This is stated at the
button in every notebook that opens a window, and in `README.md`.

## 4. What is still not covered

Stated so the verification is not read as broader than it is.

* **No browser was driven in this pass.** The page is confirmed to be served
  and to contain all sixteen controls; that the *rendering* is correct, and
  that each button produces the right visual change, was verified in an
  earlier pass by a headless-Chrome CDP session against `newtons_cradle`
  (28/28 checks) and is not re-run here.
* **`--script` mode is not exercised for the animating notebooks**, because
  it exits immediately and takes the window with it. That is exactly why the
  documentation directs readers to `--notebook`.
* **macOS and Windows behaviour is derived, not observed.** The claim that
  nothing opens there follows from `xdg-open` being absent, and the reply
  line is now driven by whether the spawn succeeded — so those systems will
  print the honest variant. No macOS or Windows machine was available to
  confirm it end to end.
* **The notebook gate cannot see this class of defect.** `certify_clean.sh`
  fails a notebook that errors. Every notebook in the previous pass ran
  clean and still documented the arrow keys backwards, because wrong prose
  in a comment is not an error. The entity-count column in `README.md` is
  the human-checked complement to that gate.
