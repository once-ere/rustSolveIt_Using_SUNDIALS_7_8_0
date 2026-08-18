# The posim video recorder

Records a [posim](..) run into a single HTML file you can open next
month, on a machine with no Rust toolchain and no network, and still
watch the same motion, scrub it, and read the conserved quantities off
the frame you stopped on.

Commands below are run from the parent directory, the cargo workspace:

```bash
cargo build --release -p posim
recorder/src/record_video.py videos/scenes/kepler_ellipse.posim \
    -o /tmp/mine.html --frames 360 --dt 0.02 --title "Kepler orbit"
```

They work from anywhere, though. The recorder does not assume where it
has been put — see [Which workspace gets recorded](#which-workspace-gets-recorded).

That is the whole dependency list: Python 3.8+, and a built posim. No
pip install, no CDN, nothing fetched at record time or at view time.

## What it does, and what it does not

The recorder drives `posim --machine` over the documented JSONL protocol
(grammar §7, ARCHITECTURE §3.6):

1. Feed the setup script's executable lines through `{"op":"exec"}`.
2. Loop: ask `{"op":"state"}`, keep the reply, then
   `{"op":"exec","code":"step <dt>"}`.
3. Write one HTML file with the frames embedded as JSON and a
   vanilla-JS canvas player around them.

**Every advance is a real SUNDIALS step.** This tool is a camera. It
never integrates anything, never interpolates between frames, and never
smooths anything — so a recording is evidence about the solver, not an
animation of it.

## Layout

| path | what |
|---|---|
| `src/record_video.py` | the recorder, and the player template it emits |
| `src/record_all.py` | records, or checks, every entry in the manifest |
| `recordings.json` | the manifest: the parameters each shipped video was made with |
| `cmake/FindPosim.cmake` | locates a built posim and the workspace holding it |
| `tests/test_units.py` | checks needing no posim |
| `tests/test_end_to_end.py` | checks that drive a real posim |
| `docs/FRAME_FORMAT.md` | what a recorded frame carries, field by field |

## The manifest

The parameters a recording was made with — frame count, `dt`, opening
camera, title, caption — cannot be recovered from the recording without
picking the HTML apart, and guessing them wrong silently produces a
*different* video under the same name. So they live in
`recordings.json`, and re-recording is one command:

```bash
recorder/src/record_all.py            # re-record all five
recorder/src/record_all.py --check    # verify, writing nothing
```

`--check` is the interesting mode. A recording is a pure function of its
scene, its parameters and the posim binary, so if a check fails, one of
those three moved — and the mismatch is written alongside the original
as `*.html.new` so you can diff it.

```
ok       kepler_ellipse: byte-identical (142106 bytes)
ok       tumbling_racket: byte-identical (118201 bytes)
ok       box_of_shapes: byte-identical (287976 bytes)
ok       double_pendulum_hinges: byte-identical (284436 bytes)
ok       universal_joint: byte-identical (205556 bytes)

all 5 recordings reproduce byte for byte
```

## Building, testing, installing

There is no compile step. CMake is here for what it is still good at:
finding the dependency, installing the entry points, and giving the
checks one command.

```bash
cmake -S recorder -B build/recorder
ctest --test-dir build/recorder --output-on-failure
cmake --build build/recorder --target record-all       # re-record
cmake --build build/recorder --target check-recordings # verify only
cmake --install build/recorder --prefix ~/.local
```

`find_package(Posim)` is deliberately **not** `REQUIRED`: the package
configures, installs and runs its offline checks on a machine with no
Rust toolchain. Only the tests that actually drive posim are skipped,
and CTest reports them as skipped rather than passing them vacuously.

## Which workspace gets recorded

This matters more than it sounds. A checkout can hold more than one
posim workspace — a port beside the upstream it was ported from — and
the two agree bit for bit on everything the older grammar can express.
Record against the wrong one and three of the five shipped scenes come
out byte-identical anyway; only a scene using newer grammar fails, and
it fails as a parse error that looks like a typo in the scene.

So the workspace is never guessed from what sits next to what. It is
taken, first hit wins:

1. `--workspace`, or `$POSIM_WORKSPACE`
2. the `workspace` key in the manifest (how `record_all.py` gets it)
3. the **scene script's** own directory and its ancestors
4. the current directory and its ancestors

Only ancestors are searched, never siblings. The scene lives inside the
workspace it belongs to, so the scene decides.

## Pinned properties

These are contracts, not preferences. `tests/test_units.py` checks the
ones that can be checked without a solver.

- **It never integrates.** Every advance is a `step` through
  `integrate::run`.
- **The page fetches nothing.** No CDN, no font server, no `fetch`, no
  WebSocket. It must work from `file://` with the network unplugged.
- **Wall slabs are never drawn as bodies**, and the camera auto-fit
  excludes them. A `BOX` is the dashed interior wireframe only.
- **Round shapes are symmetric about their local z axis**, so the player
  rotates every ring by the w-first quaternion — which is what makes
  spin visible.
- **Joints are drawn from the protocol, not re-derived.** The state dump
  resolves each joint's world pivot and axis on the Rust side; the
  player never reconstructs geometry from body-frame arms.
- **The player keys on geometry, not on a joint's name.** A joint whose
  two ends coincide draws a ring and an axis; one whose ends differ also
  draws the strut between them. Adding a joint kind needs no player
  change.

## Licence

BSD-3-Clause, with the rest of the repository.
