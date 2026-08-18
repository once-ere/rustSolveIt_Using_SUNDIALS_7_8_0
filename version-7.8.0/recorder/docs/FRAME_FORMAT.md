# What a recording carries

A recorded page embeds three JSON values: `BODIES` (what each object
*is*, which never changes), `FRAMES` (what each object *does*, one entry
per step) and `META` (how the run was set up). Field names are short
because they repeat once per frame per body, and a long recording is
mostly this data.

## `BODIES` — static, one entry per object

| field | meaning |
|---|---|
| `name` | the name the script gave it, or `objN` |
| `shape` | the `boundary` record: sphere, cuboid, cylinder, disk, point |
| `mass` | scalar mass |
| `wall` | true for the six slabs of a `BOX`; never drawn as a body |

## `FRAMES` — one entry per recorded step

| field | meaning |
|---|---|
| `t` | simulation time, **as posim reports it** — not `frame × dt` |
| `E` | total energy |
| `P` | total momentum, 3 components |
| `L` | total angular momentum about the origin, 3 components |
| `n` | running collision count |
| `o` | per body, `[position, orientation]`; orientation is w-first |
| `c` | contacts of the step that produced this frame, `[point, normal, impulse]` |
| `j` | joints, `[point, axis, point_j]` — see below |
| `gd` | worst \|g\| over the joint set at this instant |

`t` comes from the solver rather than from arithmetic in the recorder.
That is what makes `test_the_recorder_never_integrates` meaningful: the
frame times are the solver's own, and if they ever stop being `dt`
apart, the tool has started inventing something.

## `j` — the joint entry

Each joint contributes `[point, axis, point_j]`, all in **world**
coordinates, resolved on the Rust side. The body-frame arms a joint
stores internally are no use to a viewer, and re-deriving them in
JavaScript would be a second implementation to keep honest.

| joint | rows | `point` vs `point_j` | drawn as |
|---|---|---|---|
| `CONSTRAIN` (rod) | 1 | **differ** — two points held apart | ring + strut between the ends |
| `BALL` | 3 | coincide | ring |
| `HINGE` | 5 | coincide | ring + axis line |
| `UNIVERSAL` | 4 | coincide | ring + axis line |
| `GEAR` | 1 | **differ** — each wheel's own centre | ring + strut, plus a `ratio` field |
| `RACK` | 1 | **differ** — pinion centre and rack centre | ring + strut, plus `axis`, `axis_j` and `radius` |
| `PRISMATIC` | 5 | **differ** — rail centre and slider centre | ring + strut + axis line |

The player tests the two points for equality; it does not look at the
joint's name. A rod's ends are a fixed distance apart, so the strut is
drawn and the shaft it braces does not look unsupported. Everything
else holds one shared point, so the ends coincide and no line appears.
A new joint kind therefore needs no player change — it is drawn
correctly by construction, according to whether it holds one point or
two.

`axis` is absent for joints that do not turn about one.

## `META` — how the run was set up

| field | meaning |
|---|---|
| `joints` | per joint, `{kind, rows}` — shown in the corner readout |
| `method` | the integrator: `Adams`, `Bdf`, `Ida`, an ARKODE tableau |
| `box` | the `BOX` half-width, or null |
| `gravity` | uniform gravity vector |
| `g_constant` | pairwise `G`; **1 by default**, which is not always wanted |
| `dt` | the step the recording was made with |

`g_constant` is in `META` because it has caused a real confusion: a box
of shapes with `G = 1` left on drifts 3.2 % in energy, and with `G = 0`
holds to `5.1e-16`. Neither is a solver defect — they are different
physical systems, and the recording says which one you are watching.

## Reading a recording without a browser

The data is plain JSON, so a recording is also a data file:

```python
import json, pathlib, re
html = pathlib.Path("videos/universal_joint.html").read_text()
frames = json.loads(re.search(r"const FRAMES\s*=\s*(\[.*?\]);", html, re.S).group(1))
print(max(f["gd"] for f in frames))   # worst |g| over the whole run
```

Every measurement quoted in the documentation for these recordings was
produced this way, from the committed file, rather than copied from a
console.
