# NOTEBOOKS.md — every notebook of the 2026-07-24 releases

This file documents, in its entirety, **every notebook that was created
or used to check, verify, test, or give examples** for the work of
2026-07-24 — the *shapes + rigid bounding box* release (the manager's
box-of-shapes test problem), the *Reset button* task, and the
*user-defined functions + dumbbell* task.

A **notebook** here is a posim cell session (`In[n]:=` / `Out[n]=`).
posim runs a notebook in three ways, and all three were used:

| mode | how it runs | how it appears below |
|---|---|---|
| **interactive REPL** | `cargo run -p posim` reading stdin (driven live through a FIFO during verification, with a real browser attached to the scene window) | the commands sent, then the captured `In[]/Out[]` transcript (a FIFO does not echo input text into the prompt line, so the inputs are listed first) |
| **script batch** | `cargo run -p posim -- --script <file>` | the verbatim `In[]/Out[]` transcript, exactly as printed |
| **machine mode** | `posim --machine`, one JSON request/reply per line (the JupyterLab kernel's protocol) | the requests sent and the verbatim reply lines |

Every transcript below is **captured output, reproduced verbatim** —
nothing is retyped or abridged unless explicitly marked. Deterministic
runs (all script notebooks) reproduce these outputs exactly; the
final versions were re-run inside a fresh `git clone` of the published
repository and matched byte-for-byte.

---

## Part I — the box-of-shapes task (the manager's test problem)

### Notebook 1 — `scripts/collisions/11_box_of_shapes.posim`, first capture (superseded)

*Mode:* script batch. *Purpose:* the first executed run of the
manager's demo — all six body types inside the rigid, infinitely
massive `BOX 4`; used for the initial documentation capture.
*Status:* **superseded** by Notebook 2: the adversarial review that
followed this run fixed the tier-3 candidate axes and the anti-tunnel
cap, which legitimately changed the chaotic trajectory after the first
extended-pair contact (132 → 119 collisions). Kept here because it was
used for verification at the time, and because comparing it with
Notebook 2 shows exactly what the review fixes changed — and what they
did not (E₀ = 30000, the torus inertia, the 51 pairs, the wall
inverse-mass are identical in both).

```
In[1]:= set system.g_constant = 0
In[2]:= box 4
Out[2]= box: inner size 4 x 4 x 4 — six static walls obj0, obj1, obj2, obj3, obj4, obj5 with inverse_mass = 0 (infinitely massive); objects collide elastically off the inside faces
In[3]:= new torus { mass = 1, inner_radius = 1, outer_radius = 2, orientation = [0.888073833977115, -0.325057583671868, 0.325057583671868, 0] }
Out[3]= obj6
In[4]:= new point { mass = 1, position = [1.406590, -0.995859, 0.569601], velocity = [100, 200, 100] }
Out[4]= obj7
In[5]:= new sphere { mass = 2, radius = 1/2, position = [1.424704, 1.367496, -0.493612] }
Out[5]= obj8
In[6]:= new disk { mass = 2/3, radius = 1, position = [-0.677386, -1.041493, -1.091679], orientation = [0.900447102352677, 0.307567078752479, -0.307567078752479, 0] }
Out[6]= obj9
In[7]:= new cuboid { mass = 5/3, half_extents = [0.5, 0.5, 0.5], position = [1.074397, 0.816223, 1.102099] }
Out[7]= obj10
In[8]:= new cylinder { mass = 2, radius = 1/2, height = 3/2, position = [-1.027024, -1.403890, -0.485619], orientation = [0.968912421710645, 0, 0.247403959254523, 0] }
Out[8]= obj11
In[9]:= list
Out[9]= obj0: cuboid he=[1, 4, 4], mass=0, charge=0, pos=[3, 0, 0] [wall: static, inverse_mass=0]
obj1: cuboid he=[1, 4, 4], mass=0, charge=0, pos=[-3, 0, 0] [wall: static, inverse_mass=0]
obj2: cuboid he=[4, 1, 4], mass=0, charge=0, pos=[0, 3, 0] [wall: static, inverse_mass=0]
obj3: cuboid he=[4, 1, 4], mass=0, charge=0, pos=[0, -3, 0] [wall: static, inverse_mass=0]
obj4: cuboid he=[4, 4, 1], mass=0, charge=0, pos=[0, 0, 3] [wall: static, inverse_mass=0]
obj5: cuboid he=[4, 4, 1], mass=0, charge=0, pos=[0, 0, -3] [wall: static, inverse_mass=0]
obj6: torus ring=1.5 tube=0.5, mass=1, charge=0, pos=[0, 0, 0]
obj7: point, mass=1, charge=0, pos=[1.40659, -0.995859, 0.569601]
obj8: sphere r=0.5, mass=2, charge=0, pos=[1.424704, 1.367496, -0.493612]
obj9: disk r=1, mass=0.6666666666666666, charge=0, pos=[-0.677386, -1.041493, -1.091679]
obj10: cuboid he=[0.5, 0.5, 0.5], mass=1.6666666666666667, charge=0, pos=[1.074397, 0.816223, 1.102099]
obj11: cylinder r=0.5 h=1.5, mass=2, charge=0, pos=[-1.027024, -1.40389, -0.485619]
In[10]:= collide
Out[10]= collisions ON (51 collidable pair(s); 0 impulse(s) so far)
In[11]:= get obj0.inverse_mass
Out[11]= 0
In[12]:= get obj6.inertia_tensor
Out[12]= [[1.28125, 0, 0], [0, 1.28125, 0], [0, 0, 2.4375]]
In[13]:= energy
Out[13]= 30000
In[14]:= momentum
Out[14]= [100, 200, 100]
In[15]:= run 0.1 steps 100
Out[15]= t = 0.1 (2259 solver steps, 100 snapshots, |dE/E| = 1.031e-9, 132 collision(s) — CONTACTS lists them)
In[16]:= energy
Out[16]= 29999.999969060293
In[17]:= momentum
Out[17]= [243.5927833501326, -82.35240150995011, -4.548943318788247]
In[18]:= get system.collisions
Out[18]= 132
In[19]:= get system.box
Out[19]= 4
```

### Notebook 2 — `scripts/collisions/11_box_of_shapes.posim`, final capture (canonical)

*Mode:* script batch. *Purpose:* the **canonical** demo run with the
final, reviewed binary — this is the transcript quoted in
`grammar.md` Example 13, `physical_object_simulator.md` S13 and
`collision_detection.md` Example 11, and it reproduces **byte-for-byte
in a fresh clone** of the published repository. (Re-captured after the
second adversarial review pass: its directed-support arithmetic fixes
moved only the floating-point tails — E by one ulp, the momentum
components at the 10th significant digit — while the 2121 solver
steps, the 119 collisions and |ΔE/E| = 2.040·10⁻¹⁰ are unchanged.)
*What it verifies:* E₀ = ½·1·(100²+200²+100²) = 30000 exactly;
elastic conservation |ΔE/E| = 2.040·10⁻¹⁰ through 119 collisions; the
infinitely massive walls absorb momentum ((100,200,100) →
(146.49, 102.15, 46.02)) while staying at rest (`obj0.inverse_mass = 0`);
the tilted-torus inertia diag(1.28125, 1.28125, 2.4375) matches the
analytic solid-torus formulas; 51 collidable pairs = C(12,2) − 15
static-static wall pairs.
*Reproduce:* `cargo run -p posim -- --script scripts/collisions/11_box_of_shapes.posim`

Cells `In[1]`–`In[14]` and `In[19]` are identical to Notebook 1
(same deterministic setup). The differing cells:

```
In[15]:= run 0.1 steps 100
Out[15]= t = 0.1 (2121 solver steps, 100 snapshots, |dE/E| = 2.040e-10, 119 collision(s) — CONTACTS lists them)
In[16]:= energy
Out[16]= 29999.999993880127
In[17]:= momentum
Out[17]= [146.48911126803657, 102.1478121131382, 46.01601291636828]
In[18]:= get system.collisions
Out[18]= 119
```

### Notebook 3 — interactive box-of-shapes session with the live scene window

*Mode:* interactive REPL (FIFO-driven), scene window on port 8901
opened in a real browser. *Purpose:* run the manager's demo **from the
notebook** with the graphical display, and verify the window
end-to-end: playback, contact-normal arrows, the dashed box wireframe
(pixel-scanned on the canvas: 5,904 wireframe pixels, 2,218 golden
arrow pixels, zero JS console errors), and reverse playback landing
exactly at t = 0. The event drains (`scene events` cells) exist
because the reverse verification polled the notebook side while the
window ran.

Input commands sent, in order:

```
set system.g_constant = 0
box 4
new torus { mass = 1, inner_radius = 1, outer_radius = 2, orientation = [0.888073833977115, -0.325057583671868, 0.325057583671868, 0] }
new point { mass = 1, position = [1.406590, -0.995859, 0.569601], velocity = [100, 200, 100] }
new sphere { mass = 2, radius = 1/2, position = [1.424704, 1.367496, -0.493612] }
new disk { mass = 2/3, radius = 1, position = [-0.677386, -1.041493, -1.091679], orientation = [0.900447102352677, 0.307567078752479, -0.307567078752479, 0] }
new cuboid { mass = 5/3, half_extents = [0.5, 0.5, 0.5], position = [1.074397, 0.816223, 1.102099] }
new cylinder { mass = 2, radius = 1/2, height = 3/2, position = [-1.027024, -1.403890, -0.485619], orientation = [0.968912421710645, 0, 0.247403959254523, 0] }
energy
scene create 8901
scene set_time_step 0.0002
scene zoom 2.2
scene events            # repeated while the window ran (Start / Reverse
scene pause             # were clicked IN the browser; drains show the
scene status            # 'window action' events)
scene reverse
scene events            # repeated until the reverse completed
scene status
scene events
scene close
%quit
```

Captured transcript, in entirety:

```
posim — physical_object simulator notebook (sundials_rs backend)
type HELP for the command language, %quit to leave

In[1]:= In[2]:= Out[2]= box: inner size 4 x 4 x 4 — six static walls obj0, obj1, obj2, obj3, obj4, obj5 with inverse_mass = 0 (infinitely massive); objects collide elastically off the inside faces
In[3]:= Out[3]= obj6
In[4]:= Out[4]= obj7
In[5]:= Out[5]= obj8
In[6]:= Out[6]= obj9
In[7]:= Out[7]= obj10
In[8]:= Out[8]= obj11
In[9]:= Out[9]= 30000
In[10]:= Out[10]= scene window created: http://127.0.0.1:8901/
(opened in your browser; if no window appeared, open that address yourself)
showing 12 entities; SCENE START begins the evolution — HELP lists all scene commands
In[11]:= Out[11]= scene time step dt = 0.0002
In[12]:= Out[12]= camera distance = 5.454545454545454
In[13]:= Out[13]= window connected (1 total)
window action: start
window action: reverse
In[14]:= Out[14]= (no scene events)
In[15]:= Out[15]= (no scene events)
In[16]:= Out[16]= (no scene events)
In[17]:= Out[17]= reverse: reached the beginning of recorded history — paused
In[18]:= Out[18]= scene playback: paused
In[19]:= Out[19]= scene: http://127.0.0.1:8901/  (1 window(s) connected)
mode = paused, t = 0.7937999999999452, dt = 0.0002, steps = 5045, history = 3969 frame(s)
entities = 12 (hidden: none)
camera: yaw = -60°, pitch = 55°, dist = 5.454545454545454, target = [0, 0, 0]
In[20]:= Out[20]= window action: start
In[21]:= Out[21]= scene playback: reversing
In[22]:= Out[22]= (no scene events)
In[23]:= Out[23]= (no scene events)
In[24]:= Out[24]= (no scene events)
In[25]:= Out[25]= (no scene events)
In[26]:= Out[26]= (no scene events)
In[27]:= Out[27]= (no scene events)
In[28]:= Out[28]= (no scene events)
In[29]:= Out[29]= (no scene events)
In[30]:= Out[30]= (no scene events)
In[31]:= Out[31]= (no scene events)
In[32]:= Out[32]= (no scene events)
In[33]:= Out[33]= (no scene events)
In[34]:= Out[34]= (no scene events)
In[35]:= Out[35]= (no scene events)
In[36]:= Out[36]= (no scene events)
In[37]:= Out[37]= (no scene events)
In[38]:= Out[38]= (no scene events)
In[39]:= Out[39]= (no scene events)
In[40]:= Out[40]= (no scene events)
In[41]:= Out[41]= (no scene events)
In[42]:= Out[42]= (no scene events)
In[43]:= Out[43]= (no scene events)
In[44]:= Out[44]= reverse: reached the beginning of recorded history — paused
In[45]:= Out[45]= scene: http://127.0.0.1:8901/  (1 window(s) connected)
mode = paused, t = 0, dt = 0.0002, steps = 5045, history = 0 frame(s)
entities = 12 (hidden: none)
camera: yaw = -60°, pitch = 55°, dist = 5.454545454545454, target = [0, 0, 0]
In[46]:= Out[46]= (no scene events)
In[47]:= Out[47]= scene closed (http://127.0.0.1:8901/)
In[48]:= goodbye
```

The load-bearing readings: `Out[19]` — `steps = 5045, history = 3969`,
proving the first reverse popped exactly 5045 − 3969 = 1076 frames
(everything recorded at click time); `Out[45]` — the controlled second
reverse landed at **`t = 0, history = 0`** and *stayed paused*, the
exact-restore contract. The one `window action: start` drained at
`Out[20]` was an environmental one-off from the browser pane
re-activating the still-focused Start button; the controlled repro
(`In[21]`–`In[45]`) showed no phantom command.

---

## Part II — the Reset-button task

### Notebook 4 — interactive Reset verification session

*Mode:* interactive REPL (FIFO-driven), scene window on port 8903 in a
real browser. *Purpose:* verify the permanent GUI **Reset** button and
`SCENE RESET` — the browser side clicked Start (ran to t = 0.72, body
moved), clicked **Reset** (mode stopped, t = 0, history 0, body
bit-identically back), clicked Start again (re-run confirmed); the
notebook side then exercised `SCENE RESET` + `SCENE STATUS`.
(An initial attempt on port 8902 was discarded: the running binary
predated the Reset build — the page had no `bt-reset`. The session
below is the rebuilt binary.)

Input commands sent, in order:

```
set system.g_constant = 0
box 4
new sphere { mass = 1, radius = 0.5, velocity = [3, 1.7, 0.9] }
scene create 8903
scene set_time_step 0.005
# (browser: Start → run → Reset → Start, verified via page state)
scene reset
scene status
scene close
%quit
```

Captured transcript, in entirety:

```
posim — physical_object simulator notebook (sundials_rs backend)
type HELP for the command language, %quit to leave

In[1]:= In[2]:= Out[2]= box: inner size 4 x 4 x 4 — six static walls obj0, obj1, obj2, obj3, obj4, obj5 with inverse_mass = 0 (infinitely massive); objects collide elastically off the inside faces
In[3]:= Out[3]= obj6
In[4]:= Out[4]= scene window created: http://127.0.0.1:8903/
(opened in your browser; if no window appeared, open that address yourself)
showing 7 entities; SCENE START begins the evolution — HELP lists all scene commands
In[5]:= Out[5]= scene time step dt = 0.005
In[6]:= Out[6]= scene playback reset to its initial state (t = 0, 7 entities); Start runs the simulation again from the beginning
In[7]:= Out[7]= scene: http://127.0.0.1:8903/  (1 window(s) connected)
mode = stopped, t = 0, dt = 0.005, steps = 0, history = 0 frame(s)
entities = 7 (hidden: none)
camera: yaw = -60°, pitch = 55°, dist = 12, target = [0, 0, 0]
In[8]:= Out[8]= scene closed (http://127.0.0.1:8903/)
In[9]:= goodbye
```

The browser-side measurements of the same session (read from the live
page): `running t=0.72, moved:true` → after Reset
`stopped, t:0, history:0, backToInitial:true` → after Start
`running t=0.435, movedAgain:true`.

---

## Part III — the user-functions + dumbbell task

### Notebook 5 — `create_dumbell` smoke-test script

*Mode:* script batch (scratch script, not shipped — reproduced here in
full). *Purpose:* first end-to-end check of the whole language
feature: a multi-line `DEF` with 11 defaulted parameters, a call with
a string name and partial arguments, member reads, the `.vx`/`.x`
shorthands, a member **write** rebuilding the body (m1 = 3 → total
mass 5.5), and `LIST`/`FUNCS`.

The script:

```
def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) {
  new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, rod_radius = rod_radius, length = length, position = position, velocity = velocity, angular_velocity = angular_velocity }
}
create_dumbell("dumbell0", 1, 2, 0.5)
get dumbell0.m1
get dumbell0.m2
get dumbell0.m_rod
get dumbell0.mass
set dumbell0.vx = 1.5
get dumbell0.vx
get dumbell0.x
set dumbell0.m1 = 3
get dumbell0.mass
get dumbell0.length
list
funcs
```

Captured transcript, in entirety:

```
In[1]:= def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) {
  new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, rod_radius = rod_radius, length = length, position = position, velocity = velocity, angular_velocity = angular_velocity }
}
Out[1]= function create_dumbell(11 parameter(s)) defined — 1 body line(s)
In[2]:= create_dumbell("dumbell0", 1, 2, 0.5)
Out[2]= obj0 as dumbell0
In[3]:= get dumbell0.m1
Out[3]= 1
In[4]:= get dumbell0.m2
Out[4]= 2
In[5]:= get dumbell0.m_rod
Out[5]= 0.5000000000000002
In[6]:= get dumbell0.mass
Out[6]= 3.5
In[7]:= set dumbell0.vx = 1.5
In[8]:= get dumbell0.vx
Out[8]= 1.5
In[9]:= get dumbell0.x
Out[9]= 0
In[10]:= set dumbell0.m1 = 3
In[11]:= get dumbell0.mass
Out[11]= 5.5
In[12]:= get dumbell0.length
Out[12]= 1
In[13]:= list
Out[13]= obj0: dumbbell r1=0.25 r2=0.25 rod_r=0.1 len=1, mass=5.5, charge=0, pos=[0, 0, 0]
In[14]:= funcs
Out[14]= create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) — 1 body line(s); SHOW create_dumbell prints it
```

(`Out[5]`'s `0.5000000000000002` is the floating-point recovery of
`m_rod = (1 − f1 − f2)·M` from the stored mass fractions.)

### Notebook 6 — `scripts/collisions/12_two_dumbbells.posim` (canonical demo)

*Mode:* script batch. *Purpose:* the **acceptance test** — two named
dumbbells built by the user-defined `create_dumbell`, colliding
off-center with spin; total energy, total linear momentum AND total
angular momentum verified conserved. This transcript is quoted in
`grammar.md` Example 14, `physical_object_simulator.md` S14 and
`collision_detection.md` Example 12, and reproduces byte-for-byte in a
fresh clone of the published repository.
*Reproduce:* `cargo run -p posim -- --script scripts/collisions/12_two_dumbbells.posim`

Captured transcript, in entirety:

```
In[1]:= set system.g_constant = 0
In[2]:= def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) {
  new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, rod_radius = rod_radius, length = length, position = position, velocity = velocity, angular_velocity = angular_velocity }
}
Out[2]= function create_dumbell(11 parameter(s)) defined — 1 body line(s)
In[3]:= create_dumbell("dumbell0", 1, 2, 0.5, 0.25, 0.25, 0.1, 1, [-2, 0.15, 0], [1.5, 0, 0], [0, 0, 0.6])
Out[3]= obj0 as dumbell0
In[4]:= create_dumbell("dumbell1", 2, 1, 0.4, 0.3, 0.2, 0.08, 1.2, [2, -0.15, 0], [-1.5, 0, 0], [0.4, 0, 0])
Out[4]= obj1 as dumbell1
In[5]:= list
Out[5]= obj0: dumbbell r1=0.25 r2=0.25 rod_r=0.1 len=1, mass=3.5, charge=0, pos=[-2, 0.15, 0]
obj1: dumbbell r1=0.3 r2=0.2 rod_r=0.08 len=1.2, mass=3.4, charge=0, pos=[2, -0.15, 0]
In[6]:= get dumbell0.m1
Out[6]= 1
In[7]:= get dumbell1.m_rod
Out[7]= 0.3999999999999999
In[8]:= get dumbell0.vx
Out[8]= 1.5
In[9]:= energy
Out[9]= 7.865310611764706
In[10]:= momentum
Out[10]= [0.15000000000000036, 0, 0]
In[11]:= angmom
Out[11]= [0.4443030588235295, 0, -1.5059999999999998]
In[12]:= run 3 steps 60
Out[12]= t = 3 (3861 solver steps, 60 snapshots, |dE/E| = 9.463e-11, 2 collision(s) — CONTACTS lists them)
In[13]:= energy
Out[13]= 7.865310611020375
In[14]:= momentum
Out[14]= [0.15000000000000036, 0, 0]
In[15]:= angmom
Out[15]= [0.44430305882373156, -0.000000000000009547918011776346, -1.505999999999855]
In[16]:= get system.collisions
Out[16]= 2
```

The conservation record, through 2 collisions:
**E** 7.865310611764706 → 7.865310611020375 (|ΔE/E| = 9.463·10⁻¹¹);
**P** `[0.15000000000000036, 0, 0]` → bit-identical;
**L** drift ≈ 1.5·10⁻¹³ per component (the −9.5·10⁻¹⁵ y-component is
the machine-precision fingerprint of a genuine off-center impact).

### Notebook 7 — interactive two-dumbbell session with the live scene window

*Mode:* interactive REPL (FIFO-driven), scene window on port 8905 in a
real browser. *Purpose:* the same setup **as a new notebook** (the
multi-line `DEF` shows the `...:=` continuation prompts) with the
graphical display: entity labels showed the user names
(`dumbell0`, `dumbell1`), and the window's labeled **conserved
quantities** readout was read from the live page before and after the
collision.

Input commands sent, in order:

```
set system.g_constant = 0
def create_dumbell(name, m1 = 1, m2 = 1, m_rod = 0.5, r1 = 0.25, r2 = 0.25, rod_radius = 0.1, length = 1, position = [0, 0, 0], velocity = [0, 0, 0], angular_velocity = [0, 0, 0]) {
  new dumbbell as name { m1 = m1, m2 = m2, m_rod = m_rod, r1 = r1, r2 = r2, rod_radius = rod_radius, length = length, position = position, velocity = velocity, angular_velocity = angular_velocity }
}
create_dumbell("dumbell0", 1, 2, 0.5, 0.25, 0.25, 0.1, 1, [-2, 0.15, 0], [1.5, 0, 0], [0, 0, 0.6])
create_dumbell("dumbell1", 2, 1, 0.4, 0.3, 0.2, 0.08, 1.2, [2, -0.15, 0], [-1.5, 0, 0], [0.4, 0, 0])
scene create 8905
scene set_time_step 0.01
# (browser: Start; ran through the collision to t = 2.96)
scene close
%quit
```

Captured transcript, in entirety (note the two `...:=` continuation
prompts while the `DEF` block was open):

```
posim — physical_object simulator notebook (sundials_rs backend)
type HELP for the command language, %quit to leave

In[1]:= In[2]:=   ...:=   ...:= Out[2]= function create_dumbell(11 parameter(s)) defined — 1 body line(s)
In[3]:= Out[3]= obj0 as dumbell0
In[4]:= Out[4]= obj1 as dumbell1
In[5]:= Out[5]= scene window created: http://127.0.0.1:8905/
(opened in your browser; if no window appeared, open that address yourself)
showing 2 entities; SCENE START begins the evolution — HELP lists all scene commands
In[6]:= Out[6]= scene time step dt = 0.01
In[7]:= Out[7]= scene closed (http://127.0.0.1:8905/)
In[8]:= goodbye
```

The graphical readout, read verbatim from the live page's HUD —
**identical before and after the impact** (t = 0 vs t = 2.96, past the
collision), except L's y-component showing the ~10⁻¹³ residual:

```
conserved quantities
E  = 7.86531061
P  = [0.15000, 0.00000, 0.00000]  |.| = 0.15000
L  = [0.44430, 0.00000, -1.50600]  |.| = 1.57017
```
```
conserved quantities
E  = 7.86531061
P  = [0.15000, 0.00000, 0.00000]  |.| = 0.15000
L  = [0.44430, -1.31228e-13, -1.50600]  |.| = 1.57017
```

### Notebook 8 — machine-mode session (JupyterLab protocol)

*Mode:* machine (`posim --machine`, JSONL). *Purpose:* verify that a
**multi-line DEF arrives intact through the machine protocol** (one
`exec` request carrying embedded newlines — what the JupyterLab kernel
sends), that the function call works, and that the registered name
resolves as a path.

Requests sent:

```
{"op":"exec","code":"def mk(m = 3) {\n  new sphere as pip { mass = m }\n}"}
{"op":"exec","code":"mk()"}
{"op":"get","path":"pip.mass"}
{"op":"quit"}
```

Reply lines, in entirety:

```
{"display":"function mk(1 parameter(s)) defined — 1 body line(s)","ok":true,"result":"function mk(1 parameter(s)) defined — 1 body line(s)"}
{"display":"obj0 as pip","ok":true,"result":"obj0 as pip"}
{"display":"3","ok":true,"result":3.0}
```

---

## Part IV — notebooks embedded in the automated test suite

These notebooks run as cells inside `cargo test` on every build (they
are code, not captures — quoted from the test sources):

- **`posim/src/notebook.rs::save_load_round_trips_a_multiline_def`** —
  a fresh `Notebook` executes the cells
  `def probe(m = 2) {` ⏎ `  new sphere { mass = m }` ⏎ `}` and
  `probe()`, then `%save`s to a file; a **second** fresh notebook
  `%load`s the file and the test asserts the multi-line DEF replayed as
  ONE cell, the call replayed, and the default survived. This test
  exists because the inline review found (and fixed) `%load` shattering
  saved multi-line DEFs line-by-line.
- **`posim/src/vm.rs::def_call_named_objects_and_dumbbell_members`** —
  the full `create_dumbell` flow as notebook cells: definition,
  call with a string name, member reads/writes, `.x`/`.vx` shorthands,
  DEL renumbering of names, duplicate/reserved-name refusals,
  `SHOW` + re-`DEF` editing, `LET`-backed defaults, arity errors, and
  the no-ghost guarantee of a failing call.
- **`posim/src/vm.rs::box_family_and_infinite_mass_walls`,
  `box_recreate_after_wall_deletion_leaks_nothing`,
  `torus_pair_is_order_independent_and_new_is_transactional`,
  `new_shapes_and_parameter_paths`, `collide_command_and_contact_paths`,
  `new_set_get_roundtrip`, `step_runs_sundials`,
  `observables_and_errors`, `method_switch_and_sprk_gate`** — each
  drives `execute_line` cell sequences pinning the command language
  used by every notebook above.
- **`posim/src/notebook.rs::cells_number_and_capture_output`,
  `magics_edit_and_rerun`** — the pre-existing cell-numbering and
  `%edit`/`%rerun` notebooks.
- **`posim/src/machine.rs::exec_get_set_state_flow`,
  `state_reports_box_walls_and_inverse_mass`** and
  **`jupyter/test_protocol.py` / `jupyter/test_kernel.py`** — machine-
  mode notebook sessions over the JSONL protocol and, for the kernel
  test, over real ZMQ from a JupyterLab-style client.

## Part V — verification notebooks run by subagents (transcripts not retained)

During the documentation and review phases, subagents ran additional
throwaway posim sessions purely to verify quoted strings and hunt
edge cases (their scratch files were deleted per the workspace rules;
the *findings* are recorded in the commit messages). Precisely:

- the documentation writers re-ran `scripts/collisions/11_box_of_shapes.posim`
  and `12_two_dumbbells.posim` and diffed the transcripts bit-exact
  against the captures above, and executed one-off cells to capture
  reply strings verbatim (`box: dissolved (…)`, the torus `SET radius`
  error, `inner_radius must be a finite number >= 0, got -1`,
  `torus needs 0 <= inner < outer (got inner = 2, outer = 1)`,
  `{ outer_radius = 0.5, inner_radius = 0.2 }` → ring 0.35/tube 0.15,
  the `SCENE RESET` reply, `(replacing the previous box)`);
- the adversarial-review verifiers of the box release ran edge-case
  cell sequences that CONFIRMED (and led to fixes for): the
  order-dependent torus pair, the failing-NEW ghost object,
  `box 4; del 0; box 10` leaking wall slabs, `RESET` leaving a stale
  box wireframe in an open window, `SET radius` silently converting a
  torus, and the horn-torus `inner_radius = 0` rejection. Each fix is
  pinned by a test in Part IV.

## Provenance

Raw capture files (session scratch): `demo1.log` (Notebook 1),
`demo_final.log` (2), `posim_out` (3), `posim_out2` (4),
`smoke_def.posim` + its output (5), `demo_dumbbell.log` (6),
`posim_out3` (7), `machine_session.txt` (8). Notebooks 2 and 6 were
additionally re-executed inside a fresh clone of
`https://github.com/once-ere/SolveIt_rust` (the repository this project
was exported from) and matched these
transcripts byte-for-byte; the suite in Part IV (104 tests:
40 lib + 16 collision + 9 conservation + 39 posim) is green in the
same clone with zero warnings.
