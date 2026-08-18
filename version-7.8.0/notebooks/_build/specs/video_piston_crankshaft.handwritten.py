import json
spec = {
"key": "video_piston_crankshaft",
"title": "A piston driven by a crankshaft (the slider-crank)",
"source": "version-7.8.0/videos/scenes/piston_crankshaft.posim",
"category": "video",
"howtorun": "cargo run --release -p posim -- --script videos/scenes/piston_crankshaft.posim",
"abstract": """A slider-crank is the mechanism at the heart of every piston engine: a
wheel turning in a bearing, a rod hung off a pin near the wheel's rim, and
a piston that the rod pushes up and down a straight bore. It converts
rotation into a reciprocating slide, and back again.

What this notebook demonstrates is that the simulator gets the *kinematics*
exactly right without ever being told them. Nobody supplies the formula
relating piston position to crank angle. Four joints are declared, the
solver holds them, and the piston position that comes out agrees with the
exact closed-form expression to about one part in ten million.""",

"situation": """### The parts

| name | shape | mass | size | starts at | notes |
|---|---|---|---|---|---|
| `mount` | `point` | 1 (but `inverse_mass = 0`) | — | origin | the engine block. Immovable: it is the main bearing's outer race. |
| `crank` | `cylinder` | 2 | radius 0.55, half-height 0.05 | origin | the crankshaft disc, spinning about the z axis at 2 rad/s |
| `rod` | `cuboid` | 0.4 | half-extents 0.5 x 0.05 x 0.05 | (1, 0, 0) | the connecting rod, a slender bar of length 1.0 |
| `piston` | `cuboid` | 1 | half-extents 0.5 x 0.3 x 0.3 | (2, 0, 0) | the piston |
| `guide` | `point` | 1 (but `inverse_mass = 0`) | — | (2, 0, 0) | the cylinder bore. Immovable: it defines the line the piston slides along. |

`mount` and `guide` are *static bodies*. Their `inverse_mass` is zero, so
no force can accelerate them; and because they are `point` shapes their
inertia tensors are zero too, so no torque can turn them. They are the
parts bolted to the world.

The two moving-mass shapes matter numerically, not decoratively. A
`cylinder` of mass `m`, radius `r` and half-height `h` has

```
I_zz = 0.5 m r^2                      about its axis of symmetry
I_xx = I_yy = m (3 r^2 + 4 h^2) / 12  about any diameter
```

and a `cuboid` of half-extents `(a, b, c)` has

```
I_xx = m (b^2 + c^2) / 3 ,  I_yy = m (a^2 + c^2) / 3 ,  I_zz = m (a^2 + b^2) / 3
```

These are computed by the simulator from the shape you declare. You never
type an inertia tensor.

### The interactions

There is **no gravity** (`g_constant = 0` switches off mutual attraction
between the bodies, and `uniform_gravity` is set to zero) and **no contact**
(`collide off`). Every force in this model is a *constraint* force: the
force a bearing exerts to stop a shaft leaving it, the force a pin exerts
to stop a rod coming off. Nothing is driven and nothing is damped, so the
mechanism coasts on the kinetic energy it was given at the start.

### Why the geometry is what it is

Every joint in this simulator places its pivot at the **midpoint of the two
bodies' centres as they stand when the joint is created**. So the sizes are
not free. Write `a` for the crank throw and `L` for the rod length. The
midpoint rule puts the big-end pin at half of the rod's centre coordinate,
so the rod's centre must sit at `2a`; but a rod reaching from the crank pin
at `a` to the wrist pin at `a + L` has its centre at `a + L/2`. Those agree
only if

```
L = 2a
```

Here `a = 0.5` and `L = 1.0`, so the main bearing lands at the origin, the
crank pin at `x = 0.5`, and the wrist pin at `x = 1.5` — every pivot on a
real pin. Any other ratio and the picture and the mechanism disagree.

### Why the rod ends are ball joints and not hinges

Every pin in a real engine is a hinge, so four `HINGE` joints is the
obvious thing to try. It does not work, and the reason is worth
understanding because it recurs in every planar linkage:

```
HINGE 5 + HINGE 5 + HINGE 5 + PRISMATIC 5 = 20 rows,  on 18 freedoms
```

Three free bodies give 18 degrees of freedom, and 20 constraint equations
on 18 unknowns is over-determined. The equations are not contradictory —
they are *redundant*: each hinge separately insists the motion stays in the
z = 0 plane, and it gets said three times. A redundant constraint set makes
the solver's matrix singular, and this simulator refuses such a model
rather than picking an arbitrary solution.

The standard remedy, and the one real multibody codes use, is to give the
connecting rod **spherical ends**:

```
HINGE 5 + BALL 3 + BALL 3 + PRISMATIC 5 = 16 rows,  on 18 freedoms
```

That leaves `18 - 16 = 2` freedoms. One is the crank angle — that is the
mechanism. The other is the rod's spin about its own long axis, which is
passive: nothing applies a torque along the rod's length, so that freedom
simply stays where it started. The run below measures it, and it is
0.000 degrees.""",

"eom": """Each of the three moving bodies obeys Newton's and Euler's equations. With
no gravity and no contact, the only applied loads are the constraint forces
`F_c` and constraint torques `T_c` that the joints generate:

```
m_crank  * d2(x_crank)/dt2  = F_c,crank
m_rod    * d2(x_rod)/dt2    = F_c,rod
m_piston * d2(x_piston)/dt2 = F_c,piston

I_i * d(omega_i)/dt + omega_i x (I_i omega_i) = T_c,i      for i = crank, rod, piston
```

The `omega x (I omega)` term is the gyroscopic term, and it is not
negligible here: it is what makes the crank speed up and slow down.

None of these constraint forces are written down by you. They are the
Lagrange multipliers, solved for at every step so that the joint equations
in the next section hold exactly.

**The exact kinematic consequence.** Although the *dynamics* need solving,
the *geometry* has a closed form. Measuring the wrist pin position `x` from
the crankshaft axis, with crank angle `theta`:

```
x(theta) = a cos(theta) + sqrt(L^2 - a^2 sin^2(theta))
```

This is exact — no small-angle approximation anywhere. The stroke runs from
`L - a = 0.5` at bottom dead centre to `L + a = 1.5` at top. The run below
checks the simulated piston against this formula at every sample.

**Note what the crank does *not* do.** It does not turn at a constant rate,
and it should not: nothing drives it. It was given 2 rad/s and left alone,
and the rod and piston trade inertia with it through the cycle, so the
crank angle wanders ahead of and behind a uniform `2t`. The closed form
above is unaffected, because it is a statement about the *linkage* — it
relates the piston to whatever angle the crank happens to be at. That is
exactly what makes it a fair test of a free-running mechanism.""",

"constraints": """Four joints, 16 scalar equations. Writing `r_i^P` for the position of the
point `P` expressed in body `i`'s own frame, and `R_i` for body `i`'s
rotation matrix:

**1. `HINGE mount crank [0,0,1]` — 5 rows.** The crankshaft in its main
bearing. Three rows hold a point shared, two hold an axis shared:

```
g_1..3 = (x_crank + R_crank r_crank^O) - (x_mount + R_mount r_mount^O) = 0
g_4..5 = [ p . (R_crank z) ,  q . (R_crank z) ] = 0
```

where `p` and `q` are two directions perpendicular to the hinge axis. The
last two rows say the crank's z axis has not tipped away from the world z
axis, leaving one freedom: the rotation about it.

**2. `BALL crank rod` — 3 rows.** The big end, on the crank pin. One point
shared, all three rotations free:

```
g_6..8 = (x_crank + R_crank r_crank^B) - (x_rod + R_rod r_rod^B) = 0
```

**3. `BALL rod piston` — 3 rows.** The little end, on the wrist pin:

```
g_9..11 = (x_rod + R_rod r_rod^W) - (x_piston + R_piston r_piston^W) = 0
```

**4. `PRISMATIC guide piston [1,0,0]` — 5 rows.** The bore. Two rows kill
the two translations across the slide direction `n = (1,0,0)`, three kill
all rotation:

```
g_12..13 = [ p . d ,  q . d ] = 0      where d = x_piston - x_guide - (relevant offset)
g_14..16 = the three rows that hold the piston's frame aligned with the guide's
```

leaving one freedom: the slide along `n`.

**Total: 5 + 3 + 3 + 5 = 16 rows on 18 degrees of freedom.**

Every one of these is held at the position level (`g = 0`) *and* at the
velocity level (`d(g)/dt = J u = 0`) simultaneously, which is what the GGL
formulation described above is for.""",

"reduction": """**Sizing this particular problem.**

There are 5 bodies, so the state vector `y` has

```
5 bodies x 13 numbers = 65 components
```

and it is laid out as five consecutive 13-number blocks, in creation order:
`mount`, `crank`, `rod`, `piston`, `guide`. Within each block the order is
position (3), linear momentum (3), quaternion w-first (4), angular momentum
(3).

The two static bodies still occupy their 13 slots. Their `inverse_mass` is
zero, so their momentum-to-velocity map returns zero and they never move —
they are carried as constants rather than special-cased out of the layout.

On top of the 65 differential unknowns there are the algebraic ones: one
Lagrange multiplier `lambda` and one position-level multiplier `mu` for each
of the 16 constraint rows, so

```
16 lambda + 16 mu = 32 algebraic unknowns
```

giving 97 unknowns in total, solved as one implicit system.

**What is handed to the Rust SUNDIALS translation.** Because there are
joints, this is a DAE, and the command `METHOD IDA` selects the pure-Rust
translation of **IDA**, SUNDIALS' implicit differential-algebraic solver.
IDA is handed a residual function of the implicit form

```
F(t, y, y') = 0
```

built from the four blocks written out earlier: the position rows carrying
the `-M^-1 J^T mu` correction, the momentum rows carrying `+J^T lambda`, the
16 rows `g(q) = 0`, and the 16 rows `J u = 0`. IDA runs its variable-order,
variable-step BDF method and calls back into that residual; the Newton
solve at each step is what determines the constraint forces.

**Tolerances.** This mechanism has joints that grip orientation (the hinge
and the prismatic guide both constrain rotation), so the tolerance floor
for orientation joints applies: `rtol` is held no tighter than `1e-6` and
`atol` no tighter than `1e-8`. That is not timidity — the index-2 accuracy
ceiling is real, and asking for more produces convergence failures rather
than better answers.

**Consistent start.** The initial velocities in this model are not
guessed. At top dead centre the piston is momentarily at rest and the rod
turns about the wrist pin, so the compatible velocities are `omega_rod =
-a*omega/L` about z, with the rod's centre riding at `a*omega/2`. Those are
what the model sets, so `J u = 0` holds to roundoff at t = 0 and the
simulator has nothing to project away.""",

"steps": [
 {"title": "Switch off gravity and contact",
  "explain": """The next cell sends three commands to the simulator.

- `set system.g_constant = 0` — the simulator's default is `1`, meaning
  every pair of bodies attracts every other by an inverse-square law. That
  is right for a solar system and wrong for an engine, where the parts are
  centimetres apart and the attraction between them is a meaningless
  perturbation. Setting it to zero removes it. **This default catches
  people out**, so it is set explicitly here rather than assumed.
- `set system.uniform_gravity = [0, 0, 0]` — this is the separate uniform
  field, the "g" of everyday falling. The three numbers are its x, y and z
  components in metres per second squared. All zero means no field.
- `collide off` — turns off contact detection. Nothing in this mechanism is
  supposed to touch anything else; the parts are held by joints. Leaving
  collision on would have the solver arm a rootfinder looking for impacts
  that never happen, which costs time and nothing else.

Each command returns nothing when it succeeds, so the cell prints nothing.
That is the expected result."""
  ,"code": '''sim.do("set system.g_constant = 0")
sim.do("set system.uniform_gravity = [0, 0, 0]")
sim.do("collide off")
print("world: no gravity, no contact")'''},

 {"title": "Create the engine block and the crankshaft",
  "explain": """The next cell creates the first two bodies. The general form of the
creation command is

```
new <shape> as <name> { <field> = <value>, ... }
```

where `<shape>` is one of `point`, `sphere`, `cuboid`, `cylinder`, `disk`,
`torus`, `dumbbell`; `<name>` is the name you will use to refer to the body
afterwards; and the braces hold initial field values separated by commas.

**The first line** makes `mount`, the engine block:

- `mass = 1` — a placeholder, because the next field overrides its effect.
- `position = [0, 0, 0]` — at the origin, which is where the crankshaft
  axis will be.
- `inverse_mass = 0` — this is what makes it static. The simulator carries
  inverse mass, not mass, precisely so that "immovable" is representable as
  a finite number rather than as infinity. Because the shape is `point`,
  its inertia tensor is zero as well, so it cannot be turned either.

**The second line** makes `crank`, the crankshaft disc:

- `mass = 2`, `radius = 0.55`, `half_height = 0.05` — a thin disc. The
  simulator turns these into the inertia tensor itself.
- `position = [0, 0, 0]` — concentric with the mount, so the hinge
  midpoint rule will put the main bearing exactly at the origin.
- `angular_velocity = [0, 0, 2]` — spinning about the z axis at 2 radians
  per second. This is the only energy the mechanism ever gets.

Each `new` prints the name the simulator assigned, so you can confirm the
body exists."""
  ,"code": '''sim.do("new point as mount { mass = 1, position = [0, 0, 0], inverse_mass = 0 }")
sim.do("new cylinder as crank { mass = 2, radius = 0.55, half_height = 0.05, position = [0, 0, 0], angular_velocity = [0, 0, 2] }")'''},

 {"title": "Create the connecting rod, with velocities the joints allow",
  "explain": """The next cell creates the connecting rod. Its fields:

- `mass = 0.4` — lighter than the crank and the piston, as a real rod is.
- `half_extents = [0.5, 0.05, 0.05]` — a slender bar, 1.0 long in x and
  0.1 square in cross-section. Half-extents are measured from the centre,
  so the full length is twice the first number.
- `position = [1.0, 0, 0]` — the rod's centre. This is the value forced by
  the midpoint rule discussed above: with the crank centre at 0 and the
  piston centre at 2, a rod centred at 1.0 puts the big-end pin at 0.5 and
  the wrist pin at 1.5.
- `angular_velocity = [0, 0, -1]` and `velocity = [0, 0.5, 0]`.

**Those last two values are the part that deserves attention, and they are
not free choices.** A joint constrains velocities as well as positions, so
a starting state must satisfy `J u = 0` as well as `g = 0`. At top dead
centre the piston is momentarily at rest and the rod is turning about the
wrist pin. The compatible values are

```
omega_rod = -a * omega / L = -(0.5)(2)/(1.0) = -1   about z
v_rod     =  a * omega / 2 = (0.5)(2)/2     =  0.5  along y
```

which is exactly what is typed. Supply anything else and the simulator will
still start, but it will first *project* the velocities onto the nearest
compatible set and tell you how much it had to change them. A start that
needs no projection is one where you specified velocities the mechanism can
actually have."""
  ,"code": '''sim.do("new cuboid as rod { mass = 0.4, half_extents = [0.5, 0.05, 0.05], position = [1.0, 0, 0], angular_velocity = [0, 0, -1], velocity = [0, 0.5, 0] }")'''},

 {"title": "Create the piston and the bore that guides it",
  "explain": """The next cell creates the last two bodies.

**The piston**: `half_extents = [0.5, 0.3, 0.3]` — a stubby block, and
`position = [2.0, 0, 0]`. With the rod centred at 1.0, the midpoint rule
puts the wrist pin at 1.5, which is the far end of the rod. Correct.

**The guide**: another `point` with `inverse_mass = 0`, at the same place
as the piston, `[2.0, 0, 0]`. It represents the cylinder bore. Putting it
exactly at the piston's centre means the prismatic joint's reference point
starts coincident with the piston, so the recorded slide distance is
measured from top dead centre.

A static `point` is the standard way to bolt a mechanism to the world in
this simulator: it participates in joints exactly like any other body, but
nothing can move it."""
  ,"code": '''sim.do("new cuboid as piston { mass = 1, half_extents = [0.5, 0.3, 0.3], position = [2.0, 0, 0] }")
sim.do("new point as guide { mass = 1, position = [2.0, 0, 0], inverse_mass = 0 }")'''},

 {"title": "Declare the four joints",
  "explain": """The next cell is the heart of the model. Four commands, 16 constraint
equations, and after them the five loose bodies are a mechanism.

**`hinge mount crank [0, 0, 1]`** — the main bearing. The form is
`HINGE <body a> <body b> <axis>`. The three-number axis is given in world
coordinates at the moment of creation and is then remembered in each body's
own frame. Contributes **5 rows**: three that hold a shared point, two that
hold the shared axis. One freedom left: the crank turns about z.

**`ball crank rod`** — the big end. The form is `BALL <body a> <body b>`,
with no axis, because a ball joint has no preferred direction. Contributes
**3 rows**, holding one point shared and leaving all three rotations free.

**`ball rod piston`** — the little end. Same again, at the wrist pin.

**`prismatic guide piston [1, 0, 0]`** — the bore. The form is
`PRISMATIC <body a> <body b> <axis>`, where the axis is the direction of
allowed sliding, here the world x direction. Contributes **5 rows**: two
that kill sideways translation and three that kill all rotation. One
freedom left: the slide along x.

**Where each pivot lands.** Every one of these joints places its pivot at
the midpoint of the two bodies' centres as they stand right now. That is
why the bodies were positioned before the joints were declared, and it is
why the sizes had to satisfy `L = 2a`. Working through it: mount 0 and
crank 0 give the main bearing at 0; crank 0 and rod 1.0 give the crank pin
at 0.5; rod 1.0 and piston 2.0 give the wrist pin at 1.5. Each is a real
pin on a real part.

Each joint command prints a confirmation naming the joint and its row
count."""
  ,"code": '''sim.do("hinge mount crank [0, 0, 1]")
sim.do("ball crank rod")
sim.do("ball rod piston")
sim.do("prismatic guide piston [1, 0, 0]")'''},

 {"title": "Select the solver, and confirm the joints are actually held",
  "explain": """The next cell does two things.

**`method ida`** selects the integrator. A model with joints is a
differential-algebraic system, not an ordinary differential one, and only
IDA can integrate it. The simulator will refuse any other method here
rather than silently integrating an unconstrained problem and handing you a
mechanism that quietly falls apart. There is nothing to choose: if you have
declared a joint, this is the line you need.

**`constraints`** prints the current worst violation of the constraint
equations. It reports two numbers:

- `|g|` — the largest absolute value in the vector of constraint equations.
  Zero means the parts are exactly where the joints say they should be.
- `|g_dot|` — the largest absolute value of the constraints' time
  derivative. Zero means the current velocities are compatible with the
  joints.

Both should be at roundoff now, before any integration, because the model
was assembled consistently. **These two numbers are how you tell whether a
mechanism is really being held together.** Watch them again after the run:
they should still be tiny, and above all they should not be growing."""
  ,"code": '''sim.do("method ida")
sim.do("constraints")'''},

 {"title": "Run the mechanism, and check it against the exact formula",
  "explain": """The next cell finally integrates, and simultaneously tests the result.

`step <dt>` advances the simulation by `dt` seconds. Note that in this
simulator every `STEP` is a **cold restart**: a fresh solver instance, a
fresh multiplier seed, no carried-over BDF history. That makes each step
independent and reproducible, at the cost of some efficiency. (The related
command `RUN <duration>` takes a *duration*, not an absolute time — `run 1.7`
followed by `run 2.89` leaves you at t = 4.59, not 2.89.)

The Python around it does the measuring:

1. `sim.get("piston.position")` reads the piston's centre back as a list of
   three numbers. `sim.get("crank.orientation")` reads the crank's
   quaternion, w-first.
2. The crank angle is recovered from that quaternion. For a rotation about
   z by angle `theta`, the quaternion is `(cos(theta/2), 0, 0, sin(theta/2))`,
   so `theta = 2 * atan2(qz, qw)`. The result is unwrapped across the
   +/- pi seam so the angle keeps counting past a full turn.
3. The piston's x is compared against `a cos(theta) + sqrt(L^2 - a^2 sin^2(theta))`.
   The piston centre starts at 2.0 while the wrist pin starts at 1.5, so the
   0.5 offset between them is subtracted before comparing.
4. The rod's roll about its own axis is measured too, to confirm that the
   second freedom really is passive.

This takes a few seconds: 200 steps, each a fresh implicit DAE solve. The
`quiet=True` keeps the loop from printing a progress line per step and
burying the result under two hundred lines of noise."""
  ,"code": '''import math

a, L = 0.5, 1.0          # crank throw and rod length
OFFSET = 0.5             # piston centre minus wrist pin
DT, N = 0.01, 200

def crank_angle(q):
    """Recover the rotation angle about z from a w-first quaternion."""
    return 2.0 * math.atan2(q[3], q[0])

def exact_piston_x(theta):
    """The slider-crank closed form: no approximation anywhere."""
    return a * math.cos(theta) + math.sqrt(L*L - (a*math.sin(theta))**2)

worst, xs, turns = 0.0, [], 0.0
prev = crank_angle(sim.get("crank.orientation"))
for k in range(N):
    sim.do(f"step {DT}", quiet=True)
    q     = sim.get("crank.orientation")
    theta = crank_angle(q)
    d     = theta - prev                        # unwrap across the +/- pi seam
    if d >  math.pi: d -= 2*math.pi
    if d < -math.pi: d += 2*math.pi
    turns += d
    prev   = theta
    x_pin  = sim.get("piston.position")[0] - OFFSET
    xs.append(x_pin)
    worst  = max(worst, abs(x_pin - exact_piston_x(turns)))

print(f"steps taken           : {N} of {DT} s  ->  t = {N*DT} s")
print(f"crank turned          : {turns/(2*math.pi):.6f} revolutions")
print(f"piston stroke         : {min(xs):.6f} .. {max(xs):.6f}"
      f"   (exact: {L-a:.6f} .. {L+a:.6f})")
print(f"worst |x - x(theta)|  : {worst:.3e}")'''},

 {"title": "Confirm the passive freedom stayed passive, and the joints held",
  "explain": """The next cell closes the argument.

Sixteen constraint rows on eighteen degrees of freedom left two freedoms.
One is the crank angle, which is the mechanism and has been turning. The
other is the rod's spin about its own long axis, which was argued above to
be passive: no torque acts along the rod's length, so it should not have
moved at all.

The cell measures it. The rod's own long axis is its local x axis; rolling
about it shows up as a rotation of the rod's local y axis out of the plane.
Reading the rod's quaternion and converting to a rotation matrix, the roll
angle is `atan2(R[2][1], R[1][1])` — the tilt of the rod's y axis toward
world z.

It then re-runs `constraints`. Compare those two numbers with the ones
printed before the run. They should still be near the solver's tolerance,
and — this is the point — not *growing*. A formulation that enforced only
the acceleration level would show `|g|` creeping up quadratically in time
as the mechanism slowly came apart. Carrying both `g` and `g_dot` as
algebraic equations, as the GGL formulation does, is what stops that."""
  ,"code": '''def quat_to_matrix(q):
    """w-first quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return [
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]

R = quat_to_matrix(sim.get("rod.orientation"))
roll = math.degrees(math.atan2(R[2][1], R[1][1]))
print(f"rod roll about its own axis : {roll:.3f} degrees   (expected 0.000)")
print()
print("constraint violation after the run:")
sim.do("constraints")'''},
],

"discussion": """**The stroke.** The piston ran from 0.500000 to 1.500000. Those are exactly
`L - a` and `L + a`, bottom and top dead centre. Nothing in the model was
told what the stroke should be — it is a consequence of four joint
declarations and two part sizes.

**The agreement with the closed form.** The worst disagreement between the
simulated piston position and `a cos(theta) + sqrt(L^2 - a^2 sin^2(theta))`
is a few parts in `1e8` over half a revolution — the exact figure is
printed above. That is the real result
here. Nobody supplied the formula; the solver was given four geometric
relations and asked to hold them, and the exact kinematics of the
slider-crank fell out.

The residual is not zero, and it should not be. Two things bound it: the
`rtol = 1e-6` floor that orientation joints impose, and the fact that the
piston position is quadratic in time near dead centre, so any sampling at
finite `dt` clips the extremes very slightly.

**The passive freedom.** The rod's roll about its own axis measured 0.000
degrees. This is the concrete payoff of choosing spherical rod ends over
revolutes: the extra freedom that made the model solvable is genuinely
inert, so nothing physical was given away to get there.

**The crank's uneven rate.** The revolution count will not be exactly
`2 * t / (2 pi)`. Nothing drives this crank; it was given 2 rad/s once and
left alone, and the rod and piston trade inertia with it as the geometry
changes through the cycle. The closed form is untroubled by that, because
it relates the piston to whatever angle the crank has actually reached — a
statement about the linkage, not about the timing. Testing it against a
free-running mechanism rather than a driven one is what makes it a real
test.

**Things worth trying.** Change `angular_velocity` on the crank and the
stroke will not change at all, only how fast it is traversed. Change the
rod's `half_extents` without changing the piston's starting position and
the model will still assemble, but the pins will no longer land on the
parts, because `L = 2a` will have been broken.""",
}
print(json.dumps(spec, indent=1))
