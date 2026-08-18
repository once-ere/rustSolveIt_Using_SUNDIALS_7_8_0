"""What every element of the simulator's command language means.

Used to write, for each command a notebook is about to send, an exact
statement of what that input line expects. Requirement 3 asks for a
comment before every user entry; this is where those comments come from.
"""

# ---------------------------------------------------------------- shapes
SHAPES = {
 "point": dict(
   args=[], blurb="a massive point: it has mass but no size and no inertia",
   inertia="`I = 0`. A point cannot store rotational energy, so no torque turns it.",
   why="Used for anchors, for planets whose size does not matter, and for "
       "charges. A `point` with `inverse_mass = 0` is the standard way to "
       "bolt a mechanism to the world."),
 "sphere": dict(
   args=["radius"], blurb="a uniform solid ball",
   inertia="`I = diag(2mr^2/5, 2mr^2/5, 2mr^2/5)` — isotropic, so every axis "
           "through the centre is a principal axis with the same moment.",
   why="The only shape whose inertia tensor is unchanged by any rotation, "
       "which makes it the simplest collidable body."),
 "cuboid": dict(
   args=["half_extents"], blurb="a uniform rectangular block",
   inertia="`I = (m/3) * diag(b^2+c^2, a^2+c^2, a^2+b^2)` for half-extents "
           "`(a, b, c)`. Three distinct moments in general, which is what "
           "makes a cuboid able to tumble.",
   why="Half-extents are measured from the centre, so the full side lengths "
       "are twice these numbers."),
 "cylinder": dict(
   args=["radius", "half_height"], blurb="a uniform solid cylinder",
   inertia="`I_zz = m r^2 / 2` about the axis of symmetry, and "
           "`I_xx = I_yy = m (3 r^2 + 4 h^2) / 12` about any diameter.",
   why="Symmetric about its LOCAL z axis, like every round shape here. A "
       "long thin cylinder is a rod; a short fat one is a disc."),
 "disk": dict(
   args=["radius"], blurb="an infinitely thin uniform disc",
   inertia="`I_zz = m r^2 / 2` about its axis, `I_xx = I_yy = m r^2 / 4` "
           "about any diameter.",
   why="Symmetric about its LOCAL z axis. Thin enough that face-on "
       "disc-disc contact is invisible to the rootfinder — tilt one, or use "
       "a thin cylinder, if the discs must collide face to face."),
 "torus": dict(
   args=["ring_radius", "tube_radius"], blurb="a uniform ring (a doughnut)",
   inertia="`I_zz = m (c^2 + 3a^2/4)` about the axis of revolution and "
           "`I_xx = I_yy = m (c^2/2 + 5a^2/8)` about any diameter, for ring "
           "radius `c` and tube radius `a`.",
   why="Symmetric about its LOCAL z axis, which is the axis of revolution — "
       "NOT a diameter. Getting those two confused makes a ring that spins "
       "invisibly instead of swinging. May also be given as "
       "`inner_radius`/`outer_radius`, which is the same ring described from "
       "the hole outward."),
 "dumbbell": dict(
   args=["m1", "m2", "r1", "r2", "z1", "z2", "rod_radius", "m_rod"],
   blurb="two spheres rigidly joined by a rod, treated as ONE rigid body",
   inertia="the parallel-axis sum of the two end spheres and the rod about "
           "their common centre of mass.",
   why="Not two bodies with a constraint: one body with a composite inertia "
       "tensor and a two-lobed collision shape. Its centre of mass is the "
       "mass-weighted mean of the parts, so the body's `position` is that "
       "centre, not the midpoint of the rod."),
}

# ---------------------------------------------------------------- fields
FIELDS = {
 "mass": "the body's mass, in kilograms. Setting it also sets `inverse_mass` "
         "to its reciprocal — the two are kept consistent, and the simulator "
         "carries the inverse because that is what the equations of motion "
         "actually use.",
 "inverse_mass": "one over the mass. Setting it to **0** makes the body "
                 "STATIC: no force can accelerate it. This is how a body is "
                 "bolted to the world, and it is representable as a finite "
                 "number, which infinite mass is not.",
 "position": "the centre of mass, as `[x, y, z]` in metres.",
 "velocity": "the centre-of-mass velocity, as `[vx, vy, vz]` in metres per "
             "second. Setting it sets the linear momentum to `m * v`; the "
             "momentum is what the state vector actually carries.",
 "momentum": "the linear momentum `p = m v`, as `[px, py, pz]`. This is the "
             "quantity the integrator carries, because it is conserved when "
             "no external force acts.",
 "orientation": "the attitude, as a unit quaternion `[w, x, y, z]` — **w "
                "first**, everywhere in this simulator. Quaternions are used "
                "rather than Euler angles because they have no gimbal lock "
                "and renormalise cheaply. Setting it renormalises.",
 "angular_velocity": "the spin rate `omega`, as `[wx, wy, wz]` in radians "
                     "per second, in WORLD axes. Setting it sets the angular "
                     "momentum to `L = I omega`, which is what is carried.",
 "angular_momentum": "the angular momentum `L = I omega`, as `[Lx, Ly, Lz]` "
                     "in world axes. This is the carried quantity, and it is "
                     "the one that is conserved when no torque acts — "
                     "`omega` is not.",
 "inertia_tensor": "the 3x3 inertia tensor in the body's own frame, given "
                   "row by row. Normally you do not set this: the simulator "
                   "computes it from the shape. Setting it explicitly is for "
                   "modelling a body whose mass distribution is not the "
                   "shape's.",
 "inverse_inertia_tensor": "the inverse of the inertia tensor. Setting a "
                           "row to zero makes the body infinitely resistant "
                           "to torque about that axis.",
 "radius": "the radius, in metres.",
 "half_extents": "half the side lengths, as `[a, b, c]`, measured from the "
                 "centre. The full block is `2a` by `2b` by `2c`.",
 "half_height": "half the height along the cylinder's own z axis.",
 "height": "the full height along the cylinder's own z axis.",
 "length": "the full length.",
 "ring_radius": "the distance from the torus centre to the middle of its tube.",
 "tube_radius": "the radius of the torus tube itself.",
 "inner_radius": "the radius of the hole. With `outer_radius`, an equivalent "
                 "way to describe the same ring. Zero is legal, and gives a "
                 "horn torus.",
 "outer_radius": "the overall radius of the ring, hole plus both tube walls.",
 "restitution": "the coefficient of restitution `e`, between 0 and 1. It is "
                "the ratio of separating to approaching speed along the "
                "contact normal: `e = 1` is perfectly elastic and conserves "
                "kinetic energy, `e = 0` is perfectly inelastic and the "
                "bodies leave the contact with no relative normal speed. A "
                "pair's effective value is taken from the two bodies.",
 "charge": "the electric charge, in coulombs, which is what the electric "
           "and magnetic fields grip.",
 "magnetic_moment_tensor": "a 3x3 tensor giving the body's magnetic moment. "
                           "Watch the columns: a field along z can only grip "
                           "the tensor's THIRD column, so a tensor whose "
                           "third column is zero produces exactly no torque "
                           "however antisymmetric it looks.",
 "torque": "a constant applied torque, as `[Tx, Ty, Tz]`.",
 "gravity": "a per-body gravitational parameter.",
 "m1": "the mass of the dumbbell's first end sphere.",
 "m2": "the mass of the dumbbell's second end sphere.",
 "r1": "the radius of the dumbbell's first end sphere.",
 "r2": "the radius of the dumbbell's second end sphere.",
 "z1": "where the first end sphere sits along the dumbbell's local z axis.",
 "z2": "where the second end sphere sits along the dumbbell's local z axis.",
 "rod_radius": "the radius of the dumbbell's connecting rod.",
 "m_rod": "the mass of the dumbbell's connecting rod. Zero gives a massless rod.",
}

SYSTEM_FIELDS = {
 "g_constant": ("the gravitational constant `G` used for the mutual "
   "attraction between every pair of bodies, `F = G m1 m2 / r^2`. **Its "
   "default is 1, not 0**, which catches people out: bodies rattling in a "
   "box also attract each other unless you say otherwise. Set it to 0 for a "
   "mechanism or a collision experiment; set it to a real value for an "
   "orbit."),
 "uniform_gravity": ("a uniform gravitational field applied to every body, "
   "as `[gx, gy, gz]` in metres per second squared. This is the everyday "
   "'g' of falling, entirely separate from the mutual attraction above."),
 "softening": ("a length that softens the mutual gravity at short range, "
   "replacing `1/r^2` with `1/(r^2 + s^2)`. It stops the force blowing up "
   "when two bodies pass very close. Too small a value is nearly singular "
   "when surfaces touch."),
 "rtol": ("the integrator's relative error tolerance. Smaller is more "
   "accurate and slower. Joints that grip orientation impose a floor of "
   "`1e-6`, because the index-2 accuracy ceiling is real and asking for "
   "more produces convergence failures rather than better answers."),
 "atol": ("the integrator's absolute error tolerance, which is what governs "
   "components passing near zero, where a relative tolerance means nothing. "
   "Orientation joints impose a floor of `1e-8`."),
 "collisions": ("whether contact detection is armed. The `COLLIDE` command "
   "is the usual way to set it."),
 "b_field": ("a uniform magnetic field `[Bx, By, Bz]` in tesla, which acts "
   "on any body with a charge through the Lorentz force `F = q v x B`."),
 "e_field": ("a uniform electric field `[Ex, Ey, Ez]`, which acts on any "
   "body with a charge through `F = q E`."),
 "box": ("the rigid box enclosing the scene. Set with the `BOX` command."),
}

# ---------------------------------------------------------------- joints
JOINTS = {
 "constrain": dict(rows=1, form="CONSTRAIN <a> <b> [length]",
   holds="the distance between the two centres is fixed",
   detail="With no length given it freezes whatever separation the two "
          "bodies already have, so you position them and then tie them. The "
          "constraint equation is `g = |x_b - x_a| - L`, written on the "
          "distance and not on its square: the squared form is tidier "
          "algebraically and worse numerically, because its gradient scales "
          "with `L` and the corrector stops converging for `L` far from 1.",
   frees=5),
 "ball": dict(rows=3, form="BALL <a> <b>",
   holds="the two bodies share one point, and may turn any way about it",
   detail="Three equations, one per coordinate of the shared point: "
          "`g = (x_a + R_a r_a) - (x_b + R_b r_b) = 0`, where `r_a` and "
          "`r_b` locate the pivot in each body's own frame. No axis is "
          "given, because a ball joint has no preferred direction.",
   frees=3),
 "hinge": dict(rows=5, form="HINGE <a> <b> <axis>",
   holds="the two bodies share a point AND an axis: one freedom left, the swing",
   detail="Three equations hold the shared point, exactly as a ball joint "
          "does. Two more hold the axis: taking `p` and `q` perpendicular to "
          "the hinge axis `h`, the rows are `p . (R_a h) = 0` and "
          "`q . (R_a h) = 0`, which say the axis carried by one body has not "
          "tipped away from the axis carried by the other.",
   frees=1),
 "universal": dict(rows=4, form="UNIVERSAL <a> <b> <axis u> <axis w>",
   holds="a shared point and a right angle between two carried axes",
   detail="Three equations hold the shared point. The fourth holds the "
          "cross-pin square: `g = (R_a u) . (R_b w) = 0`, which says the two "
          "arms of the cross stay perpendicular. That single row is what "
          "makes a Cardan joint transmit rotation through a bend.",
   frees=2),
 "gear": dict(rows=1, form="GEAR <a> <b> <axis> <ratio>",
   holds="two rotations stay in a fixed proportion",
   detail="One equation, `g = sin(q*theta_a + p*theta_b) = 0`, where the "
          "ratio has been written as the exact fraction `p/q`. The sine is "
          "there so the row is smooth across the angle wrap; that is also "
          "why the ratio must be rational, and the command refuses a ratio "
          "it cannot represent as a fraction rather than silently rounding.",
   frees=5),
 "rack": dict(rows=1, form="RACK <pinion> <rack> <axis> <direction> <radius>",
   holds="turning is tied to sliding: `ds = r d(theta)`",
   detail="One equation relating the rack's travel along its direction to "
          "the pinion's turn about its axis. The pinion angle is unwrapped "
          "from the unbounded travel — `k = round((ds/r - theta)/2pi)` — so "
          "the rack may run for many turns without the row losing track of "
          "which revolution it is on.",
   frees=5),
 "prismatic": dict(rows=5, form="PRISMATIC <a> <b> <axis>",
   holds="the bodies may slide along one line and do nothing else",
   detail="Two equations kill translation across the slide direction `n`, "
          "and three kill relative rotation entirely. It is the guide a rack "
          "runs in, and the bore a piston runs in.",
   frees=1),
}

# ---------------------------------------------------------------- methods
METHODS = {
 "adams": ("the pure-Rust translation of **CVODE** in Adams-Moulton mode: a "
   "variable-order, variable-step explicit-family method for non-stiff "
   "problems. This is the default, and it is the right choice for orbits "
   "and for anything smooth."),
 "bdf": ("the pure-Rust translation of **CVODE** in BDF mode: backward "
   "differentiation formulas, implicit, for stiff problems. A stiff problem "
   "is one with widely separated timescales — a fast gyration riding on a "
   "slow drift, for instance — where an explicit method is forced into "
   "absurdly small steps by stability rather than by accuracy."),
 "sprk": ("the pure-Rust translation of **ARKODE**'s symplectic partitioned "
   "Runge-Kutta method. A symplectic method preserves the geometric "
   "structure of Hamiltonian mechanics, so energy error stays BOUNDED and "
   "oscillates rather than drifting away — which matters over very long "
   "integrations even when a general-purpose method is locally more "
   "accurate."),
 "ida": ("the pure-Rust translation of **IDA**, SUNDIALS' implicit "
   "differential-algebraic solver. Any model with a joint MUST use it, and "
   "the simulator refuses every other method for such a model rather than "
   "silently integrating an unconstrained problem."),
}
