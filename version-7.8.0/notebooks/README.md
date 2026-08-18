# notebooks/ — one Jupyter notebook per example, 109 of them

Every example in this repository has exactly one Jupyter notebook, and
every notebook here pairs with exactly one example:

| prefix | count | paired with | example form |
|---|---|---|---|
| `video_*` | 13 | `videos/scenes/*.posim` | the scripts behind the recorded browser videos |
| `rust_*` | 6 | `physical_object/examples/*.rs` | the self-checking compiled Rust examples |
| `collision_*` | 12 | `scripts/collisions/*.posim` | the collision-detection walkthroughs |
| `solveit_*` | 19 | `scripts/solveit/*.posim` | the SolveIt worked examples |
| `dynamic_*` | 59 | `dynamic_notebooks/*.posim` | the dynamic notebooks, incl. the 34 Routh problems |

Each notebook is a **Python 3** notebook that starts the simulator as a
child process in machine mode (`posim --machine`) and drives it over
JSON Lines — the same wire protocol the `jupyter/` kernel uses. Each is
completely stand-alone: the launch instructions, the command-language
glossary, the second-order-to-first-order reduction, the physical
situation (objects, properties, interactions, equations of motion,
constraint equations, state-vector sizing), an explanation before every
code cell, and the name-and-save cells are all written out in full in
every notebook, on purpose. No notebook ever refers to another.

The committed notebooks carry **real outputs**: every code cell was
executed against the release build, and the `rust_*` six also run their
compiled example and assert its `SUCCESS` verdict.

## Running one

```bash
cargo build --release -p posim        # once
python3 -m pip install --user jupyterlab
jupyter lab notebooks/
```

Open any notebook and run the cells top to bottom. The final two cells
ask you to name the notebook and pick a folder in a save dialog; they
are interactive and are the only cells the batch runner skips.

## The 109, one by one

### Video scenes — 13

Paired with the scripts behind the recorded browser videos in `videos/`.

| notebook | pairs with | subject |
|---|---|---|
| `video_ball_joint_chain.ipynb` | `videos/scenes/ball_joint_chain.posim` | A four-link chain hung on BALL joints, whirling as it collapses |
| `video_box_of_shapes.ipynb` | `videos/scenes/box_of_shapes.posim` | The mixed-shape box rattle, exactly as the regression test |
| `video_cardan_compass.ipynb` | `videos/scenes/cardan_compass.posim` | A compass in a Cardan suspension, rocking back towards level |
| `video_cardan_gear.ipynb` | `videos/scenes/cardan_gear.posim` | Cardan gears: a wheel of radius r inside a ring of radius 2r, and the |
| `video_double_pendulum_hinges.ipynb` | `videos/scenes/double_pendulum_hinges.posim` | A double pendulum built out of two HINGES |
| `video_gyroscope_gimbal.ipynb` | `videos/scenes/gyroscope_gimbal.posim` | A gyroscope slung in a two-ring gimbal |
| `video_kepler_ellipse.ipynb` | `videos/scenes/kepler_ellipse.posim` | A single planet on a strongly eccentric Kepler orbit (e = 0.6) |
| `video_piston_crankshaft.ipynb` | `videos/scenes/piston_crankshaft.posim` | A piston driven by a crankshaft (the slider-crank) |
| `video_rack_and_pinion.ipynb` | `videos/scenes/rack_and_pinion.posim` | A rack and pinion drive: a falling weight turning a flywheel |
| `video_rod_pendulum_chain.ipynb` | `videos/scenes/rod_pendulum_chain.posim` | Four bobs on four rods, released from rest and left to go chaotic |
| `video_spinning_top.ipynb` | `videos/scenes/spinning_top.posim` | A spinning top, held at its tip and precessing under gravity |
| `video_tumbling_racket.ipynb` | `videos/scenes/tumbling_racket.posim` | The tennis-racket theorem (Dzhanibekov effect): a torque-free body |
| `video_universal_joint.ipynb` | `videos/scenes/universal_joint.posim` | A universal (Cardan) joint transmitting rotation between two shafts |

### Compiled Rust examples — 6

Paired with the self-checking programs in `physical_object/examples/`; each of these notebooks reproduces the same physics through the command language AND runs the compiled example itself, asserting its SUCCESS verdict.

| notebook | pairs with | subject |
|---|---|---|
| `rust_bouncing_ball_restitution.ipynb` | `physical_object/examples/bouncing_ball_restitution.rs` | A ball bouncing at the exact time of impact |
| `rust_charged_in_b_field.ipynb` | `physical_object/examples/charged_in_b_field.rs` | A charged sphere gyrating in a magnetic field |
| `rust_kepler_orbit.ipynb` | `physical_object/examples/kepler_orbit.rs` | A Kepler orbit conserving the Laplace-Runge-Lenz vector |
| `rust_newtons_cradle.ipynb` | `physical_object/examples/newtons_cradle.rs` | Newton's cradle resolved by pairwise impulses |
| `rust_outer_solar_system.ipynb` | `physical_object/examples/outer_solar_system.rs` | The outer solar system over 1,370 years |
| `rust_tumbling_body.ipynb` | `physical_object/examples/tumbling_body.rs` | The Dzhanibekov effect: tumbling about the intermediate axis |

### Collision walkthroughs — 12

Paired with the scripts in `scripts/collisions/`.

| notebook | pairs with | subject |
|---|---|---|
| `collision_01_head_on_exchange.ipynb` | `scripts/collisions/01_head_on_exchange.posim` | Equal masses, head on, perfectly elastic (e = 1) |
| `collision_02_unequal_masses.ipynb` | `scripts/collisions/02_unequal_masses.posim` | Unequal masses, 1-D elastic textbook formulas |
| `collision_03_restitution_ladder.ipynb` | `scripts/collisions/03_restitution_ladder.posim` | The restitution ladder: the SAME approach (closing speed |
| `collision_04_newtons_cradle.ipynb` | `scripts/collisions/04_newtons_cradle.posim` | Newton's cradle with five spheres: four touching balls, |
| `collision_05_billiard_break.ipynb` | `scripts/collisions/05_billiard_break.posim` | A billiard break: the cue ball strikes the apex of a |
| `collision_06_spin_up.ipynb` | `scripts/collisions/06_spin_up.posim` | Linear momentum becomes spin: a sphere strikes a free |
| `collision_07_thin_wall_toi.ipynb` | `scripts/collisions/07_thin_wall_toi.posim` | The precision differentiator: a small fast bullet |
| `collision_08_colliding_binary.ipynb` | `scripts/collisions/08_colliding_binary.posim` | Gravity AND collisions together: two spheres on a bound |
| `collision_09_spinning_target.ipynb` | `scripts/collisions/09_spinning_target.posim` | Hitting a TUMBLING target: the cuboid spins at |
| `collision_10_billiard_box.ipynb` | `scripts/collisions/10_billiard_box.posim` | The long game: an elastic (e = 1) ball ping-pongs |
| `collision_11_box_of_shapes.ipynb` | `scripts/collisions/11_box_of_shapes.posim` | The box of shapes: every body type inside a rigid, |
| `collision_12_two_dumbbells.ipynb` | `scripts/collisions/12_two_dumbbells.posim` | Two colliding dumbbells built by a USER-DEFINED function |

### SolveIt worked examples — 19

Paired with the scripts in `scripts/solveit/`.

| notebook | pairs with | subject |
|---|---|---|
| `solveit_01_elastic_head_on.ipynb` | `scripts/solveit/01_elastic_head_on.posim` | Unequal masses in a head-on elastic collision |
| `solveit_02_keplers_third_law.ipynb` | `scripts/solveit/02_keplers_third_law.posim` | Kepler's third law: T^2 / a^3 is one number for every orbit |
| `solveit_03_three_conics.ipynb` | `scripts/solveit/03_three_conics.posim` | Bound, parabolic and hyperbolic, decided by the sign of E |
| `solveit_04_restitution_ladder.ipynb` | `scripts/solveit/04_restitution_ladder.posim` | A ball with restitution e reaches e^(2n) of its drop height on |
| `solveit_05_cyclotron_bdf.ipynb` | `scripts/solveit/05_cyclotron_bdf.posim` | A charged sphere in a uniform B field |
| `solveit_06_symplectic_vs_adaptive.ipynb` | `scripts/solveit/06_symplectic_vs_adaptive.posim` | Symplectic (SPRK) versus adaptive multistep (Adams) over a long |
| `solveit_07_dzhanibekov.ipynb` | `scripts/solveit/07_dzhanibekov.posim` | The intermediate-axis (tennis racket) instability, measured |
| `solveit_08_magnetic_torque.ipynb` | `scripts/solveit/08_magnetic_torque.posim` | A magnetised body in a field. The torque is tau = (R M R^T) B, so |
| `solveit_09_newtons_cradle.ipynb` | `scripts/solveit/09_newtons_cradle.posim` | Newton's cradle. One sphere hits four touching ones. The impulse |
| `solveit_10_no_tunnelling.ipynb` | `scripts/solveit/10_no_tunnelling.posim` | A bullet at 100 m/s meets a 5 mm plate. In one output interval of |
| `solveit_11_lagrange_l4.ipynb` | `scripts/solveit/11_lagrange_l4.posim` | Lagrange's equilateral solution. Three EQUAL masses at the |
| `solveit_12_dumbbell_inertia.ipynb` | `scripts/solveit/12_dumbbell_inertia.posim` | A user-defined constructor, and the inertia tensor it produces |
| `solveit_13_tilted_torus.ipynb` | `scripts/solveit/13_tilted_torus.posim` | Geometry that only a real support function gets right. An |
| `solveit_14_particle_in_a_box.ipynb` | `scripts/solveit/14_particle_in_a_box.posim` | The infinite square well, solved inside the language |
| `solveit_15_tunnelling.ipynb` | `scripts/solveit/15_tunnelling.posim` | Tunnelling through a square barrier, checked against the textbook |
| `solveit_16_special_functions.ipynb` | `scripts/solveit/16_special_functions.posim` | The special-function library, reached through ordinary calls |
| `solveit_17_pendulum_dae.ipynb` | `scripts/solveit/17_pendulum_dae.posim` | A pendulum as a CONSTRAINT, not a force |
| `solveit_18_equilibrium_and_sensitivity.ipynb` | `scripts/solveit/18_equilibrium_and_sensitivity.posim` | Two questions that are not "what happens next" |
| `solveit_19_hinged_door.ipynb` | `scripts/solveit/19_hinged_door.posim` | A door on a hinge: an ORIENTATION constraint |

### Dynamic notebooks — 59

Paired with the scripts in `dynamic_notebooks/`, including the 34 Routh problems.

| notebook | pairs with | subject |
|---|---|---|
| `dynamic_billiard_box.ipynb` | `dynamic_notebooks/billiard_box.posim` | An elastic ball ping-pongs between two static walls forever |
| `dynamic_billiard_break.ipynb` | `dynamic_notebooks/billiard_break.posim` | A tiny billiard break: cue into a three-ball rack |
| `dynamic_bouncing_ball_restitution.ipynb` | `dynamic_notebooks/bouncing_ball_restitution.posim` | A bouncing ball with e = 0.8: apex = e^2 h |
| `dynamic_box_of_shapes.ipynb` | `dynamic_notebooks/box_of_shapes.posim` | The manager's box of shapes: every body type in a rigid BOX 4 |
| `dynamic_box_of_shapes_m32.ipynb` | `dynamic_notebooks/box_of_shapes_m32.posim` | The box of shapes with a HEAVY point (mass 32) |
| `dynamic_charged_in_b_field.ipynb` | `dynamic_notebooks/charged_in_b_field.posim` | Cyclotron motion: a charge circling in a uniform magnetic field |
| `dynamic_charged_in_e_field.ipynb` | `dynamic_notebooks/charged_in_e_field.posim` | A charge in a uniform electric field: x(t) = (qE/2m) t^2, exact |
| `dynamic_colliding_binary.ipynb` | `dynamic_notebooks/colliding_binary.posim` | Gravity + collisions: a bound binary that bounces at pericenter (e_rest = 0.6) |
| `dynamic_double_slit.ipynb` | `dynamic_notebooks/double_slit.posim` | --------------------------------------------------------------------- |
| `dynamic_head_on_exchange.ipynb` | `dynamic_notebooks/head_on_exchange.posim` | Equal masses head-on: velocities exchange exactly |
| `dynamic_kepler_orbit.ipynb` | `dynamic_notebooks/kepler_orbit.posim` | An eccentric Kepler orbit (e = 0.6) and the Runge-Lenz compass |
| `dynamic_magnetic_spin_up.ipynb` | `dynamic_notebooks/magnetic_spin_up.posim` | Magnetic torque spins a body up — and honestly grows ENERGY |
| `dynamic_newtons_cradle.ipynb` | `dynamic_notebooks/newtons_cradle.posim` | Newton's cradle: five spheres, one incomer, only the far ball exits |
| `dynamic_outer_solar_system.ipynb` | `dynamic_notebooks/outer_solar_system.posim` | The outer solar system |
| `dynamic_restitution_ladder.ipynb` | `dynamic_notebooks/restitution_ladder.posim` | The restitution ladder: e = 1.0 / 0.8 / 0.5 / 0.2 side by side |
| `dynamic_routh_double_star_period.ipynb` | `dynamic_notebooks/routh_double_star_period.posim` | Routh I.Art.400 — a double star's period depends only on the SUM of the masses |
| `dynamic_routh_p1_apsidal_symmetry.ipynb` | `dynamic_notebooks/routh_p1_apsidal_symmetry.posim` | Routh I.Arts.419-420 — an apsidal radius divides the orbit symmetrically, |
| `dynamic_routh_p1_centre_of_gravity.ipynb` | `dynamic_notebooks/routh_p1_centre_of_gravity.posim` | Routh I.Art.92 — impacts cannot move the centre of gravity |
| `dynamic_routh_p1_collinear_three_body.ipynb` | `dynamic_notebooks/routh_p1_collinear_three_body.posim` | Routh I.Arts.409-412 — the collinear three-body configuration, and its |
| `dynamic_routh_p1_equal_periods.ipynb` | `dynamic_notebooks/routh_p1_equal_periods.posim` | Routh I.Art.335 — the SPEED alone fixes the orbit's size, so four differently |
| `dynamic_routh_p1_equilateral_three_body.ipynb` | `dynamic_notebooks/routh_p1_equilateral_three_body.posim` | Routh I.Arts.407-408 + 412 — Lagrange's equilateral three-body solution, and |
| `dynamic_routh_p1_escape_velocity.ipynb` | `dynamic_notebooks/routh_p1_escape_velocity.posim` | Routh I.Art.312 + 335 — the velocity from infinity, and the three conics |
| `dynamic_routh_p1_expanding_sphere.ipynb` | `dynamic_notebooks/routh_p1_expanding_sphere.posim` | Routh I.Art.167 Ex.1 — equal speeds in all directions give an expanding |
| `dynamic_routh_p1_geometric_progression.ipynb` | `dynamic_notebooks/routh_p1_geometric_progression.posim` | Routh I.Art.88 — masses in geometric progression give velocities in |
| `dynamic_routh_p1_hodograph_circle.ipynb` | `dynamic_notebooks/routh_p1_hodograph_circle.posim` | Routh I.Art.394-398 — the hodograph of a Kepler orbit is a CIRCLE |
| `dynamic_routh_p1_kepler_equation.ipynb` | `dynamic_notebooks/routh_p1_kepler_equation.posim` | Routh I.Arts.342-346 — Kepler's equation and the equation of the centre |
| `dynamic_routh_p1_lambert_theorem.ipynb` | `dynamic_notebooks/routh_p1_lambert_theorem.posim` | Routh I.Arts.350-355 — Lambert's theorem: the time depends only on the chord, |
| `dynamic_routh_p1_oblique_impact.ipynb` | `dynamic_notebooks/routh_p1_oblique_impact.posim` | Routh I.Art.89 — oblique impact of equal smooth spheres: the velocities come |
| `dynamic_routh_p1_parabola_of_safety.ipynb` | `dynamic_notebooks/routh_p1_parabola_of_safety.posim` | Routh I.Arts.159-160 — the parabola of safety, and the two ways to hit a |
| `dynamic_routh_p1_sphere_exchange.ipynb` | `dynamic_notebooks/routh_p1_sphere_exchange.posim` | Routh I.Art.85 + 87 — equal elastic spheres EXCHANGE velocities, and a |
| `dynamic_routh_p1_three_projectiles.ipynb` | `dynamic_notebooks/routh_p1_three_projectiles.posim` | Routh I.Art.158 Ex.1 — three projectiles stay in a plane parallel to itself |
| `dynamic_routh_p1_two_trajectories.ipynb` | `dynamic_notebooks/routh_p1_two_trajectories.posim` | Routh I.Art.339 — TWO trajectories reach the same point at the same speed |
| `dynamic_routh_p2_correlated_bodies.ipynb` | `dynamic_notebooks/routh_p2_correlated_bodies.posim` | Routh II.Arts.192-195 — correlated bodies: confocal ellipsoids of gyration |
| `dynamic_routh_p2_fixed_couple.ipynb` | `dynamic_notebooks/routh_p2_fixed_couple.posim` | Routh II.Art.148 — a couple whose axis is fixed in space |
| `dynamic_routh_p2_impulsive_couple.ipynb` | `dynamic_notebooks/routh_p2_impulsive_couple.posim` | Routh II.Art.146 — an impulsive couple, and what it does to the invariable |
| `dynamic_routh_p2_invariable_line.ipynb` | `dynamic_notebooks/routh_p2_invariable_line.posim` | Routh II.Art.141 — the invariable line is absolutely fixed in space, and the |
| `dynamic_routh_p2_mean_axis_instability.ipynb` | `dynamic_notebooks/routh_p2_mean_axis_instability.posim` | Routh II.Art.155 — the three principal axes do NOT possess equal degrees of |
| `dynamic_routh_p2_period_ratio.ipynb` | `dynamic_notebooks/routh_p2_period_ratio.posim` | Routh II.Art.150b Ex.2 — the ratio of the two periods is independent of the |
| `dynamic_routh_p2_poinsot_rolling.ipynb` | `dynamic_notebooks/routh_p2_poinsot_rolling.posim` | Routh II.Art.143 — Poinsot's construction: the momental ellipsoid rolls on a |
| `dynamic_routh_p2_polhode_quarter_period.ipynb` | `dynamic_notebooks/routh_p2_polhode_quarter_period.posim` | Routh II.Art.150a — the quarter period of the polhode, as an elliptic |
| `dynamic_routh_p2_principal_axes_in_space.ipynb` | `dynamic_notebooks/routh_p2_principal_axes_in_space.posim` | Routh II.Arts.176-179 — the principal axes referred to the invariable line |
| `dynamic_routh_p2_rolling_cones.ipynb` | `dynamic_notebooks/routh_p2_rolling_cones.posim` | Routh II.Arts.157-159 — the invariable and instantaneous cones, rolling on |
| `dynamic_routh_p2_separatrix.ipynb` | `dynamic_notebooks/routh_p2_separatrix.posim` | Routh II.Arts.184-185 — the separatrix G^2 = B*T, and which axis the polhode |
| `dynamic_routh_p2_spin_stabilisation.ipynb` | `dynamic_notebooks/routh_p2_spin_stabilisation.posim` | Routh II.Art.156 — why rapid rotation steadies a body |
| `dynamic_routh_p2_sylvester_time.ipynb` | `dynamic_notebooks/routh_p2_sylvester_time.posim` | Routh II.Arts.196-198 — Sylvester's and Poinsot's measures of the time |
| `dynamic_routh_p2_thin_rod.ipynb` | `dynamic_notebooks/routh_p2_thin_rod.posim` | Routh II.Art.144 — a thin rod, whose momental ellipsoid is a circular |
| `dynamic_routh_p2_two_quadrics.ipynb` | `dynamic_notebooks/routh_p2_two_quadrics.posim` | Routh II.Arts.140-142 — the two quadrics whose intersection IS the polhode |
| `dynamic_routh_p2_uniaxal_precession.ipynb` | `dynamic_notebooks/routh_p2_uniaxal_precession.posim` | Routh II.Arts.180-183 — motion when A = B: steady precession with two |
| `dynamic_routh_rectangle_diagonal.ipynb` | `dynamic_notebooks/routh_rectangle_diagonal.posim` | Routh II.Art.150b Ex.1 — a rectangle spun about one diagonal ends up spinning |
| `dynamic_spin_up.ipynb` | `dynamic_notebooks/spin_up.posim` | Off-center impact spins up a box: L = r x J |
| `dynamic_spinning_target.ipynb` | `dynamic_notebooks/spinning_target.posim` | Hitting a tumbling target: the contact point itself is moving |
| `dynamic_static_anchor.ipynb` | `dynamic_notebooks/static_anchor.posim` | Inverse_mass = 0: a pinned anchor ignores gravity while a test body falls |
| `dynamic_thin_wall_toi.ipynb` | `dynamic_notebooks/thin_wall_toi.posim` | No tunneling: a 100 m/s bullet meets a 1 cm plate at the exact TOI |
| `dynamic_three_bodies.ipynb` | `dynamic_notebooks/three_bodies.posim` | Three gravitating bodies with zero total momentum |
| `dynamic_thrown_ball.ipynb` | `dynamic_notebooks/thrown_ball.posim` | A thrown ball vs the textbook parabola |
| `dynamic_tumbling_body.ipynb` | `dynamic_notebooks/tumbling_body.posim` | The tennis-racket theorem: unstable intermediate-axis tumbling |
| `dynamic_tunneling.ipynb` | `dynamic_notebooks/tunneling.posim` | --------------------------------------------------------------------- |
| `dynamic_two_dumbbells.ipynb` | `dynamic_notebooks/two_dumbbells.posim` | Two dumbbells from a user-defined function: E, P and L all survive the impact |
| `dynamic_unequal_masses.ipynb` | `dynamic_notebooks/unequal_masses.posim` | Unequal masses: 1-vs-3 head-on, textbook one-dimensional formulas |


## Regenerating and re-verifying all 109

```bash
cargo build --release -p posim
python3 notebooks/_build/regen.py          # specs -> .ipynb, all 109
ls notebooks/*.ipynb | POSIM_NO_BROWSER=1 xargs -P 6 -I{} \
    python3 notebooks/_build/nbrun.py {}   # execute, embed real outputs
python3 notebooks/_build/nbcheck.py        # audit the 7 requirements
```

`_build/` holds the machinery: `nbtext.py` (the invariant prose),
`lang.py` (what every command and field means), `nbgen.py` (derives a
spec from a `.posim` example), `nbbuild.py` (spec → notebook),
`nbrun.py` (executes code cells and embeds outputs), `nbcheck.py`
(audits every requirement), `specs/` (the 109 derived specs), and
`rust_equivalents/` (posim reproductions of the six compiled examples,
each verified against the same analytic anchors).
