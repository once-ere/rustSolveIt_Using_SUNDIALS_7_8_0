# dynamic_notebooks/ — MANIFEST

Provenance and verification record for every notebook in this directory.
Includes the 34 Routh solutions -- 17 from each Part: a first pair
(`routh_double_star_period`, `routh_rectangle_diagonal`), then 16 more
from each book -- each solving a numbered article and checking its
closed-form answer.
`README.md` is the reader's catalogue — what you watch and the physical
anchor. This file is the *audit trail*: where each notebook came from, and
the result of actually running it.

**Verified 2026-08-04.** Every notebook below was executed headlessly with

```bash
POSIM_NO_BROWSER=1 ./target/release/posim --script dynamic_notebooks/<name>.posim
```

and all 59 loaded with **exit status 0 and zero failing cells**. 57 reach
`SCENE CREATE`; the two quantum notebooks deliberately do not (they write an
HTML file instead). Cell counts below are the `In[n]` totals from those runs.

## Every notebook

| notebook | solves / derives from | scene | cells | verified |
|---|---|---|---|---|
| `billiard_box` | collision script 10 (`collision_detection.md` §9) | yes | 12 | ok |
| `billiard_break` | collision script 05 (`collision_detection.md` §9) | yes | 9 | ok |
| `bouncing_ball_restitution` | Rust example `physical_object/examples/bouncing_ball_restitution.rs` | yes | 15 | ok |
| `box_of_shapes` | collision script 11; grammar.md Ex13; user guide S13 | yes | 17 | ok |
| `box_of_shapes_m32` | variant of `box_of_shapes` with a mass-32 point | yes | 17 | ok |
| `charged_in_b_field` | Rust example `charged_in_b_field.rs`; grammar.md Ex3 | yes | 12 | ok |
| `charged_in_e_field` | user guide S4 | yes | 8 | ok |
| `colliding_binary` | collision script 08 (`collision_detection.md` §9) | yes | 7 | ok |
| `double_slit` | quantum: 2-D `QM2` double slit (no scene; writes `double_slit.html`) | **no** | 28 | ok |
| `head_on_exchange` | collision script 01 (`collision_detection.md` §9) | yes | 7 | ok |
| `kepler_orbit` | Rust example `kepler_orbit.rs`; grammar.md Ex1 + Ex11; user guide S2 | yes | 7 | ok |
| `magnetic_spin_up` | grammar.md Ex6 | yes | 6 | ok |
| `newtons_cradle` | Rust example `newtons_cradle.rs`; collision script 04 | yes | 11 | ok |
| `outer_solar_system` | Rust example `outer_solar_system.rs` (live at 50 days/tick) | yes | 15 | ok |
| `restitution_ladder` | collision script 03, adapted — four simultaneous lanes | yes | 17 | ok |
| `routh_double_star_period` | **Routh, Part I, Art. 400** (Dynamics of a Particle, 1898) | yes | 59 | ok |
| `routh_rectangle_diagonal` | **Routh, Part II, Art. 150b Ex. 1** (Advanced Rigid Dynamics, 1905) | yes | 26 | ok |
| `routh_p1_hodograph_circle` | **Routh, Part I, Arts. 394-398** (Dynamics of a Particle, 1898) | yes | 19 | ok |
| `routh_p1_two_trajectories` | **Routh, Part I, Art. 339** (Dynamics of a Particle, 1898) | yes | 29 | ok |
| `routh_p1_lambert_theorem` | **Routh, Part I, Arts. 350-355** (Dynamics of a Particle, 1898) | yes | 32 | ok |
| `routh_p1_collinear_three_body` | **Routh, Part I, Arts. 409-412** (Dynamics of a Particle, 1898) | yes | 42 | ok |
| `routh_p2_poinsot_rolling` | **Routh, Part II, Art. 143** (Advanced Rigid Dynamics, 1905) | yes | 25 | ok |
| `routh_p2_uniaxal_precession` | **Routh, Part II, Arts. 180-183** (Advanced Rigid Dynamics, 1905) | yes | 23 | ok |
| `routh_p2_impulsive_couple` | **Routh, Part II, Art. 146** (Advanced Rigid Dynamics, 1905) | yes | 24 | ok |
| `routh_p2_thin_rod` | **Routh, Part II, Art. 144** (Advanced Rigid Dynamics, 1905) | yes | 21 | ok |
| `routh_p2_rolling_cones` | **Routh, Part II, Arts. 157-159** (Advanced Rigid Dynamics, 1905) | yes | 24 | ok |
| `routh_p2_principal_axes_in_space` | **Routh, Part II, Arts. 176-179** (Advanced Rigid Dynamics, 1905) | yes | 24 | ok |
| `routh_p2_correlated_bodies` | **Routh, Part II, Arts. 192-195** (Advanced Rigid Dynamics, 1905) | yes | 29 | ok |
| `routh_p2_sylvester_time` | **Routh, Part II, Arts. 196-198** (Advanced Rigid Dynamics, 1905) | yes | 31 | ok |
| `routh_p2_polhode_quarter_period` | **Routh, Part II, Art. 150a** (Advanced Rigid Dynamics, 1905) | yes | 22 | ok |
| `routh_p2_period_ratio` | **Routh, Part II, Art. 150b Ex. 2** (Advanced Rigid Dynamics, 1905) | yes | 22 | ok |
| `routh_p1_parabola_of_safety` | **Routh, Part I, Arts. 159-160** (Dynamics of a Particle, 1898) | yes | 45 | ok |
| `routh_p1_apsidal_symmetry` | **Routh, Part I, Arts. 419-420** (Dynamics of a Particle, 1898) | yes | 40 | ok |
| `routh_p2_invariable_line` | **Routh, Part II, Art. 141** (Advanced Rigid Dynamics, 1905) | yes | 23 | ok |
| `routh_p2_separatrix` | **Routh, Part II, Arts. 184-185** (Advanced Rigid Dynamics, 1905) | yes | 36 | ok |
| `routh_p1_kepler_equation` | **Routh, Part I, Arts. 342-346** (Dynamics of a Particle, 1898) | yes | 28 | ok |
| `routh_p1_equilateral_three_body` | **Routh, Part I, Arts. 407-408 + 412** (Dynamics of a Particle, 1898) | yes | 45 | ok |
| `routh_p2_spin_stabilisation` | **Routh, Part II, Art. 156** (Advanced Rigid Dynamics, 1905) | yes | 25 | ok |
| `routh_p2_two_quadrics` | **Routh, Part II, Arts. 140-142** (Advanced Rigid Dynamics, 1905) | yes | 28 | ok |
| `routh_p2_fixed_couple` | **Routh, Part II, Art. 148** (Advanced Rigid Dynamics, 1905) | yes | 23 | ok |
| `routh_p2_mean_axis_instability` | **Routh, Part II, Art. 155** (Advanced Rigid Dynamics, 1905) | yes | 30 | ok |
| `routh_p1_three_projectiles` | **Routh, Part I, Art. 158 Ex. 1** (Dynamics of a Particle, 1898) | yes | 22 | ok |
| `routh_p1_expanding_sphere` | **Routh, Part I, Art. 167 Ex. 1** (Dynamics of a Particle, 1898) | yes | 29 | ok |
| `routh_p1_equal_periods` | **Routh, Part I, Art. 335** (Dynamics of a Particle, 1898) | yes | 34 | ok |
| `routh_p1_oblique_impact` | **Routh, Part I, Art. 89** (Dynamics of a Particle, 1898) | yes | 26 | ok |
| `routh_p1_escape_velocity` | **Routh, Part I, Arts. 312 + 335** (Dynamics of a Particle, 1898) | yes | 30 | ok |
| `routh_p1_sphere_exchange` | **Routh, Part I, Arts. 85 + 87** (Dynamics of a Particle, 1898) | yes | 31 | ok |
| `routh_p1_geometric_progression` | **Routh, Part I, Art. 88** (Dynamics of a Particle, 1898) | yes | 30 | ok |
| `routh_p1_centre_of_gravity` | **Routh, Part I, Art. 92** (Dynamics of a Particle, 1898) | yes | 27 | ok |
| `spin_up` | collision script 06 (`collision_detection.md` §9) | yes | 8 | ok |
| `spinning_target` | collision script 09 (`collision_detection.md` §9) | yes | 8 | ok |
| `static_anchor` | user guide S5, adapted — a free test body beside the pinned anchor | yes | 17 | ok |
| `thin_wall_toi` | collision script 07 (`collision_detection.md` §9) | yes | 14 | ok |
| `three_bodies` | grammar.md Ex5 | yes | 8 | ok |
| `thrown_ball` | grammar.md Ex2 | yes | 12 | ok |
| `tumbling_body` | Rust example `tumbling_body.rs`; grammar.md Ex4; user guide S11 | yes | 16 | ok |
| `tunneling` | quantum: 1-D `QM` barrier (no scene; writes `scatter.html`) | **no** | 32 | ok |
| `two_dumbbells` | collision script 12; grammar.md Ex14; user guide S14 | yes | 12 | ok |
| `unequal_masses` | collision script 02 (`collision_detection.md` §9) | yes | 6 | ok |

## Launching

Any notebook, by bare name:

```bash
tools/posim_notebook <name>
```

or without the launcher on your PATH:

```bash
cargo run -p posim --release -- --notebook dynamic_notebooks/<name>.posim
```

Both open the scene window and leave you at an interactive `In[]` prompt;
press **Start** in the window or type `scene start`. Set `POSIM_NO_BROWSER=1`
to print the URL instead of opening a browser. The port is chosen per run —
read it from the output rather than reusing a previous one.

## Routh solutions — analytic prediction vs measured result

These two notebooks each solve a numbered problem from Routh and check the
book's closed-form answer against the integrator. Both were re-measured on
2026-08-04 with `rtol = 1e-13`, `atol = 1e-15`.

### `routh_double_star_period` — Part I, Art. 400

> "the periodic time of a double star does not depend on the mass of either
> constituent, but on the sum of the masses."

Three binaries share total mass `M = 2` and a circular relative orbit of
separation `r = 1`, differing only in mass ratio. With `G = 1`,

    T = 2*pi*sqrt(r^3 / (G*M)) = pi*sqrt(2) = 4.442882938158366

| case | mass ratio | distance from start after `T` |
|---|---|---|
| A | 1 : 1 | 1.62e-11 |
| B | 3 : 1 | 2.03e-11 |
| C | 19 : 1 | 3.36e-11 |
| control | 19 : 1, **total mass 2.5** | **0.696** — fails to close, as it must |
| control | same, run for its own `2*pi/sqrt(2.5)` | 4.57e-11 |

The control is the point: change the *sum* and `pi*sqrt(2)` no longer closes
the orbit; change only the *ratio* and it still does.

### `routh_rectangle_diagonal` — Part II, Art. 150b, Ex. 1 [Coll. Exam. 1903]

> "A rectangle is set rotating about a diagonal with angular velocity O, prove
> that it will be rotating about the other diagonal after a time
> (2/(O*sqrt(cos 2a))) * Integral_0^{pi/2} dphi/sqrt(1 - sin^2 a sin^2 phi),
> where tan a is the ratio of the smaller to the longer side."

A 2 x 1 lamina, mass 3, exact inertia `diag(0.25, 1, 1.25)`; `tan a = 1/2`, so
`k^2 = sin^2 a = 0.2`, `cos 2a = 0.6`, `K(k) = 1.6596235986105279` and

    T = 2*K / (O*sqrt(cos 2a)) = 4.285129705594264      (O = 1)

Axis read in the body as `w_body = conj(q) * w_world * q`:

| time | `w_body` | expected | error |
|---|---|---|---|
| 0 | `(0.894427191, +0.447213595, 0)` | diagonal 1 | — |
| `T` | `(0.8944271910010517, -0.4472135954998159, 3.1e-13)` | diagonal 2 | 1.2e-12 |
| `2T` | `(0.8944271910001502, +0.4472135954999288, -9.2e-13)` | diagonal 1 | 9.5e-13 |
| `T/2` | `(1.0000000000011926, 3.0e-13, -0.3464101615136378)` | `(1, 0, -sqrt(0.12))` — principal plane, Art. 150a | 1.4e-13 |

`|w| = sqrt(1.12)` at `T/2`: the magnitude of the angular velocity is not
conserved (only `T` and `G` are), so this is genuine polhode motion.

## Adding a notebook

A notebook is complete when all of the following hold. Each is checkable.

1. It solves the problem outright, with the reasoning in comments.
2. It is stand-alone: **every fact needed to understand it is stated inside
   it.** No "see <other>.posim", no deferring to `grammar.md`. A reader
   opening this one file needs nothing else.
3. Its header carries the exact launch syntax, including the directory to run
   from and how to start the scene.
4. It ends at `t = 0`, Stopped, ready — run any verification cells first, then
   `reset` and rebuild before `scene create`, or the window opens mid-motion.
5. It includes a visual simulation, or states plainly why it cannot.
6. It is listed in `README.md`'s catalogue and in the table above.
7. It has been launched, and the scene confirmed to *advance* — `scene status`
   showing `mode = running` with `t` and `steps` climbing, not merely a window
   created.
