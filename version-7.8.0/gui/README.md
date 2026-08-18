# gui/ — thirteen live GUI web pages, one per recorded video scene

Each directory here is a self-contained live GUI for one of the thirteen
scenes in `videos/scenes/`. Every GUI is the same two files:

- **`server.py`** — standard-library Python only. It owns one
  `posim --machine` child running the **exact model from the paired
  scene**, steps it one `step` per frame at real-time pacing, and serves
  the page plus a small JSON API. All integration happens inside the
  pure-Rust SUNDIALS engine; the server never computes physics.
- **`index.html`** — vanilla JS + canvas, no external resources. It polls
  `/api/state` on animation frames and draws whatever the solver
  returned, with Start / Pause / Reset buttons, a speed slider (seconds
  of simulation per frame), and live readouts that check the scene's own
  claims against its closed forms and recorded anchors.

## Running one

```bash
cargo build --release -p posim        # once
python3 gui/<name>/server.py          # prints its URL, serve_forever
```

Each GUI has a fixed port, so any subset can run side by side:

| GUI | port | model | what the readouts verify (measured live) |
|---|---|---|---|
| `piston_crankshaft` | 8895 | HINGE + BALL + BALL + PRISMATIC, 16 rows / 18 DOF | piston x vs the exact x(θ) = a·cosθ + √(L²−a²sin²θ): 7.5e-10 at the sampled instant after ~31 revolutions; \|g\| not drifting |
| `rack_and_pinion` | 8896 | HINGE + PRISMATIC + RACK, 11 rows / 12 DOF | the g/2 fall law y = −0.1t² (4.8e-4, the per-restart figure) and the tooth mesh Δs = rθ under the joint's own unwrap (1.2e-10) |
| `gyroscope_gimbal` | 8897 | three HINGEs on perpendicular axes, 15 rows / 18 DOF | L·ŷ conserved to 4.4e-16; rotor spin I₃ω₃ to 1.8e-5; max tilt 16.32° / yaw 7.40° vs the manifest's 16.33° / 7.43° |
| `cardan_compass` | 8898 | two HINGEs, pendulous bowl, 10 rows / 12 DOF | live-timed periods: pitch 1.8833 vs the recording's 1.883426, roll 2.3138/2.3132 vs 2.313653 |
| `universal_joint` | 8899 | HINGE + UNIVERSAL + ROD, 10 rows / 12 DOF | bend folds to exactly its 53.130° limit; trunnion right angle \|u·w\| = 2.6e-7 and its exact time derivative; the Cardan speed-ratio swing |
| `spinning_top` | 8900 | one BALL joint, axis horizontal, rtol 1e-5 by hand | precession Ω vs the exact Mgr/I₃ω₃ = 1.020408 (rel. err 8.1e-5 over a full turn); no nutation (8.6e-5); spin conserved |
| `ball_joint_chain` | 8901 | four BALLs, 12 rows / 24 DOF, rigid whirl start | \|g\| = \|g_dot\| = 0.0 literally at t = 0; the chain leaves the hinge-forbidden plane by 1.96; tip reach never exceeds 2.0 |
| `cardan_gear` | 8902 | HINGE + HINGE + GEAR, 11 rows / 12 DOF | the Tusi couple: two rim markers ride straight diameters to 2.0e-9; gear row \|θₚ+θ꜀\| wrapped to 2.7e-13 |
| `rod_pendulum_chain` | 8903 | four CONSTRAIN rods, 4 rows / 24 DOF, rtol 1e-6 by hand | worst rod stretch 2.3e-8 over 654 cold restarts; rest start exactly on the manifold |
| `double_pendulum_hinges` | 8904 | two HINGEs about z, 10 rows / 12 DOF | out-of-plane worst 0.00e+00 — exactly zero, the contrast with the ball chain; shared hinge coincides to 5.7e-8 |
| `tumbling_racket` | 8905 | one free cuboid, CVODE Adams — no joints | the Dzhanibekov flips counted live; \|dL\|/\|L\| = 0.00e+00 bit-exact while ∠(ω, L) swings to 33.7° |
| `kepler_ellipse` | 8906 | two gravitating spheres, CVODE Adams — no joints | retraces the dashed analytic a = 2.5, e = 0.6 ellipse to 2.7e-6 over 10 orbits; eccentricity vector still to 2.0e-7 (no precession) |
| `box_of_shapes` | 8907 | BOX 4 + three shapes, CVODE Adams + rootfinding | 98 collision events with \|dE/E\| ≤ 1.6e-8; wall drift 0.00e+00; nothing tunnels |

## The API every server speaks

```
GET  /            the page
GET  /api/state   latest state as JSON (always served, never blocks)
POST /api/start   run          POST /api/stop    pause
POST /api/reset   fresh solver, fresh model, t = 0
POST /api/dt/<x>  seconds of simulated time per frame (0.001 .. 0.05)
```

## Design notes, shared by all thirteen

- **The engine does all the physics.** The server relays commands and
  state over the JSON-lines machine protocol; the page draws bodies from
  their live positions and w-first quaternions. No integrator, no
  approximation, exists outside the Rust engine.
- **Every `step` is a cold restart** (fresh solver, fresh multiplier
  seed), the same discipline as the video recordings — so the readouts
  reproduce the recordings' per-restart accuracy figures, not the tighter
  continuous-run ones, and the two scenes that need non-default
  tolerances (`spinning_top`, `rod_pendulum_chain`) carry them in their
  MODEL with the reason commented.
- **A busy engine can never freeze the page**: `/api/state` serves an
  atomically-swapped snapshot without touching the stepping lock, reset
  kills the child first (unblocking any solve in flight), and the
  stepping thread is restarted if it died.
- **Hidden tabs pause drawing, not physics.** The page polls on
  animation frames, which browsers suspend for hidden views; the engine
  keeps running. Cumulative page-side counters (revolution counts,
  worst-so-far figures) can miss what happened while hidden — Reset
  clears them. Checks that matter are computed from wrapped angles or
  engine reports, which stay correct regardless.
