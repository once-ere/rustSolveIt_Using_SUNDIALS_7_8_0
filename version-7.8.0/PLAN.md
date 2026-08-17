# physical_object_simulator — integration plan and design record

A pure-Rust physics simulator whose **only** numerical-integration
backend is the local [`../sundials_rs`](../sundials_rs) workspace (a
pure-Rust, zero-`unsafe`, zero-external-dependency translation of
SUNDIALS 7.8.0). Constraints honored throughout: zero `unsafe`, zero
external crate dependencies, zero warnings, zero errors; crate roots
allow `non_snake_case`, `non_camel_case_types`, `non_upper_case_globals`.

## 1. The unique union

`physical_object/src/physical_object.rs` defines
`pub struct physical_object`, the **unique union** of the three legacy
types (`src/solver.rs:PointParticle`, `RigidBody.rs:RigidBody`,
`RigidBody3D.rs:RigidBody3D`). Every donor data member and property
appears exactly once:

| union field | PointParticle | RigidBody | RigidBody3D | resolution |
|---|---|---|---|---|
| `id: usize` | `id` | — | — | |
| `mass: f64` | `mass` | `mass` | `mass` | coupled with `inverse_mass` |
| `inverse_mass: f64` | — | — | `inverse_mass` | `m <= 0 → 0` (static body) |
| `charge: f64` | — | `charge` | `charge` | |
| `position: Vec3` | ✓ (nalgebra) | ✓ (nalgebra) | ✓ (`[f64;3]`) | own `linalg::Vec3` |
| `orientation: Quat` | — | `UnitQuaternion` | — | setter renormalizes |
| `momentum: Vec3` | *(via `velocity`)* | ✓ | ✓ | **canonical**; PointParticle's `velocity` member became the derived `get_velocity`/`set_velocity` property (`p = m v`) — no duplicated state |
| `angular_momentum: Vec3` | *(method)* | ✓ (spin) | ✓ | field = spin `L`; PointParticle's orbital method renamed `orbital_angular_momentum(com)` (collision resolution); `total_angular_momentum(com)` = orbital + spin |
| `inertia_tensor: Mat3` | — | `local_inertia_tensor` | `inertia_tensor` | one body-frame tensor; coupled inverse |
| `inverse_inertia_tensor: Mat3` | — | `local_inertia_tensor_inverse` | `inverse_inertia_tensor` | singular → zero (non-rotating) |
| `magnetic_moment_tensor: Mat3` | — | ✓ | — | torque `(R M Rᵀ)·B` |
| `boundary: Boundary` | — (`Point`) | `Box<dyn Boundary>` (SDF trait) | `enum Boundary` | enum keeps the name (`Point`/`Sphere`/`Cuboid`); the SDF trait survives as `Sdf` (verbatim central-difference `surface_normal`) implemented for the enum — no `dyn`, stays `Copy`/`Clone` |

Union methods carried over verbatim (arithmetic order preserved):
`momentum()`, `orbital_angular_momentum(com)`, `laplace_vector(com, k)`
(incl. the `r = 0` guard), `linear_velocity()`, `angular_velocity()`
(world `R I⁻¹ Rᵀ L`), `kinetic_energy()`, `to_local_space` /
`to_world_space`, `signed_distance` / `surface_normal`,
`recompute_inertia_from_boundary()`, plus one constructor per donor:
`new_point`, `new_rigid`, `new_from_shape`.

**Get/set:** every stored field has `get_x()` / `set_x()`; coupled
invariants (mass ↔ inverse, inertia ↔ inverse, velocity ↔ momentum,
quaternion normalization) are maintained by the setters.

`GravitationalSystem` became `PhysicalObjectSystem`
(`particles` → `objects`, `SOFTENING` const → `softening` field,
`center_of_mass` / `total_mass` / `compute_accelerations` verbatim)
plus uniform gravity/E/B fields, per-object external force/torque,
solver settings, and `pack_state`/`unpack_state` for the 13N solver
layout `[pos 3 | momentum 3 | quat 4 | L 3]` per object.

## 2. Integration: sundials_rs only — no exceptions

The legacy hand-rolled steppers (velocity Verlet, semi-implicit Euler,
explicit Euler) were **not** ported. `physical_object/src/integrate.rs`
is the only place time integration happens:

- **CVODE** (`cvode_rs`): Adams (default) or BDF, Newton iteration +
  dense linear solver with DQ Jacobian — the general path for the full
  13N state (equations: `dq/dt = p m⁻¹`; `dp/dt` = softened pairwise
  gravity + `m g` + `qE` + `q v×B` + `F_ext`; `dq̂/dt = ½(0,ω)⊗q̂` with
  `ω = R I⁻¹ Rᵀ L`; `dL/dt = τ_ext + (R M Rᵀ)B`). Quaternions are
  renormalized at output points only, followed by `CVodeReInit`
  (mutating `y` invalidates the multistep history; stats are
  accumulated first since ReInit zeroes the counters).
- **ARKODE SPRKStep** (`arkode_rs`): symplectic fixed-step methods for
  separable systems (translational point-mass dynamics; no B field, no
  torques). The legacy velocity Verlet is exactly
  `ARKODE_SPRK_LEAPFROG_2_2`. Non-separable systems get a descriptive
  error naming the offending feature.
- `physical_object::integrate(force, torque, dt)` (the union of the two
  legacy `integrate` methods) delegates to a one-object CVODE solve.

Callbacks are plain `fn` pointers (`user_data: Option<Box<dyn Any>>`,
downcast inside; failure returns the unrecoverable flag `-1`).

Validation (all automated in `tests/` + `examples/`):
- `outer_solar_system` reproduces the donor
  `cvode_rs/examples/solar_system.rs` run — Pluto's position matches to
  8 decimals over 500 000 days; energy drift 7.8e-7 < 1e-6.
- `kepler_orbit` (e = 0.6): energy, `|L|`, and Laplace-vector drift all
  < 1e-6 on both CVODE and SPRK paths.
- `tumbling_body` (Dzhanibekov): `|ΔL| = 0`, KE drift 5e-9, `|q| ≡ 1`.
- `charged_in_b_field`: measured gyroradius = `m v/(qB)` to 1e-4,
  orbit closes to 7e-9 after one analytic period.

## 3. The command language (replacing direct struct manipulation)

`posim/` replaces the archaic hard-coded `main.rs` driving with a
**lexer → grammar compiler → stack machine** pipeline (flex/bison
style):

- `lexer.rs`: tokens = case-insensitive keywords (`NEW SET GET DEL LIST
  STEP RUN STEPS METHOD ADAMS BDF SPRK ENERGY COM MOMENTUM ANGMOM
  LAPLACE HELP POINT SPHERE CUBOID RESET`), identifiers, numbers
  (scientific), `[ ] { } ( ) , . = + - * /`, `#` comments; every token
  carries its column.
- `parser.rs`: recursive-descent over the EBNF documented in its
  header; compiles to a **postfix instruction program**.
- `vm.rs`: the stack machine — `Value` operand stack (`Num`, `Vec3`,
  `Quat`, `Mat3`, `List`, `Str`, `Unit`) executing `Instr` programs;
  all state access goes through the `physical_object` get/set API.
  `STEP`/`RUN` call the sundials drivers.

```
In[1]:= new sphere { mass = 2, radius = 0.5, charge = -1.5,
                     position = [0, 10, 0], velocity = [1, 0, -0.5] }
Out[1]= obj0
In[2]:= set system.gravity = [0, -9.81, 0]
In[3]:= step 1
Out[3]= t = 1 (advanced by 1, 12 solver steps)
In[4]:= get obj0.position.y
Out[4]= 5.095000000000006
```

- `notebook.rs`: the `In[n]`/`Out[n]` cell REPL; Enter executes (the
  terminal's shift-enter); `%history`, `%edit n <text>`, `%rerun n`,
  `%save`, `%load`, `%reset`, `%quit`. Pure `std` has no raw terminal
  mode, so cursor-key navigation is delegated to the JupyterLab front
  end. `--script` runs files in batch.
- `machine.rs`: `--machine` line-delimited JSON protocol (hand-rolled
  minimal JSON, zero deps) for programmatic get/set/exec/state.

## 4. JupyterLab

`jupyter/` holds a ~100-line `ipykernel` **wrapper kernel** that
subprocesses `posim --machine` (see `jupyter/README.md`). Verified:
protocol test (stdlib only), full ZMQ kernel test via `jupyter_client`,
kernelspec visible in JupyterLab's launcher, and cells executed through
a live JupyterLab server (websocket API + UI).

## 5. Verification gates

`cargo build --workspace` warning-free; `cargo test --workspace` green
(17 lib + 9 conservation + 26 posim tests = 52); 4 examples self-check;
`grep -rn unsafe` over the new crates only hits the `#![forbid]`
attributes; `Cargo.lock` contains exactly the 5 local crates
(physical_object, posim, sundials_core, cvode_rs, arkode_rs) — proof of
zero external dependencies.
